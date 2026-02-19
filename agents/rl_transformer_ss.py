"""
Score-and-Select Transformer Agent (RL_Transformer_SS).

Architecture:
  1. Transformer encoder processes all N players once → contextualized embeddings
  2. At each step t (forming raid t):
     - Context = step_embedding + mean(available_embeddings)
     - Score each available player via MLP([embedding; context]) → scalar
     - Select 12 players using Plackett-Luce (sequential softmax w/o replacement)
     - Mask selected players
  3. After all raids formed: simulate with common random numbers
  4. Terminal reward = mean reward across ALL raids
  5. PPO update with GAE (gamma=1.0, lam=0.95)

Terminal reward forces the agent to plan ahead: it cannot cherry-pick
good matches early and ignore later raids, because the reward is the
average quality across all lobbies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import List

from agents.base import Agent
from agents.networks import (
    TransformerEncoder, Critic, extract_features,
    TrajectoryBuffer, sample_plackett_luce, recompute_plackett_luce_log_prob,
)
from player import Player, PlayerPool
from simulator import compute_reward
from config import ExperimentConfig


class ScoreSelectPolicy(nn.Module):
    """
    Policy head: scores each player given encoder output and step context.
    Score determines selection probability via Plackett-Luce.
    """

    def __init__(self, d_model: int = 128, num_raids: int = 10):
        super().__init__()
        self.step_emb = nn.Embedding(num_raids + 1, d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Small init → near-uniform initial selection
        nn.init.orthogonal_(self.score_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, embeddings: torch.Tensor, available_mask: torch.Tensor,
                step: int) -> torch.Tensor:
        """
        Returns (N,) scores. Unavailable players get -inf.
        """
        avail_embs = embeddings[available_mask]
        if avail_embs.shape[0] > 0:
            pool_ctx = avail_embs.mean(dim=0)
        else:
            pool_ctx = torch.zeros(embeddings.shape[1], device=embeddings.device)

        step_ctx = self.step_emb(torch.tensor(step, device=embeddings.device))
        context = pool_ctx + step_ctx  # (d_model,)

        context_exp = context.unsqueeze(0).expand(embeddings.shape[0], -1)
        scores = self.score_head(torch.cat([embeddings, context_exp], dim=-1)).squeeze(-1)
        scores[~available_mask] = float('-inf')
        return scores


class RLTransformerSSAgent(Agent):
    """RL Transformer Score-and-Select agent with PPO training."""

    def __init__(self, config: ExperimentConfig, input_dim: int = 7, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, lr: float = 3e-4,
                 gamma: float = 1.0, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, ppo_epochs: int = 4,
                 pretrained_encoder_path: str = None, encoder_lr_scale: float = 0.1):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_raids = config.queue_size // config.lobby_size

        # Load pretrained encoder if available (overrides arch params)
        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            ckpt = torch.load(pretrained_encoder_path, map_location=self.device)
            d_model = ckpt['d_model']
            nhead = ckpt['nhead']
            num_layers = ckpt['num_layers']
            dim_ff = ckpt['dim_feedforward']
            self.encoder = TransformerEncoder(
                input_dim, d_model, nhead, num_layers, dim_ff
            ).to(self.device)
            self.encoder.load_state_dict(ckpt['encoder_state_dict'])
            print(f"  Loaded pretrained encoder: d={d_model}, h={nhead}, L={num_layers}, ff={dim_ff}")
        else:
            self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers).to(self.device)

        self.policy = ScoreSelectPolicy(d_model, num_raids).to(self.device)
        self.critic = Critic(d_model, num_raids).to(self.device)

        # Separate LR groups: slow encoder, fast policy/critic
        encoder_lr = lr * encoder_lr_scale if pretrained_encoder_path else lr
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr},
        ])

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        self.buffer = TrajectoryBuffer()
        self.best_reward = -float("inf")
        self.reward_window: List[float] = []

    @property
    def name(self) -> str:
        return "RL_Transformer_SS"

    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        """Deterministic inference: greedy top-k selection."""
        feat_t = extract_features(players, self.device)
        with torch.no_grad():
            embeddings = self.encoder(feat_t)
        num_raids = len(players) // lobby_size
        available = torch.ones(len(players), dtype=torch.bool, device=self.device)
        lobbies = []
        for step in range(num_raids):
            with torch.no_grad():
                scores = self.policy(embeddings, available, step)
            _, top_indices = scores.topk(lobby_size)
            selected = top_indices.cpu().tolist()
            lobbies.append([players[i] for i in selected])
            available[selected] = False
        return lobbies

    def run_episode(self, pool: PlayerPool, episode_num: int) -> dict:
        players = pool.sample_queue(self.config.queue_size)
        lobby_size = self.config.lobby_size
        num_raids = len(players) // lobby_size

        # --- Phase 1: Form all raids sequentially ---
        feat_t = extract_features(players, self.device)
        embeddings = self.encoder(feat_t)  # (N, d_model), with grad

        available = torch.ones(len(players), dtype=torch.bool, device=self.device)
        lobbies = []
        self.buffer.clear()

        for step in range(num_raids):
            step_avail = available.clone()
            scores = self.policy(embeddings, step_avail, step)
            value = self.critic(embeddings.detach(), step_avail, step)
            selected, log_prob, entropy = sample_plackett_luce(scores, lobby_size, step_avail)

            self.buffer.store(log_prob, value, entropy, available, selected)
            lobbies.append([players[i] for i in selected])
            for idx in selected:
                available[idx] = False

        # --- Phase 2: Simulate all raids with common random numbers ---
        all_results = self.raid_runner.run_lobbies(lobbies, episode_num)

        all_rewards = []
        total_ext, total_deaths, total_kills = 0, 0, 0
        total_looted = 0.0

        for lobby, results in zip(lobbies, all_results):
            results_list = list(results.values())
            reward_info = compute_reward(results_list, lobby_size)

            all_rewards.append(reward_info['reward'])
            total_looted += reward_info['extracted_loot']
            total_ext += reward_info['total_extractions']
            total_kills += reward_info['total_kills']

            for pid, data in results.items():
                total_deaths += int(not data['extracted'])

            for p in lobby:
                r = results[p.id]
                p.record_raid(r['extracted'], r['stash'], r['damage_dealt'],
                              r['damage_received'], int(round(r['kills'])), r['aggression_used'])

        # Terminal reward = mean across ALL raids
        terminal_reward = float(np.mean(all_rewards))
        self.buffer.set_terminal_reward(terminal_reward)

        # --- Phase 3: PPO update ---
        advantages, returns = self.buffer.compute_gae(self.gamma, self.gae_lambda)
        old_log_probs = torch.stack([t.log_prob for t in self.buffer.transitions]).detach()
        adv_t = torch.stack(advantages).detach()
        ret_t = torch.stack(returns).detach()

        if adv_t.shape[0] > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        policy_loss_total, value_loss_total, entropy_total = 0.0, 0.0, 0.0

        for epoch in range(self.ppo_epochs):
            new_embeddings = self.encoder(feat_t)
            new_log_probs_list, new_values_list, new_entropies_list = [], [], []
            avail = torch.ones(len(players), dtype=torch.bool, device=self.device)

            for step, trans in enumerate(self.buffer.transitions):
                # Clone avail before use — in-place updates below must not
                # mutate tensors that participated in this step's graph
                step_avail = avail.clone()
                new_scores = self.policy(new_embeddings, step_avail, step)
                new_value = self.critic(new_embeddings.detach(), step_avail, step)
                new_lp, new_ent = recompute_plackett_luce_log_prob(
                    new_scores, trans.selected_indices, step_avail
                )
                new_log_probs_list.append(new_lp)
                new_values_list.append(new_value)
                new_entropies_list.append(new_ent)
                for idx in trans.selected_indices:
                    avail[idx] = False

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            new_entropies = torch.stack(new_entropies_list)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (new_values - ret_t).pow(2).mean()
            entropy_loss = -new_entropies.mean()

            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.policy.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            entropy_total += new_entropies.mean().item()

        n_ep = self.ppo_epochs
        return {
            'avg_reward': terminal_reward,
            'std_reward': float(np.std(all_rewards)),
            'total_extractions': total_ext,
            'total_deaths': total_deaths,
            'total_kills': total_kills,
            'total_looted': total_looted,
            'train/policy_loss': policy_loss_total / n_ep,
            'train/value_loss': value_loss_total / n_ep,
            'train/entropy': entropy_total / n_ep,
            'train/grad_norm': grad_norm.item(),
        }

    def before_training(self, pool: PlayerPool):
        n_params = sum(p.numel() for p in self.encoder.parameters()) + \
                   sum(p.numel() for p in self.policy.parameters()) + \
                   sum(p.numel() for p in self.critic.parameters())
        print(f"  Device: {self.device} | Parameters: {n_params:,}")

    def after_episode(self, episode: int, metrics: dict):
        self.reward_window.append(metrics['avg_reward'])
        if len(self.reward_window) > 100:
            self.reward_window.pop(0)
        running_avg = np.mean(self.reward_window)
        if running_avg > self.best_reward and episode >= 100:
            self.best_reward = running_avg
            self.save("rl_transformer_ss_best.pt")
        if (episode + 1) % 1000 == 0:
            self.save(f"rl_transformer_ss_ep{episode+1}.pt")

    def after_training(self):
        self.save("rl_transformer_ss_final.pt")

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy.load_state_dict(ckpt['policy'])
        self.critic.load_state_dict(ckpt['critic'])