"""
Autoregressive Pointer Transformer Agent (RL_Transformer_AR).

Architecture:
  1. Transformer encoder processes all N players once → contextualized embeddings
  2. At each step t (forming raid t):
     - Initialize context = step_embedding + mean(available_embeddings)
     - For pick k = 0..11:
       a. Pointer attention: query=context, keys=embeddings → logits
       b. Sample player from Categorical(logits) over available players
       c. Update context via GRU(selected_embedding, context)
       d. Mask selected player
  3. After all raids formed: simulate with common random numbers
  4. Terminal reward = mean reward across ALL raids
  5. PPO update with trajectory replay

Key difference from Score-and-Select: the GRU context updates after each
pick within a raid, so the 7th player selected depends on the previous 6.
This is more expressive (can model within-raid synergies) but ~12x more
decoder passes per raid.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import List, Tuple

from agents.base import Agent
from agents.networks import TransformerEncoder, Critic, extract_features, TrajectoryBuffer
from player import Player, PlayerPool
from simulator import compute_reward
from config import ExperimentConfig


# =========================================================================
# Pointer Decoder
# =========================================================================

class PointerDecoder(nn.Module):
    """
    Autoregressive decoder using pointer attention.

    At each pick, the context vector queries all available player embeddings
    via scaled dot-product attention. After selecting, the context is updated
    with a GRU cell conditioned on the selected player's embedding.
    """

    def __init__(self, d_model: int = 128, num_raids: int = 10, clip_logits: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.clip_logits = clip_logits

        self.step_emb = nn.Embedding(num_raids + 1, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.context_update = nn.GRUCell(d_model, d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_context(self, embeddings: torch.Tensor, available_mask: torch.Tensor,
                     step: int) -> torch.Tensor:
        """Initialize context for a new raid from step embedding + pool summary."""
        avail_embs = embeddings[available_mask]
        if avail_embs.shape[0] > 0:
            pool_summary = avail_embs.mean(dim=0)
        else:
            pool_summary = torch.zeros(self.d_model, device=embeddings.device)
        step_vec = self.step_emb(torch.tensor(step, device=embeddings.device))
        return pool_summary + step_vec

    def compute_logits(self, context: torch.Tensor, embeddings: torch.Tensor,
                       available_mask: torch.Tensor) -> torch.Tensor:
        """Compute pointer attention logits over available players."""
        query = self.query_proj(context)
        keys = self.key_proj(embeddings)
        logits = (keys @ query) / math.sqrt(self.d_model)
        logits = self.clip_logits * torch.tanh(logits)
        logits[~available_mask] = float('-inf')
        return logits

    def update_context(self, context: torch.Tensor, selected_emb: torch.Tensor) -> torch.Tensor:
        """Update context after selecting a player via GRU."""
        return self.context_update(selected_emb.unsqueeze(0), context.unsqueeze(0)).squeeze(0)


# =========================================================================
# Sampling helpers
# =========================================================================

def sample_pointer_raid(decoder: PointerDecoder, embeddings: torch.Tensor,
                        available_mask: torch.Tensor, step: int,
                        lobby_size: int) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Autoregressively select lobby_size players for one raid.

    Returns:
        selected_indices: ordered list of selected player indices
        total_log_prob: sum of log-probs across all picks
        total_entropy: sum of entropies across all picks
    """
    context = decoder.init_context(embeddings, available_mask, step)
    mask = available_mask.clone()
    selected = []
    total_log_prob = torch.tensor(0.0, device=embeddings.device)
    total_entropy = torch.tensor(0.0, device=embeddings.device)

    for _ in range(lobby_size):
        logits = decoder.compute_logits(context, embeddings, mask)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()

        total_log_prob = total_log_prob + dist.log_prob(idx)
        total_entropy = total_entropy + dist.entropy()

        selected.append(idx.item())
        mask[idx] = False
        context = decoder.update_context(context, embeddings[idx])

    return selected, total_log_prob, total_entropy


def recompute_pointer_log_prob(decoder: PointerDecoder, embeddings: torch.Tensor,
                               available_mask: torch.Tensor, step: int,
                               selected_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replay pointer selections with updated weights for PPO."""
    context = decoder.init_context(embeddings, available_mask, step)
    mask = available_mask.clone()
    total_log_prob = torch.tensor(0.0, device=embeddings.device)
    total_entropy = torch.tensor(0.0, device=embeddings.device)

    for idx in selected_indices:
        logits = decoder.compute_logits(context, embeddings, mask)
        dist = torch.distributions.Categorical(logits=logits)
        idx_t = torch.tensor(idx, device=embeddings.device)

        total_log_prob = total_log_prob + dist.log_prob(idx_t)
        total_entropy = total_entropy + dist.entropy()

        mask[idx] = False
        context = decoder.update_context(context, embeddings[idx])

    return total_log_prob, total_entropy


# =========================================================================
# Agent
# =========================================================================

class RLTransformerARAgent(Agent):
    """RL Transformer Autoregressive Pointer agent with PPO training."""

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

        self.decoder = PointerDecoder(d_model, num_raids).to(self.device)
        self.critic = Critic(d_model, num_raids).to(self.device)

        # Separate LR groups: slow encoder, fast decoder/critic
        encoder_lr = lr * encoder_lr_scale if pretrained_encoder_path else lr
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.decoder.parameters(), 'lr': lr},
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
        return "RL_Transformer_AR"

    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        """Deterministic inference: greedy decoding (argmax at each pick)."""
        feat_t = extract_features(players, self.device)
        with torch.no_grad():
            embeddings = self.encoder(feat_t)
        num_raids = len(players) // lobby_size
        available = torch.ones(len(players), dtype=torch.bool, device=self.device)
        lobbies = []

        for step in range(num_raids):
            context = self.decoder.init_context(embeddings, available, step)
            mask = available.clone()
            selected = []

            for _ in range(lobby_size):
                logits = self.decoder.compute_logits(context, embeddings, mask)
                idx = logits.argmax().item()
                selected.append(idx)
                mask[idx] = False
                context = self.decoder.update_context(context, embeddings[idx])

            lobbies.append([players[i] for i in selected])
            available[selected] = False

        return lobbies

    def run_episode(self, pool: PlayerPool, episode_num: int) -> dict:
        players = pool.sample_queue(self.config.queue_size)
        lobby_size = self.config.lobby_size
        num_raids = len(players) // lobby_size

        # --- Phase 1: Form all raids autoregressively ---
        feat_t = extract_features(players, self.device)
        embeddings = self.encoder(feat_t)

        available = torch.ones(len(players), dtype=torch.bool, device=self.device)
        lobbies = []
        self.buffer.clear()

        for step in range(num_raids):
            step_avail = available.clone()
            value = self.critic(embeddings.detach(), step_avail, step)
            selected, log_prob, entropy = sample_pointer_raid(
                self.decoder, embeddings, step_avail, step, lobby_size
            )
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
                new_value = self.critic(new_embeddings.detach(), step_avail, step)
                new_lp, new_ent = recompute_pointer_log_prob(
                    self.decoder, new_embeddings, step_avail, step, trans.selected_indices
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
                list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.critic.parameters()),
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
                   sum(p.numel() for p in self.decoder.parameters()) + \
                   sum(p.numel() for p in self.critic.parameters())
        print(f"  Device: {self.device} | Parameters: {n_params:,}")

    def after_episode(self, episode: int, metrics: dict):
        self.reward_window.append(metrics['avg_reward'])
        if len(self.reward_window) > 100:
            self.reward_window.pop(0)
        running_avg = np.mean(self.reward_window)
        if running_avg > self.best_reward and episode >= 100:
            self.best_reward = running_avg
            self.save("rl_transformer_ar_best.pt")
        if (episode + 1) % 1000 == 0:
            self.save(f"rl_transformer_ar_ep{episode+1}.pt")

    def after_training(self):
        self.save("rl_transformer_ar_final.pt")

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self.critic.load_state_dict(ckpt['critic'])