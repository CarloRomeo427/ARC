"""
RL Bandit Agent for single-raid matchmaking (v3).

Uses compute_reward_sr (loot + encounters) and entropy annealing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

from agents.sr_base import SingleRaidAgent, GIF_LOG_INTERVAL
from agents.networks import TransformerEncoder, extract_features, sample_plackett_luce
from player import Player, PlayerPool
from sr_reward import compute_reward_sr
from config import SingleRaidConfig


class SingleRaidPolicy(nn.Module):
    def __init__(self, input_dim: int = 7, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 256):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.orthogonal_(self.score_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.encoder(x)).squeeze(-1)


class SRRLBanditAgent(SingleRaidAgent):

    def __init__(self, config: SingleRaidConfig,
                 input_dim: int = 7, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3,
                 lr: float = 3e-4,
                 entropy_coef_start: float = 0.05,
                 entropy_coef_end: float = 0.005,
                 max_grad_norm: float = 1.0):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SingleRaidPolicy(input_dim, d_model, nhead, num_layers).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end   = entropy_coef_end
        self.max_grad_norm      = max_grad_norm
        self.rloo_k             = config.rloo_k
        self.best_reward_avg    = -float("inf")
        self.reward_window: List[float] = []

    @property
    def name(self) -> str:
        return "SR_RL_Bandit"

    def _entropy_coef(self, episode: int) -> float:
        t = min(episode / max(self.config.num_episodes - 1, 1), 1.0)
        return self.entropy_coef_start + t * (self.entropy_coef_end - self.entropy_coef_start)

    def select_lobby(self, players: List[Player]) -> List[Player]:
        feat_t = extract_features(players, self.device)
        with torch.no_grad():
            scores = self.policy(feat_t)
        _, top_indices = scores.topk(self.config.lobby_size)
        return [players[i] for i in top_indices.cpu().tolist()]

    def run_episode(self, pool: PlayerPool, episode_num: int) -> dict:
        queue      = pool.sample_queue(self.config.queue_size)
        N          = len(queue)
        k          = self.rloo_k
        lobby_size = self.config.lobby_size

        feat_t = extract_features(queue, self.device)
        scores = self.policy(feat_t)

        all_lobbies:   List[List[Player]] = []
        all_log_probs: List[torch.Tensor] = []
        all_entropies: List[torch.Tensor] = []

        for _ in range(k):
            available = torch.ones(N, dtype=torch.bool, device=self.device)
            idx, lp, ent = sample_plackett_luce(scores, lobby_size, available)
            all_lobbies.append([queue[i] for i in idx])
            all_log_probs.append(lp)
            all_entropies.append(ent)

        all_results = self._simulate_lobbies(all_lobbies, episode_num)

        rewards: List[float] = []
        all_ris = []
        for results in all_results:
            ri = compute_reward_sr(results)
            rewards.append(ri['reward'])
            all_ris.append(ri)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # RLOO gradient
        reward_sum    = rewards_t.sum()
        policy_loss   = torch.tensor(0.0, device=self.device)
        entropy_total = torch.stack(all_entropies).mean()
        ec            = self._entropy_coef(episode_num)

        for j in range(k):
            baseline  = (reward_sum - rewards_t[j]) / (k - 1)
            advantage = (rewards_t[j] - baseline).detach()
            policy_loss = policy_loss - all_log_probs[j] * advantage

        policy_loss = policy_loss / k
        total_loss  = policy_loss - ec * entropy_total

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Best lobby for stat updates
        best_j       = int(np.argmax(rewards))
        best_lobby   = all_lobbies[best_j]
        best_results = all_results[best_j]
        best_ri      = all_ris[best_j]

        for p in best_lobby:
            r = best_results[p.id]
            p.record_raid(
                r['extracted'], r['stash'], r['damage_dealt'],
                r['damage_received'], int(round(r['kills'])), r['aggression_used'],
            )

        if (episode_num + 1) % self.config.churn_interval == 0:
            pool.churn(self.config.churn_count)

        self._maybe_log_gif(best_lobby, episode_num)

        lobby_aggr = [p.running_aggression for p in best_lobby]

        return {
            'reward':              best_ri['reward'],
            'loot_score':          best_ri['loot_score'],
            'fight_score':         best_ri['fight_score'],
            'total_extracted_loot':best_ri['total_extracted_loot'],
            'total_encounters':    best_ri['total_encounters'],
            'total_kills':         best_ri['total_kills'],
            'total_extractions':   best_ri['total_extractions'],
            'reward_case':         best_ri.get('reward_case', 'unknown'),
            'reward_mean_samples': float(rewards_t.mean().item()),
            'reward_std_samples':  float(rewards_t.std().item()),
            'lobby_aggr_mean':     float(np.mean(lobby_aggr)),
            'lobby_aggr_std':      float(np.std(lobby_aggr)),
            'lobby_n_passive':     sum(1 for p in best_lobby if p.classification=='passive'),
            'lobby_n_neutral':     sum(1 for p in best_lobby if p.classification=='neutral'),
            'lobby_n_aggressive':  sum(1 for p in best_lobby if p.classification=='aggressive'),
            '_lobby_aggr':         lobby_aggr,
            'train/policy_loss':   policy_loss.item(),
            'train/entropy':       entropy_total.item(),
            'train/entropy_coef':  ec,
            'train/total_loss':    total_loss.item(),
            'train/grad_norm':     grad_norm.item(),
        }

    def before_training(self, pool: PlayerPool):
        n = sum(p.numel() for p in self.policy.parameters())
        print(f"  Device: {self.device} | Params: {n:,} | RLOO k={self.rloo_k}")
        print(f"  Entropy coef: {self.entropy_coef_start:.3f} → {self.entropy_coef_end:.4f}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    def after_episode(self, episode: int, metrics: dict):
        self.reward_window.append(metrics['reward'])
        if len(self.reward_window) > 100:
            self.reward_window.pop(0)
        if episode >= 100:
            avg = np.mean(self.reward_window)
            if avg > self.best_reward_avg:
                self.best_reward_avg = avg
                self.save("sr_rl_bandit_best.pt")
        if (episode + 1) % 5000 == 0:
            self.save(f"sr_rl_bandit_ep{episode+1}.pt")

    def after_training(self):
        self.save("sr_rl_bandit_final.pt")

    def save(self, path: str):
        torch.save({'policy_state_dict':    self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])