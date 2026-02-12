"""
RL Bandit Agent - Learned matchmaking via policy gradient with DeepSets.

Overrides run_episode() for custom REINFORCE training with per-lobby credit
assignment. Policy forward/backward passes run on GPU when available;
simulation remains CPU-bound.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple

from agents.base import Agent
from player import Player, PlayerPool
from simulator import compute_reward
from config import ExperimentConfig


# =========================================================================
# Network
# =========================================================================

class DeepSetsPolicy(nn.Module):
    """
    DeepSets encoder for permutation-equivariant player scoring.
    Input:  (N, D) player features
    Output: (N,)   sorting scores
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.player_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.score_head[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.player_encoder(x)                   # (N, H)
        g = self.global_encoder(h.mean(dim=0))        # (H,)
        g_exp = g.unsqueeze(0).expand_as(h)           # (N, H)
        return self.score_head(torch.cat([h, g_exp], dim=-1)).squeeze(-1)  # (N,)


# =========================================================================
# Agent
# =========================================================================

class RLBanditAgent(Agent):
    """RL Bandit matchmaking agent with REINFORCE training."""

    def __init__(
        self,
        config: ExperimentConfig,
        input_dim: int = 7,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        sigma_init: float = 0.1,
        sigma_min: float = 0.01,
        sigma_decay: float = 0.995,
        baseline_decay: float = 0.95,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(config)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy
        self.policy = DeepSetsPolicy(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Exploration
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

        # Baseline
        self.baseline = 0.0
        self.baseline_decay = baseline_decay

        # Training config
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Tracking
        self.best_reward = -float("inf")
        self.reward_window: List[float] = []

        # Pre-allocate feature buffer on device (updated each episode)
        self._feat_buffer: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "RL_Bandit"

    # ------------------------------------------------------------------
    # Feature extraction (returns tensor already on device)
    # ------------------------------------------------------------------

    def _extract_features_tensor(self, players: List[Player]) -> torch.Tensor:
        """Extract (N, 7) features directly into a GPU tensor."""
        n = len(players)
        feats = np.empty((n, 7), dtype=np.float32)
        for i, p in enumerate(players):
            total_raids = p.total_raids or 1
            feats[i] = [
                p.aggression,
                p.running_aggression,
                p.extraction_rate,
                min(p.kills_per_raid / 2.0, 1.0),
                p.deaths_per_raid,
                min(p.avg_stash / 100_000, 1.0),
                min(total_raids / 1000, 1.0),
            ]
        # Single CPU→GPU transfer
        return torch.from_numpy(feats).to(self.device, non_blocking=True)

    # ------------------------------------------------------------------
    # Lobby creation (inference, no grad)
    # ------------------------------------------------------------------

    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        """Deterministic lobby creation for evaluation (no exploration noise)."""
        feat_t = self._extract_features_tensor(players)
        with torch.no_grad():
            scores = self.policy(feat_t).cpu().numpy()
        sorted_idx = np.argsort(scores)
        sorted_p = [players[i] for i in sorted_idx]
        return [sorted_p[i:i+lobby_size] for i in range(0, len(sorted_p), lobby_size)
                if i + lobby_size <= len(sorted_p)]

    # ------------------------------------------------------------------
    # Training episode (overrides base)
    # ------------------------------------------------------------------

    def run_episode(self, pool: PlayerPool, episode_num: int) -> dict:
        # Sample who's online this episode
        players = pool.sample_queue(self.config.queue_size)
        lobby_size = self.config.lobby_size

        # --- Forward pass on GPU ---
        feat_t = self._extract_features_tensor(players)
        mean_scores = self.policy(feat_t)  # (N,) on device, with grad
        dist = Normal(mean_scores, self.sigma)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)  # (N,) on device, with grad

        # Transfer scores to CPU for sorting (single GPU→CPU transfer)
        actions_np = actions.detach().cpu().numpy()
        sorted_indices = np.argsort(actions_np)
        sorted_players = [players[i] for i in sorted_indices]

        num_lobbies = len(players) // lobby_size
        lobbies = [sorted_players[k*lobby_size:(k+1)*lobby_size] for k in range(num_lobbies)]

        # Map lobby→original indices for credit assignment
        lobby_original_indices = [
            sorted_indices[k*lobby_size:(k+1)*lobby_size].tolist()
            for k in range(num_lobbies)
        ]

        # --- Simulate raids (CPU) ---
        classifications = {p.id: p.classification for p in players}
        lobby_rewards = []
        all_passive, all_aggressive = [], []
        total_ext, total_deaths, total_kills = 0, 0, 0

        for raid_idx, lobby in enumerate(lobbies):
            raid_seed = episode_num * 10000 + raid_idx * 100
            results = self.raid_runner.run_averaged_raid(lobby, raid_seed)
            results_list = [{'persistent_id': pid, **data} for pid, data in results.items()]
            reward_info = compute_reward(results_list, classifications)

            lobby_rewards.append(reward_info['pareto'])
            all_passive.append(reward_info['passive_score'])
            all_aggressive.append(reward_info['aggressive_score'])

            for pid, data in results.items():
                total_ext += int(data['extracted'])
                total_deaths += int(not data['extracted'])
                total_kills += data['kills']

            for p in lobby:
                r = results[p.id]
                p.record_raid(r['extracted'], r['stash'], r['damage_dealt'],
                              r['damage_received'], int(round(r['kills'])), r['aggression_used'])

        # --- Policy gradient update (GPU) ---
        mean_reward = np.mean(lobby_rewards)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_reward

        # Build per-lobby loss entirely on device
        policy_loss = torch.tensor(0.0, device=self.device)
        entropy_total = torch.tensor(0.0, device=self.device)

        for indices, reward in zip(lobby_original_indices, lobby_rewards):
            advantage = reward - self.baseline
            lp = log_probs[indices]
            policy_loss = policy_loss - lp.sum() * advantage
            entropy_total = entropy_total + (-lp).sum()

        policy_loss = policy_loss / num_lobbies
        entropy_total = entropy_total / num_lobbies
        total_loss = policy_loss - self.entropy_coef * entropy_total

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Decay exploration
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        return {
            'avg_reward': mean_reward,
            'std_reward': float(np.std(lobby_rewards)),
            'avg_passive_score': np.mean(all_passive),
            'avg_aggressive_score': np.mean(all_aggressive),
            'total_extractions': total_ext,
            'total_deaths': total_deaths,
            'total_kills': total_kills,
            'train/policy_loss': policy_loss.item(),
            'train/entropy': entropy_total.item(),
            'train/total_loss': total_loss.item(),
            'train/grad_norm': grad_norm.item(),
            'train/baseline': self.baseline,
            'train/sigma': self.sigma,
            'zero_reward_lobbies': sum(1 for r in lobby_rewards if r == 0),
        }

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def before_training(self, pool: PlayerPool):
        print(f"  Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    def after_episode(self, episode: int, metrics: dict):
        self.reward_window.append(metrics['avg_reward'])
        if len(self.reward_window) > 100:
            self.reward_window.pop(0)
        running_avg = np.mean(self.reward_window)
        if running_avg > self.best_reward and episode >= 100:
            self.best_reward = running_avg
            self.save("rl_bandit_best.pt")
        if (episode + 1) % 1000 == 0:
            self.save(f"rl_bandit_ep{episode+1}.pt")

    def after_training(self):
        self.save("rl_bandit_final.pt")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sigma': self.sigma,
            'baseline': self.baseline,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.sigma = ckpt['sigma']
        self.baseline = ckpt['baseline']