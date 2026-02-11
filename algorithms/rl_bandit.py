"""
RL Bandit Matchmaker - Learned matchmaking via policy gradient with DeepSets.

Architecture: DeepSets encoder for permutation-equivariant player scoring.
Training: REINFORCE with per-lobby credit assignment and baseline subtraction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List, Optional, Tuple

from .base import Matchmaker


# =============================================================================
# Network
# =============================================================================

class DeepSetsPolicy(nn.Module):
    """
    DeepSets encoder for permutation-equivariant player scoring.
    
    Input:  (N, 7) player features
    Output: (N,)   sorting scores
    
    Architecture:
        1. Per-player MLP: x_i -> h_i  (shared weights)
        2. Global context:  mean(h) -> g  (pool + transform)
        3. Score head:      [h_i; g] -> s_i
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        
        # Per-player encoder (shared across all players)
        self.player_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Global context encoder (operates on pooled representation)
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Score head (per-player, conditioned on global context)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small init for score head output to start near-uniform
        nn.init.orthogonal_(self.score_head[-1].weight, gain=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 7) player features
        Returns:
            scores: (N,) sorting scores
        """
        # Per-player encoding
        h = self.player_encoder(x)              # (N, hidden)
        
        # Global context via mean pooling
        g = self.global_encoder(h.mean(dim=0))   # (hidden,)
        g_expanded = g.unsqueeze(0).expand_as(h)  # (N, hidden)
        
        # Combine local + global, produce scores
        c = torch.cat([h, g_expanded], dim=-1)   # (N, 2*hidden)
        scores = self.score_head(c).squeeze(-1)   # (N,)
        
        return scores


# =============================================================================
# Matchmaker (for inference / integration with ExperimentRunner)
# =============================================================================

class RLBanditMatchmaker(Matchmaker):
    """RL Bandit Matchmaker - wraps trained DeepSetsPolicy for lobby creation."""
    
    def __init__(self, policy: DeepSetsPolicy, device: str = "cpu"):
        self.policy = policy
        self.device = torch.device(device)
        self.policy.to(self.device)
    
    @property
    def name(self) -> str:
        return "RL_Bandit"
    
    def create_lobbies(self, players: list, lobby_size: int) -> List[list]:
        """Create lobbies using learned policy (no exploration noise)."""
        features = self._extract_features(players)
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            scores = self.policy(features_t).cpu().numpy()
        
        sorted_indices = np.argsort(scores)
        sorted_players = [players[i] for i in sorted_indices]
        
        return [
            sorted_players[i:i + lobby_size]
            for i in range(0, len(sorted_players), lobby_size)
            if i + lobby_size <= len(sorted_players)
        ]
    
    @staticmethod
    def _extract_features(players: list) -> np.ndarray:
        """Extract normalized (N, 7) feature matrix from player objects."""
        feats = np.zeros((len(players), 7), dtype=np.float32)
        for i, p in enumerate(players):
            total_raids = getattr(p, 'total_raids', 0) or getattr(p, 'raids', 0) or 1
            feats[i] = [
                p.aggression,
                p.running_aggression,
                getattr(p, 'extraction_rate', 0.0),
                min(getattr(p, 'kills_per_raid', 0.0) / 2.0, 1.0),
                getattr(p, 'deaths_per_raid', 0.0),
                min(getattr(p, 'avg_stash', 0.0) / 100_000, 1.0),
                min(total_raids / 1000, 1.0),
            ]
        return feats


# =============================================================================
# Trainer
# =============================================================================

class RLBanditTrainer:
    """
    REINFORCE trainer with per-lobby credit assignment.
    
    Each episode:
        1. Extract player features → forward pass → mean scores
        2. Add Gaussian noise → sort → create lobbies
        3. Simulate raids → per-lobby Pareto rewards
        4. Policy gradient with per-lobby credit assignment + baseline
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        sigma_init: float = 0.1,
        sigma_min: float = 0.01,
        sigma_decay: float = 0.995,
        baseline_decay: float = 0.95,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy = DeepSetsPolicy(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Exploration
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        
        # Baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        
        # Training config
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def get_matchmaker(self) -> RLBanditMatchmaker:
        """Return a matchmaker wrapping the current policy (for evaluation)."""
        return RLBanditMatchmaker(self.policy, str(self.device))
    
    def select_action(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Forward pass + exploration noise.
        
        Returns:
            actions: (N,) numpy array of noisy scores (used for sorting)
            mean_scores: (N,) tensor with grad (for policy gradient)
            log_probs: (N,) tensor of log π(a_i | s) (for REINFORCE)
        """
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        mean_scores = self.policy(features_t)  # (N,)
        
        dist = Normal(mean_scores, self.sigma)
        actions = dist.sample()          # (N,) — no grad through sample
        log_probs = dist.log_prob(actions)  # (N,) — grad through mean_scores
        
        return actions.detach().cpu().numpy(), mean_scores, log_probs
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        lobby_indices: List[List[int]],
        lobby_rewards: List[float],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Per-lobby credit assignment policy gradient loss.
        
        loss = -(1/K) Σ_k [ Σ_{i ∈ lobby_k} log π(a_i|s) ] × advantage_k
        """
        # Update baseline
        mean_reward = np.mean(lobby_rewards)
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * mean_reward
        )
        
        # Per-lobby policy gradient
        policy_loss = torch.tensor(0.0, device=self.device)
        entropy_total = torch.tensor(0.0, device=self.device)
        
        for indices, reward in zip(lobby_indices, lobby_rewards):
            advantage = reward - self.baseline
            lobby_log_probs = log_probs[indices]
            policy_loss -= lobby_log_probs.sum() * advantage
            entropy_total += (-lobby_log_probs).sum()
        
        num_lobbies = len(lobby_indices)
        policy_loss /= num_lobbies
        entropy_total /= num_lobbies
        
        # Entropy bonus (encourages exploration)
        total_loss = policy_loss - self.entropy_coef * entropy_total
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "entropy": entropy_total.item(),
            "total_loss": total_loss.item(),
            "baseline": self.baseline,
            "sigma": self.sigma,
            "mean_reward": mean_reward,
            "std_reward": float(np.std(lobby_rewards)),
        }
        
        return total_loss, metrics
    
    def update(
        self,
        log_probs: torch.Tensor,
        lobby_indices: List[List[int]],
        lobby_rewards: List[float],
    ) -> dict:
        """Single gradient step."""
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(log_probs, lobby_indices, lobby_rewards)
        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        metrics["grad_norm"] = grad_norm.item()
        
        self.optimizer.step()
        
        # Decay exploration
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        
        return metrics
    
    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "sigma": self.sigma,
            "baseline": self.baseline,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.sigma = checkpoint["sigma"]
        self.baseline = checkpoint["baseline"]
