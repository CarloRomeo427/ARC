"""
Shared neural network components for Transformer-based RL matchmaking agents.

Contains:
  - TransformerEncoder: permutation-equivariant set encoder (no positional encoding)
  - Critic: value function V(state) for PPO baseline
  - Feature extraction from Player objects
  - GAE computation and PPO trajectory storage
  - Plackett-Luce sampling (used by Score-and-Select agent)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from player import Player


# =========================================================================
# Transformer Encoder (shared by both agents)
# =========================================================================

class TransformerEncoder(nn.Module):
    """
    Permutation-equivariant Transformer encoder for player sets.

    No positional encoding — players have no natural order.
    Self-attention captures pairwise player interactions, which is exactly
    what matchmaking needs: a player's value depends on who else is in
    the pool.
    """

    def __init__(self, input_dim: int = 7, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (N, D) player features
            mask: (N,) bool, True = IGNORE this player (padding mask)
        Returns:
            (N, d_model) contextualized player embeddings
        """
        h = self.input_proj(x).unsqueeze(0)  # (1, N, d_model)
        if mask is not None:
            h = self.encoder(h, src_key_padding_mask=mask.unsqueeze(0))
        else:
            h = self.encoder(h)
        return h.squeeze(0)  # (N, d_model)


# =========================================================================
# Critic (shared by both agents)
# =========================================================================

class Critic(nn.Module):
    """
    Value function V(s_t) = expected total reward given remaining players + step.
    State representation: [mean(available_embeddings), step_embedding]
    """

    def __init__(self, d_model: int = 128, num_raids: int = 10):
        super().__init__()
        self.step_emb = nn.Embedding(num_raids + 1, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, embeddings: torch.Tensor, available_mask: torch.Tensor,
                step: int) -> torch.Tensor:
        """
        Args:
            embeddings: (N, d_model)
            available_mask: (N,) bool, True = available
            step: current raid step (0-indexed)
        Returns:
            scalar value estimate
        """
        avail_embs = embeddings[available_mask]
        if avail_embs.shape[0] == 0:
            pool_summary = torch.zeros(embeddings.shape[1], device=embeddings.device)
        else:
            pool_summary = avail_embs.mean(dim=0)
        step_vec = self.step_emb(torch.tensor(step, device=embeddings.device))
        state = torch.cat([pool_summary, step_vec])
        return self.value_head(state).squeeze(-1)


# =========================================================================
# Feature extraction
# =========================================================================

def extract_features(players: List[Player], device: torch.device) -> torch.Tensor:
    """Extract (N, 7) normalized features from Player objects → GPU tensor."""
    n = len(players)
    feats = np.empty((n, 7), dtype=np.float32)
    for i, p in enumerate(players):
        feats[i] = [
            p.aggression,
            p.running_aggression,
            p.extraction_rate,
            min(p.kills_per_raid / 2.0, 1.0),
            p.deaths_per_raid,
            min(p.avg_stash / 100_000, 1.0),
            min((p.total_raids or 1) / 1000, 1.0),
        ]
    return torch.from_numpy(feats).to(device, non_blocking=True)


# =========================================================================
# PPO Trajectory Buffer
# =========================================================================

@dataclass
class Transition:
    """Single timestep (one raid formation) in the trajectory."""
    log_prob: torch.Tensor          # Sum of log-probs for selecting this lobby
    value: torch.Tensor             # V(s_t) at this step
    entropy: torch.Tensor           # Sum of entropies for this selection
    available_mask: torch.Tensor    # (N,) bool mask at this step
    selected_indices: List[int]     # Which player indices were selected


class TrajectoryBuffer:
    """Stores one episode trajectory for PPO updates."""

    def __init__(self):
        self.transitions: List[Transition] = []
        self.terminal_reward: float = 0.0

    def store(self, log_prob: torch.Tensor, value: torch.Tensor, entropy: torch.Tensor,
              available_mask: torch.Tensor, selected_indices: List[int]):
        self.transitions.append(Transition(
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            available_mask=available_mask.clone(),
            selected_indices=list(selected_indices),
        ))

    def set_terminal_reward(self, reward: float):
        self.terminal_reward = reward

    def compute_gae(self, gamma: float = 1.0, lam: float = 0.95) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute GAE advantages and returns.

        Terminal reward only: r_t = 0 for t < T-1, r_{T-1} = R_total.
        gamma=1.0 because there's no discounting within a single matchmaking
        episode (all raids matter equally).
        """
        T = len(self.transitions)
        rewards = [0.0] * T
        rewards[-1] = self.terminal_reward

        values = [t.value.detach() for t in self.transitions]
        values.append(torch.tensor(0.0, device=values[0].device))  # V(s_T) = 0

        advantages = []
        gae = torch.tensor(0.0, device=values[0].device)

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        returns = [adv + values[t] for t, adv in enumerate(advantages)]
        return advantages, returns

    def clear(self):
        self.transitions.clear()
        self.terminal_reward = 0.0

    def __len__(self):
        return len(self.transitions)


# =========================================================================
# Plackett-Luce sampling (used by Score-and-Select agent)
# =========================================================================

def sample_plackett_luce(scores: torch.Tensor, k: int,
                         available_mask: torch.Tensor) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Sample k items without replacement using Plackett-Luce (sequential softmax).

    Defines a proper probability distribution over ordered k-subsets.
    Scores are fixed (computed once per step), unlike the autoregressive
    pointer where context updates after each pick.

    Args:
        scores: (N,) raw scores for all players
        k: number to select (lobby_size)
        available_mask: (N,) bool, True = available
    Returns:
        selected: list of k selected indices
        total_log_prob: sum of log-probs across all k selections
        total_entropy: sum of entropies across all k selections
    """
    selected = []
    total_log_prob = torch.tensor(0.0, device=scores.device)
    total_entropy = torch.tensor(0.0, device=scores.device)
    mask = available_mask.clone()

    for _ in range(k):
        masked_scores = scores.clone()
        masked_scores[~mask] = float('-inf')
        log_probs = F.log_softmax(masked_scores, dim=-1)
        probs = log_probs.exp()

        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()

        total_log_prob = total_log_prob + dist.log_prob(idx)
        total_entropy = total_entropy + dist.entropy()

        selected.append(idx.item())
        mask[idx] = False

    return selected, total_log_prob, total_entropy


def recompute_plackett_luce_log_prob(scores: torch.Tensor, selected_indices: List[int],
                                     available_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute log_prob and entropy for a stored action with updated policy weights."""
    total_log_prob = torch.tensor(0.0, device=scores.device)
    total_entropy = torch.tensor(0.0, device=scores.device)
    mask = available_mask.clone()

    for idx in selected_indices:
        masked_scores = scores.clone()
        masked_scores[~mask] = float('-inf')
        log_probs = F.log_softmax(masked_scores, dim=-1)
        probs = log_probs.exp()

        dist = torch.distributions.Categorical(probs=probs)
        idx_t = torch.tensor(idx, device=scores.device)
        total_log_prob = total_log_prob + dist.log_prob(idx_t)
        total_entropy = total_entropy + dist.entropy()

        mask[idx] = False

    return total_log_prob, total_entropy