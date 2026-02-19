"""
Diverse baseline for single-raid: maximise aggression spread.

Sorts the pool by aggression and picks lobby_size players at evenly-spaced
quantile positions, guaranteeing maximum diversity of playstyle within the
selected lobby.

Rationale: if the optimal raid composition requires multiple archetypes,
forced diversity is a strong heuristic.
"""

import numpy as np
from typing import List

from agents.sr_base import SingleRaidAgent
from player import Player
from config import SingleRaidConfig


class SRDiverseAgent(SingleRaidAgent):

    def __init__(self, config: SingleRaidConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "SR_Diverse"

    def select_lobby(self, players: List[Player]) -> List[Player]:
        sorted_players = sorted(players, key=lambda p: p.running_aggression)
        n = len(sorted_players)
        k = self.config.lobby_size
        # Evenly-spaced quantile indices across the sorted pool
        indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        return [sorted_players[i] for i in indices]
