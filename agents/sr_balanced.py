"""
Balanced baseline for single-raid.

Selects an equal share of passive (aggression < 0.4), neutral (0.4–0.6),
and aggressive (> 0.6) players. If a tercile is under-represented in the
pool, the deficit is filled from the largest remaining tercile.

Rationale: the reward function rewards BOTH extractions and kills.
Passive players extract; aggressive players kill. Neutral players do
both moderately. A balanced lobby hedges across both components.
"""

import numpy as np
from typing import List

from agents.sr_base import SingleRaidAgent
from player import Player
from config import SingleRaidConfig


class SRBalancedAgent(SingleRaidAgent):

    def __init__(self, config: SingleRaidConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "SR_Balanced"

    def select_lobby(self, players: List[Player]) -> List[Player]:
        passive    = [p for p in players if p.running_aggression < 0.4]
        neutral    = [p for p in players if 0.4 <= p.running_aggression <= 0.6]
        aggressive = [p for p in players if p.running_aggression > 0.6]

        # Shuffle within groups for unbiased selection
        np.random.shuffle(passive)
        np.random.shuffle(neutral)
        np.random.shuffle(aggressive)

        k = self.config.lobby_size
        target = k // 3
        remainder = k - target * 3        # 0, 1, or 2 extra slots

        groups = [passive, neutral, aggressive]
        selected = []
        shortfalls = []

        for group in groups:
            take = min(target, len(group))
            selected.extend(group[:take])
            shortfalls.append(target - take)

        # Fill shortfalls from other groups, largest-first
        deficit = sum(shortfalls) + remainder
        if deficit > 0:
            # Pool of candidates not yet selected
            used_ids = {p.id for p in selected}
            remaining = [p for p in players if p.id not in used_ids]
            np.random.shuffle(remaining)
            selected.extend(remaining[:deficit])

        return selected[:k]
