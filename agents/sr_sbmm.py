"""
SBMM baseline for single-raid: skill-based matchmaking.

Selects lobby_size players whose aggression values cluster most tightly
around the pool median — i.e. picks the players closest to median
aggression, minimising within-lobby skill variance.

Rationale: traditional SBMM groups players of equal skill together to
ensure fair, evenly-matched games.
"""

import numpy as np
from typing import List

from agents.sr_base import SingleRaidAgent
from player import Player
from config import SingleRaidConfig


class SRSBMMAgent(SingleRaidAgent):

    def __init__(self, config: SingleRaidConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "SR_SBMM"

    def select_lobby(self, players: List[Player]) -> List[Player]:
        aggression = np.array([p.running_aggression for p in players])
        median = np.median(aggression)
        distance = np.abs(aggression - median)
        # Take lobby_size players nearest to the pool median
        indices = np.argsort(distance)[:self.config.lobby_size]
        return [players[i] for i in indices]
