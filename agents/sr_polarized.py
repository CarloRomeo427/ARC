"""
Polarized baseline for single-raid.

Selects the lobby_size//2 most passive and lobby_size//2 most aggressive
players from the pool, placing predators and prey in the same raid.

Rationale: this is the configuration that maximises PvP encounters
(aggressive players have passive targets), and should therefore maximise
the kills component of the reward while relying on passive players for
extractions.
"""

import numpy as np
from typing import List

from agents.sr_base import SingleRaidAgent
from player import Player
from config import SingleRaidConfig


class SRPolarizedAgent(SingleRaidAgent):

    def __init__(self, config: SingleRaidConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "SR_Polarized"

    def select_lobby(self, players: List[Player]) -> List[Player]:
        sorted_players = sorted(players, key=lambda p: p.running_aggression)
        k = self.config.lobby_size
        n_passive   = k // 2
        n_aggressive = k - n_passive          # handles odd lobby sizes
        passive   = sorted_players[:n_passive]
        aggressive = sorted_players[-n_aggressive:]
        return passive + aggressive
