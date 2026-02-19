"""Random baseline: select lobby_size players uniformly at random from pool."""

import numpy as np
from typing import List

from agents.sr_base import SingleRaidAgent
from player import Player
from config import SingleRaidConfig


class SRRandomAgent(SingleRaidAgent):

    def __init__(self, config: SingleRaidConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "SR_Random"

    def select_lobby(self, players: List[Player]) -> List[Player]:
        indices = np.random.choice(len(players), size=self.config.lobby_size, replace=False)
        return [players[i] for i in indices]
