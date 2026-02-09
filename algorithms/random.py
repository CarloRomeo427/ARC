"""
Random Matchmaker - Completely random player assignment.
"""

import numpy as np
from typing import List

from .base import Matchmaker
from player import Player


class RandomMatchmaker(Matchmaker):
    """Completely random matchmaking."""
    
    @property
    def name(self) -> str:
        return "Random"
    
    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        shuffled = players.copy()
        np.random.shuffle(shuffled)
        return [shuffled[i:i+lobby_size] for i in range(0, len(shuffled), lobby_size) 
                if i + lobby_size <= len(shuffled)]
