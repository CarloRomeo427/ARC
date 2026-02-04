"""
SBMM Matchmaker - Skill-Based Matchmaking (group similar aggression).
"""

from typing import List

from .base import Matchmaker
from ..player import Player


class SBMMMatchmaker(Matchmaker):
    """Skill-Based Matchmaking - group similar aggression levels."""
    
    @property
    def name(self) -> str:
        return "SBMM"
    
    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        sorted_players = sorted(players, key=lambda p: p.running_aggression)
        return [sorted_players[i:i+lobby_size] for i in range(0, len(sorted_players), lobby_size)
                if i + lobby_size <= len(sorted_players)]
