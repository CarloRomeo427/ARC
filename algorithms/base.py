"""
Base matchmaker class and common utilities.
"""

from abc import ABC, abstractmethod
from typing import List

from ..player import Player


class Matchmaker(ABC):
    """Abstract base class for matchmaking strategies."""
    
    @abstractmethod
    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        """Create lobbies from player list."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the matchmaker."""
        pass
