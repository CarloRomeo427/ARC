"""
Polarized Matchmaker - Segregate players by type.
"""

import numpy as np
from typing import List

from .base import Matchmaker
from ..player import Player


class PolarizedMatchmaker(Matchmaker):
    """Segregated matchmaking - passive with passive, aggressive with aggressive."""
    
    @property
    def name(self) -> str:
        return "Polarized"
    
    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        passive = [p for p in players if p.classification == 'passive']
        neutral = [p for p in players if p.classification == 'neutral']
        aggressive = [p for p in players if p.classification == 'aggressive']
        
        np.random.shuffle(passive)
        np.random.shuffle(neutral)
        np.random.shuffle(aggressive)
        
        lobbies = []
        
        # Create pure lobbies from each group
        for group in [passive, aggressive, neutral]:
            for i in range(0, len(group), lobby_size):
                if i + lobby_size <= len(group):
                    lobbies.append(group[i:i+lobby_size])
        
        # Handle remaining players
        remaining = []
        for group in [passive, aggressive, neutral]:
            leftover = len(group) % lobby_size
            if leftover > 0:
                remaining.extend(group[-leftover:])
        
        np.random.shuffle(remaining)
        for i in range(0, len(remaining), lobby_size):
            if i + lobby_size <= len(remaining):
                lobbies.append(remaining[i:i+lobby_size])
        
        return lobbies
