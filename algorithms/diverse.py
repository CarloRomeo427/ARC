"""
Diverse Matchmaker - Force mix of passive, neutral, aggressive.
"""

import numpy as np
from typing import List

from .base import Matchmaker
from player import Player


class DiverseMatchmaker(Matchmaker):
    """Force diversity - mix passive, neutral, aggressive in each lobby."""
    
    @property
    def name(self) -> str:
        return "Diverse"
    
    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        passive = [p for p in players if p.classification == 'passive']
        neutral = [p for p in players if p.classification == 'neutral']
        aggressive = [p for p in players if p.classification == 'aggressive']
        
        np.random.shuffle(passive)
        np.random.shuffle(neutral)
        np.random.shuffle(aggressive)
        
        num_lobbies = len(players) // lobby_size
        lobbies = []
        
        p_idx, n_idx, a_idx = 0, 0, 0
        
        for _ in range(num_lobbies):
            lobby = []
            
            # Target: 4 passive, 4 aggressive, 4 neutral
            for group, idx_name in [(passive, 'p'), (aggressive, 'a'), (neutral, 'n')]:
                idx = {'p': p_idx, 'a': a_idx, 'n': n_idx}[idx_name]
                take = min(4, len(group) - idx, lobby_size - len(lobby))
                lobby.extend(group[idx:idx+take])
                if idx_name == 'p':
                    p_idx += take
                elif idx_name == 'a':
                    a_idx += take
                else:
                    n_idx += take
            
            # Fill if needed
            while len(lobby) < lobby_size:
                filled = False
                for group, idx_name in [(passive, 'p'), (aggressive, 'a'), (neutral, 'n')]:
                    idx = {'p': p_idx, 'a': a_idx, 'n': n_idx}[idx_name]
                    if idx < len(group):
                        lobby.append(group[idx])
                        if idx_name == 'p':
                            p_idx += 1
                        elif idx_name == 'a':
                            a_idx += 1
                        else:
                            n_idx += 1
                        filled = True
                        break
                if not filled:
                    break
            
            if len(lobby) == lobby_size:
                lobbies.append(lobby)
        
        return lobbies
