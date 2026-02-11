"""Diverse Agent - Force mix of passive, neutral, aggressive."""

import numpy as np
from typing import List

from agents.base import Agent
from player import Player
from config import ExperimentConfig


class DiverseAgent(Agent):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

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
            target_per_group = lobby_size // 3  # 4 each for lobby_size=12

            for group, idx_name in [(passive, 'p'), (aggressive, 'a'), (neutral, 'n')]:
                idx = {'p': p_idx, 'a': a_idx, 'n': n_idx}[idx_name]
                take = min(target_per_group, len(group) - idx, lobby_size - len(lobby))
                lobby.extend(group[idx:idx+take])
                if idx_name == 'p': p_idx += take
                elif idx_name == 'a': a_idx += take
                else: n_idx += take

            while len(lobby) < lobby_size:
                filled = False
                for group, idx_name in [(passive, 'p'), (aggressive, 'a'), (neutral, 'n')]:
                    idx = {'p': p_idx, 'a': a_idx, 'n': n_idx}[idx_name]
                    if idx < len(group):
                        lobby.append(group[idx])
                        if idx_name == 'p': p_idx += 1
                        elif idx_name == 'a': a_idx += 1
                        else: n_idx += 1
                        filled = True
                        break
                if not filled:
                    break

            if len(lobby) == lobby_size:
                lobbies.append(lobby)

        return lobbies
