"""SBMM Agent - Skill-Based Matchmaking (group similar aggression)."""

from typing import List

from agents.base import Agent
from player import Player
from config import ExperimentConfig


class SBMMAgent(Agent):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "SBMM"

    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        sorted_players = sorted(players, key=lambda p: p.running_aggression)
        return [sorted_players[i:i+lobby_size] for i in range(0, len(sorted_players), lobby_size)
                if i + lobby_size <= len(sorted_players)]
