"""Matchmaking agents."""

from .base import Agent
from .random_agent import RandomAgent
from .sbmm import SBMMAgent
from .diverse import DiverseAgent
from .rl_bandit import RLBanditAgent

AGENT_REGISTRY = {
    'random': RandomAgent,
    'sbmm': SBMMAgent,
    'diverse': DiverseAgent,
    'rl_bandit': RLBanditAgent,
}

__all__ = ['Agent', 'RandomAgent', 'SBMMAgent', 'DiverseAgent', 'RLBanditAgent', 'AGENT_REGISTRY']
