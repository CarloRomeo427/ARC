"""
Matchmaking algorithms.
"""

from .base import Matchmaker
from .random_mm import RandomMatchmaker
from .polarized import PolarizedMatchmaker
from .sbmm import SBMMMatchmaker
from .diverse import DiverseMatchmaker

__all__ = [
    'Matchmaker',
    'RandomMatchmaker',
    'PolarizedMatchmaker',
    'SBMMMatchmaker',
    'DiverseMatchmaker',
]
