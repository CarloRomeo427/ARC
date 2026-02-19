"""Matchmaking agents."""

# ── Multi-raid agents (original) ──────────────────────────────────────────
from .base import Agent
from .random_agent import RandomAgent
from .sbmm import SBMMAgent
from .diverse import DiverseAgent
from .rl_bandit import RLBanditAgent
from .rl_transformer_ss import RLTransformerSSAgent
from .rl_transformer_ar import RLTransformerARAgent

AGENT_REGISTRY = {
    'random':           RandomAgent,
    'sbmm':             SBMMAgent,
    'diverse':          DiverseAgent,
    'rl_bandit':        RLBanditAgent,
    'rl_transformer_ss':RLTransformerSSAgent,
    'rl_transformer_ar':RLTransformerARAgent,
}

# ── Single-raid agents ────────────────────────────────────────────────────
from .sr_base import SingleRaidAgent
from .sr_random import SRRandomAgent
from .sr_sbmm import SRSBMMAgent
from .sr_diverse import SRDiverseAgent
from .sr_polarized import SRPolarizedAgent
from .sr_balanced import SRBalancedAgent
from .sr_rl_bandit import SRRLBanditAgent

SR_AGENT_REGISTRY = {
    'sr_random':    SRRandomAgent,
    'sr_sbmm':      SRSBMMAgent,
    'sr_diverse':   SRDiverseAgent,
    'sr_polarized': SRPolarizedAgent,
    'sr_balanced':  SRBalancedAgent,
    'sr_rl_bandit': SRRLBanditAgent,
}

__all__ = [
    # multi-raid
    'Agent', 'RandomAgent', 'SBMMAgent', 'DiverseAgent',
    'RLBanditAgent', 'RLTransformerSSAgent', 'RLTransformerARAgent',
    'AGENT_REGISTRY',
    # single-raid
    'SingleRaidAgent', 'SRRandomAgent', 'SRSBMMAgent',
    'SRDiverseAgent', 'SRPolarizedAgent', 'SRBalancedAgent',
    'SRRLBanditAgent', 'SR_AGENT_REGISTRY',
]
