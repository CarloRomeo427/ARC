"""
Configuration for the matchmaking experiment.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Multi-raid (original) experiment configuration."""
    pool_size: int = 1200
    queue_size: int = 120
    lobby_size: int = 12
    raids_per_episode: int = 10
    churn_interval: int = 50
    churn_count: int = 5
    raid_repetitions: int = 5
    num_episodes: int = 100_000
    master_seed: int = 42
    wandb_project: str = "ARC"
    aggression_noise_std: float = 0.05
    map_radius: float = 100.0
    num_loot_zones: int = 8
    max_ticks: int = 500
    sight_radius: float = 15.0
    extraction_time: int = 30
    extraction_cooldown: int = 30
    combat_max_ticks: int = 15
    flee_hp_threshold: float = 35.0
    flee_base: float = 0.3
    post_combat_cooldown: int = 15
    heal_on_kill: float = 5.0


@dataclass
class SingleRaidConfig:
    """
    Single-raid experiment configuration.

    Task: at each episode, sample queue_size players from the persistent
    pool of pool_size players, then select lobby_size of them for one
    optimised raid. The remaining queue players sit out.

    Using pool_size >> queue_size prevents memorisation of specific player
    IDs and forces the agent to learn general composition rules based on
    player features, not identities.
    """
    # Population
    pool_size: int  = 1200        # Persistent player population
    queue_size: int = 120         # Players online / available each episode
    lobby_size: int = 12          # Players selected into the raid

    # Training
    num_episodes: int = 50_000
    master_seed: int = 42

    # RLOO (RL bandit only)
    rloo_k: int = 8               # Parallel lobby samples per episode

    # Player churn
    churn_interval: int = 50
    churn_count: int = 10         # Slightly higher churn to keep pool fresh

    # Simulation averaging
    raid_repetitions: int = 5

    # Wandb
    wandb_project: str = "ARC_SingleRaid"

    # Physics
    aggression_noise_std: float = 0.05
    map_radius: float = 100.0
    num_loot_zones: int = 8
    max_ticks: int = 500
    sight_radius: float = 10.0
    extraction_time: int = 30
    extraction_cooldown: int = 30
    combat_max_ticks: int = 15
    flee_hp_threshold: float = 35.0
    flee_base: float = 0.3
    post_combat_cooldown: int = 15
    heal_on_kill: float = 5.0