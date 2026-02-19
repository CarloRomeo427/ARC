"""
Configuration for the matchmaking experiment.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Multi-raid (original) experiment configuration."""
    # Pool & queue
    pool_size: int = 1200
    queue_size: int = 120
    lobby_size: int = 12
    raids_per_episode: int = 10

    # Player churn
    churn_interval: int = 50
    churn_count: int = 5

    # Repetitions
    raid_repetitions: int = 5
    num_episodes: int = 100_000

    # Seeds
    master_seed: int = 42

    # Wandb
    wandb_project: str = "ARC"

    # Raid settings
    aggression_noise_std: float = 0.05

    # Map settings
    map_radius: float = 100.0
    num_loot_zones: int = 8

    # Raid settings
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

    Task: given pool_size online players, select the best lobby_size
    players to form ONE optimised raid per episode.
    The remaining pool players are not assigned to any raid that episode.
    """
    # Pool
    pool_size: int = 120          # Players "online" and available for selection
    lobby_size: int = 12          # Players selected per raid

    # Training
    num_episodes: int = 50_000
    master_seed: int = 42

    # RLOO sampling (RL bandit only)
    rloo_k: int = 8               # Number of parallel lobby samples per episode

    # Player churn
    churn_interval: int = 50
    churn_count: int = 5

    # Simulation averaging
    raid_repetitions: int = 5

    # Wandb
    wandb_project: str = "ARC_SingleRaid"

    # Physics (must match simulator expectations)
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
