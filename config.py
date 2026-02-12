"""
Configuration for the matchmaking experiment.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Standardized experiment configuration."""
    # Pool & queue
    pool_size: int = 1200          # Total player population
    queue_size: int = 120         # Players sampled per episode (who's online)
    lobby_size: int = 12
    raids_per_episode: int = 10   # queue_size / lobby_size

    # Player churn
    churn_interval: int = 50      # Every N episodes, replace some players
    churn_count: int = 5          # How many players to retire/replace per churn

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