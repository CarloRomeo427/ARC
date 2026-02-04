#!/usr/bin/env python3
"""
Main entry point for running all matchmaking experiments.
"""

import sys
sys.path.insert(0, '/home/claude/extraction_sim')

from arc_matchmaking import (
    ExperimentConfig,
    ExperimentRunner,
    RandomMatchmaker,
    PolarizedMatchmaker,
    SBMMMatchmaker,
    DiverseMatchmaker,
)


def main():
    print("=" * 70)
    print("EXTRACTION SHOOTER MATCHMAKING - STANDARDIZED EXPERIMENT")
    print("=" * 70)
    
    config = ExperimentConfig(
        num_players=1200,
        lobby_size=12,
        raids_per_episode=100,
        raid_repetitions=10,
        num_episodes=10000,
        master_seed=42,
        wandb_project="ARC",
    )
    
    print(f"\nConfiguration:")
    print(f"  Players: {config.num_players}")
    print(f"  Lobby size: {config.lobby_size}")
    print(f"  Raids per episode: {config.raids_per_episode}")
    print(f"  Repetitions per raid: {config.raid_repetitions}")
    print(f"  Total episodes: {config.num_episodes}")
    print(f"  Master seed: {config.master_seed}")
    print(f"  Wandb project: {config.wandb_project}")
    
    runner = ExperimentRunner(config)
    
    matchmakers = [
        RandomMatchmaker(),
        PolarizedMatchmaker(),
        SBMMMatchmaker(),
        DiverseMatchmaker(),
    ]
    
    print(f"\nRunning experiments for: {[m.name for m in matchmakers]}")
    print("=" * 70)
    
    for matchmaker in matchmakers:
        runner.run_experiment(matchmaker)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
