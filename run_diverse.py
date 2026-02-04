#!/usr/bin/env python3
"""
Run Diverse matchmaker experiment.
Usage: python run_diverse.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_matchmaking import ExperimentConfig, ExperimentRunner, DiverseMatchmaker


def main():
    config = ExperimentConfig(
        num_players=1200,
        lobby_size=12,
        raids_per_episode=100,
        raid_repetitions=10,
        num_episodes=10000,
        master_seed=42,
        wandb_project="ARC",
    )
    
    print("=" * 70)
    print("RUNNING: Diverse Matchmaker")
    print("=" * 70)
    
    runner = ExperimentRunner(config)
    matchmaker = DiverseMatchmaker()
    
    pool = runner.run_experiment(matchmaker)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
