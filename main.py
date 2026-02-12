#!/usr/bin/env python3
"""
Unified entry point for all matchmaking experiments.

Usage:
    python main.py --agent random
    python main.py --agent sbmm
    python main.py --agent diverse
    python main.py --agent rl_bandit
    python main.py --agent all          # run all sequentially
    python main.py --agent random sbmm  # run subset
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from agents import AGENT_REGISTRY
from simulator import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Extraction Shooter Matchmaking Experiments")
    parser.add_argument(
        "--agent", nargs="+", required=True,
        choices=list(AGENT_REGISTRY.keys()) + ["all"],
        help="Agent(s) to run. Use 'all' for every registered agent.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes")
    parser.add_argument("--pool-size", type=int, default=None, help="Override pool_size")
    parser.add_argument("--queue-size", type=int, default=None, help="Override queue_size")
    parser.add_argument("--seed", type=int, default=None, help="Override master_seed")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override wandb project name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config with optional overrides
    overrides = {}
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.pool_size is not None:
        overrides["pool_size"] = args.pool_size
    if args.queue_size is not None:
        overrides["queue_size"] = args.queue_size
        overrides["raids_per_episode"] = args.queue_size // 12
    if args.seed is not None:
        overrides["master_seed"] = args.seed
    if args.wandb_project is not None:
        overrides["wandb_project"] = args.wandb_project

    config = ExperimentConfig(**overrides)

    # Resolve agent list
    if "all" in args.agent:
        agent_names = list(AGENT_REGISTRY.keys())
    else:
        agent_names = args.agent

    print("=" * 70)
    print("EXTRACTION SHOOTER MATCHMAKING EXPERIMENT")
    print("=" * 70)
    print(f"  Pool size:  {config.pool_size}")
    print(f"  Queue size: {config.queue_size}")
    print(f"  Lobby size: {config.lobby_size}")
    print(f"  Raids/ep:   {config.raids_per_episode}")
    print(f"  Reps/raid:  {config.raid_repetitions}")
    print(f"  Episodes:   {config.num_episodes}")
    print(f"  Churn:      {config.churn_count} every {config.churn_interval} eps")
    print(f"  Seed:       {config.master_seed}")
    print(f"  Agents:     {agent_names}")
    print("=" * 70)

    # Seed all PRNGs for reproducibility
    seed_everything(config.master_seed)

    for agent_name in agent_names:
        agent_cls = AGENT_REGISTRY[agent_name]
        agent = agent_cls(config)
        agent.run()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()