#!/usr/bin/env python3
"""
Entry point for single-raid matchmaking experiments.

Task: given pool_size online players, select the best lobby_size subset
to form one optimised raid per episode. Unselected players sit out.

Usage:
    python main_single_raid.py --agent sr_random
    python main_single_raid.py --agent sr_rl_bandit
    python main_single_raid.py --agent all
    python main_single_raid.py --agent sr_random sr_sbmm sr_diverse sr_polarized sr_balanced
    python main_single_raid.py --agent sr_rl_bandit --episodes 30000 --rloo-k 8
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SingleRaidConfig
from agents import SR_AGENT_REGISTRY
from simulator import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-raid matchmaking experiments",
    )
    parser.add_argument(
        "--agent", nargs="+", required=True,
        choices=list(SR_AGENT_REGISTRY.keys()) + ["all"],
        help="Agent(s) to run. 'all' runs every registered single-raid agent.",
    )
    parser.add_argument("--episodes",    type=int,   default=None)
    parser.add_argument("--pool-size",   type=int,   default=None)
    parser.add_argument("--lobby-size",  type=int,   default=None)
    parser.add_argument("--rloo-k",      type=int,   default=None,
                        help="RLOO sample count (RL bandit only)")
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.episodes     is not None: overrides["num_episodes"]   = args.episodes
    if args.pool_size    is not None: overrides["pool_size"]      = args.pool_size
    if args.lobby_size   is not None: overrides["lobby_size"]     = args.lobby_size
    if args.rloo_k       is not None: overrides["rloo_k"]         = args.rloo_k
    if args.seed         is not None: overrides["master_seed"]    = args.seed
    if args.wandb_project is not None: overrides["wandb_project"] = args.wandb_project

    config = SingleRaidConfig(**overrides)

    agent_names = list(SR_AGENT_REGISTRY.keys()) if "all" in args.agent else args.agent

    print("=" * 70)
    print("SINGLE-RAID MATCHMAKING EXPERIMENT")
    print("=" * 70)
    print(f"  Pool size:  {config.pool_size}  (online players to choose from)")
    print(f"  Lobby size: {config.lobby_size}  (players selected per raid)")
    print(f"  Episodes:   {config.num_episodes}")
    print(f"  Reps/raid:  {config.raid_repetitions}")
    print(f"  RLOO k:     {config.rloo_k}  (RL bandit only)")
    print(f"  Churn:      {config.churn_count} every {config.churn_interval} eps")
    print(f"  Seed:       {config.master_seed}")
    print(f"  Agents:     {agent_names}")
    print("=" * 70)

    seed_everything(config.master_seed)

    for agent_name in agent_names:
        agent_cls = SR_AGENT_REGISTRY[agent_name]
        agent = agent_cls(config)
        agent.run()

    print("\n" + "=" * 70)
    print("ALL SINGLE-RAID EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
