#!/usr/bin/env python3
"""
Train RL Bandit Matchmaker.

Custom training loop that hooks into the simulation for per-lobby rewards.

Usage: python run_rl_bandit.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import wandb
import time

from config import ExperimentConfig
from player import PlayerPool
from simulator import RaidRunner, compute_reward
from algorithms.rl_bandit import RLBanditMatchmaker, RLBanditTrainer


def extract_features(players) -> np.ndarray:
    """Extract normalized (N, 7) feature matrix."""
    return RLBanditMatchmaker._extract_features(players)


def run_training_episode(
    trainer: RLBanditTrainer,
    pool: PlayerPool,
    raid_runner: RaidRunner,
    config: ExperimentConfig,
    episode_num: int,
) -> dict:
    """
    One training episode:
      1. Get players, extract features
      2. Policy forward pass + exploration noise
      3. Sort by noisy scores → lobbies
      4. Simulate raids, collect per-lobby rewards
      5. Policy gradient update with per-lobby credit assignment
      6. Update player persistent stats
    """
    players = pool.get_all_players()
    num_players = len(players)
    lobby_size = config.lobby_size
    
    # --- Forward pass ---
    features = extract_features(players)
    actions, mean_scores, log_probs = trainer.select_action(features)
    
    # --- Sort by scores → create lobbies ---
    sorted_indices = np.argsort(actions)
    sorted_players = [players[i] for i in sorted_indices]
    
    num_lobbies = num_players // lobby_size
    lobbies = [
        sorted_players[k * lobby_size : (k + 1) * lobby_size]
        for k in range(num_lobbies)
    ]
    
    # Track which original indices ended up in which lobby
    # (needed for credit assignment back to log_probs)
    lobby_original_indices = []
    for k in range(num_lobbies):
        start = k * lobby_size
        original_idxs = sorted_indices[start : start + lobby_size].tolist()
        lobby_original_indices.append(original_idxs)
    
    # --- Simulate raids & compute rewards ---
    classifications = {p.id: p.classification for p in players}
    
    lobby_rewards = []
    all_passive_scores = []
    all_aggressive_scores = []
    total_extractions = 0
    total_deaths = 0
    total_kills = 0
    
    for raid_idx, lobby in enumerate(lobbies):
        raid_seed = episode_num * 10000 + raid_idx * 100
        results = raid_runner.run_averaged_raid(lobby, raid_seed)
        
        results_list = [
            {"persistent_id": pid, **data} for pid, data in results.items()
        ]
        reward_info = compute_reward(results_list, classifications)
        
        lobby_rewards.append(reward_info["pareto"])
        all_passive_scores.append(reward_info["passive_score"])
        all_aggressive_scores.append(reward_info["aggressive_score"])
        
        # Accumulate stats
        for pid, data in results.items():
            if data["extracted"]:
                total_extractions += 1
            else:
                total_deaths += 1
            total_kills += data["kills"]
        
        # Update player persistent stats
        for p in lobby:
            r = results[p.id]
            p.record_raid(
                r["extracted"],
                r["stash"],
                r["damage_dealt"],
                r["damage_received"],
                int(round(r["kills"])),
                r["aggression_used"],
            )
            p.update_aggression(
                r["extracted"],
                int(round(r["kills"])),
                r["damage_dealt"],
                r["damage_received"],
                r["aggression_used"],
            )
    
    # --- Policy gradient update ---
    train_metrics = trainer.update(log_probs, lobby_original_indices, lobby_rewards)
    
    # --- Compose episode metrics ---
    metrics = {
        "avg_reward": np.mean(lobby_rewards),
        "std_reward": np.std(lobby_rewards),
        "min_reward": np.min(lobby_rewards),
        "max_reward": np.max(lobby_rewards),
        "avg_passive_score": np.mean(all_passive_scores),
        "avg_aggressive_score": np.mean(all_aggressive_scores),
        "zero_reward_lobbies": sum(1 for r in lobby_rewards if r == 0),
        "total_extractions": total_extractions,
        "total_deaths": total_deaths,
        "total_kills": total_kills,
        **{f"train/{k}": v for k, v in train_metrics.items()},
    }
    
    return metrics


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
    
    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Trainer ---
    trainer = RLBanditTrainer(
        input_dim=7,
        hidden_dim=128,
        lr=3e-4,
        sigma_init=0.1,
        sigma_min=0.01,
        sigma_decay=0.995,
        baseline_decay=0.95,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        device=device,
    )
    
    # --- Simulation ---
    raid_runner = RaidRunner(config)
    pool = PlayerPool(config.num_players, seed=config.master_seed)
    
    # --- wandb ---
    wandb.init(
        project=config.wandb_project,
        name="RL_Bandit",
        config={
            **config.__dict__,
            "algorithm": "RL_Bandit",
            "hidden_dim": 128,
            "lr": 3e-4,
            "sigma_init": 0.1,
            "sigma_decay": 0.995,
            "entropy_coef": 0.01,
            "device": device,
        },
    )
    
    print("=" * 70)
    print(f"RUNNING: RL Bandit Matchmaker ({device})")
    print("=" * 70)
    
    best_reward = -float("inf")
    reward_window = []
    
    for episode in range(config.num_episodes):
        t0 = time.time()
        metrics = run_training_episode(trainer, pool, raid_runner, config, episode)
        elapsed = time.time() - t0
        
        metrics["episode_time"] = elapsed
        wandb.log(metrics, step=episode)
        
        # Track running average
        reward_window.append(metrics["avg_reward"])
        if len(reward_window) > 100:
            reward_window.pop(0)
        running_avg = np.mean(reward_window)
        
        # Save best model
        if running_avg > best_reward and episode >= 100:
            best_reward = running_avg
            trainer.save("rl_bandit_best.pt")
        
        # Periodic checkpoint
        if (episode + 1) % 1000 == 0:
            trainer.save(f"rl_bandit_ep{episode+1}.pt")
        
        # Logging
        if episode % 50 == 0:
            print(
                f"Ep {episode:5d} | "
                f"R={metrics['avg_reward']:.4f} (±{metrics['std_reward']:.3f}) | "
                f"R100={running_avg:.4f} | "
                f"σ={trainer.sigma:.4f} | "
                f"loss={metrics['train/total_loss']:.4f} | "
                f"∇={metrics['train/grad_norm']:.3f} | "
                f"zero={metrics['zero_reward_lobbies']} | "
                f"{elapsed:.1f}s"
            )
    
    trainer.save("rl_bandit_final.pt")
    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
