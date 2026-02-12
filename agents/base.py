"""
Base Agent class for matchmaking strategies.

All agents inherit from this and implement create_lobbies().
Agents that need training override run_episode() and the training hooks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
import time
import wandb

from config import ExperimentConfig
from player import Player, PlayerPool
from simulator import RaidRunner, compute_reward


class Agent(ABC):
    """Base matchmaking agent."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.raid_runner = RaidRunner(config)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def create_lobbies(self, players: List[Player], lobby_size: int) -> List[List[Player]]:
        pass

    # ------------------------------------------------------------------
    # Training hooks (override in learning agents)
    # ------------------------------------------------------------------

    def before_training(self, pool: PlayerPool):
        """Called once before the training loop starts."""
        pass

    def after_episode(self, episode: int, metrics: dict):
        """Called after each episode with metrics."""
        pass

    def after_training(self):
        """Called once after the training loop ends."""
        pass

    # ------------------------------------------------------------------
    # Default episode logic (heuristic agents use this as-is)
    # ------------------------------------------------------------------

    def run_episode(self, pool: PlayerPool, episode_num: int) -> dict:
        """Run one episode. Learning agents can override this entirely."""
        # Sample who's online this episode
        players = pool.sample_queue(self.config.queue_size)
        lobbies = self.create_lobbies(players, self.config.lobby_size)
        classifications = {p.id: p.classification for p in players}

        all_rewards, all_passive, all_aggressive = [], [], []
        total_ext, total_deaths, total_kills = 0, 0, 0

        for raid_idx, lobby in enumerate(lobbies):
            raid_seed = episode_num * 10000 + raid_idx * 100
            results = self.raid_runner.run_averaged_raid(lobby, raid_seed)

            results_list = [{'persistent_id': pid, **data} for pid, data in results.items()]
            reward_info = compute_reward(results_list, classifications)

            all_rewards.append(reward_info['pareto'])
            all_passive.append(reward_info['passive_score'])
            all_aggressive.append(reward_info['aggressive_score'])

            for pid, data in results.items():
                total_ext += int(data['extracted'])
                total_deaths += int(not data['extracted'])
                total_kills += data['kills']

            for p in lobby:
                r = results[p.id]
                p.record_raid(r['extracted'], r['stash'], r['damage_dealt'],
                              r['damage_received'], int(round(r['kills'])), r['aggression_used'])

        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_passive_score': np.mean(all_passive),
            'avg_aggressive_score': np.mean(all_aggressive),
            'total_extractions': total_ext,
            'total_deaths': total_deaths,
            'total_kills': total_kills,
        }

    # ------------------------------------------------------------------
    # Full experiment loop
    # ------------------------------------------------------------------

    def run(self) -> PlayerPool:
        """Run the full experiment with wandb logging."""
        pool = PlayerPool(self.config.pool_size, self.config.master_seed)

        run = wandb.init(
            project=self.config.wandb_project,
            name=self.name,
            config={**self.config.__dict__, 'algorithm': self.name, 'seed': self.config.master_seed},
        )

        self.before_training(pool)

        print(f"\nRunning {self.name} for {self.config.num_episodes} episodes...")
        print(f"  Pool: {self.config.pool_size} | Queue: {self.config.queue_size} | "
              f"Churn: {self.config.churn_count} every {self.config.churn_interval} eps")
        start_time = time.time()

        for episode in range(self.config.num_episodes):
            # Player churn: retire and replace periodically
            if episode > 0 and episode % self.config.churn_interval == 0:
                pool.churn(self.config.churn_count)

            metrics = self.run_episode(pool, episode)
            pool_stats = pool.get_stats()
            metrics = self.run_episode(pool, episode)
            pool_stats = pool.get_stats()

            log_data = {
                'episode': episode,
                'avg_reward': metrics['avg_reward'],
                'std_reward': metrics['std_reward'],
                'avg_passive_score': metrics['avg_passive_score'],
                'avg_aggressive_score': metrics['avg_aggressive_score'],
                'total_extractions': metrics['total_extractions'],
                'total_deaths': metrics['total_deaths'],
                'total_kills': metrics['total_kills'],
                'extraction_rate': metrics['total_extractions'] / max(1, metrics['total_extractions'] + metrics['total_deaths']),
                'pool_aggression_mean': pool_stats['aggression_mean'],
                'pool_aggression_std': pool_stats['aggression_std'],
                'pool_passive_count': pool_stats['passive_count'],
                'pool_neutral_count': pool_stats['neutral_count'],
                'pool_aggressive_count': pool_stats['aggressive_count'],
                'pool_size': pool_stats['pool_size'],
                'total_players_seen': pool_stats['total_players_seen'],
            }
            # Merge any extra keys from learning agents
            for k, v in metrics.items():
                if k not in log_data:
                    log_data[k] = v

            wandb.log(log_data, step=episode)
            self.after_episode(episode, metrics)

            if (episode + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed
                eta = (self.config.num_episodes - episode - 1) / eps_per_sec
                print(f"  Ep {episode+1}/{self.config.num_episodes} | "
                      f"R={metrics['avg_reward']:.3f} | ETA: {eta/60:.1f}min")

        self.after_training()
        wandb.finish()

        total_time = time.time() - start_time
        print(f"  {self.name} completed in {total_time/60:.1f} minutes")
        return pool