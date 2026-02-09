"""
Experiment runner with wandb logging.
"""

import numpy as np
import wandb
import time
from typing import Dict

from config import ExperimentConfig
from player import PlayerPool
from simulator import RaidRunner, compute_reward
from algorithms import Matchmaker


class ExperimentRunner:
    """Runs standardized experiments with wandb logging."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.raid_runner = RaidRunner(config)
    
    def run_episode(self, pool: PlayerPool, matchmaker: Matchmaker, 
                    episode_num: int) -> dict:
        """
        Run one episode:
        - Create 100 lobbies from 1200 players
        - Run each raid 10 times
        - Average rewards
        - Update player stats
        """
        players = pool.get_all_players()
        lobbies = matchmaker.create_lobbies(players, self.config.lobby_size)
        
        classifications = {p.id: p.classification for p in players}
        
        all_rewards = []
        all_passive_scores = []
        all_aggressive_scores = []
        total_extractions = 0
        total_deaths = 0
        total_kills = 0
        
        for raid_idx, lobby in enumerate(lobbies):
            raid_seed = episode_num * 10000 + raid_idx * 100
            
            results = self.raid_runner.run_averaged_raid(lobby, raid_seed)
            
            results_list = [{'persistent_id': pid, **data} for pid, data in results.items()]
            reward_info = compute_reward(results_list, classifications)
            
            all_rewards.append(reward_info['pareto'])
            all_passive_scores.append(reward_info['passive_score'])
            all_aggressive_scores.append(reward_info['aggressive_score'])
            
            for pid, data in results.items():
                if data['extracted']:
                    total_extractions += 1
                else:
                    total_deaths += 1
                total_kills += data['kills']
            
            for p in lobby:
                r = results[p.id]
                p.record_raid(
                    r['extracted'], r['stash'], r['damage_dealt'],
                    r['damage_received'], int(round(r['kills'])), r['aggression_used']
                )
                p.update_aggression(
                    r['extracted'], int(round(r['kills'])),
                    r['damage_dealt'], r['damage_received'], r['aggression_used']
                )
        
        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_passive_score': np.mean(all_passive_scores),
            'avg_aggressive_score': np.mean(all_aggressive_scores),
            'total_extractions': total_extractions,
            'total_deaths': total_deaths,
            'total_kills': total_kills,
        }
    
    def run_experiment(self, matchmaker: Matchmaker) -> PlayerPool:
        """Run full experiment for one matchmaker with wandb logging."""
        
        run = wandb.init(
            project=self.config.wandb_project,
            name=f"{matchmaker.name}",
            config={
                'matchmaker': matchmaker.name,
                'num_players': self.config.num_players,
                'lobby_size': self.config.lobby_size,
                'raids_per_episode': self.config.raids_per_episode,
                'raid_repetitions': self.config.raid_repetitions,
                'num_episodes': self.config.num_episodes,
                'master_seed': self.config.master_seed,
            }
        )
        
        pool = PlayerPool(self.config.num_players, self.config.master_seed)
        
        print(f"\nRunning {matchmaker.name} for {self.config.num_episodes} episodes...")
        
        start_time = time.time()
        
        for episode in range(self.config.num_episodes):
            metrics = self.run_episode(pool, matchmaker, episode)
            pool_stats = pool.get_stats()
            
            wandb.log({
                'episode': episode,
                'avg_reward': metrics['avg_reward'],
                'std_reward': metrics['std_reward'],
                'avg_passive_score': metrics['avg_passive_score'],
                'avg_aggressive_score': metrics['avg_aggressive_score'],
                'total_extractions': metrics['total_extractions'],
                'total_deaths': metrics['total_deaths'],
                'total_kills': metrics['total_kills'],
                'extraction_rate': metrics['total_extractions'] / (metrics['total_extractions'] + metrics['total_deaths']),
                'pool_aggression_mean': pool_stats['aggression_mean'],
                'pool_aggression_std': pool_stats['aggression_std'],
                'pool_passive_count': pool_stats['passive_count'],
                'pool_neutral_count': pool_stats['neutral_count'],
                'pool_aggressive_count': pool_stats['aggressive_count'],
            })
            
            if (episode + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed
                eta = (self.config.num_episodes - episode - 1) / eps_per_sec
                print(f"  Episode {episode+1}/{self.config.num_episodes} | "
                      f"Reward: {metrics['avg_reward']:.3f} | "
                      f"ETA: {eta/60:.1f}min")
        
        wandb.finish()
        
        total_time = time.time() - start_time
        print(f"  Completed in {total_time/60:.1f} minutes")
        
        return pool
