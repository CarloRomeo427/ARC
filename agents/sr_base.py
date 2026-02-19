"""
Base class for all single-raid matchmaking agents.

Changes in this version
-----------------------
  1. Reward: uses compute_reward_sr from sr_reward.py (loot + encounters).
  2. Aggression distribution: logs wandb.Histogram for both the full pool
     and the selected lobby every episode.
  3. GIF logging: every GIF_LOG_INTERVAL episodes, renders one raid from
     the current episode to a GIF and logs it to wandb.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
import random
import time
import os
import wandb

from config import SingleRaidConfig
from player import Player, PlayerPool
from simulator import RaidRunner
from sr_reward import compute_reward_sr

GIF_LOG_INTERVAL = 1000   # log a raid GIF every this many episodes


class SingleRaidAgent(ABC):

    def __init__(self, config: SingleRaidConfig):
        self.config = config
        self.raid_runner = RaidRunner(config)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select_lobby(self, players: List[Player]) -> List[Player]:
        pass

    # ------------------------------------------------------------------
    # Simulation helper
    # ------------------------------------------------------------------

    def _simulate_lobbies(self, lobbies: List[List[Player]],
                          episode_num: int) -> List[Dict[int, dict]]:
        n_reps    = self.config.raid_repetitions
        n_lobbies = len(lobbies)

        accs = []
        for lobby in lobbies:
            acc = {p.id: {'extracted': 0, 'stash': 0.0, 'items_looted': 0.0,
                          'dmg_dealt': 0.0, 'dmg_recv': 0.0, 'kills': 0.0,
                          'encounters': 0.0, 'aggr': 0.0}
                   for p in lobby}
            accs.append(acc)

        for rep in range(n_reps):
            rep_seed = self.config.master_seed + episode_num * n_reps + rep
            for lobby_idx, lobby in enumerate(lobbies):
                np.random.seed(rep_seed)
                random.seed(rep_seed)
                results_list = self.raid_runner.run_single_raid(lobby)
                for r in results_list:
                    pid = r['persistent_id']
                    a   = accs[lobby_idx][pid]
                    a['extracted']    += 1 if r['extracted'] else 0
                    a['stash']        += r['stash']
                    a['items_looted'] += r.get('items_looted', 0.0)
                    a['dmg_dealt']    += r['damage_dealt']
                    a['dmg_recv']     += r['damage_received']
                    a['kills']        += r['kills']
                    a['encounters']   += r.get('encounters', 0.0)
                    a['aggr']         += r['aggression_used']

        all_results = []
        for lobby_idx in range(n_lobbies):
            lobby_results = {}
            for pid, a in accs[lobby_idx].items():
                lobby_results[pid] = {
                    'persistent_id':   pid,
                    'extracted':       a['extracted'] > n_reps / 2,
                    'stash':           a['stash']        / n_reps,
                    'items_looted':    a['items_looted'] / n_reps,
                    'damage_dealt':    a['dmg_dealt']    / n_reps,
                    'damage_received': a['dmg_recv']     / n_reps,
                    'kills':           a['kills']        / n_reps,
                    'encounters':      a['encounters']   / n_reps,
                    'aggression_used': a['aggr']         / n_reps,
                }
            all_results.append(lobby_results)

        return all_results

    # ------------------------------------------------------------------
    # GIF logging
    # ------------------------------------------------------------------

    def _maybe_log_gif(self, lobby: List[Player], episode_num: int):
        """Render and log one raid GIF to wandb every GIF_LOG_INTERVAL eps."""
        if episode_num % GIF_LOG_INTERVAL != 0:
            return
        try:
            from raid_visualizer import render_raid_gif
            gif_seed = self.config.master_seed + episode_num
            gif_path = render_raid_gif(lobby, self.config, gif_seed)
            if gif_path and os.path.exists(gif_path):
                wandb.log(
                    {'raid_replay': wandb.Video(gif_path, fps=10, format='gif')},
                    step=episode_num,
                )
                os.unlink(gif_path)
        except Exception as e:
            print(f"  [gif] Failed at episode {episode_num}: {e}")

    # ------------------------------------------------------------------
    # Default episode logic (heuristic agents)
    # ------------------------------------------------------------------

    def run_episode(self, pool: PlayerPool, episode_num: int) -> dict:
        queue = pool.sample_queue(self.config.queue_size)
        lobby = self.select_lobby(queue)

        all_results = self._simulate_lobbies([lobby], episode_num)
        results     = all_results[0]
        ri          = compute_reward_sr(results)

        for p in lobby:
            r = results[p.id]
            p.record_raid(
                r['extracted'], r['stash'], r['damage_dealt'],
                r['damage_received'], int(round(r['kills'])), r['aggression_used'],
            )

        if (episode_num + 1) % self.config.churn_interval == 0:
            pool.churn(self.config.churn_count)

        self._maybe_log_gif(lobby, episode_num)

        lobby_aggr = [p.running_aggression for p in lobby]
        return {
            'reward':              ri['reward'],
            'loot_score':          ri['loot_score'],
            'fight_score':         ri['fight_score'],
            'total_extracted_loot':ri['total_extracted_loot'],
            'total_encounters':    ri['total_encounters'],
            'total_kills':         ri['total_kills'],
            'total_extractions':   ri['total_extractions'],
            'reward_case':         ri.get('reward_case', 'unknown'),
            'lobby_aggr_mean':     float(np.mean(lobby_aggr)),
            'lobby_aggr_std':      float(np.std(lobby_aggr)),
            'lobby_n_passive':     sum(1 for p in lobby if p.classification=='passive'),
            'lobby_n_neutral':     sum(1 for p in lobby if p.classification=='neutral'),
            'lobby_n_aggressive':  sum(1 for p in lobby if p.classification=='aggressive'),
            '_lobby_aggr':         lobby_aggr,
        }

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def before_training(self, pool: PlayerPool):
        pass

    def after_episode(self, episode: int, metrics: dict):
        pass

    def after_training(self):
        pass

    # ------------------------------------------------------------------
    # Full experiment loop
    # ------------------------------------------------------------------

    def run(self) -> PlayerPool:
        pool = PlayerPool(self.config.pool_size, self.config.master_seed)

        wandb.init(
            project=self.config.wandb_project,
            name=self.name,
            config={**self.config.__dict__, 'algorithm': self.name,
                    'seed': self.config.master_seed},
        )

        self.before_training(pool)
        print(f"\nRunning {self.name} for {self.config.num_episodes} episodes...")
        start_time    = time.time()
        reward_window: List[float] = []

        for episode in range(self.config.num_episodes):
            metrics    = self.run_episode(pool, episode)
            pool_stats = pool.get_stats()

            reward_window.append(metrics['reward'])
            if len(reward_window) > 100:
                reward_window.pop(0)

            # Pull and remove the raw arrays before passing to wandb
            lobby_aggr = metrics.pop('_lobby_aggr', [])
            pool_aggr  = [p.running_aggression for p in pool.get_all_players()]

            log_data = {
                'episode':               episode,
                'reward':                metrics['reward'],
                'reward_100ep_avg':      np.mean(reward_window),
                'loot_score':            metrics['loot_score'],
                'fight_score':           metrics['fight_score'],
                'total_extracted_loot':  metrics['total_extracted_loot'],
                'total_encounters':      metrics['total_encounters'],
                'total_kills':           metrics['total_kills'],
                'total_extractions':     metrics['total_extractions'],
                'reward_case':           metrics.get('reward_case', 'unknown'),
                # Lobby composition
                'lobby_aggr_mean':       metrics['lobby_aggr_mean'],
                'lobby_aggr_std':        metrics['lobby_aggr_std'],
                'lobby_n_passive':       metrics['lobby_n_passive'],
                'lobby_n_neutral':       metrics['lobby_n_neutral'],
                'lobby_n_aggressive':    metrics['lobby_n_aggressive'],
                # Aggression distributions
                'dist/lobby_aggression': wandb.Histogram(lobby_aggr),
                'dist/pool_aggression':  wandb.Histogram(pool_aggr),
                # Pool stats
                'pool_aggression_mean':  pool_stats['aggression_mean'],
                'pool_aggression_std':   pool_stats['aggression_std'],
                'pool_passive_count':    pool_stats['passive_count'],
                'pool_neutral_count':    pool_stats['neutral_count'],
                'pool_aggressive_count': pool_stats['aggressive_count'],
            }
            # Merge any extra keys from learning agents
            for k, v in metrics.items():
                if k not in log_data:
                    log_data[k] = v

            wandb.log(log_data, step=episode)
            self.after_episode(episode, metrics)

            if (episode + 1) % 500 == 0:
                elapsed = time.time() - start_time
                eps_sec = (episode + 1) / elapsed
                eta     = (self.config.num_episodes - episode - 1) / eps_sec
                print(f"  Ep {episode+1:6d}/{self.config.num_episodes} | "
                      f"R={metrics['reward']:.4f} | avg100={np.mean(reward_window):.4f} | "
                      f"loot={metrics['loot_score']:.3f} fight={metrics['fight_score']:.3f} | "
                      f"P/N/A={metrics['lobby_n_passive']}/"
                      f"{metrics['lobby_n_neutral']}/"
                      f"{metrics['lobby_n_aggressive']} | "
                      f"ETA: {eta/60:.1f}min")

        self.after_training()
        wandb.finish()
        print(f"  {self.name} completed in {(time.time()-start_time)/60:.1f} minutes")
        return pool