"""
Raid visualiser for wandb GIF logging.

Records per-tick player state during a live raid and renders an animated
GIF using matplotlib. Called from sr_base.py every 1000 episodes.

Colour scheme:
  Cyan  (#4cc9f0) — passive  (aggression < 0.4)
  Orange (#f8961e) — neutral  (0.4 ≤ aggression ≤ 0.6)
  Pink   (#f72585) — aggressive (aggression > 0.6)
  Yellow (#ffff00) — in combat (overrides archetype colour)
  Green  (#00ff88) — extracted
  Grey   (#555555) — dead
  Teal   (#00b4d8) — extraction point (square marker)
  Fading gold     — loot zones (alpha tracks remaining items)
"""

import os
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from player import Player, RaidPlayer
from config import SingleRaidConfig


# =========================================================================
# State capture
# =========================================================================

@dataclass
class _PlayerSnap:
    x: float
    y: float
    alive: bool
    extracted: bool
    in_combat: bool
    aggression: float


@dataclass
class _Frame:
    tick: int
    players: List[_PlayerSnap]
    zone_remaining: List[int]


class _RecordingRaid:
    """
    Thin wrapper around Raid that captures player positions every
    `record_interval` ticks without modifying the simulator source.
    """

    def __init__(self, game_map, players, config, record_interval: int = 5):
        from simulator import Raid
        self._raid = Raid(game_map, players, config)
        self._interval = record_interval
        self.frames: List[_Frame] = []

    def run(self):
        while (self._raid.tick < self._raid.config.max_ticks
               and self._raid.get_alive()):
            self._raid.run_tick()
            if self._raid.tick % self._interval == 0:
                self._capture()
        self._capture()   # always capture final state
        return [p.to_result() for p in self._raid.players]

    def _capture(self):
        snaps = [
            _PlayerSnap(
                x=p.x, y=p.y,
                alive=p.alive, extracted=p.extracted,
                in_combat=(p.in_combat_with is not None),
                aggression=p.aggression,
            )
            for p in self._raid.players
        ]
        zone_rem = [z.remaining_items for z in self._raid.map.loot_zones]
        self.frames.append(_Frame(self._raid.tick, snaps, zone_rem))


# =========================================================================
# Rendering
# =========================================================================

def _aggr_color(a: float) -> str:
    if a < 0.4:
        return '#4cc9f0'
    elif a <= 0.6:
        return '#f8961e'
    return '#f72585'


def _render_to_gif(frames: List[_Frame], game_map, output_path: str, fps: int = 10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    R = game_map.radius
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.set_xlim(-R * 1.08, R * 1.08)
    ax.set_ylim(-R * 1.08, R * 1.08)
    ax.set_aspect('equal')
    ax.axis('off')

    import matplotlib.patches as mpatches
    boundary = plt.Circle((0, 0), R, fill=False,
                           edgecolor='#0f3460', linewidth=1.5, linestyle='--')
    ax.add_patch(boundary)

    max_items = [max(z.total_items, 1) for z in game_map.loot_zones]
    loot_patches = []
    for z in game_map.loot_zones:
        p = plt.Circle((z.x, z.y), z.radius, alpha=0.45,
                        facecolor='#e2b96f', edgecolor='none', zorder=1)
        ax.add_patch(p)
        loot_patches.append(p)

    for e in game_map.extraction_points:
        ax.plot(e.x, e.y, 's', color='#00b4d8', ms=9,
                markeredgecolor='white', markeredgewidth=0.4, zorder=2)

    f0 = frames[0]
    xs = [s.x for s in f0.players]
    ys = [s.y for s in f0.players]
    cs = [_aggr_color(s.aggression) for s in f0.players]
    scat = ax.scatter(xs, ys, c=cs, s=38, zorder=5,
                      edgecolors='white', linewidths=0.25)

    tick_txt = ax.text(0, R * 1.04, "Tick 0",
                       ha='center', va='bottom', color='white',
                       fontsize=8, fontweight='bold')

    import matplotlib.lines as mlines
    legend_items = [
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#4cc9f0',
                      ms=7, label='Passive', linestyle=''),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#f8961e',
                      ms=7, label='Neutral', linestyle=''),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#f72585',
                      ms=7, label='Aggressive', linestyle=''),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#ffff00',
                      ms=7, label='In combat', linestyle=''),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#00ff88',
                      ms=7, label='Extracted', linestyle=''),
        mlines.Line2D([], [], marker='s', color='w', markerfacecolor='#00b4d8',
                      ms=7, label='Extraction pt', linestyle=''),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=5.5,
              framealpha=0.25, labelcolor='white',
              facecolor='#1a1a2e', edgecolor='none')

    def update(fi):
        frame = frames[fi]
        for i, (patch, mx) in enumerate(zip(loot_patches, max_items)):
            rem = frame.zone_remaining[i] if i < len(frame.zone_remaining) else 0
            patch.set_alpha(0.08 + 0.42 * (rem / mx))

        xs_f, ys_f, cs_f, ss_f = [], [], [], []
        for s in frame.players:
            xs_f.append(s.x)
            ys_f.append(s.y)
            if s.extracted:
                cs_f.append('#00ff88'); ss_f.append(75)
            elif not s.alive:
                cs_f.append('#555555'); ss_f.append(15)
            elif s.in_combat:
                cs_f.append('#ffff00'); ss_f.append(60)
            else:
                cs_f.append(_aggr_color(s.aggression)); ss_f.append(38)

        scat.set_offsets(np.c_[xs_f, ys_f])
        scat.set_color(cs_f)
        scat.set_sizes(ss_f)
        tick_txt.set_text(f"Tick {frame.tick}")
        return scat, tick_txt

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=75)
    plt.close(fig)


# =========================================================================
# Public entry point
# =========================================================================

def render_raid_gif(lobby: List[Player], config: SingleRaidConfig,
                   seed: int) -> Optional[str]:
    """
    Run one raid with state recording and render to a temporary GIF.

    Mirrors run_single_raid exactly:
      - game_map.reset(stash_budget)   — budget-proportional loot distribution
      - spawn permutation              — seed-dependent position assignment
      - stash_threshold per player     — inherited from persistent Player
      - personal_zone_scores           — lognormal(0, 0.65) per player per zone

    Returns the path to the GIF file, or None if rendering fails.
    The caller is responsible for logging and cleanup.
    """
    try:
        import random
        from simulator import GameMap

        np.random.seed(seed)
        random.seed(seed)

        game_map = GameMap(config, map_seed=config.master_seed)

        # Budget-proportional loot distribution (required by reset v2)
        stash_budget = sum(p.stash_threshold for p in lobby)
        game_map.reset(stash_budget)

        spawns = game_map.get_spawn_positions(len(lobby))
        # Permute spawn assignment — identical logic to run_single_raid
        perm = np.random.permutation(len(spawns))

        # Per-player zone score perturbations — same sigma as run_single_raid
        SCORE_SIGMA = 0.65
        raid_players = []
        for i, p in enumerate(lobby):
            personal_scores = {
                z.zone_id: z.expected_value_score * np.random.lognormal(0.0, SCORE_SIGMA)
                for z in game_map.loot_zones
            }
            raid_players.append(RaidPlayer(
                id=i,
                persistent_id=p.id,
                x=spawns[perm[i]][0],
                y=spawns[perm[i]][1],
                aggression=p.get_raid_aggression(config.aggression_noise_std),
                stash_threshold=p.stash_threshold,
                personal_zone_scores=personal_scores,
            ))

        recorder = _RecordingRaid(game_map, raid_players, config, record_interval=8)
        recorder.run()

        if not recorder.frames:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
        tmp.close()
        _render_to_gif(recorder.frames, game_map, tmp.name, fps=10)
        return tmp.name

    except Exception as e:
        print(f"  [raid_visualizer] GIF rendering failed: {e}")
        return None