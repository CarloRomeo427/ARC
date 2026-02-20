"""
Player classes for persistent and transient (raid) players.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Player:
    """
    Persistent player with running statistics.

    Aggression is fixed at init and never changes. Per-raid noise is
    applied transiently via get_raid_aggression().

    stash_threshold: the loot value at which this player decides to
    head to extraction. Sampled once at creation [80_000, 200_000] and
    constant across all raids, including repetitions of the same episode.
    """
    id: int
    aggression: float
    stash_threshold: float = 140_000.0   # set by PlayerPool._spawn_player

    total_raids: int = 0
    total_extractions: int = 0
    total_deaths: int = 0
    total_kills: int = 0
    total_stash: float = 0.0
    total_damage_dealt: float = 0.0
    total_damage_received: float = 0.0

    aggression_sum: float = 0.0
    aggression_count: int = 0

    @property
    def running_aggression(self) -> float:
        if self.aggression_count == 0:
            return self.aggression
        return self.aggression_sum / self.aggression_count

    @property
    def classification(self) -> str:
        if self.aggression < 0.4:
            return "passive"
        elif self.aggression > 0.6:
            return "aggressive"
        return "neutral"

    @property
    def extraction_rate(self) -> float:
        if self.total_raids == 0:
            return 0.0
        return self.total_extractions / self.total_raids

    @property
    def kills_per_raid(self) -> float:
        if self.total_raids == 0:
            return 0.0
        return self.total_kills / self.total_raids

    @property
    def deaths_per_raid(self) -> float:
        if self.total_raids == 0:
            return 0.0
        return self.total_deaths / self.total_raids

    @property
    def avg_stash(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.total_stash / self.total_extractions

    def get_raid_aggression(self, noise_std: float = 0.05) -> float:
        noisy = self.aggression + np.random.normal(0, noise_std)
        return np.clip(noisy, 0.0, 1.0)

    def record_raid(self, extracted: bool, stash: float, damage_dealt: float,
                    damage_received: float, kills: int, aggression_used: float):
        self.total_raids += 1
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received
        self.total_kills += kills
        self.aggression_sum += aggression_used
        self.aggression_count += 1
        if extracted:
            self.total_extractions += 1
            self.total_stash += stash
        else:
            self.total_deaths += 1

    def copy(self) -> 'Player':
        p = Player(id=self.id, aggression=self.aggression,
                   stash_threshold=self.stash_threshold)
        for attr in ['total_raids', 'total_extractions', 'total_deaths', 'total_kills',
                     'total_stash', 'total_damage_dealt', 'total_damage_received',
                     'aggression_sum', 'aggression_count']:
            setattr(p, attr, getattr(self, attr))
        return p


class PlayerPool:
    """
    Pool of persistent players with queue sampling and churn.

    Each episode, a random subset (the queue) is drawn from the pool.
    Periodically, some players are retired and replaced with fresh ones,
    simulating population churn in a live game.
    """

    def __init__(self, pool_size: int, seed: int):
        np.random.seed(seed)
        self.players: Dict[int, Player] = {}
        self._next_id = 0
        for _ in range(pool_size):
            self._spawn_player()

    def _spawn_player(self) -> Player:
        """Create a new player with random aggression and zero history."""
        aggr      = np.random.uniform(0.05, 0.95)
        threshold = np.random.uniform(80_000, 200_000)
        p = Player(id=self._next_id, aggression=aggr, stash_threshold=threshold)
        self.players[self._next_id] = p
        self._next_id += 1
        return p

    def sample_queue(self, queue_size: int) -> List[Player]:
        all_players = list(self.players.values())
        indices = np.random.choice(len(all_players), size=min(queue_size, len(all_players)),
                                   replace=False)
        return [all_players[i] for i in indices]

    def churn(self, count: int):
        pids = list(self.players.keys())
        if count >= len(pids):
            return
        retire_ids = np.random.choice(pids, size=count, replace=False)
        for pid in retire_ids:
            del self.players[pid]
            self._spawn_player()

    def get_all_players(self) -> List[Player]:
        return list(self.players.values())

    def get_players_by_classification(self) -> Dict[str, List[Player]]:
        groups = {'passive': [], 'neutral': [], 'aggressive': []}
        for p in self.players.values():
            groups[p.classification].append(p)
        return groups

    def get_stats(self) -> dict:
        players = self.get_all_players()
        aggr = [p.aggression for p in players]
        groups = self.get_players_by_classification()
        experienced = [p for p in players if p.total_raids > 0]
        return {
            'aggression_mean': np.mean(aggr),
            'aggression_std': np.std(aggr),
            'passive_count': len(groups['passive']),
            'neutral_count': len(groups['neutral']),
            'aggressive_count': len(groups['aggressive']),
            'total_extractions': sum(p.total_extractions for p in players),
            'total_deaths': sum(p.total_deaths for p in players),
            'avg_extraction_rate': np.mean([p.extraction_rate for p in experienced]) if experienced else 0,
            'pool_size': len(self.players),
            'total_players_seen': self._next_id,
        }

    def copy(self) -> 'PlayerPool':
        new_pool = PlayerPool.__new__(PlayerPool)
        new_pool.players = {pid: p.copy() for pid, p in self.players.items()}
        new_pool._next_id = self._next_id
        return new_pool


@dataclass
class RaidPlayer:
    """
    Transient player for a single raid.

    visited_zones: set of zone_id integers the player has physically
    reached this raid. Used for partial-observability targeting — the
    player knows which zones they've already found empty and avoids
    revisiting them.

    stash_threshold: inherited from the persistent Player. When stash
    reaches this value the player voluntarily heads to extraction,
    regardless of remaining loot zones.
    """
    id: int
    persistent_id: int
    x: float
    y: float
    aggression: float
    stash_threshold: float = 140_000.0

    hp: float = 100.0
    stash: float = 0.0
    alive: bool = True
    extracted: bool = False
    speed: float = 2.0

    target_x: float = 0.0
    target_y: float = 0.0
    target_type: str = "loot"

    extraction_ticks: int = 0
    extracting_at: Optional[str] = None

    combat_cooldown: int = 0
    no_fight_cooldowns: dict = field(default_factory=dict)
    in_combat_with: Optional[int] = None
    is_attacker: bool = False
    combat_ticks: int = 0

    damage_dealt: float = 0.0
    damage_received: float = 0.0
    kills: int = 0
    items_looted: float = 0.0
    encounters: int = 0

    # Partial observability: zones this player has physically visited
    visited_zones: Set[int] = field(default_factory=set)

    # Per-player private valuation of zones, sampled once at raid start.
    # Dict[zone_id -> float].  Drives divergent routing between players.
    personal_zone_scores: dict = field(default_factory=dict)

    # Zone the player is currently committed to heading toward.
    # None = no commitment (re-evaluate on next tick).
    # Cleared when the player arrives at the zone (visits it).
    current_zone_target: int = -1   # -1 == no target locked

    # Countdown ticks before the current item is collected.
    # Reset to loot_ticks_per_item when a new item starts; 0 = ready.
    loot_ticks_remaining: int = 0

    def is_alive(self) -> bool:
        return self.alive and self.hp > 0

    def is_extracting(self) -> bool:
        return self.extracting_at is not None

    def is_in_combat(self) -> bool:
        return self.in_combat_with is not None

    def can_fight(self, other_id: int) -> bool:
        if self.combat_cooldown > 0 or self.in_combat_with is not None:
            return False
        return other_id not in self.no_fight_cooldowns or self.no_fight_cooldowns[other_id] <= 0

    def decide_to_fight(self) -> bool:
        return np.random.random() < self.aggression

    def deal_damage(self) -> float:
        return np.random.uniform(1 + 4 * self.aggression, 5 + 15 * self.aggression)

    def take_damage(self, dmg: float):
        self.hp -= dmg
        self.damage_received += dmg
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def heal(self, amount: float):
        self.hp = min(100.0, self.hp + amount)

    def distance_to(self, x: float, y: float) -> float:
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)

    def move_toward(self, tx: float, ty: float, map_radius: float):
        dx, dy = tx - self.x, ty - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < self.speed:
            self.x, self.y = tx, ty
        else:
            self.x += self.speed * dx / dist
            self.y += self.speed * dy / dist
        dist_center = np.sqrt(self.x**2 + self.y**2)
        if dist_center > map_radius:
            scale = (map_radius - 0.1) / dist_center
            self.x *= scale
            self.y *= scale

    def update_cooldowns(self):
        if self.combat_cooldown > 0:
            self.combat_cooldown -= 1
        expired = [pid for pid, cd in self.no_fight_cooldowns.items() if cd <= 1]
        for pid in expired:
            del self.no_fight_cooldowns[pid]
        for pid in self.no_fight_cooldowns:
            self.no_fight_cooldowns[pid] -= 1

    def start_combat(self, other_id: int, as_attacker: bool):
        self.in_combat_with = other_id
        self.is_attacker = as_attacker
        self.combat_ticks = 0

    def end_combat(self):
        self.in_combat_with = None
        self.is_attacker = False
        self.combat_ticks = 0

    def set_no_fight_cooldown(self, other_id: int, steps: int = 30):
        self.no_fight_cooldowns[other_id] = steps

    def to_result(self) -> dict:
        return {
            'persistent_id':   self.persistent_id,
            'extracted':       self.extracted,
            'stash':           self.stash if self.extracted else 0,
            'items_looted':    self.items_looted if self.extracted else 0.0,
            'damage_dealt':    self.damage_dealt,
            'damage_received': self.damage_received,
            'kills':           self.kills,
            'encounters':      self.encounters,
            'aggression_used': self.aggression,
        }