"""
Raid simulation: map, loot zones, extraction points, and raid logic.
"""

import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from player import Player, RaidPlayer
from config import ExperimentConfig


def seed_everything(seed: int):
    """Set all PRNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class LootZone:
    x: float
    y: float
    radius: float
    total_items: int
    remaining_items: int
    value_multiplier: float

    def loot_item(self) -> Optional[float]:
        if self.remaining_items <= 0:
            return None
        self.remaining_items -= 1
        return np.random.exponential(400 * self.value_multiplier)

    def is_empty(self) -> bool:
        return self.remaining_items <= 0

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2

    def reset(self):
        self.remaining_items = self.total_items


@dataclass
class ExtractionPoint:
    x: float
    y: float
    radius: float = 5.0
    name: str = ""
    cooldown: int = 0

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2

    def is_available(self) -> bool:
        return self.cooldown <= 0

    def start_cooldown(self, ticks: int):
        self.cooldown = ticks

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def reset(self):
        self.cooldown = 0


class GameMap:
    """
    Game map with fixed structure and per-raid loot randomization.

    Like real extraction shooters: the map layout (loot zone positions,
    extraction points) is generated once from a seed and never changes.
    Only the loot contents (item counts, value multipliers) are
    re-randomized each raid via reset().
    """

    def __init__(self, config: ExperimentConfig, map_seed: int):
        self.config = config
        self.radius = config.map_radius
        self.loot_zones: List[LootZone] = []
        self.extraction_points: List[ExtractionPoint] = []
        self._generate_fixed_layout(map_seed)

    def _generate_fixed_layout(self, seed: int):
        """Generate map structure once. Positions and radii are permanent."""
        rng = np.random.RandomState(seed)  # local RNG, won't touch global state
        max_dist = 0.5 * self.radius

        self.loot_zones = []
        for _ in range(self.config.num_loot_zones):
            for _ in range(100):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, max_dist)
                x, y = dist * np.cos(angle), dist * np.sin(angle)
                radius = rng.uniform(5, 15)
                if all(np.sqrt((x - z.x)**2 + (y - z.y)**2) >= radius + z.radius + 5
                       for z in self.loot_zones):
                    if dist + radius <= max_dist + 10:
                        # Position and radius are fixed; items/value set per-raid in reset()
                        self.loot_zones.append(LootZone(
                            x=x, y=y, radius=radius,
                            total_items=0, remaining_items=0,
                            value_multiplier=1.0,
                        ))
                        break

        d = 0.9 * self.radius
        self.extraction_points = [
            ExtractionPoint(0, d, 5.0, "N"),
            ExtractionPoint(0, -d, 5.0, "S"),
            ExtractionPoint(d, 0, 5.0, "E"),
            ExtractionPoint(-d, 0, 5.0, "W"),
        ]

    def reset(self):
        """
        Randomize loot contents for a new raid. Map structure stays fixed.

        Uses the current global numpy state, which should be seeded
        deterministically by the raid runner before calling this.
        """
        for z in self.loot_zones:
            base_items = max(3, int(np.pi * z.radius**2 * 0.3))
            z.total_items = max(1, int(np.random.normal(base_items, base_items * 0.2)))
            z.remaining_items = z.total_items
            z.value_multiplier = max(0.3, np.random.normal(z.radius / 5, 0.3))
        for e in self.extraction_points:
            e.reset()

    def get_spawn_positions(self, n: int) -> List[Tuple[float, float]]:
        spawn_radius = 0.9 * self.radius
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n
        return [(spawn_radius * np.cos(a), spawn_radius * np.sin(a)) for a in angles]

    def get_closest_loot_zone(self, x: float, y: float) -> Optional[LootZone]:
        valid = [z for z in self.loot_zones if not z.is_empty()]
        if not valid:
            return None
        return min(valid, key=lambda z: np.sqrt((z.x-x)**2 + (z.y-y)**2))

    def get_closest_extraction(self, x: float, y: float) -> ExtractionPoint:
        available = [e for e in self.extraction_points if e.is_available()]
        if available:
            return min(available, key=lambda e: np.sqrt((e.x-x)**2 + (e.y-y)**2))
        return min(self.extraction_points, key=lambda e: np.sqrt((e.x-x)**2 + (e.y-y)**2))

    def tick_extractions(self):
        for ext in self.extraction_points:
            ext.tick_cooldown()


class Raid:
    def __init__(self, game_map: GameMap, players: List[RaidPlayer], config: ExperimentConfig):
        self.map = game_map
        self.players = players
        self.config = config
        self.tick = 0

    def get_alive(self) -> List[RaidPlayer]:
        return [p for p in self.players if p.is_alive() and not p.extracted]

    def get_visible(self, player: RaidPlayer) -> List[RaidPlayer]:
        return [p for p in self.get_alive()
                if p.id != player.id and player.distance_to(p.x, p.y) <= self.config.sight_radius]

    def should_extract(self, p: RaidPlayer) -> bool:
        ext = self.map.get_closest_extraction(p.x, p.y)
        ticks_needed = (p.distance_to(ext.x, ext.y) / p.speed + self.config.extraction_time) * 1.5
        caution = 1 + (1 - p.aggression) * 0.5
        return (self.config.max_ticks - self.tick) <= ticks_needed * caution

    def select_target(self, p: RaidPlayer) -> Tuple[float, float, str]:
        if self.should_extract(p):
            ext = self.map.get_closest_extraction(p.x, p.y)
            return ext.x, ext.y, "extraction"
        if p.aggression > 0.7:
            visible = self.get_visible(p)
            if visible:
                t = max(visible, key=lambda x: x.stash)
                return t.x, t.y, "player"
            zones = [z for z in self.map.loot_zones if not z.is_empty()]
            if zones:
                z = max(zones, key=lambda x: x.radius)
                return z.x, z.y, "loot"
        zone = self.map.get_closest_loot_zone(p.x, p.y)
        if zone:
            return zone.x, zone.y, "loot"
        ext = self.map.get_closest_extraction(p.x, p.y)
        return ext.x, ext.y, "extraction"

    def resolve_encounter(self, p1: RaidPlayer, p2: RaidPlayer):
        if not p1.can_fight(p2.id) or not p2.can_fight(p1.id):
            return
        f1, f2 = p1.decide_to_fight(), p2.decide_to_fight()
        if not f1 and not f2:
            p1.set_no_fight_cooldown(p2.id, 30)
            p2.set_no_fight_cooldown(p1.id, 30)
            return
        if f1 and not f2:
            a, d = p1, p2
        elif f2 and not f1:
            a, d = p2, p1
        else:
            a, d = (p1, p2) if np.random.random() < p1.aggression/(p1.aggression+p2.aggression+0.01) else (p2, p1)
        a.start_combat(d.id, True)
        d.start_combat(a.id, False)

    def process_combat(self, p: RaidPlayer):
        if not p.is_in_combat():
            return
        other = next((x for x in self.players if x.id == p.in_combat_with), None)
        if not other or not other.is_alive():
            p.end_combat()
            return
        if not p.is_attacker:
            return
        p.combat_ticks += 1
        other.combat_ticks += 1
        dmg = p.deal_damage()
        other.take_damage(dmg)
        p.damage_dealt += dmg
        if not other.is_alive():
            p.stash += other.stash
            other.stash = 0
            p.kills += 1
            p.heal(self.config.heal_on_kill)
            p.end_combat()
            other.end_combat()
            p.combat_cooldown = self.config.post_combat_cooldown
            return
        if other.hp < self.config.flee_hp_threshold:
            if np.random.random() < self.config.flee_base + (1-other.aggression)*0.4:
                p.end_combat()
                other.end_combat()
                p.combat_cooldown = self.config.post_combat_cooldown
                other.combat_cooldown = self.config.post_combat_cooldown
                p.set_no_fight_cooldown(other.id, 50)
                other.set_no_fight_cooldown(p.id, 50)
                return
        if p.combat_ticks >= self.config.combat_max_ticks:
            p.end_combat()
            other.end_combat()
            p.combat_cooldown = self.config.post_combat_cooldown
            other.combat_cooldown = self.config.post_combat_cooldown
            p.set_no_fight_cooldown(other.id, 40)
            other.set_no_fight_cooldown(p.id, 40)
            return
        p.is_attacker = False
        other.is_attacker = True

    def tick_player(self, p: RaidPlayer):
        if not p.is_alive() or p.extracted:
            return
        p.update_cooldowns()
        if p.is_in_combat():
            if p.is_extracting():
                p.extraction_ticks = 0
                p.extracting_at = None
            self.process_combat(p)
            return
        for other in self.get_visible(p):
            if other.is_alive() and not other.is_in_combat() and p.can_fight(other.id):
                self.resolve_encounter(p, other)
                if not p.is_alive() or p.is_in_combat():
                    return
        ext = next((e for e in self.map.extraction_points if e.contains_point(p.x, p.y)), None)
        if ext and p.target_type == "extraction":
            if ext.is_available():
                if p.extracting_at == ext.name:
                    p.extraction_ticks += 1
                else:
                    p.extracting_at = ext.name
                    p.extraction_ticks = 1
                if p.extraction_ticks >= self.config.extraction_time:
                    p.extracted = True
                    ext.start_cooldown(self.config.extraction_cooldown)
                return
            else:
                p.extracting_at = None
                p.extraction_ticks = 0
        else:
            if p.is_extracting():
                p.extracting_at = None
                p.extraction_ticks = 0
        p.target_x, p.target_y, p.target_type = self.select_target(p)
        zone = next((z for z in self.map.loot_zones if z.contains_point(p.x, p.y)), None)
        if zone and not zone.is_empty() and p.target_type == "loot":
            v = zone.loot_item()
            if v:
                p.stash += v
            return
        p.move_toward(p.target_x, p.target_y, self.map.radius)

    def run_tick(self):
        self.map.tick_extractions()
        alive = self.get_alive()
        np.random.shuffle(alive)
        for p in alive:
            self.tick_player(p)
        self.tick += 1

    def run(self) -> List[dict]:
        while self.tick < self.config.max_ticks and self.get_alive():
            self.run_tick()
        return [p.to_result() for p in self.players]


class RaidRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Map layout is fixed for the entire experiment, seeded from master_seed
        self.game_map = GameMap(config, map_seed=config.master_seed)

    def run_single_raid(self, lobby: List[Player]) -> List[dict]:
        self.game_map.reset()
        spawns = self.game_map.get_spawn_positions(len(lobby))
        raid_players = [
            RaidPlayer(
                id=i, persistent_id=p.id,
                x=x, y=y,
                aggression=p.get_raid_aggression(self.config.aggression_noise_std),
            )
            for i, (p, (x, y)) in enumerate(zip(lobby, spawns))
        ]
        return Raid(self.game_map, raid_players, self.config).run()

    def run_averaged_raid(self, lobby: List[Player], base_seed: int) -> Dict[int, dict]:
        acc = {p.id: {'extracted': 0, 'stash': 0, 'dmg_dealt': 0,
                      'dmg_recv': 0, 'kills': 0, 'aggr': 0}
               for p in lobby}
        for rep in range(self.config.raid_repetitions):
            raid_seed = base_seed + rep
            np.random.seed(raid_seed)
            random.seed(raid_seed)
            results = self.run_single_raid(lobby)
            for r in results:
                pid = r['persistent_id']
                acc[pid]['extracted'] += 1 if r['extracted'] else 0
                acc[pid]['stash'] += r['stash']
                acc[pid]['dmg_dealt'] += r['damage_dealt']
                acc[pid]['dmg_recv'] += r['damage_received']
                acc[pid]['kills'] += r['kills']
                acc[pid]['aggr'] += r['aggression_used']
        n = self.config.raid_repetitions
        return {
            pid: {
                'extracted': a['extracted'] > n / 2,
                'stash': a['stash'] / n,
                'damage_dealt': a['dmg_dealt'] / n,
                'damage_received': a['dmg_recv'] / n,
                'kills': a['kills'] / n,
                'aggression_used': a['aggr'] / n,
            }
            for pid, a in acc.items()
        }


def compute_reward(raid_results: List[dict], classifications: Dict[int, str]) -> dict:
    passive_results = []
    aggressive_results = []
    for r in raid_results:
        cls = classifications.get(r['persistent_id'], 'neutral')
        if cls == 'passive':
            passive_results.append(r)
        elif cls == 'aggressive':
            aggressive_results.append(r)

    if passive_results:
        p_ext = sum(1 for r in passive_results if r['extracted'])
        p_ext_rate = p_ext / len(passive_results)
        p_stash = sum(r['stash'] for r in passive_results if r['extracted'])
        p_avg_stash = p_stash / max(1, p_ext)
        p_stash_score = min(p_avg_stash / 100000, 1.0)
        passive_score = p_ext_rate * (0.5 + 0.5 * p_stash_score)
    else:
        passive_score = 0.0

    if aggressive_results:
        a_kills = sum(r['kills'] for r in aggressive_results)
        a_kills_per = a_kills / len(aggressive_results)
        a_ext = sum(1 for r in aggressive_results if r['extracted'])
        a_ext_rate = a_ext / len(aggressive_results)
        a_kill_score = min(a_kills_per / 2, 1.0)
        aggressive_score = a_kill_score * (0.5 + 0.5 * a_ext_rate)
    else:
        aggressive_score = 0.0

    if passive_score > 0 and aggressive_score > 0:
        pareto = np.sqrt(passive_score * aggressive_score)
    else:
        pareto = 0.0

    return {
        'pareto': pareto,
        'passive_score': passive_score,
        'aggressive_score': aggressive_score,
        'passive_count': len(passive_results),
        'aggressive_count': len(aggressive_results),
    }