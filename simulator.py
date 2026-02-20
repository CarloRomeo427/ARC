"""
Raid simulation: map, loot zones, extraction points, and raid logic.

Loot zone redesign (v2)
-----------------------
  - Zones are clustered near the map centre using an exponential distance
    distribution. Central zones are hot (high value_multiplier); outer
    zones cool off with a radial decay + log-normal noise so the gradient
    is non-trivial to exploit.

  - Each LootZone carries an expected_value_score fixed at layout
    generation. Players target zones by this score (partial observability):
    they know which zones are generally valuable from meta-knowledge, but
    cannot see whether a zone has been depleted by other players. They
    discover depletion on arrival and mark the zone as visited.

  - Per-raid value_multiplier is derived from expected_value_score plus
    small per-raid noise, so actual item values vary each rep while
    preserving the hot-zone hierarchy.

  - Mean item value base raised from 400 → 5000, giving per-item values
    of ~1500 (outer) to ~15000 (central). Total map supply is intentionally
    less than total player demand (12 × avg_threshold ≈ 1.7 M vs supply
    ≈ 700 k), forcing competition.

Spawn permutation (v2)
----------------------
  - Spawn positions are computed once (fixed geometry, N evenly-spaced
    points on the perimeter). Each call to run_single_raid shuffles the
    assignment of positions to players using the current numpy seed, which
    is set by the caller before the call. Repetitions with different seeds
    therefore produce different spawn assignments.

Extraction logic (v2)
---------------------
  should_extract now returns True in three cases:
    1. Tick budget: not enough ticks to reach extraction and wait.
    2. Stash threshold: player has accumulated >= stash_threshold loot and
       voluntarily leaves.
    3. All zones visited: player has physically visited every loot zone
       and found them all — nothing left to do.
"""

import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

from player import Player, RaidPlayer
from config import ExperimentConfig


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================================
# Map primitives
# =========================================================================

@dataclass
class LootZone:
    x: float
    y: float
    radius: float
    total_items: int
    remaining_items: int
    value_multiplier: float          # per-raid, set in reset()
    expected_value_score: float      # fixed at layout gen, player-visible prior
    zone_id: int                     # stable identity for visited_zones tracking

    def loot_item(self) -> Optional[float]:
        if self.remaining_items <= 0:
            return None
        self.remaining_items -= 1
        # Item value uniform [500, 10000] — slows looting, forces multi-zone foraging
        return np.random.uniform(500, 10_000)

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


# =========================================================================
# GameMap
# =========================================================================

class GameMap:
    """
    Fixed map layout with per-raid loot randomisation.

    Layout (zone positions, radii, expected_value_scores) is generated
    once from map_seed and never changes. Per-raid content (remaining
    items, actual value_multiplier with noise) is re-randomised in
    reset() before each raid.
    """

    def __init__(self, config, map_seed: int):
        self.config = config
        self.radius = config.map_radius
        self.loot_zones: List[LootZone] = []
        self.extraction_points: List[ExtractionPoint] = []
        self._generate_fixed_layout(map_seed)

    def _generate_fixed_layout(self, seed: int):
        """
        Place loot zones with strong exponential clustering toward the centre.

        Spatial layout (unchanged from v1):
          max_dist    = 0.75 * radius  — zones confined to inner 75 % of map
          decay_scale = max_dist * 0.30 — tight exponential; median zone at ~21 units
          min_spacing = zone_radius + 4 — zones are visually distinct

        EVS gradient:
          exp(-d / 0.45*max_dist) * LogNormal(0, 0.35)
          Centre d=0  → evs ≈ 1.0  (hot)
          Edge   d=75 → evs ≈ 0.11 (cold)

        Item counts (set in reset()) are proportional to EVS weight so that
        hot central zones contain more loot than cold outer zones, and the
        total supply stays fixed regardless of num_loot_zones.
        """
        rng = np.random.RandomState(seed)
        max_dist    = 0.75 * self.radius
        decay_scale = max_dist * 0.30

        def evs_to_radius(evs: float) -> float:
            """Hot zones are visually larger; cold zones are small dots."""
            t = float(np.clip(evs / 1.2, 0.0, 1.0))
            return 3.0 + t * 11.0   # range [3, 14]

        self.loot_zones = []
        for zone_id in range(self.config.num_loot_zones):
            placed = False
            for _ in range(300):
                d     = min(rng.exponential(decay_scale), max_dist)
                angle = rng.uniform(0, 2 * np.pi)
                x, y  = d * np.cos(angle), d * np.sin(angle)
                noise = rng.lognormal(mean=0.0, sigma=0.35)
                evs   = float(np.clip(np.exp(-d / (max_dist * 0.45)) * noise, 0.05, 4.0))
                zone_radius = evs_to_radius(evs)

                if all(
                    np.sqrt((x - z.x)**2 + (y - z.y)**2) >= zone_radius + z.radius + 4
                    for z in self.loot_zones
                ):
                    self.loot_zones.append(LootZone(
                        x=x, y=y, radius=zone_radius,
                        total_items=0, remaining_items=0,
                        value_multiplier=1.0,
                        expected_value_score=evs,
                        zone_id=zone_id,
                    ))
                    placed = True
                    break

            if not placed:
                d     = min(rng.exponential(decay_scale), max_dist)
                angle = rng.uniform(0, 2 * np.pi)
                x, y  = d * np.cos(angle), d * np.sin(angle)
                noise = rng.lognormal(mean=0.0, sigma=0.35)
                evs   = float(np.clip(np.exp(-d / (max_dist * 0.45)) * noise, 0.05, 4.0))
                self.loot_zones.append(LootZone(
                    x=x, y=y, radius=evs_to_radius(evs),
                    total_items=0, remaining_items=0,
                    value_multiplier=1.0,
                    expected_value_score=evs,
                    zone_id=zone_id,
                ))

        # Four cardinal extraction points near the perimeter
        d_ext = 0.9 * self.radius
        self.extraction_points = [
            ExtractionPoint(0,      d_ext, 5.0, "N"),
            ExtractionPoint(0,     -d_ext, 5.0, "S"),
            ExtractionPoint(d_ext,  0,     5.0, "E"),
            ExtractionPoint(-d_ext, 0,     5.0, "W"),
        ]

    def reset(self, stash_budget: float = 0.0):
        """
        Distribute a fixed loot budget across zones proportional to their EVS.

        stash_budget  : sum of stash_threshold for the current lobby.
                        0.0 falls back to a sensible per-zone default so
                        the map can still be reset without a lobby (tests).

        Total loot value = BUDGET_FRACTION * stash_budget.
        Each zone receives:  items = max(1, round(evs_weight * total_items))
        where total_items = total_loot_value / BASE_ITEM_VALUE.

        Hot zones get more items (higher item count × same BASE_ITEM_VALUE),
        cold zones get fewer.  Total supply is constant regardless of how
        many zones exist — adding more zones redistributes, not inflates.
        """
        # Budget: total raid loot < sum of player stash thresholds.
        # BUDGET_FRACTION < 1.0 guarantees scarcity: only a minority of
        # players can fill their stash from looting alone. The rest must
        # keep foraging — spending more time on the map and accumulating
        # fight risk — until the tick budget forces them to extract empty.
        #
        # Mean item value = (500 + 10000) / 2 = 5250 (uniform distribution).
        # BUDGET_FRACTION = 0.65: total loot ≈ 65% of total demand.
        # With 12 players and mean threshold 140k → budget ≈ 1.09M → ~208
        # items across 20 zones → ~10 items/zone on average.
        # Hot central zones get ~15-20 items; cold outer zones get 1-3.
        BUDGET_FRACTION = 0.65   # < 1.0: guaranteed scarcity; ~40% of players can't fill threshold from loot
        BASE_ITEM_VALUE = 5_250  # mean of Uniform(500, 10000)

        if stash_budget > 0:
            total_loot_value  = BUDGET_FRACTION * stash_budget
            total_items_float = total_loot_value / BASE_ITEM_VALUE
        else:
            # Fallback: ~10 items per zone
            total_items_float = 10.0 * len(self.loot_zones)

        evs_sum    = sum(z.expected_value_score for z in self.loot_zones) or 1.0
        evs_weights = [z.expected_value_score / evs_sum for z in self.loot_zones]

        for z, w in zip(self.loot_zones, evs_weights):
            # Per-raid noise ±10 % preserves hot/cold hierarchy across repetitions
            per_raid_noise    = max(0.1, np.random.normal(1.0, 0.10))
            z.value_multiplier = per_raid_noise   # fixed BASE_ITEM_VALUE; multiplier is noise only
            n_items            = max(0, round(w * total_items_float * per_raid_noise))
            z.total_items      = n_items
            z.remaining_items  = n_items

        for e in self.extraction_points:
            e.reset()

    # ------------------------------------------------------------------
    # Targeting helpers
    # ------------------------------------------------------------------

    def get_best_zone_for_player(
        self,
        px: float, py: float,
        aggression: float,
        visited_zone_ids: Set[int],
        personal_zone_scores: dict,
    ) -> Optional[LootZone]:
        """
        Return the best unvisited zone for a specific player.

        Score = personal_evs[zone] * exp(-distance * (1 - aggression) * k)

        k = 0.015 calibrated so that at max map distance (~140 units) and
        zero aggression the distance penalty reduces a zone's score by ~88 %,
        while at full aggression (1.0) the penalty vanishes and the player
        targets purely by expected value.

        personal_zone_scores contains per-player lognormal perturbations of
        the global expected_value_score, sampled once at raid start so the
        ranking is stable across the raid but differs between players.

        Players cannot see depletion — they discover it on arrival.
        """
        DIST_K = 0.015
        unvisited = [z for z in self.loot_zones if z.zone_id not in visited_zone_ids]
        if not unvisited:
            return None
        best, best_score = None, -1.0
        for z in unvisited:
            personal_evs = personal_zone_scores.get(z.zone_id, z.expected_value_score)
            dist         = np.sqrt((px - z.x)**2 + (py - z.y)**2)
            dist_penalty = np.exp(-dist * (1.0 - aggression) * DIST_K)
            score        = personal_evs * dist_penalty
            if score > best_score:
                best_score = score
                best       = z
        return best

    def get_closest_extraction(self, x: float, y: float) -> 'ExtractionPoint':
        available = [e for e in self.extraction_points if e.is_available()]
        pool = available if available else self.extraction_points
        return min(pool, key=lambda e: np.sqrt((e.x - x)**2 + (e.y - y)**2))

    def get_spawn_positions(self, n: int) -> List[Tuple[float, float]]:
        spawn_radius = 0.9 * self.radius
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
        return [(spawn_radius * np.cos(a), spawn_radius * np.sin(a)) for a in angles]

    def tick_extractions(self):
        for ext in self.extraction_points:
            ext.tick_cooldown()


# =========================================================================
# Raid
# =========================================================================

class Raid:
    def __init__(self, game_map: GameMap, players: List[RaidPlayer], config):
        self.map     = game_map
        self.players = players
        self.config  = config
        self.tick    = 0

    def get_alive(self) -> List[RaidPlayer]:
        return [p for p in self.players if p.is_alive() and not p.extracted]

    def get_visible(self, player: RaidPlayer) -> List[RaidPlayer]:
        return [
            p for p in self.get_alive()
            if p.id != player.id
            and player.distance_to(p.x, p.y) <= self.config.sight_radius
        ]

    def should_extract(self, p: RaidPlayer) -> bool:
        """
        Return True if the player should head to extraction now.

        Three independent triggers:
          1. Tick budget  — too few ticks remaining to loot and still extract.
          2. Stash filled — player has reached their personal stash threshold.
          3. Map exhausted — player has visited every loot zone already.
        """
        ext = self.map.get_closest_extraction(p.x, p.y)
        ticks_to_extract = (
            p.distance_to(ext.x, ext.y) / p.speed + self.config.extraction_time
        )
        caution = 1.0 + (1.0 - p.aggression) * 0.5   # passive players leave earlier
        ticks_remaining = self.config.max_ticks - self.tick

        # 1. Tick budget (hard constraint — leave now or never make it)
        if ticks_remaining <= ticks_to_extract * caution:
            return True

        # 2. Stash threshold — voluntary departure
        if p.stash >= p.stash_threshold:
            return True

        # Note: "all zones visited" is NOT a trigger here.
        # When every zone is visited, get_best_zone_for_player returns None
        # and select_target naturally falls through to extraction. Triggering
        # here caused premature exits when players had only visited empty
        # cold outer zones, not the actual hot loot zones.
        return False

    def select_target(self, p: RaidPlayer) -> Tuple[float, float, str]:
        """
        Decide the player's next movement target.

        Priority order:
          1. Extraction  — if should_extract() is True.
          2. Enemy player — aggressive players (>0.7) chase visible targets.
          3. Best unvisited loot zone — all players target by expected value.
          4. Extraction fallback — if every zone has been visited.
        """
        if self.should_extract(p):
            ext = self.map.get_closest_extraction(p.x, p.y)
            return ext.x, ext.y, "extraction"

        # Aggressive players hunt visible enemies first
        if p.aggression > 0.7:
            visible = self.get_visible(p)
            if visible:
                t = max(visible, key=lambda x: x.stash)
                return t.x, t.y, "player"

        # All players: target best unvisited zone, personalised by position + aggression
        # Commit to current locked zone if still unvisited (avoid jitter)
        if p.current_zone_target != -1 and p.current_zone_target not in p.visited_zones:
            locked = next(
                (z for z in self.map.loot_zones if z.zone_id == p.current_zone_target),
                None,
            )
            if locked is not None:
                return locked.x, locked.y, "loot"

        zone = self.map.get_best_zone_for_player(
            p.x, p.y, p.aggression, p.visited_zones, p.personal_zone_scores
        )
        if zone is not None:
            p.current_zone_target = zone.zone_id
            return zone.x, zone.y, "loot"

        # No unvisited zones remain — go extract
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
            ratio = p1.aggression / (p1.aggression + p2.aggression + 0.01)
            a, d  = (p1, p2) if np.random.random() < ratio else (p2, p1)
        a.start_combat(d.id, True)
        d.start_combat(a.id, False)
        a.encounters += 1
        d.encounters += 1

    def process_combat(self, p: RaidPlayer):
        if not p.is_in_combat():
            return
        other = next((x for x in self.players if x.id == p.in_combat_with), None)
        if not other or not other.is_alive():
            p.end_combat()
            return
        if not p.is_attacker:
            return
        p.combat_ticks  += 1
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
            flee_prob = self.config.flee_base + (1 - other.aggression) * 0.4
            if np.random.random() < flee_prob:
                p.end_combat();     other.end_combat()
                p.combat_cooldown   = self.config.post_combat_cooldown
                other.combat_cooldown = self.config.post_combat_cooldown
                p.set_no_fight_cooldown(other.id, 50)
                other.set_no_fight_cooldown(p.id, 50)
                return

        if p.combat_ticks >= self.config.combat_max_ticks:
            p.end_combat();     other.end_combat()
            p.combat_cooldown   = self.config.post_combat_cooldown
            other.combat_cooldown = self.config.post_combat_cooldown
            p.set_no_fight_cooldown(other.id, 40)
            other.set_no_fight_cooldown(p.id, 40)
            return

        p.is_attacker     = False
        other.is_attacker = True

    def tick_player(self, p: RaidPlayer):
        if not p.is_alive() or p.extracted:
            return

        p.update_cooldowns()

        # --- Combat phase ---
        if p.is_in_combat():
            if p.is_extracting():
                p.extraction_ticks = 0
                p.extracting_at    = None
            self.process_combat(p)
            return

        # --- Encounter check with visible players ---
        for other in self.get_visible(p):
            if other.is_alive() and not other.is_in_combat() and p.can_fight(other.id):
                self.resolve_encounter(p, other)
                if not p.is_alive() or p.is_in_combat():
                    return

        # --- Extraction phase ---
        ext = next(
            (e for e in self.map.extraction_points if e.contains_point(p.x, p.y)),
            None,
        )
        if ext and p.target_type == "extraction":
            if ext.is_available():
                if p.extracting_at == ext.name:
                    p.extraction_ticks += 1
                else:
                    p.extracting_at    = ext.name
                    p.extraction_ticks = 1
                if p.extraction_ticks >= self.config.extraction_time:
                    p.extracted = True
                    ext.start_cooldown(self.config.extraction_cooldown)
                return
            else:
                p.extracting_at    = None
                p.extraction_ticks = 0
        elif p.is_extracting():
            p.extracting_at    = None
            p.extraction_ticks = 0

        # --- Loot phase (partial observability) ---
        # Check whether the player is currently inside any loot zone.
        current_zone = next(
            (z for z in self.map.loot_zones if z.contains_point(p.x, p.y)),
            None,
        )
        if current_zone is not None and p.target_type == "loot":
            # Mark as visited the moment the player arrives — they now
            # know the real state of this zone.
            p.visited_zones.add(current_zone.zone_id)
            # Clear zone lock — player has arrived, re-evaluate next tick
            if p.current_zone_target == current_zone.zone_id:
                p.current_zone_target = -1

            if not current_zone.is_empty():
                # Multi-tick looting: player spends TICKS_PER_ITEM ticks
                # crouching over each item before picking it up.
                # This slows extraction and increases fight exposure time.
                TICKS_PER_ITEM = self.config.loot_ticks_per_item
                p.loot_ticks_remaining -= 1
                if p.loot_ticks_remaining > 0:
                    return  # still working on current item
                # Item collected — reset counter and loot
                p.loot_ticks_remaining = TICKS_PER_ITEM
                v = current_zone.loot_item()
                if v:
                    p.stash        += v
                    p.items_looted += v
                return
            else:
                # Zone depleted — reset loot counter and pick next target
                p.loot_ticks_remaining = 0
                p.target_x, p.target_y, p.target_type = self.select_target(p)
        else:
            # Not in any loot zone: reset loot counter and move
            p.loot_ticks_remaining = 0
            p.target_x, p.target_y, p.target_type = self.select_target(p)

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


# =========================================================================
# RaidRunner
# =========================================================================

class RaidRunner:
    def __init__(self, config):
        self.config   = config
        self.game_map = GameMap(config, map_seed=config.master_seed)

    def run_single_raid(self, lobby: List[Player]) -> List[dict]:
        """
        Run one raid.

        Spawn positions are fixed geometry (N evenly-spaced perimeter
        points), but the assignment of positions to players is shuffled
        using the current numpy RNG state. Callers must seed numpy before
        calling this method to get reproducible, seed-varying assignments.
        """
        stash_budget = sum(p.stash_threshold for p in lobby)
        self.game_map.reset(stash_budget)
        spawns = self.game_map.get_spawn_positions(len(lobby))
        # Permute spawn assignment — varies with seed, cheap, one line
        perm = np.random.permutation(len(spawns))
        # Per-player lognormal perturbation of zone scores.
        # σ=0.4 produces ~50 % std relative to mean — enough for players
        # to disagree on zone ranking without being completely uncorrelated.
        SCORE_SIGMA = 0.65  # raised: creates stronger per-player zone ranking divergence
        raid_players = []
        for i, p in enumerate(lobby):
            personal_scores = {
                z.zone_id: z.expected_value_score * np.random.lognormal(0.0, SCORE_SIGMA)
                for z in self.game_map.loot_zones
            }
            raid_players.append(RaidPlayer(
                id=i,
                persistent_id=p.id,
                x=spawns[perm[i]][0],
                y=spawns[perm[i]][1],
                aggression=p.get_raid_aggression(self.config.aggression_noise_std),
                stash_threshold=p.stash_threshold,
                personal_zone_scores=personal_scores,
            ))
        return Raid(self.game_map, raid_players, self.config).run()

    def run_averaged_raid(self, lobby: List[Player], base_seed: int) -> Dict[int, dict]:
        """Legacy helper used by multi-raid agents."""
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
                acc[pid]['stash']     += r['stash']
                acc[pid]['dmg_dealt'] += r['damage_dealt']
                acc[pid]['dmg_recv']  += r['damage_received']
                acc[pid]['kills']     += r['kills']
                acc[pid]['aggr']      += r['aggression_used']
        n = self.config.raid_repetitions
        return {
            pid: {
                'extracted':       a['extracted'] > n / 2,
                'stash':           a['stash']     / n,
                'damage_dealt':    a['dmg_dealt'] / n,
                'damage_received': a['dmg_recv']  / n,
                'kills':           a['kills']     / n,
                'aggression_used': a['aggr']      / n,
            }
            for pid, a in acc.items()
        }


# =========================================================================
# Legacy multi-raid reward (unchanged — used by non-SR agents)
# =========================================================================

def compute_reward(raid_results, classifications) -> dict:
    passive_results    = []
    aggressive_results = []
    for r in raid_results:
        cls = classifications.get(r['persistent_id'], 'neutral')
        if cls == 'passive':
            passive_results.append(r)
        elif cls == 'aggressive':
            aggressive_results.append(r)

    if passive_results:
        p_ext      = sum(1 for r in passive_results if r['extracted'])
        p_ext_rate = p_ext / len(passive_results)
        p_stash    = sum(r['stash'] for r in passive_results if r['extracted'])
        p_avg_stash    = p_stash / max(1, p_ext)
        p_stash_score  = min(p_avg_stash / 100_000, 1.0)
        passive_score  = p_ext_rate * (0.5 + 0.5 * p_stash_score)
    else:
        passive_score  = 0.0

    if aggressive_results:
        a_kills        = sum(r['kills'] for r in aggressive_results)
        a_kills_per    = a_kills / len(aggressive_results)
        a_ext          = sum(1 for r in aggressive_results if r['extracted'])
        a_ext_rate     = a_ext / len(aggressive_results)
        a_kill_score   = min(a_kills_per / 2, 1.0)
        aggressive_score = a_kill_score * (0.5 + 0.5 * a_ext_rate)
    else:
        aggressive_score = 0.0

    pareto = (
        np.sqrt(passive_score * aggressive_score)
        if passive_score > 0 and aggressive_score > 0
        else 0.0
    )
    return {
        'pareto':            pareto,
        'passive_score':     passive_score,
        'aggressive_score':  aggressive_score,
        'passive_count':     len(passive_results),
        'aggressive_count':  len(aggressive_results),
    }