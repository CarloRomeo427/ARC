"""
Single-raid reward function v7.

Reward structure
----------------
  TWO REQUIRED BEHAVIOURS:
    Passive:    extracted loot value  → total items_looted by extracted players
    Aggressive: combat intensity      → total kills across all players

  NORMALISERS — derived from simulation constants, NO calibration:

    LOOT:  total_extracted_loot / stash_budget
      stash_budget = sum(p.stash_threshold for p in lobby)
      Semantics: "what fraction of what players came to collect was actually
      extracted?"
      Range: [0, 1]. Can never exceed 1 because stash_budget is the maximum
      loot demand.
      Floor: supply_ratio * extraction_rate ≈ 0.65 * 0.54 ≈ 0.28 for a
      random lobby. Worst observed (near-wipeout): 0.097.

    FIGHT: total_kills / (lobby_size - 1)
      lobby_size - 1 = 11 = maximum possible kills (one survivor left).
      Semantics: "what fraction of possible PvP deaths actually occurred?"
      Range: [0, 1]. Physically bounded.
      Floor: 2.2 / 11 ≈ 0.20 (incidental combat even in passive lobbies).
      Max observed: 8.6 / 11 ≈ 0.78 (aggressive lobbies with high kill rate).

  Why NOT kills/sum_aggression_used:
    Passive lobbies (low aggression) still generate kills due to forced
    proximity at hot loot zones. kills/aggression is higher for passive
    lobbies (≈0.99) than aggressive ones (≈0.90) — inverse of desired
    signal. Useless as a differentiator.

  Why NOT damage_dealt:
    Range 855–1128 across 300 random lobbies, CV=7%. Essentially constant
    regardless of lobby composition. Provides zero gradient.

  ABSENCE THRESHOLDS:
    LOOT_THRESHOLD  = 0.15 — penalizes lobbies extracting < 15 % of demand.
                             Hits ≈4.6 % of random lobbies (near-wipeouts
                             and very passive+low-extract compositions).

    FIGHT_THRESHOLD = 0.42 — penalizes lobbies with < 4.6 kills.
                             Hits ≈12 % of random lobbies (passive-only
                             lobbies where incidental kills barely fire).
                             Floor score 0.20 is well below threshold,
                             so the penalty is reachable and discriminative.

  REWARD FORMULA:
    both_present (loot > T_l AND fight > T_f):
        reward = loot_score * fight_score + 0.1 * (loot_score + fight_score)
        Typical max ≈ 0.53 * 0.78 + 0.1*(0.53+0.78) ≈ 0.54

    loot_absent  (loot ≤ T_l):
        reward = -0.6 * (1 - loot_score / T_l)

    fight_absent (fight ≤ T_f):
        reward = -0.6 * (1 - fight_score / T_f)

    both_absent:
        reward = -1.0

  Random lobby baseline (500 lobbies, uniform(500,10k) items, 3-tick
  looting, 20 zones, radial hot/cold):
    loot_score:  mean=0.278  std=0.071
    fight_score: mean=0.511  std=0.075
    both_present: 87 %   fight_absent: 12 %   loot_absent: 1 %
    reward:  mean=0.199  std=0.081

  NOTE on stash_budget parameter:
    compute_reward_sr now requires stash_budget as a second argument.
    Callers already have the lobby object:
      stash_budget = sum(p.stash_threshold for p in lobby)
    This replaces all LOOT_NORM magic numbers.
"""

import numpy as np
from typing import Dict

# ── Thresholds ─────────────────────────────────────────────────────────────
LOOT_THRESHOLD  = 0.15    # < 15 % of stash demand extracted → penalty
FIGHT_THRESHOLD = 0.42    # < 4.6 kills (out of 11 possible) → penalty

# ── Penalty magnitudes ─────────────────────────────────────────────────────
SINGLE_MISS_PENALTY = 0.6
BOTH_MISS_PENALTY   = 1.0

# ── Fight normaliser (simulation constant, not a calibration number) ────────
# Maximum kills possible = lobby_size - 1 (one survivor).
# Passed as parameter so the function works for any lobby_size.
# Default 11 = 12 - 1 for the standard SingleRaidConfig.
DEFAULT_LOBBY_SIZE = 12


def compute_reward_sr(results: Dict[int, dict], stash_budget: float,
                      lobby_size: int = DEFAULT_LOBBY_SIZE) -> dict:
    """
    Compute single-raid reward.

    Args:
        results:      {persistent_id: {'extracted': bool, 'items_looted': float,
                                       'kills': float, 'encounters': float, ...}}
        stash_budget: sum(p.stash_threshold for p in lobby)
                      The economic demand of this lobby. Used to normalise
                      extracted loot without any calibrated magic numbers.
        lobby_size:   number of players in the lobby (default 12).
                      Used to compute max_possible_kills = lobby_size - 1.
    """
    total_extracted_loot = sum(
        r.get('items_looted', 0.0)
        for r in results.values()
        if r.get('extracted', False)
    )
    total_kills       = sum(r.get('kills',      0.0) for r in results.values())
    total_encounters  = sum(r.get('encounters', 0.0) for r in results.values())
    total_extractions = sum(1 for r in results.values() if r.get('extracted', False))

    if total_extracted_loot == 0.0 and total_kills == 0.0:
        return _fallback_reward(results, stash_budget, lobby_size, total_extractions)

    max_kills   = max(lobby_size - 1, 1)
    loot_score  = min(total_extracted_loot / max(stash_budget, 1.0), 1.0)
    fight_score = min(total_kills / max_kills, 1.0)

    reward, reward_case = _apply_formula(loot_score, fight_score)

    return {
        'reward':               reward,
        'loot_score':           loot_score,
        'fight_score':          fight_score,
        'total_extracted_loot': total_extracted_loot,
        'total_kills':          total_kills,
        'total_encounters':     total_encounters,
        'total_extractions':    total_extractions,
        'reward_case':          reward_case,
    }


def _apply_formula(loot_score: float, fight_score: float):
    loot_absent  = loot_score  <= LOOT_THRESHOLD
    fight_absent = fight_score <= FIGHT_THRESHOLD

    if loot_absent and fight_absent:
        return -BOTH_MISS_PENALTY, 'both_absent'
    elif loot_absent:
        severity = 1.0 - (loot_score / LOOT_THRESHOLD)
        return -SINGLE_MISS_PENALTY * severity, 'loot_absent'
    elif fight_absent:
        severity = 1.0 - (fight_score / FIGHT_THRESHOLD)
        return -SINGLE_MISS_PENALTY * severity, 'fight_absent'
    else:
        reward = loot_score * fight_score + 0.1 * (loot_score + fight_score)
        return reward, 'both_present'


def _fallback_reward(results: Dict[int, dict], stash_budget: float,
                     lobby_size: int, total_extractions: int) -> dict:
    """Fallback when items_looted is unavailable — uses stash value instead."""
    total_stash = sum(
        r.get('stash', 0.0)
        for r in results.values()
        if r.get('extracted', False)
    )
    total_kills = sum(r.get('kills', 0.0) for r in results.values())

    max_kills   = max(lobby_size - 1, 1)
    loot_score  = min(total_stash / max(stash_budget, 1.0), 1.0)
    fight_score = min(total_kills / max_kills, 1.0)
    reward, reward_case = _apply_formula(loot_score, fight_score)

    return {
        'reward':               reward,
        'loot_score':           loot_score,
        'fight_score':          fight_score,
        'total_extracted_loot': total_stash,
        'total_kills':          total_kills,
        'total_encounters':     0.0,
        'total_extractions':    total_extractions,
        'reward_case':          reward_case,
    }