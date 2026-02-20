"""
Single-raid reward function v5.

Reward structure
----------------
  TWO REQUIRED BEHAVIOURS:
    Passive:    extracted loot value  → total items_looted by extracted players
    Aggressive: combat intensity      → total kills across all players

  Normalisers (calibrated from 1,500 random lobbies, 3 seeds):
    LOOT_NORM  = 176,000   (p90 of non-zero extracted loot)
    FIGHT_NORM = 9.6       (p90 of total kills)

  Component scores:
    loot_score  = clip(total_extracted_loot / LOOT_NORM,  0, 1)
    fight_score = clip(total_kills          / FIGHT_NORM, 0, 1)

  ABSENCE THRESHOLDS — deliberately asymmetric:
    LOOT_THRESHOLD  = 0.05  — loot CAN reach 0 (everyone dies → 0 extractions)
                               so 0.05 is a genuine near-zero floor check.
                               Empirically: 2.1% of random lobbies hit this.

    FIGHT_THRESHOLD = 0.65  — kills minimum is ~5.0, fight_score minimum is
                               5.0/9.6 = 0.52. Setting threshold=0.50 never
                               triggers. At 0.65 the penalty triggers below
                               6.24 kills, covering truly passive-only lobbies
                               (kills ≈ 5–6) which are ~10–15% of random draws.
                               This creates the required hard signal.

  REWARD FORMULA:
    If loot_score > LOOT_THRESHOLD AND fight_score > FIGHT_THRESHOLD:
        reward = loot_score * fight_score
               + 0.1 * (loot_score + fight_score)
        range: (0, 1.2]

    If loot_score ≤ LOOT_THRESHOLD:
        miss_severity = 1 - (loot_score / LOOT_THRESHOLD)
        reward = -SINGLE_MISS_PENALTY * miss_severity
        range: [-0.6, 0)

    If fight_score ≤ FIGHT_THRESHOLD:
        miss_severity = 1 - (fight_score / FIGHT_THRESHOLD)
        reward = -SINGLE_MISS_PENALTY * miss_severity
        range: [-0.6, 0)

    If both absent:
        reward = -BOTH_MISS_PENALTY = -1.0

  Signal semantics:
    reward > 0    → both objectives achieved, magnitude reflects quality
    reward ∈ (-0.6, 0) → one objective failed, severity reflects how badly
    reward = -0.6 → one objective completely absent
    reward = -1.0 → both absent (e.g. all-neutral lobby that neither
                    extracts nor fights effectively)
"""

import numpy as np
from typing import Dict

LOOT_NORM       = 325_807.0
FIGHT_NORM      = 8.6
LOOT_THRESHOLD  = 0.05
FIGHT_THRESHOLD = 0.66
SINGLE_MISS_PENALTY = 0.6
BOTH_MISS_PENALTY   = 1.0


def compute_reward_sr(results: Dict[int, dict]) -> dict:
    total_extracted_loot = sum(
        r.get('items_looted', 0.0)
        for r in results.values()
        if r.get('extracted', False)
    )
    total_kills       = sum(r.get('kills',      0.0) for r in results.values())
    total_encounters  = sum(r.get('encounters', 0.0) for r in results.values())
    total_extractions = sum(1 for r in results.values() if r.get('extracted', False))

    if total_extracted_loot == 0.0 and total_kills == 0.0:
        return _fallback_reward(results, total_extractions)

    loot_score  = min(total_extracted_loot / LOOT_NORM, 1.0)
    fight_score = min(total_kills          / FIGHT_NORM, 1.0)

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


def _fallback_reward(results: Dict[int, dict], total_extractions: int) -> dict:
    """Fallback when items_looted unavailable — uses stash/kills."""
    total_stash = sum(
        r.get('stash', 0.0)
        for r in results.values()
        if r.get('extracted', False)
    )
    total_kills = sum(r.get('kills', 0.0) for r in results.values())

    loot_score  = min(total_stash / LOOT_NORM,  1.0)
    fight_score = min(total_kills / FIGHT_NORM, 1.0)
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