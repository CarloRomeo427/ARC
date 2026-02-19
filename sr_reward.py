"""
Single-raid reward function v3.

Reward structure
----------------
  TWO REQUIRED BEHAVIOURS:
    Passive players' key feature:    extracted loot VALUE (items_looted for
                                     players who successfully extracted)
    Aggressive players' key feature: combat ENCOUNTERS (raw engagement count,
                                     double-counted for both participants)

  Normalisers (empirical targets, not hard caps):
    LOOT_NORM  = 15_000  ~6 passive players extracting ~2,500 value each
    FIGHT_NORM = 48      ~4 aggressive players × ~12 engagements × 2 sides

  Scoring:
    loot_score  = clip(total_extracted_loot / LOOT_NORM,  0, 1)
    fight_score = clip(total_encounters      / FIGHT_NORM, 0, 1)

  REWARD FORMULA:
    If BOTH components > ABSENCE_THRESHOLD (= 0.05):
        reward = loot_score * fight_score          # product: requires BOTH
               + 0.1 * (loot_score + fight_score)  # partial credit for each
        range: (0, 1.2]

    If ONE component is at or below ABSENCE_THRESHOLD:
        reward = -SINGLE_MISS_PENALTY * (1 - present_score)
        The penalty scales with how badly the missing component is absent.
        A near-miss (fight_score=0.04) still punishes less than a total miss
        (fight_score=0.0), but both are clearly negative.
        range: [-SINGLE_MISS_PENALTY, 0)

    If BOTH components are at or below ABSENCE_THRESHOLD:
        reward = -BOTH_MISS_PENALTY
        Worst case: the lobby generated neither loot nor combat. This
        indicates a completely miscalibrated composition.
        range: -BOTH_MISS_PENALTY (constant)

  Signal semantics:
    reward > 0   → valid composition, both objectives achieved to some degree
    reward < 0   → at least one objective completely failed
    reward = -1.0 → lobby was entirely useless (no loot, no fights)

  This asymmetry ensures:
    1. The RLOO advantage is strongly negative for homogeneous lobbies,
       not just "less positive" — the policy has a clear cliff to escape.
    2. Near-misses (one score barely above threshold) still receive a
       positive signal, so the policy isn't discouraged from making progress
       on one objective while still building the other.
    3. Missing both is strictly worse than missing one, preserving correct
       ordering for gradient updates.
"""

import numpy as np
from typing import Dict

LOOT_NORM           = 15_000.0
FIGHT_NORM          = 48.0
ABSENCE_THRESHOLD   = 0.05   # below this a component counts as absent
SINGLE_MISS_PENALTY = 0.6    # max penalty when one objective is absent
BOTH_MISS_PENALTY   = 1.0    # penalty when both objectives are absent


def compute_reward_sr(results: Dict[int, dict]) -> dict:
    """
    Compute single-raid reward from _simulate_lobbies output.

    Args:
        results: {persistent_id: result_dict}
                 Required keys: extracted, items_looted, encounters, stash, kills

    Returns:
        dict with: reward, loot_score, fight_score,
                   total_extracted_loot, total_encounters,
                   total_kills, total_extractions, reward_case
    """
    total_extracted_loot = sum(
        r.get('items_looted', 0.0)
        for r in results.values()
        if r.get('extracted', False)
    )
    total_encounters  = sum(r.get('encounters', 0.0) for r in results.values())
    total_kills       = sum(r.get('kills',      0.0) for r in results.values())
    total_extractions = sum(1 for r in results.values() if r.get('extracted', False))

    # Fallback to stash/kills if simulator doesn't track the new fields
    if total_extracted_loot == 0.0 and total_encounters == 0.0:
        return _fallback_reward(results, total_kills, total_extractions)

    loot_score  = min(total_extracted_loot / LOOT_NORM,  1.0)
    fight_score = min(total_encounters      / FIGHT_NORM, 1.0)

    loot_absent  = loot_score  <= ABSENCE_THRESHOLD
    fight_absent = fight_score <= ABSENCE_THRESHOLD

    if loot_absent and fight_absent:
        # Both objectives failed — worst case
        reward      = -BOTH_MISS_PENALTY
        reward_case = 'both_absent'

    elif loot_absent:
        # No one extracted useful loot: passive players failed entirely
        # Penalty scales: full penalty when loot_score=0, 0 as loot_score→threshold
        miss_severity = 1.0 - (loot_score / ABSENCE_THRESHOLD)
        reward        = -SINGLE_MISS_PENALTY * miss_severity
        reward_case   = 'loot_absent'

    elif fight_absent:
        # No combat occurred: aggressive players either weren't selected or were passive
        miss_severity = 1.0 - (fight_score / ABSENCE_THRESHOLD)
        reward        = -SINGLE_MISS_PENALTY * miss_severity
        reward_case   = 'fight_absent'

    else:
        # Both objectives achieved: product reward + partial credit
        reward      = loot_score * fight_score + 0.1 * (loot_score + fight_score)
        reward_case = 'both_present'

    return {
        'reward':               reward,
        'loot_score':           loot_score,
        'fight_score':          fight_score,
        'total_extracted_loot': total_extracted_loot,
        'total_encounters':     total_encounters,
        'total_kills':          total_kills,
        'total_extractions':    total_extractions,
        'reward_case':          reward_case,
    }


def _fallback_reward(results: Dict[int, dict],
                     total_kills: float, total_extractions: int) -> dict:
    """
    Fallback for simulators that don't track items_looted / encounters.
    Uses stash value and kills as proxies. Same penalty logic applies.
    """
    STASH_NORM = 15_000.0
    KILL_NORM  = 6.0

    total_stash = sum(
        r.get('stash', 0.0)
        for r in results.values()
        if r.get('extracted', False)
    )
    loot_score  = min(total_stash / STASH_NORM, 1.0)
    fight_score = min(total_kills / KILL_NORM,  1.0)

    loot_absent  = loot_score  <= ABSENCE_THRESHOLD
    fight_absent = fight_score <= ABSENCE_THRESHOLD

    if loot_absent and fight_absent:
        reward      = -BOTH_MISS_PENALTY
        reward_case = 'both_absent'
    elif loot_absent:
        miss_severity = 1.0 - (loot_score / ABSENCE_THRESHOLD)
        reward        = -SINGLE_MISS_PENALTY * miss_severity
        reward_case   = 'loot_absent'
    elif fight_absent:
        miss_severity = 1.0 - (fight_score / ABSENCE_THRESHOLD)
        reward        = -SINGLE_MISS_PENALTY * miss_severity
        reward_case   = 'fight_absent'
    else:
        reward      = loot_score * fight_score + 0.1 * (loot_score + fight_score)
        reward_case = 'both_present'

    return {
        'reward':               reward,
        'loot_score':           loot_score,
        'fight_score':          fight_score,
        'total_extracted_loot': total_stash,   # proxy
        'total_encounters':     total_kills,   # proxy
        'total_kills':          total_kills,
        'total_extractions':    total_extractions,
        'reward_case':          reward_case,
    }