#!/usr/bin/env python3
"""
Calibrate sr_reward normalisers by measuring actual simulation output ranges.

Detects which fields the simulator tracks (items_looted/encounters vs
stash/kills fallback) and reports the appropriate distributions.
Runs n_lobbies per seed and averages statistics across all seeds.

Usage:
    python calibrate_reward.py
    python calibrate_reward.py --n-lobbies 500 --seeds 42 123 7 99 2024
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
from config import SingleRaidConfig
from player import PlayerPool
from simulator import RaidRunner


def run_single_seed(seed: int, n_lobbies: int, config: SingleRaidConfig,
                    use_new_fields: bool) -> dict:
    """Run n_lobbies random raids for one seed, return raw metric arrays."""
    pool   = PlayerPool(config.pool_size, seed)
    runner = RaidRunner(config)

    loot_vals, encounter_vals, kill_vals, extraction_vals = [], [], [], []

    for ep in range(n_lobbies):
        queue   = pool.sample_queue(config.queue_size)
        indices = np.random.choice(len(queue), config.lobby_size, replace=False)
        lobby   = [queue[i] for i in indices]

        acc = {p.id: {'extracted': 0, 'items_looted': 0.0, 'stash': 0.0,
                      'encounters': 0.0, 'kills': 0.0}
               for p in lobby}

        for rep in range(config.raid_repetitions):
            np.random.seed(seed + ep * config.raid_repetitions + rep)
            random.seed(seed + ep * config.raid_repetitions + rep)
            results = runner.run_single_raid(lobby)
            for r in results:
                pid = r['persistent_id']
                acc[pid]['extracted']    += 1 if r['extracted'] else 0
                acc[pid]['items_looted'] += r.get('items_looted', 0.0)
                acc[pid]['stash']        += r.get('stash', 0.0)
                acc[pid]['encounters']   += r.get('encounters', 0.0)
                acc[pid]['kills']        += r.get('kills', 0.0)

        n = config.raid_repetitions
        extracted_pids = {pid for pid, a in acc.items() if a['extracted'] > n / 2}

        if use_new_fields:
            total_loot = sum(
                a['items_looted'] / n for pid, a in acc.items() if pid in extracted_pids
            )
            total_enc = sum(a['encounters'] / n for a in acc.values())
        else:
            # fallback: stash for extracted players, kills as fight proxy
            total_loot = sum(
                a['stash'] / n for pid, a in acc.items() if pid in extracted_pids
            )
            total_enc = sum(a['kills'] / n for a in acc.values())

        total_kills       = sum(a['kills']   / n for a in acc.values())
        total_extractions = len(extracted_pids)

        loot_vals.append(total_loot)
        encounter_vals.append(total_enc)
        kill_vals.append(total_kills)
        extraction_vals.append(total_extractions)

    return {
        'loot':       np.array(loot_vals),
        'encounters': np.array(encounter_vals),
        'kills':      np.array(kill_vals),
        'extractions':np.array(extraction_vals),
    }


def detect_fields(config: SingleRaidConfig, seed: int) -> bool:
    """Return True if simulator tracks items_looted and encounters."""
    pool   = PlayerPool(config.pool_size, seed)
    runner = RaidRunner(config)
    np.random.seed(seed)
    random.seed(seed)
    queue  = pool.sample_queue(config.queue_size)
    lobby  = [queue[i] for i in np.random.choice(len(queue), config.lobby_size, replace=False)]
    results = runner.run_single_raid(lobby)
    has_items = any('items_looted' in r for r in results)
    has_enc   = any('encounters'   in r for r in results)
    return has_items and has_enc


def print_stats(name: str, arr: np.ndarray) -> dict:
    s = {
        'min': arr.min(), 'p25': np.percentile(arr, 25),
        'median': np.median(arr), 'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90), 'p95': np.percentile(arr, 95),
        'max': arr.max(), 'mean': arr.mean(), 'std': arr.std(),
    }
    print(f"\n{name}:")
    print(f"  min={s['min']:.1f}  p25={s['p25']:.1f}  median={s['median']:.1f}  "
          f"p75={s['p75']:.1f}  p90={s['p90']:.1f}  p95={s['p95']:.1f}  "
          f"max={s['max']:.1f}  mean={s['mean']:.1f}  std={s['std']:.1f}")
    return s


def run_calibration(n_lobbies: int, seeds: list):
    config = SingleRaidConfig()

    # Detect which fields are available
    np.random.seed(seeds[0])
    use_new = detect_fields(config, seeds[0])
    loot_label = ("total_extracted_loot  [items_looted]" if use_new
                  else "total_extracted_loot  [stash fallback — simulator missing items_looted]")
    enc_label  = ("total_encounters       [encounters]" if use_new
                  else "total_encounters       [kills fallback — simulator missing encounters]")

    print(f"Simulator fields: items_looted={'YES' if use_new else 'NO'}, "
          f"encounters={'YES' if use_new else 'NO'}")
    if not use_new:
        print("  ⚠  Fallback to stash/kills. Apply simulator_patch.py for accurate results.\n")

    total = n_lobbies * len(seeds)
    print(f"Running {n_lobbies} lobbies × {len(seeds)} seeds = {total:,} total "
          f"(lobby_size={config.lobby_size}, reps={config.raid_repetitions})...")

    all_loot, all_enc, all_kills, all_ext = [], [], [], []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        print(f"  seed={seed}...", flush=True)
        data = run_single_seed(seed, n_lobbies, config, use_new)
        all_loot.append(data['loot'])
        all_enc.append(data['encounters'])
        all_kills.append(data['kills'])
        all_ext.append(data['extractions'])

    loot_arr = np.concatenate(all_loot)
    enc_arr  = np.concatenate(all_enc)
    kill_arr = np.concatenate(all_kills)
    ext_arr  = np.concatenate(all_ext)

    print(f"\n{'='*65}")
    print(f"CALIBRATION RESULTS  ({len(loot_arr):,} lobbies, {len(seeds)} seeds)")
    print(f"{'='*65}")

    print_stats(loot_label, loot_arr)
    print_stats(enc_label,  enc_arr)
    print_stats("total_kills",       kill_arr)
    print_stats("total_extractions", ext_arr)

    frac_zero_loot = np.mean(loot_arr == 0.0)
    frac_zero_enc  = np.mean(enc_arr  == 0.0)
    print(f"\n  Lobbies with zero loot   (everyone died): {frac_zero_loot:.1%}")
    print(f"  Lobbies with zero fights (no combat):     {frac_zero_enc:.1%}")

    # Compute normalisers on non-zero samples for loot (zero = valid failure, not outlier)
    loot_nonzero = loot_arr[loot_arr > 0]
    if len(loot_nonzero) == 0:
        print("\n  ERROR: all lobbies produced zero loot. Check simulator patch.")
        return

    loot_p75  = np.percentile(loot_nonzero, 75)
    loot_p90  = np.percentile(loot_nonzero, 90)
    kill_p75  = np.percentile(kill_arr, 75)
    kill_p90  = np.percentile(kill_arr, 90)
    kill_min  = kill_arr.min()

    # Cross-seed stability (kills-based fight metric)
    per_seed_loot_p90 = [np.percentile(d[d > 0], 90) if np.any(d > 0) else 0
                         for d in all_loot]
    per_seed_kill_p90 = [np.percentile(d, 90) for d in all_kills]
    loot_cv  = np.std(per_seed_loot_p90) / np.mean(per_seed_loot_p90) if np.mean(per_seed_loot_p90) > 0 else float('nan')
    fight_cv = np.std(per_seed_kill_p90) / np.mean(per_seed_kill_p90) if np.mean(per_seed_kill_p90) > 0 else float('nan')

    # Minimum fight_score with p90 normaliser — determines viable FIGHT_THRESHOLD
    min_fight_score = kill_min / kill_p90

    # FIGHT_THRESHOLD must be above the floor score to ever trigger.
    # We target: ~10-15% of random lobbies penalised for fight_absent.
    # Empirically sample what threshold achieves this.
    target_rate = 0.12
    thresholds  = np.arange(min_fight_score + 0.05, 0.90, 0.01)
    fight_threshold = FIGHT_THRESHOLD_DEFAULT = 0.65
    for t in thresholds:
        rate = np.mean((kill_arr / kill_p90) <= t)
        if rate >= target_rate:
            fight_threshold = round(t, 2)
            break
    fight_abs_rate = np.mean((kill_arr / kill_p90) <= fight_threshold)

    loot_abs  = np.mean((loot_arr / loot_p90) <= 0.05)

    print(f"""
{'='*65}
RECOMMENDED NORMALISERS  (sr_reward.py uses KILLS for fight)
{'='*65}

  FIGHT metric: total_kills (not encounters — encounters have hard
  floor of ~42 making ABSENCE_THRESHOLD unreachable for encounters)
  kills floor={kill_min:.1f}, so fight_score floor={min_fight_score:.3f} with p90 norm.

  ┌──────────────────┬──────────────┬──────────────┐
  │                  │  p75 target  │  p90 target  │
  ├──────────────────┼──────────────┼──────────────┤
  │ LOOT_NORM        │ {loot_p75:>12,.0f} │ {loot_p90:>12,.0f} │
  │ FIGHT_NORM       │ {kill_p75:>12.1f} │ {kill_p90:>12.1f} │
  └──────────────────┴──────────────┴──────────────┘

  Thresholds (absence penalty zone):
    LOOT_THRESHOLD  = 0.05   → penalises {loot_abs:.1%} of random lobbies
                               (matches zero-loot rate of {frac_zero_loot:.1%})
    FIGHT_THRESHOLD = {fight_threshold}  → penalises {fight_abs_rate:.1%} of random lobbies
                               (above floor of {min_fight_score:.3f}, targets ~{target_rate:.0%})

  Cross-seed stability (CV of p90):
    LOOT_NORM  CV = {loot_cv:.3f}
    FIGHT_NORM CV = {fight_cv:.3f}

  Update sr_reward.py:
    LOOT_NORM       = {loot_p90:,.0f}
    FIGHT_NORM      = {kill_p90:.1f}
    LOOT_THRESHOLD  = 0.05
    FIGHT_THRESHOLD = {fight_threshold}
""")


def verify_reward_distribution(n_lobbies: int, seeds: list):
    """
    Verify reward distribution under current sr_reward.py parameters.
    Reports reward cases, component score distributions, and penalty rates.
    """
    from sr_reward import (compute_reward_sr, LOOT_NORM, FIGHT_NORM,
                           LOOT_THRESHOLD, FIGHT_THRESHOLD)

    config = SingleRaidConfig()
    print(f"\n{'='*65}")
    print(f"REWARD DISTRIBUTION VERIFICATION")
    print(f"{'='*65}")
    print(f"  LOOT_NORM={LOOT_NORM:,.0f}   FIGHT_NORM={FIGHT_NORM}")
    print(f"  LOOT_THRESHOLD={LOOT_THRESHOLD}   FIGHT_THRESHOLD={FIGHT_THRESHOLD}")

    all_rewards, cases = [], []
    all_loot_scores, all_fight_scores = [], []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        pool   = PlayerPool(config.pool_size, seed)
        runner = RaidRunner(config)

        for ep in range(n_lobbies):
            queue   = pool.sample_queue(config.queue_size)
            indices = np.random.choice(len(queue), config.lobby_size, replace=False)
            lobby   = [queue[i] for i in indices]

            acc = {p.id: {'extracted': 0, 'items_looted': 0.0,
                          'kills': 0.0, 'encounters': 0.0, 'stash': 0.0}
                   for p in lobby}

            for rep in range(config.raid_repetitions):
                np.random.seed(seed + ep * config.raid_repetitions + rep)
                random.seed(seed + ep * config.raid_repetitions + rep)
                res = runner.run_single_raid(lobby)
                for r in res:
                    pid = r['persistent_id']
                    acc[pid]['extracted']    += 1 if r['extracted'] else 0
                    acc[pid]['items_looted'] += r.get('items_looted', 0.0)
                    acc[pid]['kills']        += r.get('kills', 0.0)
                    acc[pid]['encounters']   += r.get('encounters', 0.0)
                    acc[pid]['stash']        += r.get('stash', 0.0)

            n = config.raid_repetitions
            results_dict = {
                pid: {
                    'persistent_id':   pid,
                    'extracted':       a['extracted'] > n / 2,
                    'items_looted':    a['items_looted'] / n,
                    'kills':           a['kills']        / n,
                    'encounters':      a['encounters']   / n,
                    'stash':           a['stash']        / n,
                }
                for pid, a in acc.items()
            }
            ri = compute_reward_sr(results_dict)
            all_rewards.append(ri['reward'])
            cases.append(ri['reward_case'])
            all_loot_scores.append(ri['loot_score'])
            all_fight_scores.append(ri['fight_score'])

    arr   = np.array(all_rewards)
    loot  = np.array(all_loot_scores)
    fight = np.array(all_fight_scores)
    case_counts = {c: cases.count(c) for c in set(cases)}
    total = len(cases)

    print(f"\n  Component score distributions ({total:,} random lobbies):")
    print(f"  loot_score:  min={loot.min():.3f}  p25={np.percentile(loot,25):.3f}  "
          f"median={np.median(loot):.3f}  p75={np.percentile(loot,75):.3f}  "
          f"max={loot.max():.3f}  [threshold={LOOT_THRESHOLD}]")
    print(f"  fight_score: min={fight.min():.3f}  p25={np.percentile(fight,25):.3f}  "
          f"median={np.median(fight):.3f}  p75={np.percentile(fight,75):.3f}  "
          f"max={fight.max():.3f}  [threshold={FIGHT_THRESHOLD}]")
    print(f"\n  Fraction below threshold (penalty zone):")
    print(f"    loot_score  ≤ {LOOT_THRESHOLD}: {np.mean(loot  <= LOOT_THRESHOLD):.1%}")
    print(f"    fight_score ≤ {FIGHT_THRESHOLD}: {np.mean(fight <= FIGHT_THRESHOLD):.1%}")

    print(f"\n  Reward distribution:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}")
    print(f"    p25={np.percentile(arr,25):.3f}  median={np.median(arr):.3f}  "
          f"p75={np.percentile(arr,75):.3f}  p90={np.percentile(arr,90):.3f}")
    print(f"\n  Reward cases:")
    for case in ['both_present', 'loot_absent', 'fight_absent', 'both_absent']:
        n_case = case_counts.get(case, 0)
        print(f"    {case:20s}: {n_case:5d} / {total} = {n_case/total:.1%}")
    print(f"\n  Fraction reward > 0 (both met):  {np.mean(arr > 0):.1%}")
    print(f"  Fraction reward < 0 (penalised): {np.mean(arr < 0):.1%}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lobbies", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 777])
    parser.add_argument("--verify", action="store_true",
                        help="Also run reward distribution verification")
    args = parser.parse_args()
    run_calibration(args.n_lobbies, args.seeds)
    if args.verify:
        verify_reward_distribution(min(args.n_lobbies, 200), args.seeds)