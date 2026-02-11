#!/usr/bin/env python3
"""
Plot all wandb-logged metrics for every run in the ARC project into a single figure.

Usage:
    python plot_wandb.py                          # defaults to project "ARC"
    python plot_wandb.py --project MyProject
    python plot_wandb.py --entity my-team --project ARC
    python plot_wandb.py --smooth 50              # rolling window size
"""

import argparse
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Metrics logged by all agents (base.run)
SHARED_METRICS = [
    "avg_reward",
    "std_reward",
    "avg_passive_score",
    "avg_aggressive_score",
    "total_extractions",
    "total_deaths",
    "total_kills",
    "extraction_rate",
    "pool_aggression_mean",
    "pool_aggression_std",
    "pool_passive_count",
    "pool_neutral_count",
    "pool_aggressive_count",
]

# Extra metrics logged only by RL_Bandit
RL_METRICS = [
    "train/policy_loss",
    "train/entropy",
    "train/total_loss",
    "train/grad_norm",
    "train/baseline",
    "train/sigma",
    "zero_reward_lobbies",
]


def fetch_runs(entity, project):
    """Fetch all runs from wandb project, return {run_name: DataFrame}."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)

    data = {}
    for run in runs:
        name = run.name
        print(f"  Fetching: {name} ({run.state})")
        hist = run.history(samples=50000, pandas=True)
        if hist.empty:
            print(f"    -> empty, skipping")
            continue
        data[name] = hist
    return data


def smooth(series, window):
    """Rolling mean, ignoring NaNs."""
    return series.rolling(window, min_periods=1).mean()


def plot_all(run_data, window, output_path):
    """Plot all metrics in a single figure."""
    # Determine which metrics are actually present across all runs
    all_keys = set()
    for df in run_data.values():
        all_keys.update(df.columns)

    metrics_to_plot = [m for m in SHARED_METRICS + RL_METRICS if m in all_keys]
    n = len(metrics_to_plot)
    if n == 0:
        print("No plottable metrics found.")
        return

    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle("Matchmaking Experiment — All Logged Metrics", fontsize=16, y=1.01)

    colors = plt.cm.tab10.colors
    run_names = sorted(run_data.keys())
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(run_names)}

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // cols][idx % cols]
        for name in run_names:
            df = run_data[name]
            if metric not in df.columns:
                continue
            series = df[metric].dropna()
            if series.empty:
                continue
            x = series.index if "episode" not in df.columns else df["episode"].iloc[series.index]
            smoothed = smooth(series, window)
            ax.plot(smoothed.values, label=name, color=color_map[name], linewidth=1.2)
            # Light raw data behind
            ax.plot(series.values, color=color_map[name], alpha=0.12, linewidth=0.5)

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot all wandb metrics for ARC experiments")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity (team/user)")
    parser.add_argument("--project", type=str, default="ARC", help="Wandb project name")
    parser.add_argument("--smooth", type=int, default=20, help="Rolling average window")
    parser.add_argument("--output", type=str, default="wandb_metrics.png", help="Output file")
    args = parser.parse_args()

    print(f"Fetching runs from wandb project '{args.project}'...")
    run_data = fetch_runs(args.entity, args.project)

    if not run_data:
        print("No runs found.")
        return

    print(f"\nPlotting {len(run_data)} runs (smooth window={args.smooth})...")
    plot_all(run_data, args.smooth, args.output)


if __name__ == "__main__":
    main()