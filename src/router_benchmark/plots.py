"""Deliverable (4): plots/visualizations generated from harness output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def plot_pareto_frontier(overall_df: pd.DataFrame, out_path: Path, label: str = "simulated evaluation") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    frontier = overall_df[overall_df["is_pareto_optimal"]].sort_values("mean_cost_per_task_usd")

    def display_name(name: str) -> str:
        return (
            name.replace(" (live)", "")
            .replace(" Semantic Router", " SR")
            .replace(" Router", "")
        )

    ax.scatter(
        overall_df["mean_cost_per_task_usd"],
        overall_df["mean_success_rate"],
        s=60,
        c="#4C72B0",
        alpha=0.85,
        zorder=3,
        label="Router operating point",
    )
    ax.plot(
        frontier["mean_cost_per_task_usd"],
        frontier["mean_success_rate"],
        c="#DD8452",
        lw=1.5,
        ls="--",
        zorder=2,
        label="Pareto frontier",
    )
    for _, row in overall_df.iterrows():
        name = display_name(row["router_name"])
        offset = {
            "LiteLLM": (4, 8),
            "RouteLLM": (4, -12),
            "vLLM SR": (4, 6),
            "Aurelio SR": (4, 6),
        }.get(name, (4, 4))
        ax.annotate(
            name,
            (row["mean_cost_per_task_usd"], row["mean_success_rate"]),
            fontsize=7,
            xytext=offset,
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per task (USD, log scale)")
    ax.set_ylabel("Mean task success rate")
    ax.set_title(f"Cost vs. Quality Pareto Frontier Across Routers\n({label})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_success_heatmap(per_bm_df: pd.DataFrame, out_path: Path, label: str = "simulated evaluation") -> None:
    pivot = per_bm_df.pivot(index="router_name", columns="benchmark_name", values="success_rate")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Task success rate")
    ax.set_title(f"Router Success Rate by Benchmark ({label})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_latency_distribution(results_df: pd.DataFrame, out_path: Path, label: str = "simulated evaluation") -> None:
    routers = sorted(results_df["router_name"].unique())
    data = [results_df.loc[results_df["router_name"] == r, "latency_ms"] for r in routers]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8CB9BD")
        patch.set_alpha(0.7)
    ax.set_yticklabels(routers, fontsize=8)
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_title(f"Latency Distribution by Router, All Benchmarks Pooled\n({label})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fallback_and_stability(overall_df: pd.DataFrame, out_path: Path, label: str = "simulated evaluation") -> None:
    df = overall_df.sort_values("mean_fallback_rate", ascending=False)
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df["mean_fallback_rate"], width, label="Fallback rate", color="#C44E52")
    ax.bar(x + width / 2, df["mean_route_stability"], width, label="Route stability", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["router_name"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_title(f"Fallback Frequency vs. Route Stability by Router\n({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_all_plots(results_df: pd.DataFrame, per_bm_df: pd.DataFrame, overall_df: pd.DataFrame, out_dir: Path, label: str = "simulated evaluation") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pareto_frontier.png": lambda p: plot_pareto_frontier(overall_df, p, label),
        "success_heatmap.png": lambda p: plot_success_heatmap(per_bm_df, p, label),
        "latency_distribution.png": lambda p: plot_latency_distribution(results_df, p, label),
        "fallback_stability.png": lambda p: plot_fallback_and_stability(overall_df, p, label),
    }
    written = []
    for fname, fn in paths.items():
        p = out_dir / fname
        fn(p)
        written.append(p)
    return written
