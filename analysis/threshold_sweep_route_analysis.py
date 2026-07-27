#!/usr/bin/env python3
"""Route-level re-analysis of the operating-point threshold sweep (paper Table 14,
tab:threshold-sweep).

Lineage: reads the threshold_sweep_v1 dir directly (NOT the canonical bundle).
For each router x threshold x benchmark it computes tier shares, tasks-changing-tier
vs the router's default threshold, and success rate with a paired-bootstrap 95% CI
clustered by task_id.

Output: output/live/threshold_sweep_v1/route_level_analysis.csv
"""
import re
import numpy as np
import pandas as pd

SWEEP_DIR = "/home/kiran/projects/agentic/router_benchmark/output/live/threshold_sweep_v1"
RESULTS = f"{SWEEP_DIR}/results.csv"
OUT = f"{SWEEP_DIR}/route_level_analysis.csv"

TIERS = ["cheap-small", "mid-general", "strong-frontier"]
DEFAULT_THR = {"RouteLLM": 0.50, "Aurelio": 0.30}
N_BOOT = 10000
SEED = 1234


def parse_router(name):
    """Return (family, threshold_float) from strings like
    'RouteLLM (thr=0.50)' or 'Aurelio (score_thr=0.30)'."""
    fam = "RouteLLM" if name.startswith("RouteLLM") else "Aurelio"
    m = re.search(r"=([0-9.]+)", name)
    thr = float(m.group(1)) if m else np.nan
    return fam, thr


def paired_bootstrap_ci(success_by_task, rng, n_boot=N_BOOT):
    """Cluster/pair bootstrap by task_id. success_by_task: list of arrays, one per
    task, holding that task's per-trial success (0/1). Resample tasks with
    replacement; mean is over all rows in the resampled tasks."""
    tasks = success_by_task
    n = len(tasks)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    point = np.mean(np.concatenate(tasks))
    idx = np.arange(n)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=n, replace=True)
        vals = np.concatenate([tasks[i] for i in pick])
        boot[b] = vals.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (point, lo, hi)


def main():
    df = pd.read_csv(RESULTS)
    df["success_int"] = df["success"].map({True: 1, False: 0, "True": 1, "False": 0})
    # Any residual NaN success -> treat as failure (none expected in this sweep)
    n_nan = df["success_int"].isna().sum()
    if n_nan:
        print(f"WARNING: {n_nan} rows with blank success -> counted as failure")
    df["success_int"] = df["success_int"].fillna(0).astype(int)

    fam_thr = df["router_name"].map(parse_router)
    df["family"] = fam_thr.map(lambda x: x[0])
    df["threshold"] = fam_thr.map(lambda x: x[1])

    # ---- Per-task modal tier at each router x benchmark (for tasks-changing-tier).
    # A task has 2 trials; router selection can differ across trials. Use the modal
    # (most frequent) tier per task; ties broken by cheaper tier.
    tier_rank = {t: i for i, t in enumerate(TIERS)}

    def modal_tier(sub):
        vc = sub["selected_candidate"].value_counts()
        top = vc[vc == vc.max()].index
        return sorted(top, key=lambda t: tier_rank.get(t, 99))[0]

    modal = (
        df.groupby(["family", "threshold", "benchmark_name", "task_id"])
        .apply(modal_tier, include_groups=False)
        .rename("modal_tier")
        .reset_index()
    )

    # default-threshold modal tier per family x benchmark x task
    default_rows = []
    for fam, dthr in DEFAULT_THR.items():
        d = modal[(modal.family == fam) & (np.isclose(modal.threshold, dthr))]
        d = d[["benchmark_name", "task_id", "modal_tier"]].rename(
            columns={"modal_tier": "default_tier"}
        )
        d["family"] = fam
        default_rows.append(d)
    default_tier = pd.concat(default_rows, ignore_index=True)

    modal = modal.merge(
        default_tier, on=["family", "benchmark_name", "task_id"], how="left"
    )
    modal["changed"] = modal["modal_tier"] != modal["default_tier"]

    rng = np.random.default_rng(SEED)
    out_rows = []

    for (fam, thr, bench), sub in df.groupby(["family", "threshold", "benchmark_name"]):
        n_rows = len(sub)
        n_tasks = sub["task_id"].nunique()
        n_trials = sub.groupby("task_id").size().max()

        # tier shares over all rows (task x trial)
        shares = {
            f"share_{t}": (sub["selected_candidate"] == t).mean() for t in TIERS
        }

        # tasks-changing-tier vs default threshold (modal tier basis)
        msub = modal[
            (modal.family == fam)
            & (np.isclose(modal.threshold, thr))
            & (modal.benchmark_name == bench)
        ]
        n_changed = int(msub["changed"].sum())
        frac_changed = n_changed / len(msub) if len(msub) else np.nan
        is_default = bool(np.isclose(thr, DEFAULT_THR[fam]))

        # mixed-split flag: does this operating point give a genuine mix?
        # (no single tier holds >=95% of routes AND at least 2 tiers each >=10%)
        share_vals = np.array([shares[f"share_{t}"] for t in TIERS])
        max_share = share_vals.max()
        n_tiers_ge10 = int((share_vals >= 0.10).sum())
        is_mixed = (max_share < 0.95) and (n_tiers_ge10 >= 2)

        # paired bootstrap CI on success, clustered by task
        succ_by_task = [
            g["success_int"].to_numpy()
            for _, g in sub.groupby("task_id")
        ]
        point, lo, hi = paired_bootstrap_ci(succ_by_task, rng)

        row = {
            "family": fam,
            "threshold": thr,
            "benchmark_name": bench,
            "is_default_threshold": is_default,
            "n_tasks": n_tasks,
            "n_trials_per_task": int(n_trials),
            "n_rows": n_rows,
            **{k: round(v, 4) for k, v in shares.items()},
            "max_tier_share": round(float(max_share), 4),
            "n_tiers_ge10pct": n_tiers_ge10,
            "is_mixed_split": is_mixed,
            "tasks_changed_tier": n_changed,
            "frac_tasks_changed_tier": round(float(frac_changed), 4),
            "success_rate": round(float(point), 4),
            "success_ci_lo": round(float(lo), 4),
            "success_ci_hi": round(float(hi), 4),
        }
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values(
        ["benchmark_name", "family", "threshold"], ascending=[True, True, False]
    )
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  ({len(out)} rows)")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for bench in sorted(out.benchmark_name.unique()):
        print(f"\n===== {bench} =====")
        cols = [
            "family", "threshold", "share_cheap-small", "share_mid-general",
            "share_strong-frontier", "is_mixed_split", "tasks_changed_tier",
            "frac_tasks_changed_tier", "success_rate", "success_ci_lo", "success_ci_hi",
        ]
        print(out[out.benchmark_name == bench][cols].to_string(index=False))


if __name__ == "__main__":
    main()
