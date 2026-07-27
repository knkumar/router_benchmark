#!/usr/bin/env python3
"""Generate Paper 1 LaTeX fragments from canonical rebuild artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from router_benchmark.scripts._paths import repository_root

ROOT = repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.protocol.canonical import validate_bundle


REQUIRED_ANALYSIS_FILES = (
    "paired_effects.csv",
    "paired_draws.json",
    "rank_uncertainty.csv",
    "pareto_uncertainty.csv",
    "candidate_tier_summary.csv",
    "baseline_summary.csv",
    "baseline_reconciliation.json",
    "bfcl_route_equivalence.csv",
    "ablation_registry.json",
    "vllm_share_permutation.csv",
    "vllm_share_permutation.json",
    "canonical_metric_suite.csv",
    "pool_ablation_comparison.csv",
    "expected_utility.csv",
    "cascade_operating_metrics.csv",
)

COMPARATIVE_EXCLUDED_BENCHMARKS: set[str] = set()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical analysis artifacts: " + ", ".join(missing))


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _number(value: object, digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _usd(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _tabular(headers: list[str], rows: list[list[object]], align: str) -> str:
    if not rows:
        raise ValueError("LaTeX tables cannot be empty")
    body = [
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(body)


def _short_benchmark(name: str) -> str:
    return {
        "RouterBench (live)": "RouterBench",
        "BFCL v4 (live)": "BFCL v4",
        "tau2-bench (live)": "tau2-bench",
        "WebArena (live)": "WebArena",
    }.get(name, name)


def _short_router(name: str) -> str:
    return (
        name.replace(" Router (live)", "")
        .replace(" Semantic Router (live)", "")
        .replace(" (live)", "")
    )


def _write_rebuild_summary(bundle: Path, analysis_dir: Path, output: Path) -> None:
    manifest = _read_json(bundle / "manifest.json")
    provenance = _read_json(bundle / "provenance.json")
    candidate_rows = _read_csv(bundle / "candidate_outcomes.csv")
    route_rows = _read_csv(bundle / "routes.csv")
    candidate_spend = sum(float(row["model_api_cost_usd"]) for row in candidate_rows)
    router_spend = sum(float(row["router_service_usd"]) for row in route_rows)
    external_spend = float(provenance.get("external_metered_usd", 0.0) or 0.0)
    infrastructure_spend = provenance.get("infrastructure_usd", "not recorded")
    if infrastructure_spend == "not recorded":
        total_spend = candidate_spend + router_spend + external_spend
    else:
        total_spend = candidate_spend + router_spend + external_spend + float(infrastructure_spend)
    rows: list[list[object]] = []
    for benchmark, counts in sorted(manifest["benchmark_counts"].items()):
        rows.append([
            _short_benchmark(benchmark),
            counts["unique_tasks"],
            counts["outcome_replicates_per_task_candidate"],
            counts["total_outcome_rows"],
            counts["total_route_rows"],
        ])
    output.write_text(
        _tabular(
            ["Benchmark", "Tasks", "Replicates", "Candidate rows", "Route rows"],
            rows,
            "lrrrr",
        ),
        encoding="utf-8",
    )
    spend_output = output.with_name("canonical_spend_summary.tex")
    spend_output.write_text(
        _tabular(
            ["Ledger", "Value"],
            [
                ["Bundle", str(bundle)],
                ["Analysis directory", str(analysis_dir)],
                ["Candidate rows observed", len(candidate_rows)],
                ["Route rows observed", len(route_rows)],
                ["Candidate model API USD", _usd(candidate_spend, 8)],
                ["Router service USD", _usd(router_spend, 8)],
                ["External metered USD", _usd(external_spend, 8)],
                ["Infrastructure USD", infrastructure_spend],
                ["Total observed USD", _usd(total_spend, 8)],
            ],
            "lp{5.6cm}",
        ),
        encoding="utf-8",
    )


def _write_candidate_tiers(rows: list[dict[str, str]], output: Path) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: (item["benchmark_id"], item["candidate_id"])):
        table_rows.append([
            _short_benchmark(row["benchmark_id"]),
            row["candidate_id"],
            row["candidate_rows"],
            _number(row["success_rate"]),
            _usd(row["model_api_cost_usd_mean"]),
            _number(row["generation_latency_ms_mean"], 1),
        ])
    output.write_text(
        _tabular(
            ["Benchmark", "Candidate", "n", "Success", "Mean USD", "Mean ms"],
            table_rows,
            "llrrrr",
        ),
        encoding="utf-8",
    )


def _write_baselines(rows: list[dict[str, str]], output: Path) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: (item["baseline_id"], item["benchmark_id"])):
        baseline = row["baseline_id"].replace(" Baseline (live)", "")
        table_rows.append([
            baseline,
            _short_benchmark(row["benchmark_id"]),
            row["candidate_rows"],
            _number(row["success_rate"]),
            _usd(row["model_api_cost_usd_mean"]),
        ])
    output.write_text(
        _tabular(["Baseline", "Benchmark", "n", "Success", "Mean USD"], table_rows, "llrrr"),
        encoding="utf-8",
    )


def _write_router_summary(bundle: Path, output: Path) -> None:
    """Summarize joined router outcomes without regenerating candidate outputs."""
    router_configs = _read_json(bundle / "router_configs.json")
    candidates = {
        f"{row['benchmark_id']}|{row['task_id']}|{row['candidate_id']}|{row['outcome_replicate']}": row
        for row in _read_csv(bundle / "candidate_outcomes.csv")
    }
    routes = _read_csv(bundle / "routes.csv")
    route_costs = {
        (row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"]): float(row["router_service_usd"])
        for row in routes
    }
    grouped: dict[tuple[str, str], list[tuple[dict[str, str], float]]] = defaultdict(list)
    for outcome in _read_csv(bundle / "outcomes.csv"):
        candidate = candidates[outcome["candidate_outcome_key"]]
        route_cost = route_costs[(outcome["router_config_id"], outcome["benchmark_id"], outcome["task_id"], outcome["routing_seed"])]
        grouped[(outcome["benchmark_id"], outcome["router_config_id"])].append((candidate, route_cost))

    table_rows = []
    for (benchmark, router_id), values in sorted(grouped.items()):
        if benchmark in COMPARATIVE_EXCLUDED_BENCHMARKS:
            continue
        candidate_rows = [candidate for candidate, _ in values]
        router_name = _short_router(router_configs[router_id]["router_name"])
        table_rows.append([
            _short_benchmark(benchmark),
            router_name,
            len(values),
            _number(sum(row["success"] == "true" for row in candidate_rows) / len(candidate_rows)),
            _usd(sum(float(row["model_api_cost_usd"]) for row in candidate_rows) / len(candidate_rows)),
            _usd(sum(route_cost for _, route_cost in values) / len(values)),
        ])
    output.write_text(
        _tabular(
            ["Benchmark", "Router", "n", "Success", "Candidate USD", "Router USD"],
            table_rows,
            "llrrrr",
        ),
        encoding="utf-8",
    )


def _write_route_selection_summary(bundle: Path, output: Path) -> None:
    router_configs = _read_json(bundle / "router_configs.json")
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for route in _read_csv(bundle / "routes.csv"):
        grouped[(route["benchmark_id"], route["router_config_id"])][route["selected_candidate"]] += 1

    table_rows = []
    for (benchmark, router_id), counts in sorted(grouped.items()):
        if benchmark in COMPARATIVE_EXCLUDED_BENCHMARKS:
            continue
        total = sum(counts.values())
        table_rows.append([
            _short_benchmark(benchmark),
            _short_router(router_configs[router_id]["router_name"]),
            total,
            _number(counts["cheap-small"] / total),
            _number(counts["mid-general"] / total),
            _number(counts["strong-frontier"] / total),
        ])
    output.write_text(
        _tabular(["Benchmark", "Router", "Routes", "Cheap", "Mid", "Strong"], table_rows, "llrrrr"),
        encoding="utf-8",
    )


def _write_paired_effects(rows: list[dict[str, str]], output: Path) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: (item["benchmark_id"], item["router_1"], item["router_2"])):
        if row["benchmark_id"] in COMPARATIVE_EXCLUDED_BENCHMARKS:
            continue
        interval = (
            f"{_number(row['risk_difference'])} "
            f"[{_number(row['risk_difference_ci_low'])}, {_number(row['risk_difference_ci_high'])}]"
        )
        table_rows.append([
            _short_benchmark(row["benchmark_id"]),
            _short_router(row["router_1"]),
            _short_router(row["router_2"]),
            row["paired_n"],
            row["n01"],
            row["n10"],
            interval,
            _number(row["adjusted_p_value"]),
            row["reject_null"],
        ])
    output.write_text(
        _tabular(
            ["Benchmark", "Router 1", "Router 2", "n", "n01", "n10", "RD [CI]", "p adj", "Reject"],
            table_rows,
            "lllrrrlrl",
        ),
        encoding="utf-8",
    )


def _write_vllm_share_permutation(rows: list[dict[str, str]], output: Path) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: item["benchmark_id"]):
        table_rows.append([
            _short_benchmark(row["benchmark_id"]),
            row["route_positions"],
            f"{row['cheap_routes']}/{row['mid_routes']}/{row['strong_routes']}",
            _number(row["actual_success"]),
            _number(row["null_mean_success"]),
            _number(row["actual_minus_null"]),
            f"[{_number(row['null_ci_low'])}, {_number(row['null_ci_high'])}]",
            _number(row["holm_adjusted_two_sided_p_value"]),
        ])
    output.write_text(
        _tabular(
            ["Benchmark", "Routes", "C/M/S", "Actual", "Null mean", "Delta", "Null 95%", "Holm p"],
            table_rows,
            "lrrrrrrr",
        ),
        encoding="utf-8",
    )


def _write_rank_uncertainty(rows: list[dict[str, str]], output: Path) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: (item["benchmark_id"], float(item["mean_rank"]), item["router_name"])):
        if row["benchmark_id"] in COMPARATIVE_EXCLUDED_BENCHMARKS:
            continue
        table_rows.append([
            _short_benchmark(row["benchmark_id"]),
            _short_router(row["router_name"]),
            row["draws"],
            _number(row["mean_rank"]),
            _number(row["rank_variance"]),
        ])
    output.write_text(
        _tabular(["Benchmark", "Router", "Draws", "Mean rank", "Rank var."], table_rows, "llrrr"),
        encoding="utf-8",
    )


def _write_pareto_uncertainty(rows: list[dict[str, str]], output: Path) -> None:
    table_rows = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["benchmark_id"],
            -float(item["pareto_nondominance_probability"]),
            item["router_name"],
        ),
    ):
        if row["benchmark_id"] in COMPARATIVE_EXCLUDED_BENCHMARKS:
            continue
        table_rows.append([
            _short_benchmark(row["benchmark_id"]),
            _short_router(row["router_name"]),
            row["draws"],
            _number(row["pareto_nondominance_probability"]),
        ])
    output.write_text(
        _tabular(["Benchmark", "Router", "Draws", "Pareto prob."], table_rows, "llrr"),
        encoding="utf-8",
    )


_ALL_POLICY_ORDER = (
    ("Aurelio Semantic Router (live)", "Aurelio Sem.", "router"),
    ("vLLM Semantic Router (live)", "vLLM Sem.", "router"),
    ("LiteLLM Router (live)", "LiteLLM", "router"),
    ("RouteLLM (live)", "RouteLLM", "router"),
    ("Always-Cheapest", "Always-Cheap", "fixed-tier"),
    ("Always-Mid", "Always-Mid", "fixed-tier"),
    ("Always-Strongest", "Always-Strong", "fixed-tier"),
)

_ALL_POLICY_BENCHMARKS = (
    ("RouterBench (live)", "RouterB."),
    ("BFCL v4 (live)", "BFCL"),
    ("tau2-bench (live)", "tau2"),
    ("WebArena (live)", "WebA."),
)


def _write_pareto_uncertainty_all_policy(rows: list[dict[str, str]], output: Path) -> None:
    """All-policy Pareto non-dominance frontier: one row per policy (routers first,
    then a rule, then fixed tiers), one column per benchmark. The per-column
    leader(s) are bolded; exact ties are all bolded."""
    value: dict[tuple[str, str], float] = {}
    for row in rows:
        value[(row["policy_name"], row["benchmark_id"])] = float(
            row["pareto_nondominance_probability"]
        )
    leaders: dict[str, set[str]] = {}
    for benchmark_id, _ in _ALL_POLICY_BENCHMARKS:
        present = {
            policy: value[(policy, benchmark_id)]
            for policy, _short, _kind in _ALL_POLICY_ORDER
            if (policy, benchmark_id) in value
        }
        if present:
            best = max(present.values())
            leaders[benchmark_id] = {p for p, v in present.items() if v >= best - 1e-9}
        else:
            leaders[benchmark_id] = set()

    def _cells(policy: str, short: str) -> str:
        cells = [short]
        for benchmark_id, _ in _ALL_POLICY_BENCHMARKS:
            prob = value.get((policy, benchmark_id))
            text = "NA" if prob is None else _number(prob, 3)
            if policy in leaders[benchmark_id]:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        return " & ".join(cells) + r" \\"

    headers = ["Policy"] + [short for _bid, short in _ALL_POLICY_BENCHMARKS]
    body = [
        rf"\begin{{tabular}}{{@{{}}l{'r' * len(_ALL_POLICY_BENCHMARKS)}@{{}}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    routers = [(p, s) for p, s, kind in _ALL_POLICY_ORDER if kind == "router"]
    fixed = [(p, s) for p, s, kind in _ALL_POLICY_ORDER if kind == "fixed-tier"]
    body += [_cells(p, s) for p, s in routers]
    body.append(r"\midrule")
    body += [_cells(p, s) for p, s in fixed]
    body += [r"\bottomrule", r"\end{tabular}", ""]
    output.write_text("\n".join(body), encoding="utf-8")


def _write_bfcl_equivalence(rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"rows": 0, "equivalent": 0, "conflicts": 0})
    for row in rows:
        key = (row["selected_candidate"], row["checker"], row["checker_version"])
        grouped[key]["rows"] += 1
        if row["equivalent_outcome"] == "true":
            grouped[key]["equivalent"] += 1
        elif row["equivalent_outcome"] != "not_applicable":
            grouped[key]["conflicts"] += 1
    table_rows = []
    for (candidate, checker, version), counts in sorted(grouped.items()):
        table_rows.append([candidate, checker, version, counts["rows"], counts["equivalent"], counts["conflicts"]])
    output.write_text(
        _tabular(
            ["Candidate", "Checker", "Version", "Rows", "Equivalent", "Conflicts"],
            table_rows,
            "lllrrr",
        ),
        encoding="utf-8",
    )


def _write_ablation(registry: dict[str, Any], output: Path) -> None:
    rows = [[registry["status"], registry.get("reason", ""), len(registry.get("canonical_ablation_claims", []))]]
    output.write_text(
        _tabular(["Status", "Reason", "Claims"], rows, "lp{6.2cm}r"),
        encoding="utf-8",
    )


def _write_metric_suite(rows: list[dict[str, str]], output: Path) -> None:
    """Compact deployment metric suite: the reliability/robustness columns that
    do not appear in the main success/cost tables."""
    table_rows = []
    for row in rows:
        table_rows.append([
            row["router"],
            _number(row["tool_call_accuracy"], 2),
            _number(row["fallback_rate"], 2),
            _number(row["route_stability"], 2),
            _number(row["mean_confidence"], 2),
            _number(row["easy_success"], 2),
            _number(row["medium_success"], 2),
            _number(row["hard_success"], 2),
            _number(row["robustness_std"], 3),
        ])
    output.write_text(
        _tabular(
            ["Router", "Tool acc.", "Fallback", "Stability", "Conf.",
             "Easy", "Med.", "Hard", "Robust. std"],
            table_rows,
            "lrrrrrrrr",
        ),
        encoding="utf-8",
    )


def _write_pool_ablation(rows: list[dict[str, str]], output: Path) -> None:
    """Wide-gap vs narrow-gap candidate-pool ablation (auxiliary two-benchmark
    lineage): static routers are pool-invariant in selection, so their cost
    tracks the exogenous price of the tier they are pinned to."""
    table_rows = []
    for row in rows:
        table_rows.append([
            row["router"],
            _number(row["widegap_success"], 3),
            _number(row["narrowgap_success"], 3),
            _usd(row["widegap_cost_per_task"], 6),
            _usd(row["narrowgap_cost_per_task"], 6),
            _number(row["cost_ratio_narrow_over_wide"], 2),
        ])
    output.write_text(
        _tabular(
            ["Router", "Wide succ.", "Narrow succ.", "Wide USD/task",
             "Narrow USD/task", "Narrow/Wide cost"],
            table_rows,
            "lrrrrr",
        ),
        encoding="utf-8",
    )


_UTILITY_ROUTER_SHORT = {
    "Aurelio Semantic Router": "Aurelio Sem.",
    "vLLM Semantic Router": "vLLM Sem.",
    "LiteLLM Router": "LiteLLM",
    "RouteLLM": "RouteLLM",
    "Always-Cheapest": "Always-Cheap",
    "Always-Mid": "Always-Mid",
    "Always-Strongest": "Always-Strong",
}


def _utility_column_label(value: float, latency_price: float) -> str:
    v = str(int(value)) if float(value).is_integer() else f"{value:g}"
    return rf"$V{{=}}{v},\lambda_\ell$" if latency_price else rf"$V{{=}}{v}$"


def _write_expected_utility(
    rows: list[dict[str, str]], output: Path, cost_basis: str = "candidate_plus_service"
) -> None:
    """Cost- and latency-aware utility U(R) from expected_utility.csv (long form:
    one row per policy x cost-basis x (value-of-success, latency-price) cell), for a
    single ``cost_basis``. Rendered wide, with the per-column leader(s) in bold;
    exact ties are all bolded. Rows without a ``cost_basis`` field (legacy schema)
    are treated as matching so old fixtures still render."""
    rows = [r for r in rows if r.get("cost_basis", cost_basis) == cost_basis]
    # Preserve first-seen order for both columns and routers.
    columns: list[tuple[str, str]] = []
    routers: list[str] = []
    value: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        col = (row["value_of_success_usd"], row["latency_price_usd_per_s"])
        if col not in columns:
            columns.append(col)
        if row["router"] not in routers:
            routers.append(row["router"])
        value.setdefault(col, {})[row["router"]] = float(row["utility_usd_per_task"])
    leaders = {
        col: {rt for rt, v in value[col].items() if v >= max(value[col].values()) - 1e-9}
        for col in columns
    }
    headers = ["Router"] + [_utility_column_label(float(v), float(lp)) for v, lp in columns]
    table_rows: list[list[str]] = []
    for rt in routers:
        cells = [_UTILITY_ROUTER_SHORT.get(rt, rt)]
        for col in columns:
            text = f"{value[col][rt]:+.3f}"
            cells.append(rf"\textbf{{{text}}}" if rt in leaders[col] else text)
        table_rows.append(cells)
    # Column labels and the bolded leaders already carry LaTeX; bypass escaping.
    body = [
        rf"\begin{{tabular}}{{@{{}}l{'r' * len(columns)}@{{}}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    body += [" & ".join(cells) + r" \\" for cells in table_rows]
    body += [r"\bottomrule", r"\end{tabular}", ""]
    output.write_text("\n".join(body), encoding="utf-8")


def _write_cascade_operating(rows: list[dict[str, str]], output: Path) -> None:
    """Idealized-cascade operating cost from cascade_operating_metrics.csv: average
    model calls, summed latency, cost per task, and success-per-dollar."""
    table_rows = []
    for row in rows:
        table_rows.append([
            row["benchmark"],
            _number(row["success"], 3),
            _number(row["avg_calls"], 2),
            _number(row["avg_latency_s"], 1),
            _usd(row["cost_per_task_usd"], 5),
            _number(row["success_per_usd"], 1),
        ])
    output.write_text(
        _tabular(
            ["Benchmark", "Success", "Avg calls", "Latency (s)", "USD/task", "Success/USD"],
            table_rows,
            "lrrrrr",
        ),
        encoding="utf-8",
    )


def _write_artifact_manifest(analysis_dir: Path, output_dir: Path, output: Path) -> None:
    rows = [["analysis/" + name, str(analysis_dir / name)] for name in REQUIRED_ANALYSIS_FILES]
    rows.extend([
        ["table/canonical_rebuild_summary", str(output_dir / "canonical_rebuild_summary.tex")],
        ["table/canonical_spend_summary", str(output_dir / "canonical_spend_summary.tex")],
        ["table/canonical_candidate_tiers", str(output_dir / "canonical_candidate_tiers.tex")],
        ["table/canonical_baselines", str(output_dir / "canonical_baselines.tex")],
        ["table/canonical_router_summary", str(output_dir / "canonical_router_summary.tex")],
        ["table/canonical_route_selection_summary", str(output_dir / "canonical_route_selection_summary.tex")],
        ["table/canonical_paired_effects", str(output_dir / "canonical_paired_effects.tex")],
        ["table/canonical_rank_uncertainty", str(output_dir / "canonical_rank_uncertainty.tex")],
        ["table/canonical_pareto_uncertainty", str(output_dir / "canonical_pareto_uncertainty.tex")],
        ["table/canonical_bfcl_equivalence", str(output_dir / "canonical_bfcl_equivalence.tex")],
        ["table/canonical_ablation_registry", str(output_dir / "canonical_ablation_registry.tex")],
        ["table/canonical_vllm_share_permutation", str(output_dir / "canonical_vllm_share_permutation.tex")],
        ["table/canonical_metric_suite", str(output_dir / "canonical_metric_suite.tex")],
        ["table/canonical_pool_ablation", str(output_dir / "canonical_pool_ablation.tex")],
        ["table/canonical_expected_utility", str(output_dir / "canonical_expected_utility.tex")],
        ["table/canonical_expected_utility_candidate", str(output_dir / "canonical_expected_utility_candidate.tex")],
        ["table/canonical_cascade_operating", str(output_dir / "canonical_cascade_operating.tex")],
    ])
    output.write_text(_tabular(["Artifact", "Path"], rows, "lp{7.4cm}"), encoding="utf-8")


def _write_appendix_fragment(output: Path) -> None:
    output.write_text(
        "\n".join([
            r"\section{Canonical Full-Rebuild Evidence}",
            r"\label{app:canonical-full-rebuild}",
            "",
            (
                "This appendix is generated from the locked full-rebuild bundle and its "
                "canonical analysis outputs. It is absent from builds that have not "
                "completed canonical validation."
            ),
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical full-rebuild matrix}",
            r"\label{tab:canonical-rebuild-summary}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_rebuild_summary.tex}}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical observed spend ledger}",
            r"\label{tab:canonical-spend-summary}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_spend_summary.tex}}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical candidate-tier outcomes}",
            r"\label{tab:canonical-candidate-tiers}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_candidate_tiers.tex}}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical deterministic baseline outcomes}",
            r"\label{tab:canonical-baselines}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_baselines.tex}}",
            r"\end{table}",
            "",
            r"\begin{table*}[t]",
            r"\caption{Canonical joined router outcomes in the four benchmarks}",
            r"\label{tab:canonical-router-summary}",
            r"\centering",
            r"\resizebox{\textwidth}{!}{\input{tables/canonical_router_summary.tex}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\caption{Canonical route-selection shares in the four benchmarks}",
            r"\label{tab:canonical-route-selection-summary}",
            r"\centering",
            r"\resizebox{\textwidth}{!}{\input{tables/canonical_route_selection_summary.tex}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\caption{Canonical paired router effects with Holm adjustment in the four benchmarks}",
            r"\label{tab:canonical-paired-effects}",
            r"\centering",
            r"\resizebox{\textwidth}{!}{\input{tables/canonical_paired_effects.tex}}",
            r"\end{table*}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical rank uncertainty in the four benchmarks}",
            r"\label{tab:canonical-rank-uncertainty}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_rank_uncertainty.tex}}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical Pareto uncertainty in the four benchmarks}",
            r"\label{tab:canonical-pareto-uncertainty}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_pareto_uncertainty.tex}}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical BFCL route-equivalence audit}",
            r"\label{tab:canonical-bfcl-equivalence}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_bfcl_equivalence.tex}}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Canonical candidate-pool ablation registry}",
            r"\label{tab:canonical-ablation-registry}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_ablation_registry.tex}}",
            r"\end{table}",
            "",
            # The vLLM share-matched permutation table lives in the Results body
            # (label tab:canonical-vllm-share-permutation); it is intentionally not
            # repeated here to avoid a multiply-defined label. Its fragment is still
            # generated by _write_vllm_share_permutation for the body to \input.
            r"\begin{table}[h]",
            r"\caption{Canonical artifact manifest}",
            r"\label{tab:canonical-artifact-manifest}",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{\input{tables/canonical_artifact_manifest.tex}}",
            r"\end{table}",
            "",
        ]),
        encoding="utf-8",
    )


def generate_tables(bundle: Path, protocol: Path, analysis_dir: Path, paper_tables_dir: Path) -> None:
    validate_bundle(bundle, protocol)
    required = [analysis_dir / filename for filename in REQUIRED_ANALYSIS_FILES]
    _require_files(required)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    _write_rebuild_summary(bundle, analysis_dir, paper_tables_dir / "canonical_rebuild_summary.tex")
    _write_candidate_tiers(
        _read_csv(analysis_dir / "candidate_tier_summary.csv"),
        paper_tables_dir / "canonical_candidate_tiers.tex",
    )
    _write_baselines(_read_csv(analysis_dir / "baseline_summary.csv"), paper_tables_dir / "canonical_baselines.tex")
    _write_router_summary(bundle, paper_tables_dir / "canonical_router_summary.tex")
    _write_route_selection_summary(bundle, paper_tables_dir / "canonical_route_selection_summary.tex")
    _write_paired_effects(_read_csv(analysis_dir / "paired_effects.csv"), paper_tables_dir / "canonical_paired_effects.tex")
    _write_rank_uncertainty(
        _read_csv(analysis_dir / "rank_uncertainty.csv"),
        paper_tables_dir / "canonical_rank_uncertainty.tex",
    )
    _write_pareto_uncertainty(
        _read_csv(analysis_dir / "pareto_uncertainty.csv"),
        paper_tables_dir / "canonical_pareto_uncertainty.tex",
    )
    all_policy_pareto = analysis_dir / "pareto_uncertainty_all_policy.csv"
    if all_policy_pareto.exists():
        _write_pareto_uncertainty_all_policy(
            _read_csv(all_policy_pareto),
            paper_tables_dir / "canonical_pareto_uncertainty_all_policy.tex",
        )
    _write_bfcl_equivalence(
        _read_csv(analysis_dir / "bfcl_route_equivalence.csv"),
        paper_tables_dir / "canonical_bfcl_equivalence.tex",
    )
    _write_ablation(_read_json(analysis_dir / "ablation_registry.json"), paper_tables_dir / "canonical_ablation_registry.tex")
    _write_vllm_share_permutation(
        _read_csv(analysis_dir / "vllm_share_permutation.csv"),
        paper_tables_dir / "canonical_vllm_share_permutation.tex",
    )
    _write_metric_suite(
        _read_csv(analysis_dir / "canonical_metric_suite.csv"),
        paper_tables_dir / "canonical_metric_suite.tex",
    )
    _write_pool_ablation(
        _read_csv(analysis_dir / "pool_ablation_comparison.csv"),
        paper_tables_dir / "canonical_pool_ablation.tex",
    )
    _utility_rows = _read_csv(analysis_dir / "expected_utility.csv")
    _write_expected_utility(
        _utility_rows,
        paper_tables_dir / "canonical_expected_utility.tex",
        cost_basis="candidate_plus_service",
    )
    _write_expected_utility(
        _utility_rows,
        paper_tables_dir / "canonical_expected_utility_candidate.tex",
        cost_basis="candidate",
    )
    _write_cascade_operating(
        _read_csv(analysis_dir / "cascade_operating_metrics.csv"),
        paper_tables_dir / "canonical_cascade_operating.tex",
    )
    _write_artifact_manifest(analysis_dir, paper_tables_dir, paper_tables_dir / "canonical_artifact_manifest.tex")
    _write_appendix_fragment(paper_tables_dir / "canonical_rebuild_appendix.tex")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis/output/paper1_canonical"))
    parser.add_argument("--paper-tables-dir", type=Path, default=Path("../paper/tables"))
    args = parser.parse_args()
    generate_tables(args.bundle, args.protocol, args.analysis_dir, args.paper_tables_dir)
    print(f"Canonical paper table fragments written to {args.paper_tables_dir}.")


if __name__ == "__main__":
    main()
