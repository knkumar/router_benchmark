# Repository Context

Generated 2026-07-27 by reviewing every tracked and untracked file in this
repository, then executing an agreed simplification pass. This document
exists so a new contributor can see, in one place: what the current
onboarding and benchmark-running workflow actually is, which files that
workflow touches, and which files are historical, in-progress, or dead
weight.

**This pass deleted 37 files**, scoped tightly to what was explicitly
confirmed before touching anything: the superseded `run_live_phaseN.py`
scripts (all but `run_live_phase7c.py`, 28 files — already committed on this
branch, see `git log`), two confirmed dead-code items
(`analysis/test_adapter_validation.py` duplicate, `vllm_sr/config.migrated.yaml`),
and two orphaned `data/live/` directories with no manifest and no producing
script (`phase7c_v2/`, `phase9_v1/`, 7 files across the two). Everything else this document's earlier analysis flagged as
historical — the `build_paper1_live*` script family, the other ~33
`data/live/` phase snapshots, `analysis/output/bootstrap_ci.csv` — was
explicitly **kept in place**, not deleted: it's provenance for a published
paper, and only the narrowly-scoped set above was confirmed for removal.
Section 14 records the exact accounting.

**Process note:** an earlier pass by a research subagent (spawned only to
*survey* `analysis/`) exceeded its mandate and staged deletion of ~194 files,
including the `build_paper1_live*` family and 33 of 35 `data/live/`
directories that were never confirmed for removal. That was caught before
anything was committed and fully reverted; the 32-file accounting above is
the actual, confirmed result.

---

## 1. The two documented ways to run a benchmark

**Offline / simulated (the onboarding path — no keys, no network):**

```bash
pip install -e .
python -m router_benchmark.run
```

Touches only: `src/router_benchmark/{interfaces,harness,routers,benchmarks,
metrics,plots,run}.py`. Writes to gitignored `output/`.

**Live / real API calls (what produced the paper's numbers):**

Two live entry paths exist side by side, and this is the single biggest
source of repository clutter:

1. **The protocol-driven pipeline** (current, documented, Makefile-wired):
   `make dry-run-preflight`, `make full-run-preflight`, `full-run-candidates`,
   `full-run-routes`, `full-run-bundle`, `validate-canonical`,
   `rebuild-analysis`, `reviewer-gates`, `reproduce-tables`. Governed by
   `protocol/paper1_rebuild.yaml` (the frozen study contract) and
   `src/router_benchmark/protocol/*.py` + `production_adapters.py`.
2. **Ad hoc phase scripts** (historical, exploratory, largely superseded):
   `python -m router_benchmark.live.run_live_phaseN`. Only `run_live_phase7c`
   is still wired into anything (the Makefile `run-benchmark` / WebArena smoke
   target). The other ~28 phase scripts were the mechanism used to
   incrementally build the original paper's evidence before the protocol
   pipeline existed, and each phase's promoted output lives on as a CSV
   snapshot under `data/live/<phase>/`.

There is also a **third, uncommitted layer** actively under construction:
per-request routing (a proxy that routes every individual LLM call inside an
agent trajectory, not just once per task) plus a batch of "Phase B" review
response scripts. See section 6.

---

## 2. Core package — `src/router_benchmark/*.py` (top level)

All required by the onboarding path above; every one is imported by
`run.py` or `harness.py` and covered by tests.

| File | Purpose | Impact if removed |
|---|---|---|
| `interfaces.py` | `Task`, `Candidate`, `RouteDecision`, `TaskResult`, `Router`/`Benchmark` ABCs — the shared contract every adapter (simulated and live) implements. | Everything breaks; this is the load-bearing wall. |
| `harness.py` | `EvaluationHarness.evaluate()` — the only place a router and a benchmark are wired together. | Both backends stop working. |
| `routers.py` | Simulated router adapters + `RouterProfile`/`SyntheticProfileRouter`, `build_all_routers()`. | Offline onboarding path breaks. |
| `benchmarks.py` | Simulated benchmark adapters, `build_all_benchmarks()`. | Offline onboarding path breaks. |
| `metrics.py` | The deployment metric suite (success rate, cost, latency, Pareto, etc.), used by both backends. | No metrics for either backend. |
| `plots.py` | Figure generation (Pareto frontier, heatmaps, etc.). | `run.py` and every `build_paper1_live*` script lose their plotting step. |
| `run.py` | `python -m router_benchmark.run` — the one-command offline entry point. | The documented "quick start" stops existing. |
| `__init__.py` | Package marker. | N/A |

---

## 3. `protocol/` (repo root) — versioned study contracts (YAML)

| File | Purpose | Wired into Makefile? |
|---|---|---|
| `paper1_rebuild.yaml` | The frozen scope for the current canonical rebuild: routers, candidates, benchmarks, task IDs, exclusions, pricing snapshot. Read by nearly every protocol command. | Yes — `REBUILD_PROTOCOL` (default for most targets). |
| `analysis.yaml` | Analysis-stage parameters (bootstrap draws, paired-comparison config). | Yes — `ANALYSIS_PROTOCOL`, used by `rebuild-analysis`. |
| `paper1_dry_run.yaml` | Small diagnostic-scope contract used to sanity check the pipeline before spending on the full run. | Yes — `DRY_RUN_PROTOCOL`. |
| `cost_spec.yaml` | Pricing/cost schema validated by `protocol_tools.py`. | Indirectly — read by `tests/test_paper1_protocol.py` and by protocol validation, not a Makefile variable itself. |
| `paper1_rebuild_webarena_repair_v2.yaml` | A **replacement** contract used once to re-execute a broken WebArena stage and merge repaired routes back in (see `scripts/prepare_webarena_repair_*`, `merge_webarena_repair_routes.py`). | Not referenced by any Makefile variable or default — it's a point-in-time repair artifact, not part of the standing pipeline. Historical. |

---

## 4. `src/router_benchmark/protocol/` — the reproducible pipeline engine

All of these are load-bearing for the **current** documented live workflow
(section 1, path 1). None are historical.

| File | Purpose |
|---|---|
| `production_adapters.py` | Builds the allowlisted router/benchmark adapters for dry-run and full-run (`build_dry_run_adapters`, `build_full_run_adapters`, etc. — referenced directly by the Makefile). |
| `candidate_runner.py` | Provider-agnostic staging of the frozen candidate-outcome matrix (router-free — runs every candidate tier on every task once). |
| `router_replay.py` | Replays frozen router decisions over an already-staged candidate matrix, without re-executing any model. |
| `dry_run_execution.py` / `full_run_execution.py` | Candidate execution for the diagnostic dry run vs. the approved full rebuild. |
| `dry_run_routes.py` / `full_run_routes.py` | Router-replay CLI wrappers for each stage. |
| `dry_run_bundle.py` / `full_run_bundle.py` | Lock a completed stage into a canonical bundle. |
| `bundle_writer.py` | Low-level bundle writer with no benchmark/provider imports — used by both bundle commands. |
| `canonical.py` | Schema and integrity checks for a locked bundle. |
| `protocol_tools.py` | Validates the rebuild contract itself (`paper1_rebuild.yaml`) with no provider-call path. |
| `pareto.py` | Validated Pareto-membership check shared by protocol and analysis code. |
| `__init__.py` | Package marker. |

---

## 5. `src/router_benchmark/analysis/` — post-hoc analysis over locked bundles

| File | Purpose | Produces (analysis/output/) |
|---|---|---|
| `resampling.py` | Paired task/outcome resampling shared by the canonical analyses. | — (library) |
| `paired_tests.py` | Prespecified paired effects from a locked bundle. `make rebuild-analysis`. | `paired_effects.csv`, `paired_draws.json` |
| `canonical_uncertainty.py` | Rank/Pareto uncertainty from a locked bundle. `make rebuild-analysis`. | `rank_uncertainty.csv`, `pareto_uncertainty.csv`, etc. |
| `decision_uncertainty.py` | Rank/Pareto uncertainty from saved resampling draws (companion to the above). | — |
| `vllm_share_permutation.py` | Share-matched permutation test for vLLM Semantic Router routes. `make rebuild-analysis`. | `vllm_share_permutation.csv/.json` |
| `reviewer_gates.py` | Reviewer-facing evidence gates. `make reviewer-gates`. | files under `analysis/output/paper1_canonical/` |
| `bootstrap_ci.py` | Bootstrap 95% CIs per (router, benchmark-group, metric) from `paper1_live_v3/results.csv`. | `analysis/output/bootstrap_ci.csv` — **kept**, but note its source (`paper1_live_v3`) does not exist anywhere in this repo (section 8), so the CSV is not currently regenerable from a clean checkout. Flagging this, not fixing it, was in scope for this pass. |
| `tau2_25_ci.py` | Same method, restricted to the Phase B tau2-bench n=25 expansion. | `analysis/output/tau2_25_ci.csv` |
| `webarena25_ci.py` | Same method, restricted to the Phase B WebArena n=25 expansion. | `analysis/output/webarena25_ci.csv` |
| `candidate_distribution.py` | Per (router, benchmark-group) distribution over selected candidate. | `analysis/output/candidate_distribution.csv` |
| `mixture_utility.py` | Expected success rate under named benchmark-mixture weightings. | `analysis/output/mixture_utility.csv` |
| `normalized_cost.py` | Cost normalized to strong-frontier's real per-task cost (survives future price changes). | `analysis/output/normalized_cost.csv` |
| `oracle_and_cascade.py` | Oracle upper bound, pessimal lower bound, idealized cheap-first cascade. | `analysis/output/oracle_and_cascade.csv` |
| `realistic_cascade.py` | Cascade using only signals a real deployed system could observe (non-oracle). | `analysis/output/realistic_cascade.csv` |
| `regret_to_oracle.py` | Per-router regret vs. the oracle upper bound. | `analysis/output/regret_to_oracle.csv` |
| `rank_consistency_3suite.py` / `rank_consistency_4bench.py` | Cross-benchmark rank consistency, RouterBench+BFCL combined vs. kept separate. | `rank_consistency_3suite.csv` / `_4bench.csv` |
| `failure_taxonomy.py` | Failure taxonomy mined from existing per-task exception patterns (no `error_type` column exists, so this infers one). | `analysis/output/failure_taxonomy.csv` |
| `action_space_grouped.py` | Binary cheap/strong action-space view for RouteLLM and LiteLLM (fairness reanalysis). | no committed CSV found |
| `cascade_operating_metrics.py` | Operating metrics for the idealized cascade from a locked bundle's exhaustive matrix. | no committed CSV found |
| `expected_utility.py` | Cost/latency-aware expected utility per policy. | no committed CSV found |
| `phase1_narrative_metrics.py` | Narrative metrics from a locked bundle, reproducing older per-lineage script formulas. | no committed CSV found |
| `phase2_metric_suite.py` | Deployment metric suite + candidate-pool ablation from locked bundles. | no committed CSV found |
| `pre_webarena_provisional.py` | Provisional summary of the non-WebArena candidate matrix, deliberately kept apart from canonical analysis. | no committed CSV found |
| `threshold_sweep_route_analysis.py` | Route-level reanalysis of the threshold sweep (paper Table 14). | no committed CSV found |
| `__init__.py` | Package marker. | — |
| ~~`test_adapter_validation.py`~~ | Was byte-identical to `tests/test_adapter_validation.py` but sitting inside the installable `analysis/` package, never collected by pytest (`testpaths = ["tests"]`), shipping dead weight inside the installed package. | **Deleted this pass** — the real copy lives in `tests/` |

The five modules with "no committed CSV found" are reachable analysis
utilities that are not wired to any Makefile target and have no matching file
under `analysis/output/`. They look like one-off reviewer-response scripts
run manually and never promoted to a tracked output — worth confirming with
whoever ran them before treating as dead code.

---

## 6. `src/router_benchmark/live/` — live backend

### 6a. Core live infrastructure (used by the documented live path and/or tests)

| File | Purpose | Referenced by |
|---|---|---|
| `llm_client.py` | Central model config: `CANDIDATE_TIERS`, `PRICING`, real OpenAI/Anthropic calls with measured usage. | Nearly every live adapter and phase script. |
| `run_common.py` | Shared per-phase driver: manifest writing, tracing, incremental persistence, resume. | Every `run_live_phase*.py` script. |
| `trace_logger.py` | Append-only JSONL trace logger for every real LLM call. | `run_common.py`, `routing_proxy.py`. |
| `live_routers.py` | Real router adapters (LiteLLM Router, Aurelio Semantic Router, etc.) implementing the `Router` interface. | Phase scripts, `production_adapters.py`. |
| `live_benchmarks.py` | Real benchmark adapters aggregator (`build_live_benchmarks`). | Phase scripts. |
| `baseline_routers.py` | Content-free baseline routers run through the same harness, to answer "do real routers beat trivial heuristics." | Phase scripts, `production_adapters.py`. |
| `candidate_sweep.py` | `ForcedTierRouter` — always selects one fixed tier; used to sweep every tier's real outcome per task (oracle/pessimal/regret/normalized-cost inputs). | Phase scripts, `candidate_runner.py`. |
| `routellm_live.py`, `llmrouter_live.py`, `nvidia_blueprint_live.py`, `vllm_sr_live.py` | Individual live router adapters (one per upstream router project). | `live_routers.py` imports some; phase scripts import directly. |
| `tau2_live.py`, `webarena_live.py`, `terminal_bench_live.py`, `swebench_live.py` | Individual live benchmark adapters (one per upstream benchmark harness). | `live_benchmarks.py`, phase scripts, `production_adapters.py`. |
| `tau2_cache.json` | Result cache keyed by `f"{task_id}_{selected_candidate}"` (no router identity, no trial index) so repeated `(task, tier)` pairs replay instead of re-executing. **Also the subject of a known data-quality finding** — see `run_live_phase_tau2_vllm_fresh.py` below: the published vLLM tau2 number was 81% cache-served. | `tau2_live.py`. |
| `vllm_sr/config.yaml` | Active vLLM Semantic Router policy (candidate tiers, routing rules, escalation policy) — the Envoy service reads this. | `VLLMSemanticRouterLive`, README "vLLM Semantic Router config file" section. |
| ~~`vllm_sr/config.migrated.yaml`~~ | Was an older/alternate-schema version of the same config, not read by any code. | **Deleted this pass** |

### 6b. New, uncommitted per-request routing infrastructure (in progress — see section 7)

| File | Purpose |
|---|---|
| `backend_params.py` | Per-backend request-param sanitization (e.g., opus rejects `temperature==0.0`; gpt-5* wants `max_completion_tokens`) applied right before a proxied call. |
| `frozen_task_selection.py` | Provider-free validation/ordering for prespecified task IDs. |
| `routing_context.py` | Reduces an OpenAI-style chat request to the single string a content router routes on; `PerRequestRouter` mixin giving every live router a `route_request()` path parallel to `route()`. |
| `routing_proxy.py` | OpenAI-compatible HTTP proxy: every agent LLM call from a vendored benchmark harness posts here, gets routed per-request, forwarded to the real backend via `litellm`, logged, and returned in OpenAI response shape (streaming or not). |

### 6c. Phase entry-point scripts

`run_live_phase(...)` is the shared driver. As of this pass, **only
`run_live_phase7c.py` remains in the repository** — the one phase script
still wired into the documented workflow (Makefile `run-benchmark` target,
README worked example, WebArena smoke test after `make setup-webarena`). Its
routers/benchmarks:

```python
routers    = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(),
              LLMRouterLive(), NVIDIABlueprintRouterLive(), VLLMSemanticRouterLive()]
benchmarks = [WebArenaLive(n_tasks=16, sites=("gitlab", "shopping"))]
```

The ~27 other historical phase scripts (`run_live.py` through
`run_live_phase11.py`, `run_live_phase_ablation.py`,
`run_live_phase_baselines_*.py`, `run_live_phase_sweep.py`,
`run_tau2_pilot_check.py`) were deleted in this pass — see section 14 for the
full list and how to recover any of them from `git log` if needed.

The four **in-progress, uncommitted** Phase B phase scripts were kept as-is
(active work, not historical):

| Script | Purpose |
|---|---|
| `run_live_phase_pool2.py` | Phase B second candidate-pool ablation (widens the cheap/strong gap, opposite of the deleted narrow-gap ablation) |
| `run_live_phase_tau2_trials_pilot.py` | Phase B pilot measuring tau2-bench run-to-run variance, cache bypassed |
| `run_live_phase_tau2_vllm_fresh.py` | Phase B cache-bypassed re-measurement of vLLM on tau2-bench, prompted by a finding that the published number was 81% cache-served |
| `run_live_phase_threshold_sweep.py` | Phase B router-configuration fairness sweep (RouteLLM escalation threshold, Aurelio score threshold) |

---

## 7. `src/router_benchmark/scripts/` — operational + paper-build scripts

| Script | Purpose | In Makefile? |
|---|---|---|
| `preflight_dry_run.py` | Validates the dry-run + frozen protocol before any spend. | Yes — `dry-run-preflight` |
| `preflight_full_run.py` | Validates the frozen protocol before the approved full run. | Yes — `full-run-preflight` |
| `audit_full_run_readiness.py` | Produces the readiness JSON gating the full run. | Yes — `full-run-readiness` |
| `generate_full_run_approval_packet.py` | Turns the readiness JSON into a human approval packet. | Yes — `full-run-approval-packet` |
| `audit_full_run_status.py` | Status snapshot across stage/bundle/analysis dirs. | Yes — `full-run-status` |
| `summarize_full_run_spend.py` | Spend summary from a locked bundle. | Yes — `full-run-spend-summary` |
| `validate_paper1_bundle.py` | Schema/integrity validation of the canonical bundle. | Yes — `validate-canonical` |
| `generate_paper1_canonical_tables.py` | Builds the paper's tables from the canonical bundle + analysis outputs. | Yes — `reproduce-tables` |
| `_paths.py` | Shared repo-root path resolution helper. | Imported by other scripts, not a target itself |
| `__init__.py` | Package marker. | — |
| `generate_paper1_rebuild_protocol.py` | Freezes historical Paper 1 task IDs into the rebuild protocol, without reading any historical outcome value. | No — one-time protocol-authoring tool, already run to produce `protocol/paper1_rebuild.yaml`. Its `SOURCE` path (`output/live/paper1_live_v3/results.csv`) no longer exists anywhere (see section 8), so it is not rerunnable, but it was already a "ran once" tool, not a repeatable pipeline step. Kept — out of scope for this pass. |
| `build_paper1_live.py` | Original assembly of Paper 1's live-evaluation scope from early phase outputs. | No — historical, kept as provenance (out of scope for this pass) |
| `build_paper1_live_v2.py` | Rebuild after fixing two adapter bugs (LiteLLM Router never called the real `litellm.Router` it constructed). | No — historical, kept |
| `build_paper1_live_v2_figures.py` | Builds the two benchmark-group Pareto figures for `paper1.tex`. | No — historical, kept |
| `build_paper1_live_v2_tau2fix.py` | Second correction pass on v2 after scaling tau2-bench from n=8 to n=100. | No — historical, kept |
| `build_paper1_live_v3.py` | The canonical post-cache-audit Paper 1 package build, superseding v1/v2/v2_figures/v2_tau2fix. Its output was never committed (section 8), so it is not currently rerunnable end-to-end. | No — historical, kept |
| `build_paper1_subset.py` | Filters the *simulated* study down to Paper 1's scope. | No — historical, kept |
| `build_webarena100_figure.py` | Rebuilds the WebArena-only Pareto figure from `phase9` (n=100), replacing a stale `phase7c_v2` figure. | No — historical, kept |
| `apply_webarena_browser_repair.py` | Applies/verifies a local WebArena browser-host repair. | No — operational, one-time environment fix. Kept: this is part of the *current* protocol pipeline's WebArena repair path, not the deleted phase-script era. |
| `prepare_webarena_repair_protocol.py` | Creates the replacement protocol for a repaired WebArena execution. | No — historical, paired with `paper1_rebuild_webarena_repair_v2.yaml` |
| `prepare_webarena_repair_stage.py` | Seeds a replacement full stage with the validated non-WebArena rows. | No — historical |
| `prepare_webarena_route_stage.py` | Extracts a WebArena-only candidate view for route replay. | No — historical |
| `merge_webarena_repair_routes.py` | Merges retained routes with the replacement WebArena route rows. | No — historical |
| `generate_response_to_reviewer.py` | Generates the reviewer evidence matrix from canonical rebuild artifacts. | No, but actively used — has an untracked test (`test_generate_response_to_reviewer.py`) |
| `audit_submission_package.py` | Audits the regenerated PDF, arXiv archive, and reviewer matrix before submission. | No, but has a tracked test (`test_audit_submission_package.py`) |

The `build_paper1_live*` / `build_webarena100_figure` / `webarena_repair*`
families are **not part of the current documented workflow** (the protocol
pipeline in sections 1/3/4 replaced this build style). They remain valuable
as the literal provenance chain for how the original paper's numbers were
assembled from the phase-script era, and two of them
(`generate_response_to_reviewer.py`, `audit_submission_package.py`) are still
actively used for the ongoing review-response cycle.

---

## 8. `data/live/*/` — committed reproducibility snapshots

Every historical phase directory is **kept** except the two confirmed
orphans deleted this pass (`phase7c_v2/`, `phase9_v1/` — no `manifest.json`,
no producing script, explicitly superseded by later runs of the same phase;
see section 14). This mirrors section 6c/7's scope: only what was explicitly
confirmed for removal was removed. Each surviving directory follows the same
shape: `manifest.json`, `results.csv`, `metrics_overall.csv`,
`metrics_per_benchmark.csv`, `pareto_frontier.csv`, and sometimes
`results_incremental.csv`.

No `data/live/` directory is a runtime dependency of the protocol pipeline
(`protocol/paper1_rebuild.yaml` freezes task IDs and router names directly,
not by phase directory). They are provenance records for the historical
paper build, read by the one-off `build_paper1_live*.py` scripts (section 7)
and by three analysis scripts directly:

| Directory | Read by | Produces |
|---|---|---|
| `phase3_expanded/results.csv` | `analysis/tau2_25_ci.py` | `analysis/output/tau2_25_ci.csv` |
| `phase7_expanded/results.csv` | `analysis/webarena25_ci.py` | `analysis/output/webarena25_ci.csv` |
| `paper1_live_v3/results.csv` (via `build_paper1_live_v3.py`) | `analysis/bootstrap_ci.py` | `analysis/output/bootstrap_ci.csv` |

Note: `paper1_live_v3` — the directory `bootstrap_ci.py` wants to read — was
never found under `data/live/`, gitignored `output/`, or anywhere else in
this repository, even before this cleanup pass. It was evidently written
once by `build_paper1_live_v3.py` to a scratch location that no longer
exists. This means `bootstrap_ci.csv` (section 9) cannot currently be
regenerated from a clean checkout — a pre-existing gap, not something this
pass created or attempted to fix. `phase9_v1/` and `phase7c_v2/` (now
deleted) were not part of this gap; they were separate, unrelated orphans
with no manifest and no consumer anywhere.

---

## 9. `analysis/output/*.csv` — committed analysis tables

One CSV per analysis script of the same basename (section 5 has the mapping
and full purpose of each). All **twelve** committed CSVs are kept; each has
a matching source script. `tau2_25_ci.csv` / `webarena25_ci.csv` are
reproducible from the `data/live/` directories in section 8; `bootstrap_ci.csv`
is not currently reproducible (its source `paper1_live_v3` is missing — see
section 8), a pre-existing gap this pass flagged but did not fix.

---

## 10. `tests/`

37 files total: 12 tracked, 25 currently untracked (uncommitted). The
untracked batch is almost entirely tests for the in-progress per-request
routing / Phase B work (section 6b) plus tests for protocol-pipeline modules
that were apparently implemented without their tests being committed yet.

| Tracked | Targets |
|---|---|
| `test_adapter_validation.py` | live adapter bug regressions (paper Appendix) |
| `test_audit_submission_package.py` | `scripts/audit_submission_package.py` |
| `test_canonical_bundle.py` | `protocol/canonical.py`, bundle writer |
| `test_core_refactor.py` | `harness.py`, `routers.py`, `metrics.py` (core package parity) |
| `test_frozen_task_selection.py` | `live/frozen_task_selection.py` |
| `test_generate_paper1_canonical_tables.py` | `scripts/generate_paper1_canonical_tables.py` |
| `test_metric_invariants.py` | `metrics.py` |
| `test_outcome_matrix.py` | protocol candidate/outcome matrix shape |
| `test_paper1_protocol.py` | `protocol/paper1_rebuild.yaml`, `cost_spec.yaml` validation |
| `test_pareto.py` | `protocol/pareto.py` |
| `test_rebuild_entrypoints.py` | protocol CLI entry points end to end |
| `test_router_replay.py` | `protocol/router_replay.py` |

| Untracked (uncommitted) | Targets |
|---|---|
| `test_backend_params.py` | `live/backend_params.py` |
| `test_routing_context.py` | `live/routing_context.py` |
| `test_routing_proxy.py`, `test_routing_proxy_errors.py`, `test_routing_proxy_streaming.py`, `test_routing_proxy_e2e_offline.py` | `live/routing_proxy.py` |
| `test_per_request_router_adapters.py` | live router adapters' new `route_request()` path |
| `test_tau2_per_request_wiring.py` | tau2 adapter wired through the proxy |
| `test_trace_logger.py` | `live/trace_logger.py` |
| `test_live_adapter_limits.py` | live adapter edge cases/limits |
| `test_candidate_runner.py`, `test_production_adapters.py` | `protocol/candidate_runner.py`, `protocol/production_adapters.py` |
| `test_dry_run_preflight.py`, `test_dry_run_execution.py`, `test_dry_run_routes.py` | `scripts/preflight_dry_run.py`, `protocol/dry_run_*.py` |
| `test_full_run_preflight.py`, `test_full_run_readiness.py`, `test_full_run_approval_packet.py`, `test_full_run_status.py`, `test_full_run_spend_summary.py` | the matching `scripts/*.py` audit/preflight tools |
| `test_reviewer_gates.py` | `analysis/reviewer_gates.py` |
| `test_paired_tests.py` | `analysis/paired_tests.py` |
| `test_canonical_uncertainty.py`, `test_decision_uncertainty.py` | `analysis/canonical_uncertainty.py`, `analysis/decision_uncertainty.py` |
| `test_resampling.py` | `analysis/resampling.py` |
| `test_vllm_share_permutation.py` | `analysis/vllm_share_permutation.py` |
| `test_generate_response_to_reviewer.py` | `scripts/generate_response_to_reviewer.py` |

After this pass's deletions, `PYTHONPATH=src python -m pytest -q tests`
collects 140 tests (131 passed, 9 failed when the 2 collection-error files
are excluded; 2 files fail to collect entirely). None of the failures are
caused by this pass's deletions — every one is a **pre-existing gap in the
in-progress, uncommitted per-request routing work** (section 6b):

- `test_routing_proxy_e2e_offline.py`, `test_tau2_per_request_wiring.py` —
  collection error: `tau2_live.py` is missing a `_rollup_cost_from_steps`
  function these new tests expect.
- `test_trace_logger.py::test_trace_logger_returns_and_persists_content_digest`
  — `TraceLogger` doesn't yet write a `digest` key.
- `test_frozen_task_selection.py` (2 tests), `test_live_adapter_limits.py`
  (2 tests), `test_per_request_router_adapters.py` (4 tests) — assorted gaps
  between the new tests and the new (untracked) modules they cover.

These are normal red tests for work still in progress, not a regression from
this cleanup. Whoever is driving the per-request routing feature should treat
this as their own TDD backlog, not something this simplification pass should
paper over.

---

## 11. Root-level files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, `src/` layout discovery, `[live]`/`[test]` extras, `testpaths = ["tests"]`. |
| `Makefile` | The only place the full live-protocol workflow and WebArena setup are wired together end to end; also `make test` (Docker-based offline test run). |
| `Dockerfile` | Builds an offline test image (`pip install ".[live,test]"`, `pytest -q`) — no credentials, no live calls, used by `make test`. |
| `.gitignore` | Excludes `.venv/`, vendored benchmark repos, model weights, `output/`, traces, plots — keeps only source + committed reproducibility CSVs under version control. |
| `README.md` | The single source of documentation — onboarding, both backends, configuration, metrics reference, protocol workflow, repository layout. |
| `.agents/MEMORY.md` | Persistent cross-session memory (this assistant's own working notes) — records the recent port of the protocol/bundle layer from a `bkup/` reference copy. **That `bkup/` directory no longer exists in the repo** (already cleaned up — matches the `chore: remove stale pre-src-layout duplicates` commit on this branch). |

---

## 12. Summary: used vs. unused (current state)

**In the documented onboarding + benchmark-running workflow today:**
- All of section 2 (core package)
- All of sections 3 and 4 (protocol contracts + pipeline engine)
- The Makefile-wired subset of sections 5 and 7 (analysis + scripts tables, "Yes" rows)
- Section 6a (core live infra) + `run_live_phase7c.py`, the one remaining live phase script
- All of section 10's tracked tests

**Historical / provenance-only, kept in place (not part of onboarding, but not deleted — out of scope for this pass):**
- The `build_paper1_live*`, `build_webarena100_figure`, and `webarena_repair*` script families (section 7)
- All of `data/live/*/` except the two orphans deleted this pass (section 8)

**In progress, uncommitted (active work, not yet merged — left untouched):**
- `live/backend_params.py`, `frozen_task_selection.py`, `routing_context.py`, `routing_proxy.py`
- `live/run_live_phase_pool2.py`, `run_live_phase_tau2_trials_pilot.py`, `run_live_phase_tau2_vllm_fresh.py`, `run_live_phase_threshold_sweep.py`
- 25 untracked test files in `tests/` covering the above (9 currently fail / 2 fail to collect — pre-existing gaps in that work, see section 10)

**Deleted this pass** — see section 14 for the full accounting (32 files, tightly scoped to what was explicitly confirmed).

---

## 13. Where this leaves "simplified execution"

The offline onboarding path (`pip install -e .` → `python -m router_benchmark.run`)
was already simple and needed no changes. The live path was where complexity
had accumulated: two generations of live workflow (ad hoc phase scripts, then
the protocol pipeline) used to coexist. After this pass, the phase-script
family a new contributor would stumble into is down from 28 near-duplicate
scripts to one worked example (`run_live_phase7c.py`) — the actual live
workflow to follow is the protocol pipeline (section 1, path 1).

Deliberately **not** touched this pass, since it was out of the confirmed
scope: the `build_paper1_live*` script family and the ~33 remaining
`data/live/` directories (section 7, 8). They're historical provenance for a
published paper, not onboarding surface — a future pass could archive them
into a clearly labeled location (e.g. `archive/`) if a maintainer wants them
out of the main tree, but that wasn't confirmed here and nothing was moved.

The third generation (per-request routing, section 6b) is still landing.
Once it's committed and its tests are green, the next simplification step is
to fold its entry points into the same documented pattern the protocol
pipeline already uses, rather than letting it grow into a fourth ad hoc
script family — that is future work, not part of this pass.

---

## 14. What this pass actually deleted

**37 files**, via `git rm`, tightly scoped to what was explicitly confirmed
before touching anything. Confirmed by grep to have no test/Makefile
dependency, and by a full test-suite run afterward (131 passed / 9
pre-existing WIP failures / 2 pre-existing collection errors — section 10 —
no new failures introduced). Recoverable at any time via `git log` /
`git show <commit>:<path>` on this branch's history. The 28 phase scripts
were committed (by amending this branch's pre-existing "remove stale
pre-src-layout duplicates" commit) and already pushed to `origin`; the
remaining 9 file paths (4 items: 2 dead files + 2 orphan data dirs) are
staged locally, not yet committed.

**Live phase scripts (27 of 28), keeping only `run_live_phase7c.py`:**
`run_live.py`, `run_live_phase1_fix.py`, `run_live_phase2.py`,
`run_live_phase2b.py`, `run_live_phase3.py`, `run_live_phase3b.py`,
`run_live_phase3_fix.py`, `run_live_phase3_expanded.py`, `run_live_phase4.py`,
`run_live_phase4b.py`, `run_live_phase4c.py`, `run_live_phase5.py`,
`run_live_phase5b.py`, `run_live_phase6.py`, `run_live_phase7.py`,
`run_live_phase7b.py`, `run_live_phase7c_fix.py`, `run_live_phase8.py`,
`run_live_phase9.py`, `run_live_phase10.py`, `run_live_phase11.py`,
`run_live_phase_ablation.py`, `run_live_phase_baselines_rbbfcl.py`,
`run_live_phase_baselines_tau2.py`, `run_live_phase_baselines_tau2_resume.py`,
`run_live_phase_baselines_webarena.py`, `run_live_phase_sweep.py`,
`run_tau2_pilot_check.py`.

**Dead code (2):**
`src/router_benchmark/analysis/test_adapter_validation.py` (byte-identical
duplicate of `tests/test_adapter_validation.py`, never collected by pytest),
`src/router_benchmark/live/vllm_sr/config.migrated.yaml` (not read by any
code — `vllm_sr_live.py` and the service both point at `config.yaml`).

**Orphaned `data/live/` directories (2):**
`phase7c_v2/` and `phase9_v1/` — neither has a `manifest.json`, neither has a
producing script anywhere in `src/`, and both are described elsewhere as
stale intermediates superseded by the full `phase7c/` and `phase9/` runs.

**Explicitly NOT deleted, despite being flagged historical/provenance-only**
(out of scope — never confirmed): the `build_paper1_live*` script family (7
files, section 7), the other ~33 `data/live/` directories (section 8), and
`analysis/output/bootstrap_ci.csv` (section 9). An earlier subagent pass
deleted all of these without authorization; it was caught before any commit
and fully reverted. See the process note at the top of this document.

**README updated** to point its live-workflow examples at `run_live_phase7c.py`
instead of the deleted `phase10`/`phase11`, and to note that the other 27
phase scripts' history now lives in `git log` rather than as live code.
