# router_benchmark

A common-interface evaluation harness for comparing agentic-routing routers
across benchmarks, built from the shortlist in
`../agentic_routing_router_benchmark_shortlist.md`.

## What this is

- `interfaces.py` — the shared `Router` / `Benchmark` / `Task` / `RouteDecision`
  / `TaskResult` contract everything else is normalized behind.
- `routers.py` — adapters for the six shortlisted routers (RouteLLM, LiteLLM
  Router, vLLM Semantic Router, Aurelio Semantic Router, LLMRouter, NVIDIA AI
  Blueprint LLM Router) plus five baselines (always-cheapest, always-strongest,
  random, heuristic-difficulty, oracle).
- `benchmarks.py` — adapters for the six shortlisted benchmarks (RouterBench,
  SWE-bench Verified, BFCL v4, tau2-bench, WebArena, Terminal-Bench 2.0).
- `harness.py` — `EvaluationHarness.evaluate(routers, benchmarks)`, the
  interface requested in deliverable (1). Runs every router against every
  benchmark for `n_trials` repeated trials and returns a tidy `pandas`
  DataFrame.
- `metrics.py` — deliverable (2): success rate, cost per task/success,
  latency p50/p95, tool-call accuracy, fallback rate, route stability,
  per-difficulty-band success rate, robustness (std across bands), and
  Pareto-frontier membership on cost vs. quality.
- `plots.py` — deliverable (4): Pareto frontier, per-benchmark success-rate
  heatmap, latency distribution, fallback-vs-stability comparison.
- `run.py` — runs everything end to end and writes `output/`.

## IMPORTANT: simulated backend

This environment has no network access and no configured model-provider
credentials, so no adapter calls a real router or executes a real benchmark
task. Every router is instead driven by an explicit, documented
`RouterProfile` (see the module docstring and per-class docstrings in
`routers.py`), and every benchmark generates seeded synthetic tasks and
scores them with a shared logistic success model (see `benchmarks.py`). All
randomness is derived from a deterministic seed, so results are fully
reproducible (`EvaluationHarness(seed=...)`).

The `Router` and `Benchmark` ABCs are the real interface: to run this against
live systems, replace a given adapter's `route()` (and a benchmark's
`generate_tasks()` / `score()`) with real calls to the upstream project or
benchmark harness — the harness, metrics, and plotting code do not change.

See `paper/paper.md`, Section IV, for the same caveat as it appears in the
write-up.

## Usage

```bash
python3 -m router_benchmark.run
```

Writes to `router_benchmark/output/`:
- `results.csv` — one row per (router, benchmark, task, trial)
- `metrics_per_benchmark.csv` — one row per (router, benchmark)
- `metrics_overall.csv` — one row per router, aggregated across benchmarks
- `pareto_frontier.csv` — Pareto-optimal routers on cost vs. quality
- `plots/*.png` — the four figures referenced in the paper

## Extending

```python
from router_benchmark.harness import EvaluationHarness
from router_benchmark.routers import build_all_routers
from router_benchmark.benchmarks import build_all_benchmarks

harness = EvaluationHarness(seed=1234, n_trials=3)
df = harness.evaluate(build_all_routers(), build_all_benchmarks())
```

Add a new router by implementing `Router.route()`; add a new benchmark by
implementing `Benchmark.generate_tasks()` and `Benchmark.score()`. Both slot
directly into `EvaluationHarness.evaluate()` without other changes.

## Repository layout & live-run setup

This repo ships the **code** plus the small live-result CSVs under
`router_benchmark/output/live/*/` that reproduce the paper's tables. Several
large or regenerable inputs are intentionally **not** version-controlled (see
`.gitignore`) and must be provisioned locally before running the `live/`
evaluation:

- `.venv/` — create with your own environment; install `pandas`, `litellm`,
  `openai`, `anthropic`, `semantic-router`, etc.
- `router_benchmark/live/tau2env/`, `router_benchmark/live/tbench_vendor/` —
  vendored third-party benchmark harnesses (tau2-bench, Terminal-Bench);
  clone the upstream projects into these paths.
- `router_benchmark/live/vllm_sr/models/` — embedding/intent-classifier
  weights downloaded at setup time.
- Live runs require `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in the
  environment. No API keys, tokens, or request/response traces are committed.

The **simulated** backend (`python -m router_benchmark.run`) needs none of the
above and runs offline with deterministic seeds.
