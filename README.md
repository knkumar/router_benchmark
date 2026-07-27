# router_benchmark

A common-interface evaluation harness for comparing **agentic-routing routers**
across benchmarks. It normalizes every router and every benchmark behind one
`Router` / `Benchmark` contract, runs an all-pairs evaluation, and produces a
deployment-focused metric suite plus figures — so you can compare routers on
**cost vs. quality**, not just a single leaderboard number.

It ships two interchangeable backends behind the same interface:

- a **simulated backend** — offline, deterministic, zero credentials, for
  developing and testing the harness itself;
- a **live backend** (`src/router_benchmark/live/`) — real, paid API calls and
  real benchmark execution, which is what produces the paper's numbers.

---

## Table of contents

1. [Quick start (offline)](#quick-start-offline)
2. [The two backends](#the-two-backends)
3. [Installation](#installation)
4. [Core concepts](#core-concepts)
5. [Configuration](#configuration)
   - [The candidate pool (shared model config)](#the-candidate-pool-shared-model-config)
   - [Configuring simulated routers & benchmarks](#configuring-simulated-routers--benchmarks)
   - [Configuring live router adapters](#configuring-live-router-adapters)
   - [Configuring live benchmark adapters](#configuring-live-benchmark-adapters)
   - [The vLLM Semantic Router config file](#the-vllm-semantic-router-config-file)
6. [Running the evaluation](#running-the-evaluation)
7. [Programmatic API](#programmatic-api)
8. [Metrics reference](#metrics-reference)
9. [Analysis and protocol workflow](#analysis-and-protocol-workflow)
10. [Extending the harness](#extending-the-harness)
11. [Reproducibility & determinism](#reproducibility--determinism)
12. [Repository layout](#repository-layout)

---

## Quick start (offline)

No API keys, no network, no benchmark downloads required:

```bash
git clone https://github.com/knkumar/router_benchmark.git
cd router_benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                             # installs the `router_benchmark` package + core deps

python -m router_benchmark.run               # runs the full simulated evaluation
```

This evaluates all routers against all benchmarks and writes results, metrics,
and plots under the repo-root `output/` (gitignored). Because every result
derives from a fixed seed, the run is fully reproducible.

---

## The two backends

The harness, metrics, and plotting code are **identical** for both backends —
only the adapter bodies differ.

| | Simulated backend | Live backend (`live/`) |
|---|---|---|
| Where | `routers.py`, `benchmarks.py` | `live/*_live.py`, `live/run_live_phaseN.py` |
| Router behavior | documented `RouterProfile` | real calls into the upstream router project |
| Task scoring | seeded synthetic tasks + logistic success model | real benchmark execution (tau2 CLI, WebArena Chromium, …) |
| Needs network / API keys | **No** | **Yes** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) |
| Cost | free | real \$ per run (see warnings below) |
| Reproducible from seed alone | Yes (bit-identical) | Task *selection* is seeded; outcomes carry real-world variance |
| Command | `python -m router_benchmark.run` | `python -m router_benchmark.live.run_live_phaseN` |
| Powers the paper's tables | No | **Yes** |

> **Note on the simulated backend.** With no network and no credentials, no
> adapter calls a real router or executes a real task. Each router is driven by
> an explicit `RouterProfile` (see `routers.py`), and each benchmark generates
> seeded synthetic tasks scored by a shared logistic success model (see
> `benchmarks.py`). The `Router` and `Benchmark` ABCs are the real interface:
> to go live you swap an adapter's `route()` / `generate_tasks()` / `score()`
> for real calls — the harness, metrics, and plots do not change. The `live/`
> subsystem in this repo is exactly that swap, already done.

---

## Installation

### Simulated backend (offline)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # core deps (pandas, numpy, matplotlib) via pyproject.toml
```

Requires **Python 3.10+** (the codebase uses `X | None` type syntax). The
package uses a `src/` layout, so an editable install (`-e`) is the simplest way
to make `import router_benchmark` resolve.

### Offline tests

Run the project test suite in the supplied container. It installs the live
adapter imports but does not make provider calls:

```bash
docker build --tag router-benchmark-test:local .
docker run --rm router-benchmark-test:local
```

### Live backend (real API calls)

In addition to the above:

1. **Install the live extras** (real router/benchmark stacks):
   ```bash
   pip install -e ".[live]"   # adds openai, anthropic, litellm, semantic-router
   ```
2. **Export provider credentials:**
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. **Provision the inputs that are _not_ version-controlled** (see `.gitignore`):
   - `src/router_benchmark/live/tau2env/`, `src/router_benchmark/live/tbench_vendor/` —
     vendored upstream benchmark harnesses (tau2-bench, Terminal-Bench); clone
     the upstream projects into these paths.
   - `src/router_benchmark/live/vllm_sr/models/` — embedding / intent-classifier
     weights, downloaded at setup time.
   - The vLLM Semantic Router service (Envoy) and the WebArena Docker sites
     (see [Configuration](#configuration) and the `Makefile`).

---

## Core concepts

Everything is normalized behind five types in `router_benchmark/interfaces.py`:

- **`Task`** — one benchmark instance: `task_id`, `benchmark_name`, `domain`,
  `difficulty` (0–1), `requires_tool_call`, and the routable `candidates`.
- **`Candidate`** — one routable option (a model/tool/policy) with a tier
  (`cheap` / `mid` / `strong`), per-token cost, base quality, and base latency.
- **`RouteDecision`** — what a router chose: `selected_candidate`, `confidence`,
  `fallback_used`, and free-form `metadata`.
- **`TaskResult`** — the outcome of one (router, task, trial): success, cost,
  latency, tool-call correctness, plus the routing decision fields.
- **`Router` / `Benchmark`** — the two abstract base classes you implement:

  ```python
  class Router(ABC):
      name: str
      def route(self, task, context, rng) -> RouteDecision: ...

  class Benchmark(ABC):
      name: str
      def generate_tasks(self, rng) -> list[Task]: ...
      def score(self, task, decision, rng) -> dict: ...   # success, cost_usd, latency_ms, tool_call_correct
  ```

The **`EvaluationHarness`** (`harness.py`) is the only place the two are wired
together: for every (benchmark → router → trial → task) it calls `route()` then
`score()`, and returns a tidy `pandas` DataFrame with one row per
(router, benchmark, task, trial).

**Routers.** RouteLLM, LiteLLM Router, vLLM Semantic Router, Aurelio Semantic
Router, LLMRouter, NVIDIA AI Blueprint LLM Router — plus five baselines:
always-cheapest, always-strongest, random, heuristic-difficulty, and oracle.

**Benchmarks.** RouterBench, SWE-bench Verified, BFCL v4, tau2-bench, WebArena,
Terminal-Bench 2.0.

---

## Configuration

### The candidate pool (shared model config)

Both backends route among the **same three tiers**. The live pool is defined in
one place — `src/router_benchmark/live/llm_client.py`:

```python
CANDIDATE_TIERS = {
    "cheap-small":     "gpt-5.4-nano",       # OpenAI
    "mid-general":     "claude-sonnet-4-6",  # Anthropic
    "strong-frontier": "claude-opus-4-8",    # Anthropic
}

PRICING_ASOF = "2026-07-02"                  # rate-card date recorded in every manifest
PRICING = {                                  # (input_usd_per_token, output_usd_per_token)
    "gpt-5.4-nano":      (0.20 / 1e6, 1.25 / 1e6),
    "claude-sonnet-4-6": (3.00 / 1e6, 15.00 / 1e6),
    "claude-opus-4-8":   (5.00 / 1e6, 25.00 / 1e6),
}
```

**To change the models or prices**, edit `CANDIDATE_TIERS`, `PRICING`, and
`_PROVIDER_OF` in `llm_client.py`. Cost is computed from each API's real `usage`
block (not estimated), and the pricing table + as-of date are written into every
run's `manifest.json` for provenance.

### Configuring simulated routers & benchmarks

**Routers** are instances of `SyntheticProfileRouter` driven by a
`RouterProfile` (`routers.py`). The knobs:

| Field | Meaning |
|---|---|
| `quality_bias` | probability mass toward the best-fit candidate |
| `cost_preference` | 0 = always cheapest … 1 = always strongest |
| `domain_affinity` | per-domain nudge `{TaskDomain: float}` |
| `base_fallback_rate` | baseline P(fallback) |
| `tool_reliability` | baseline P(tool call correct \| routed well) |
| `stability` | P(same candidate again on a repeat trial) |
| `confidence_noise` | Gaussian noise on reported confidence |

```python
from router_benchmark.routers import SyntheticProfileRouter, RouterProfile
from router_benchmark.interfaces import TaskDomain

my_router = SyntheticProfileRouter("My Router", RouterProfile(
    quality_bias=0.7, cost_preference=0.3,
    domain_affinity={TaskDomain.CODE_REPAIR: 0.15}, stability=0.95,
))
```

**Benchmarks** are `SyntheticBenchmark`s configured with a `name`, `n_tasks`,
and a `difficulty_range` (`benchmarks.py`). Defaults mirror the real suites,
e.g. RouterBench `n_tasks=400, difficulty=(0.05, 0.95)`, SWE-bench Verified
`n_tasks=200, difficulty=(0.3, 1.0)`, tau2-bench `n_tasks=150, (0.25, 0.9)`.

Seeds and repeats are set on the harness: `EvaluationHarness(seed=1234, n_trials=3)`.

### Configuring live router adapters

Each live adapter implements the same `Router` interface, backed by a real
package, and exposes its own knobs via the constructor:

| Adapter | Constructor | Configuration |
|---|---|---|
| `LiteLLMRouterLive()` | — | Uses `litellm.Router` cost-based-routing over `PRICING`; selects the cheapest deployment (content-independent, a verified property of the strategy). |
| `RouteLLMLive(threshold=0.5)` | win-rate threshold | Routes to `strong-frontier` iff the computed strong-win-rate ≥ `threshold`, else `cheap-small`. |
| `AurelioSemanticRouterLive()` | — | `semantic-router` with `aggregation="max"` over reference-utterance scores; genuine `fallback_used=True, confidence=0.0` on no-match. |
| `VLLMSemanticRouterLive(envoy_url="http://localhost:8909/…", timeout_s=30.0)` | Envoy endpoint | Requires the vLLM-SR service running (see below); routes over HTTP by the config-file rules. |
| `NVIDIABlueprintRouterLive(classifier_model="gpt-5.4-nano")` | classifier model | LLM-classifier picks the tier per task. |
| `LLMRouterLive(model_path=…)` | kNN checkpoint path | Routes via the trained kNN router checkpoint. |

All six draw from the shared `LIVE_CANDIDATES` (the three tiers above). A run
selects which routers to include in its phase script (see below).

### Configuring live benchmark adapters

| Adapter | Constructor | Notes |
|---|---|---|
| `Tau2BenchLive(n_tasks=8, domain="retail")` | sample size, tau2 domain | drives the real `tau2` CLI |
| `WebArenaLive(n_tasks=8, sites=("gitlab","shopping"))` | sample size, which sites | needs the Docker sites up (`make setup-webarena`) |
| `TerminalBenchLive(n_tasks=8)` | sample size | uses the vendored Terminal-Bench harness |
| `SWEBenchLive(instance_ids=None, n_tasks=2)` | explicit instances or count | SWE-bench Verified |

### The vLLM Semantic Router config file

`src/router_benchmark/live/vllm_sr/config.yaml` configures the vLLM Semantic Router
service that `VLLMSemanticRouterLive` calls. It declares:

- **`providers.models`** — the same three tiers, each with `provider_model_id`,
  `api_format`, `pricing`, and a `backend_refs` block (`base_url`,
  `auth_header`, and **`api_key_env`** — the env var to read the key from, never
  a literal key).
- **`routing.modelCards`** — a `quality_score` and capability tags per tier.
- **`routing.decisions`** — domain → model rules (e.g. `law`/`health` →
  `strong-frontier`, `business`/`psychology` → `mid-general`, catch-all →
  `cheap-small`), plus a priority-based `strategy`.
- **`global.router.model_selection.automix`** — escalation policy
  (`verification_threshold`, `max_escalations`, `cost_quality_tradeoff`, …).

Edit this file to change the vLLM-SR routing policy; start the service (Envoy
listener on port `8899`) before running `VLLMSemanticRouterLive`.

---

## Running the evaluation

### Simulated (offline)

```bash
python -m router_benchmark.run
```

Writes to the repo-root `output/` (gitignored) and prints the overall ranking + Pareto set:

| File | One row per |
|---|---|
| `results.csv` | (router, benchmark, task, trial) |
| `metrics_per_benchmark.csv` | (router, benchmark) |
| `metrics_overall.csv` | router (aggregated across benchmarks) |
| `pareto_frontier.csv` | Pareto-optimal router on cost vs. quality |
| `plots/*.png` | Pareto frontier, success-rate heatmap, latency distribution, fallback-vs-stability |

### Live (real API calls)

> ⚠️ **Live phases spend real money.** Per-task cost ranges from ~\$0.00005 to
> ~\$0.5+ depending on the router; a single 100-task phase can cost tens of
> dollars. Read each `run_live_phaseN.py` docstring for its cost estimate first.

A **phase** is one script that fixes a set of routers × benchmarks and calls the
shared driver `run_live_phase(...)`. For example:

```python
# src/router_benchmark/live/run_live_phase10.py (excerpt)
routers    = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(),
              LLMRouterLive(), NVIDIABlueprintRouterLive(), VLLMSemanticRouterLive()]
benchmarks = [Tau2BenchLive(n_tasks=100)]
run_live_phase("phase10", routers, benchmarks, seed=1234, n_trials=1)
```

Run it:

```bash
python -m router_benchmark.live.run_live_phase10
python -m router_benchmark.live.run_live_phase11 --fresh   # omit --fresh to resume
```

Each phase writes the repo-root `output/live/<phase>/` (gitignored runtime
output; promote a phase into the tracked `data/live/` to make it a committed
reproducibility artifact):

- `manifest.json` — models, per-token pricing (+ as-of date), seeds, sample
  sizes, package versions.
- `traces.jsonl` — every real API call: full request + response + cost/latency
  (**not committed**; regenerated per run).
- `results.csv`, `metrics_*.csv`, `pareto_frontier.csv`, `plots/*.png`.

Rows are persisted incrementally and `fsync`'d, so a crash loses at most the
in-flight row. `run_live_phase(..., resume=True)` continues a phase from its own
incremental file **without re-spending** on completed tasks (the skip-set is
read only from that phase's own file — never a hardcoded path).

To define your own phase, copy an existing `run_live_phaseN.py`, pick your
routers/benchmarks, and call `run_live_phase("myphase", routers, benchmarks)`.

---

## Programmatic API

```python
from router_benchmark.harness import EvaluationHarness
from router_benchmark.routers import build_all_routers
from router_benchmark.benchmarks import build_all_benchmarks
from router_benchmark.metrics import (
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
    compute_pareto_frontier,
)

harness = EvaluationHarness(seed=1234, n_trials=3)
results = harness.evaluate(build_all_routers(), build_all_benchmarks())

per_benchmark = compute_router_benchmark_metrics(results)
overall       = compute_router_overall_metrics(results)
frontier      = compute_pareto_frontier(overall)
```

**Resuming an interrupted run:** `evaluate()` accepts an optional
`completed_keys` set of `(router, benchmark, task_id, trial)` tuples to skip. The
harness reads no files itself — the caller owns the resume state — keeping it a
pure, reusable component. The live driver uses this for opt-in, per-phase resume.

---

## Metrics reference

`compute_router_benchmark_metrics` / `compute_router_overall_metrics` produce a
deployment-focused suite (see `metrics.py`):

- **Quality:** `success_rate`, and per-band `easy_/medium_/hard_success_rate`.
- **Cost:** `cost_per_task_usd`, `cost_per_success_usd`.
- **Latency:** `latency_p50_ms`, `latency_p95_ms`.
- **Reliability:** `tool_call_accuracy`, `fallback_rate`.
- **Routing behavior:** `route_stability` (same decision on repeat trials),
  `mean_confidence`.
- **Robustness:** `robustness_std` (std of success across difficulty bands).
- **Pareto:** `is_pareto_optimal` — non-dominated on cost vs. quality.

The design premise: no single scalar can rank routers once rankings invert
across benchmarks, so choose by **operating point** (cost/quality) and
**stability across a task mix**, not one leaderboard number.

---

## Analysis and protocol workflow

The analysis, canonical-bundle, and protocol commands are installed as package
modules. Versioned contracts live in `protocol/`; they identify the frozen
task scope, candidate tiers, route configurations, and budget reservations.

```bash
# Local validation only. These commands make no provider calls.
make dry-run-preflight
make full-run-preflight

# After a provider-backed stage has produced a locked canonical bundle:
make validate-canonical
make rebuild-analysis
make reviewer-gates
make reproduce-tables
```

`dry-run-candidates`, `dry-run-routes`, `full-run-candidates`, and
`full-run-routes` execute benchmark adapters. They require their respective
preflight, an explicitly approved protocol, provider credentials, and any
benchmark environments. The build does not run those targets automatically.

The same analysis modules can be invoked directly, for example:

```bash
python -m router_benchmark.analysis.paired_tests \
  --bundle output/live/paper1_canonical_v1 \
  --protocol protocol/paper1_rebuild.yaml \
  --output analysis/output/paper1_canonical/paired_effects.csv \
  --draws-output analysis/output/paper1_canonical/paired_draws.json
```

---

## Extending the harness

**Add a router** — implement one method:

```python
from router_benchmark.interfaces import Router, RouteDecision

class MyRouter(Router):
    name = "My Router"
    def route(self, task, context, rng) -> RouteDecision:
        choice = pick_candidate(task)                     # your logic / real API call
        return RouteDecision(selected_candidate=choice.name,
                             confidence=0.9, fallback_used=False, metadata={})
```

**Add a benchmark** — implement two methods:

```python
from router_benchmark.interfaces import Benchmark

class MyBenchmark(Benchmark):
    name = "My Benchmark"
    def generate_tasks(self, rng): ...                    # -> list[Task]
    def score(self, task, decision, rng):                 # -> {success, cost_usd, latency_ms, tool_call_correct}
        ...
```

Both slot directly into `EvaluationHarness.evaluate()` with no other changes.
Writing a *live* adapter is the same interface, backed by real calls — that's
all the `live/` directory is.

---

## Reproducibility & determinism

- Every simulated result derives from `EvaluationHarness(seed=...)`; the same
  seed yields **bit-identical** results across machines. (The only
  non-deterministic column is `router_decision_latency_ms`, measured wall-clock
  time — expected to vary.)
- Live runs seed task *selection* deterministically and record full provenance
  (`manifest.json` + `traces.jsonl`), but outcomes reflect real API variance.
- Task RNG is derived per (router, benchmark, trial) via a CRC32 of those keys,
  so trials are independent yet reproducible.

---

## Repository layout

Standard `src/` layout — the Python package is isolated from scripts, data, and
generated output:

```
router_benchmark/                     # repo root
├── pyproject.toml                    # installable `router-benchmark` (src layout)
├── README.md                         # this guide (the single source of docs)
├── Makefile                          # webarena setup + reproduce-tables targets
├── src/router_benchmark/             # THE PACKAGE (library code only)
│   ├── interfaces.py                 #   Router/Benchmark/Task/RouteDecision/TaskResult contract
│   ├── harness.py                    #   EvaluationHarness.evaluate(routers, benchmarks)
│   ├── metrics.py                    #   the deployment metric suite
│   ├── routers.py                    #   simulated router adapters + baselines (RouterProfile)
│   ├── benchmarks.py                 #   simulated benchmark adapters
│   ├── protocol/                      #   reproducibility and validation primitives
│   ├── plots.py                      #   figure generation
│   ├── run.py                        #   `python -m router_benchmark.run`
│   └── live/                         #   LIVE backend: real API calls & benchmark execution
│       ├── llm_client.py             #     candidate pool + pricing (central model config)
│       ├── run_common.py             #     per-phase driver (manifest, tracing, resume)
│       ├── *_live.py                 #     live router & benchmark adapters
│       ├── vllm_sr/config.yaml       #     vLLM Semantic Router policy
│       └── run_live_phaseN.py        #     phase entry points
├── experiments/                      # one-off paper table/figure build scripts
├── analysis/                         # bootstrap CIs, oracle/cascade, rank consistency, …
├── tests/                            # offline unit and adapter-validation tests
├── data/live/*/                      # COMMITTED reproducibility CSVs (analysis inputs)
└── output/                           # GENERATED runtime artifacts (gitignored)
```

Not version-controlled (provisioned locally; see `.gitignore`): `.venv/`, the
vendored benchmark repos under `src/router_benchmark/live/`, downloaded model
weights, the runtime `output/` tree, `traces.jsonl`, and generated `*.png` plots.
