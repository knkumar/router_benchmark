"""Live (non-simulated) evaluation components.

Everything under router_benchmark/ (interfaces.py, routers.py, benchmarks.py,
harness.py, metrics.py, plots.py) is the SIMULATED backend described in the
paper. This subpackage is the real backend: real API calls to real models,
real routing packages where available, and real benchmark data/graders.

Phase 1 (this file set): live LLM client + LiteLLM Router + Aurelio Semantic
Router, against real RouterBench (logged HF dataset) and real BFCL v4
(pip-installed dataset + live grading).

See router_benchmark/live/run_live.py to execute, and
router_benchmark/live/README.md for scope, cost, and what's still simulated.
"""
