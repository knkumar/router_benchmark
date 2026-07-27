"""Real benchmark adapters.

RouterBenchLive: uses the actual published RouterBench logged-outcome
dataset (withmartian/routerbench on Hugging Face, routerbench_0shot.pkl,
36,497 real prompts x 11 real model outcomes with real scores and real
USD costs, per Hu et al. 2024). No LLM calls needed -- the router picks a
tier, and we look up that tier's real logged score/cost for the real
prompt. This is genuinely real data at near-zero marginal cost.

BFCLLive: uses the actual BFCL v4 "simple" single-turn function-calling
test data and ground truth shipped inside the bfcl-eval pip package
(Patil et al., Gorilla project). The router picks a tier, we make a real
live LLM call with the real function schema as a tool definition, and
grade the real tool-call output against the real ground truth.

LIMITATION: the BFCL grader here is a simplified required-argument exact/
membership check against BFCL's own possible_answer ground truth format,
restricted to the single-turn "simple" categories (python/java/javascript).
It is not the full official bfcl_eval AST-based checker (which also covers
multi-turn, parallel, and irrelevance categories) -- see
router_benchmark/live/README.md.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence

import pandas as pd
from huggingface_hub import hf_hub_download

from router_benchmark.interfaces import Benchmark, RouteDecision, Task, TaskDomain
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.llm_client import CANDIDATE_TIERS, LiveLLMClient
from router_benchmark.live.frozen_task_selection import normalize_frozen_task_ids, select_frozen_records

_ROUTERBENCH_TIER_MODEL = {
    "cheap-small": "mistralai/mistral-7b-chat",
    "mid-general": "gpt-3.5-turbo-1106",
    "strong-frontier": "gpt-4-1106-preview",
}


class RouterBenchLive(Benchmark):
    name = "RouterBench (live)"

    def __init__(self, n_tasks: int = 60, seed: int = 1234, *, task_ids: Sequence[str] | None = None):
        self.n_tasks = n_tasks
        self.seed = seed
        self._frozen_task_ids = normalize_frozen_task_ids(task_ids, benchmark=self.name)
        self._df = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            path = hf_hub_download(
                repo_id="withmartian/routerbench", filename="routerbench_0shot.pkl", repo_type="dataset"
            )
            self._df = pd.read_pickle(path)
        return self._df

    def generate_tasks(self, rng) -> list[Task]:
        df = self._load()
        if self._frozen_task_ids is None:
            sample = df.sample(n=self.n_tasks, random_state=self.seed)
            selected = [(f"routerbench-{i:04d}", (idx, row)) for i, (idx, row) in enumerate(sample.iterrows())]
        else:
            positions = []
            for task_id in self._frozen_task_ids:
                match = re.fullmatch(r"routerbench-(\d{4})", task_id)
                if match is None:
                    raise ValueError(f"{self.name} has invalid frozen task ID: {task_id}")
                positions.append(int(match.group(1)))
            sample = df.sample(n=max(positions) + 1, random_state=self.seed)
            records = {f"routerbench-{i:04d}": (idx, row) for i, (idx, row) in enumerate(sample.iterrows())}
            selected = select_frozen_records(records, self._frozen_task_ids, benchmark=self.name)
        tier_models = list(_ROUTERBENCH_TIER_MODEL.values())
        tasks = []
        for task_id, (idx, row) in selected:
            avg_score = float(sum(row[m] for m in tier_models) / len(tier_models))
            difficulty = float(1.0 - avg_score)
            tasks.append(
                Task(
                    task_id=task_id,
                    benchmark_name=self.name,
                    domain=TaskDomain.QA_REASONING,
                    difficulty=difficulty,
                    requires_tool_call=False,
                    candidates=LIVE_CANDIDATES,
                    metadata={"row_idx": idx, "prompt": row["prompt"], "eval_name": row["eval_name"]},
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        df = self._load()
        row = df.loc[task.metadata["row_idx"]]
        model = _ROUTERBENCH_TIER_MODEL[decision.selected_candidate]
        real_score = float(row[model])
        real_cost = float(row[f"{model}|total_cost"])
        return {
            "success": bool(real_score >= 0.5),
            "cost_usd": real_cost,
            "latency_ms": float("nan"),  # not recorded in the published RouterBench table
            "tool_call_correct": None,
        }


class BFCLLive(Benchmark):
    name = "BFCL v4 (live)"
    reusable_score = False

    _CATEGORY_FILES = ["BFCL_v4_simple_python.json", "BFCL_v4_simple_java.json", "BFCL_v4_simple_javascript.json"]

    def __init__(self, n_tasks: int = 30, seed: int = 1234, *, task_ids: Sequence[str] | None = None):
        self.n_tasks = n_tasks
        self.seed = seed
        self._frozen_task_ids = normalize_frozen_task_ids(task_ids, benchmark=self.name)
        self._client: LiveLLMClient | None = None
        self._data_dir = None

    def _find_data_dir(self) -> str:
        if self._data_dir is None:
            import bfcl_eval

            self._data_dir = os.path.join(os.path.dirname(bfcl_eval.__file__), "data")
        return self._data_dir

    def _load_raw(self) -> list[dict]:
        data_dir = self._find_data_dir()
        gt_dir = os.path.join(data_dir, "possible_answer")
        items = []
        for fname in self._CATEGORY_FILES:
            with open(os.path.join(data_dir, fname)) as f:
                questions = [json.loads(line) for line in f if line.strip()]
            with open(os.path.join(gt_dir, fname)) as f:
                answers = {json.loads(line)["id"]: json.loads(line)["ground_truth"] for line in f if line.strip()}
            for q in questions:
                if q["id"] in answers:
                    items.append({"question": q, "ground_truth": answers[q["id"]]})
        return items

    @staticmethod
    def _sanitize_name(name: str) -> str:
        # Anthropic/OpenAI tool names must match ^[a-zA-Z0-9_-]{1,128}$;
        # BFCL ships dotted names like "math.factorial".
        return name.replace(".", "_")

    _TYPE_MAP = {
        "dict": "object",
        "hashmap": "object",
        "float": "number",
        "double": "number",
        "long": "integer",
        "integer": "integer",
        "tuple": "array",
        "array": "array",
        "arraylist": "array",
        "boolean": "boolean",
        "char": "string",
        "string": "string",
    }

    @classmethod
    def _sanitize_schema(cls, node):
        # BFCL's parameter schemas (spanning Python/Java/JS categories) use
        # many non-JSON-Schema type names (e.g. "float", "any", "dict",
        # "HashMap", "char", "") that Anthropic/OpenAI's tool schema
        # validators reject outright; map them to valid JSON Schema draft
        # 2020-12 types recursively, case-insensitively.
        if not isinstance(node, dict):
            return node
        node = dict(node)
        t = node.get("type")
        if t is None:
            pass
        elif t == "any" or t == "":
            node.pop("type", None)
        elif str(t).lower() in cls._TYPE_MAP:
            node["type"] = cls._TYPE_MAP[str(t).lower()]
        else:
            node.pop("type", None)
        if "properties" in node:
            node["properties"] = {k: cls._sanitize_schema(v) for k, v in node["properties"].items()}
        if "items" in node:
            node["items"] = cls._sanitize_schema(node["items"])
        return node

    @classmethod
    def _to_tool_schema(cls, fn: dict) -> dict:
        params = cls._sanitize_schema(fn["parameters"])
        return {
            "name": cls._sanitize_name(fn["name"]),
            "description": fn.get("description", ""),
            "parameters": params,
        }

    def generate_tasks(self, rng) -> list[Task]:
        items = self._load_raw()
        if self._frozen_task_ids is None:
            rnd = random.Random(self.seed)
            selected = [
                (f"bfcl-{i:04d}-{item['question']['id']}", item)
                for i, item in enumerate(rnd.sample(items, min(self.n_tasks, len(items))))
            ]
        else:
            # The frozen IDs include the historical sample position and the
            # upstream BFCL question ID, so both are checked before a call.
            records = {
                f"bfcl-{i:04d}-{item['question']['id']}": item
                for i, item in enumerate(random.Random(self.seed).sample(items, min(self.n_tasks, len(items))))
            }
            selected = select_frozen_records(records, self._frozen_task_ids, benchmark=self.name)
        tasks = []
        for task_id, item in selected:
            fn = item["question"]["function"][0]
            n_required = len(fn.get("parameters", {}).get("required", []))
            difficulty = min(1.0, 0.2 + 0.2 * n_required)
            user_msg = item["question"]["question"][0][0]["content"]
            tasks.append(
                Task(
                    task_id=task_id,
                    benchmark_name=self.name,
                    domain=TaskDomain.TOOL_USE,
                    difficulty=difficulty,
                    requires_tool_call=True,
                    candidates=LIVE_CANDIDATES,
                    metadata={
                        "user_msg": user_msg,
                        "tool_schema": self._to_tool_schema(fn),
                        "ground_truth": item["ground_truth"][0],
                    },
                )
            )
        return tasks

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        model = CANDIDATE_TIERS[decision.selected_candidate]
        tool_schema = task.metadata["tool_schema"]
        try:
            if self._client is None:
                self._client = LiveLLMClient()
            result = self._client.call(
                model=model,
                system="You must answer by calling the provided function with the correct arguments. Do not respond in plain text.",
                user=task.metadata["user_msg"],
                tools=[tool_schema],
                max_tokens=256,
                trace_context={
                    "router_name": decision.metadata.get("router_name"),
                    "benchmark_name": task.benchmark_name,
                    "task_id": task.task_id,
                    "trial": decision.metadata.get("trial"),
                    "selected_candidate": decision.selected_candidate,
                    "ground_truth": task.metadata["ground_truth"],
                },
            )
        except Exception as e:
            # A minority of BFCL's 550 real function schemas (spanning
            # Python/Java/JS categories) still fail a provider's strict
            # JSON-Schema validator after sanitization (e.g. malformed
            # nested array/object shapes). The request is rejected before
            # billing, so this costs $0; we record it as a real failure
            # rather than crashing the whole live run.
            print(f"  [BFCLLive] API error on {task.task_id} ({model}): {type(e).__name__}: {e}")
            return {"success": False, "cost_usd": 0.0, "latency_ms": float("nan"), "tool_call_correct": False}

        correct = self._grade(result.tool_calls, task.metadata["ground_truth"], tool_schema)
        return {
            "success": correct,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "tool_call_correct": correct,
        }

    @staticmethod
    def _grade(tool_calls: list, ground_truth: dict, tool_schema: dict) -> bool:
        if not tool_calls:
            return False
        fn_name = next(iter(ground_truth.keys()))
        expected_args = ground_truth[fn_name]
        call = tool_calls[0]
        if call["name"] != BFCLLive._sanitize_name(fn_name):
            return False
        args = call["arguments"]
        required = tool_schema["parameters"].get("required", [])
        for param in required:
            acceptable = [str(v) for v in expected_args.get(param, [])]
            if str(args.get(param, "__missing__")) not in acceptable:
                return False
        return True


def build_live_benchmarks(routerbench_n: int = 60, bfcl_n: int = 30) -> list[Benchmark]:
    return [RouterBenchLive(n_tasks=routerbench_n), BFCLLive(n_tasks=bfcl_n)]
