# WebArena reproduction targets. Wraps the two pre-built WebArena
# environment containers this study actually used (confirmed via `docker
# inspect` 2026-07-05: gitlab-populated-final-port8023:latest on host
# port 8023, shopping_final_0712:latest on host port 7770) -- these are
# WebArena's own official environment images (Zhou et al., see paper1.tex
# reference [14] / https://webarena.dev environment setup docs), not
# images built from a Dockerfile in this repo. This Makefile does not
# fabricate a build step for images this repo never built from source;
# it wraps what already exists and fails loudly if it doesn't.

.PHONY: setup-webarena run-benchmark reproduce-tables webarena-health test validate-canonical \
	dry-run-preflight dry-run-candidates dry-run-routes dry-run-bundle \
	full-run-preflight full-run-readiness full-run-approval-packet full-run-status full-run-spend-summary \
	full-run-candidates full-run-routes full-run-bundle rebuild-analysis reviewer-gates

PYTHON ?= python
RUN = PYTHONPATH=src $(PYTHON)

CANONICAL_BUNDLE ?= output/live/paper1_canonical_v1
REBUILD_PROTOCOL ?= protocol/paper1_rebuild.yaml
ANALYSIS_PROTOCOL ?= protocol/analysis.yaml
ANALYSIS_OUTPUT_DIR ?= analysis/output/paper1_canonical
DRY_RUN_PROTOCOL ?= protocol/paper1_dry_run.yaml
DRY_RUN_STAGE_DIR ?= output/dry_run_stage
DRY_RUN_BUNDLE_DIR ?= output/dry_run_bundle
FULL_RUN_STAGE_DIR ?= output/full_run_stage
FULL_RUN_READINESS ?= analysis/output/paper1_canonical/full_run_readiness.json
FULL_RUN_APPROVAL_PACKET ?= analysis/output/paper1_canonical/full_run_approval_packet.md
FULL_RUN_STATUS ?= analysis/output/paper1_canonical/full_run_status.json
FULL_RUN_SPEND_SUMMARY ?= analysis/output/paper1_canonical/spend_summary.json
FULL_RUN_BENCHMARK_SPEND ?= analysis/output/paper1_canonical/benchmark_spend.csv
ADAPTER_FACTORY ?= router_benchmark.protocol.production_adapters:build_dry_run_adapters
ROUTER_FACTORY ?= router_benchmark.protocol.production_adapters:build_dry_run_routers
FULL_ADAPTER_FACTORY ?= router_benchmark.protocol.production_adapters:build_full_run_adapters
FULL_ROUTER_FACTORY ?= router_benchmark.protocol.production_adapters:build_full_run_routers

GITLAB_IMAGE := gitlab-populated-final-port8023:latest
SHOPPING_IMAGE := shopping_final_0712:latest

setup-webarena:
	@echo "Checking for WebArena environment images..."
	@docker image inspect $(GITLAB_IMAGE) >/dev/null 2>&1 || \
		(echo "Missing $(GITLAB_IMAGE). Load it from WebArena's official environment distribution (see paper1.tex ref [14] / https://webarena.dev) before continuing." && exit 1)
	@docker image inspect $(SHOPPING_IMAGE) >/dev/null 2>&1 || \
		(echo "Missing $(SHOPPING_IMAGE). Load it from WebArena's official environment distribution (see paper1.tex ref [14] / https://webarena.dev) before continuing." && exit 1)
	@docker start gitlab >/dev/null 2>&1 || docker run -d --name gitlab -p 8023:8023 $(GITLAB_IMAGE)
	@docker start shopping >/dev/null 2>&1 || docker run -d --name shopping -p 7770:80 $(SHOPPING_IMAGE)
	$(MAKE) webarena-health

webarena-health:
	@echo "Waiting for gitlab (localhost:8023) and shopping (localhost:7770) to answer..."
	@for i in $$(seq 1 60); do \
		gitlab_ok=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8023 || echo "000"); \
		shop_ok=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7770 || echo "000"); \
		if [ "$$gitlab_ok" != "000" ] && [ "$$shop_ok" != "000" ]; then \
			echo "Both sites responding (gitlab=$$gitlab_ok, shopping=$$shop_ok)."; exit 0; \
		fi; \
		sleep 5; \
	done; \
	echo "Timed out waiting for both sites to respond." && exit 1

run-benchmark: setup-webarena
	python3 -m router_benchmark.live.run_live_phase7c

test:
	docker build --quiet --tag router-benchmark-test:local .
	docker run --rm router-benchmark-test:local

validate-canonical:
	$(RUN) -m router_benchmark.scripts.validate_paper1_bundle --bundle $(CANONICAL_BUNDLE) --protocol $(REBUILD_PROTOCOL)

dry-run-preflight:
	$(RUN) -m router_benchmark.scripts.preflight_dry_run --dry-protocol $(DRY_RUN_PROTOCOL) --frozen-protocol $(REBUILD_PROTOCOL)

full-run-preflight:
	$(RUN) -m router_benchmark.scripts.preflight_full_run --protocol $(REBUILD_PROTOCOL)

dry-run-candidates: dry-run-preflight
	$(RUN) -m router_benchmark.protocol.dry_run_execution --dry-protocol $(DRY_RUN_PROTOCOL) --frozen-protocol $(REBUILD_PROTOCOL) --stage-dir $(DRY_RUN_STAGE_DIR) --adapter-factory $(ADAPTER_FACTORY) $(if $(RESUME),--resume,)

dry-run-routes: dry-run-preflight
	$(RUN) -m router_benchmark.protocol.dry_run_routes --dry-protocol $(DRY_RUN_PROTOCOL) --frozen-protocol $(REBUILD_PROTOCOL) --stage-dir $(DRY_RUN_STAGE_DIR) --adapter-factory $(ADAPTER_FACTORY) --router-factory $(ROUTER_FACTORY) $(if $(OVERWRITE),--overwrite,)

dry-run-bundle: dry-run-preflight
	$(RUN) -m router_benchmark.protocol.dry_run_bundle --dry-protocol $(DRY_RUN_PROTOCOL) --frozen-protocol $(REBUILD_PROTOCOL) --stage-dir $(DRY_RUN_STAGE_DIR) --bundle-dir $(DRY_RUN_BUNDLE_DIR)

full-run-readiness:
	$(RUN) -m router_benchmark.scripts.audit_full_run_readiness --protocol $(REBUILD_PROTOCOL) --output $(FULL_RUN_READINESS) --allow-blockers

full-run-approval-packet: full-run-readiness
	$(RUN) -m router_benchmark.scripts.generate_full_run_approval_packet --protocol $(REBUILD_PROTOCOL) --readiness $(FULL_RUN_READINESS) --output $(FULL_RUN_APPROVAL_PACKET)

full-run-status: full-run-readiness
	$(RUN) -m router_benchmark.scripts.audit_full_run_status --protocol $(REBUILD_PROTOCOL) --readiness $(FULL_RUN_READINESS) --stage-dir $(FULL_RUN_STAGE_DIR) --bundle-dir $(CANONICAL_BUNDLE) --analysis-dir $(ANALYSIS_OUTPUT_DIR) --output $(FULL_RUN_STATUS) --allow-incomplete

full-run-spend-summary: validate-canonical
	$(RUN) -m router_benchmark.scripts.summarize_full_run_spend --bundle $(CANONICAL_BUNDLE) --json-output $(FULL_RUN_SPEND_SUMMARY) --benchmark-output $(FULL_RUN_BENCHMARK_SPEND)

full-run-candidates: full-run-preflight
	$(RUN) -m router_benchmark.protocol.full_run_execution --protocol $(REBUILD_PROTOCOL) --stage-dir $(FULL_RUN_STAGE_DIR) --adapter-factory $(FULL_ADAPTER_FACTORY) $(if $(RESUME),--resume,)

full-run-routes: full-run-preflight
	$(RUN) -m router_benchmark.protocol.full_run_routes --protocol $(REBUILD_PROTOCOL) --stage-dir $(FULL_RUN_STAGE_DIR) --adapter-factory $(FULL_ADAPTER_FACTORY) --router-factory $(FULL_ROUTER_FACTORY) $(if $(OVERWRITE),--overwrite,)

full-run-bundle: full-run-preflight
	$(RUN) -m router_benchmark.protocol.full_run_bundle --protocol $(REBUILD_PROTOCOL) --stage-dir $(FULL_RUN_STAGE_DIR) --bundle-dir $(CANONICAL_BUNDLE)

rebuild-analysis: validate-canonical
	$(RUN) -m router_benchmark.analysis.paired_tests --bundle $(CANONICAL_BUNDLE) --protocol $(REBUILD_PROTOCOL) --analysis-protocol $(ANALYSIS_PROTOCOL) --output $(ANALYSIS_OUTPUT_DIR)/paired_effects.csv --draws-output $(ANALYSIS_OUTPUT_DIR)/paired_draws.json
	$(RUN) -m router_benchmark.analysis.canonical_uncertainty --bundle $(CANONICAL_BUNDLE) --protocol $(REBUILD_PROTOCOL) --draws $(ANALYSIS_OUTPUT_DIR)/paired_draws.json --rank-output $(ANALYSIS_OUTPUT_DIR)/rank_uncertainty.csv --pareto-output $(ANALYSIS_OUTPUT_DIR)/pareto_uncertainty.csv --rank-consistency-output $(ANALYSIS_OUTPUT_DIR)/rank_consistency_uncertainty.csv --all-policy-pareto-output $(ANALYSIS_OUTPUT_DIR)/pareto_uncertainty_all_policy.csv
	$(RUN) -m router_benchmark.analysis.vllm_share_permutation --bundle $(CANONICAL_BUNDLE) --protocol $(REBUILD_PROTOCOL) --output $(ANALYSIS_OUTPUT_DIR)/vllm_share_permutation.csv --metadata-output $(ANALYSIS_OUTPUT_DIR)/vllm_share_permutation.json

reviewer-gates: validate-canonical
	$(RUN) -m router_benchmark.analysis.reviewer_gates --bundle $(CANONICAL_BUNDLE) --protocol $(REBUILD_PROTOCOL) --output-dir $(ANALYSIS_OUTPUT_DIR)

reproduce-tables: rebuild-analysis reviewer-gates
	$(RUN) -m router_benchmark.scripts.generate_paper1_canonical_tables --bundle $(CANONICAL_BUNDLE) --protocol $(REBUILD_PROTOCOL) --analysis-dir $(ANALYSIS_OUTPUT_DIR) --paper-tables-dir $(ANALYSIS_OUTPUT_DIR)/tables
