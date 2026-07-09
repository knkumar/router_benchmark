# WebArena reproduction targets. Wraps the two pre-built WebArena
# environment containers this study actually used (confirmed via `docker
# inspect` 2026-07-05: gitlab-populated-final-port8023:latest on host
# port 8023, shopping_final_0712:latest on host port 7770) -- these are
# WebArena's own official environment images (Zhou et al., see paper1.tex
# reference [14] / https://webarena.dev environment setup docs), not
# images built from a Dockerfile in this repo. This Makefile does not
# fabricate a build step for images this repo never built from source;
# it wraps what already exists and fails loudly if it doesn't.

.PHONY: setup-webarena run-benchmark reproduce-tables webarena-health

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

reproduce-tables:
	cd analysis && python3 bootstrap_ci.py && python3 candidate_distribution.py && \
		python3 oracle_and_cascade.py && python3 regret_to_oracle.py && \
		python3 normalized_cost.py && python3 realistic_cascade.py && \
		python3 rank_consistency_4bench.py && python3 mixture_utility.py
	cd ../paper && rm -f paper1.aux paper1.log paper1.out && pdflatex paper1.tex && pdflatex paper1.tex
