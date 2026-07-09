from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive

def main():
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(), VLLMSemanticRouterLive()]
    benchmarks = [Tau2BenchLive(n_tasks=2)]
    run_live_phase("tau2_pilot_check", routers, benchmarks, seed=1234, n_trials=1)

if __name__ == "__main__":
    main()
