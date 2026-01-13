import os

import pytest
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.llmperf.run_inference import inference_results

TIMES = 3
IN_TOKENS = [4000, 8000, 16000, 32000]
OUT_TOKENS = [1000]
CONCURRENCY = [1, 8, 16, 24]
HIT_RATE = 0

perf_scenarios = [
    (in_tokens, OUT_TOKENS[0], concurrent, concurrent, "{}", HIT_RATE)
    for in_tokens in IN_TOKENS
    for concurrent in CONCURRENCY
]
perf_scenarios = perf_scenarios * TIMES
scenario_ids = [f"in_{s[0]}-out_{s[1]}-con_{s[3]}" for s in perf_scenarios]


@pytest.mark.stage(2)
@pytest.mark.feature("uc_performance_test_0")
@pytest.mark.parametrize(
    "in_tokens, out_tokens, max_req, concurrent, sampling, hit_rate",
    perf_scenarios,
    ids=scenario_ids,
)
@export_vars
def test_performance(
    in_tokens, out_tokens, max_req, concurrent, sampling, hit_rate, request
):
    all_summaries = inference_results(
        [in_tokens], [out_tokens], [max_req], [concurrent], [sampling], [hit_rate]
    )
    summary = all_summaries[0]
    failed_cases = []

    results = summary.get("results", {})

    # Build a flattened metrics dict for analysis and reporting
    metrics = {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "concurrent": concurrent,
        "sum_requests": max_req,
        "hit_rate": hit_rate,
        "ttft_mean": results.get("ttft_s", {}).get("mean"),
        "tpot_mean": results.get("inter_token_latency_s", {}).get("mean"),
        "total_throughput": summary.get("total_throughput"),
        "e2e_mean": results.get("end_to_end_latency_s", {}).get("mean"),
        "extra_info": os.getenv("TEST_EXTRA_INFO")
        or config_instance.get_nested_config("llm_connection.extra_info"),
        "mean_input_tokens": summary.get("mean_input_tokens"),
        "mean_output_tokens": summary.get("mean_output_tokens"),
        "itl_p50": results.get("inter_token_latency_s", {})
        .get("quantiles", {})
        .get("p50"),
        "itl_p90": results.get("inter_token_latency_s", {})
        .get("quantiles", {})
        .get("p90"),
        "itl_p99": results.get("inter_token_latency_s", {})
        .get("quantiles", {})
        .get("p99"),
        "ttft_p50": results.get("ttft_s", {}).get("quantiles", {}).get("p50"),
        "ttft_p90": results.get("ttft_s", {}).get("quantiles", {}).get("p90"),
        "ttft_p99": results.get("ttft_s", {}).get("quantiles", {}).get("p99"),
        "e2e_p50": results.get("end_to_end_latency_s", {})
        .get("quantiles", {})
        .get("p50"),
        "e2e_p90": results.get("end_to_end_latency_s", {})
        .get("quantiles", {})
        .get("p90"),
        "e2e_p99": results.get("end_to_end_latency_s", {})
        .get("quantiles", {})
        .get("p99"),
        "num_completed_requests": summary.get("num_completed_requests"),
        "elapsed_time": summary.get("elapsed_time"),
        "incremental_throughput": summary.get("incremental_throughput"),
    }

    for key, val in metrics.items():
        assert val is not None, f"Metric '{key}' is missing"

    return {"_name": "llmperf", "_proj": metrics}
