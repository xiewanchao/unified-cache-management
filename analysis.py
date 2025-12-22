import pandas as pd

path = "/mnt/data/llmperf_summary_results.csv"
df = pd.read_csv(path)

group_cols = ["mean_input_tokens", "mean_output_tokens", "concurrent_requests"]
value_cols = [
    "results_ttft_s_mean",
    "results_inter_token_latency_s_mean",
    "results_end_to_end_latency_s_mean",
]

result = df.groupby(group_cols)[value_cols].agg(["max", "min", "mean"])

# flatten columns
result.columns = [
    f"{col}_{stat}" for col, stat in result.columns
]
result = result.reset_index()

out_path = "/mnt/data/llmperf_grouped_stats.xlsx"
result.to_excel(out_path, index=False)

result.head(), out_path
