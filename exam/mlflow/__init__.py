import mlflow
from mlflow.entities import SpanType


def calculate_overhead(run_id, total_time):
    """Calculate the overhead time (total - tool execution time)"""
    client = mlflow.MlflowClient()

    # Get run info
    run = client.get_run(run_id)
    run_info = run.info

    total_duration_ms = total_time * 1000

    # Get all traces for the run
    traces = client.search_traces(
        locations=[run_info.experiment_id],
        filter_string=f"attributes.run_id = '{run_id}'"
    )

    # Sum tool execution times
    tool_duration_ms = 0
    if traces:
        for trace in traces:
            for span in trace.data.spans:
                if span.span_type == SpanType.TOOL:
                    tool_duration_ms += (span.end_time_ns - span.start_time_ns) / 1_000_000

    # Calculate overhead
    overhead_ms = total_duration_ms - tool_duration_ms

    print("\n" + "=" * 60)
    print("⏱️  TIME ANALYSIS")
    print("=" * 60)
    print(f"Total Duration:       {total_duration_ms / 1000:.2f}s ({total_duration_ms / 60000:.2f}m)")
    print(f"Tool Execution Time:  {tool_duration_ms / 1000:.2f}s ({tool_duration_ms / 60000:.2f}m)")
    print(f"Overhead Time:        {overhead_ms / 1000:.2f}s ({overhead_ms / 60000:.2f}m)")
    print(f"Overhead Percentage:  {(overhead_ms / total_duration_ms * 100):.1f}%")
    print("=" * 60 + "\n")