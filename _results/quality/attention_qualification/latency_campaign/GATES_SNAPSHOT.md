# Frozen latency gates copied before the 1K/1K campaign

Source: `_results/quality/attention_qualification/GATES.md`, written before
the paired quality suite. These limits are not being revised for this run.

- C1 1K/1K output throughput must be at least 98 tok/s.
- C8 1K/1K aggregate throughput must be at least 640 tok/s.
- At C1/C4/C8, TTFT p95 may not regress more than 10% or 5 ms, whichever is
  larger, versus stock for the same context.
- Token-arrival p95 may not regress versus stock and may not exceed 11 ms
  at the 1K workload.

This campaign measures only the 1K/1K C1/C4/C8 cell, three unprofiled
repetitions, with `scripts/bench_openai_stream.py`. Graph mode, batching,
KV precision, checkpoint, and GEMM dispatch are unchanged.
