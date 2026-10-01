#!/usr/bin/env python3
"""
Unified Request & Event Telemetry Schema for LLM Tail Latency & Agent Benchmarks.

Provides a standardized, immutable event schema across all inference benchmarks:
  - Timestamping at nanosecond resolution
  - Separation of scheduled arrival vs. actual send (detects client lag & coordinated omission)
  - Explicit measurement boundaries (scheduled->first, actual->first, server->first)
  - Disaggregated milestones: first byte, first output token (TTFT), first answer token (TTFAT)
  - Uncompressed token-event timeline (every output event timestamp)
  - Decode rate: (M - 1) / (t_last - t_first) only when M > 1, using token timestamps
  - Server spans: queue, prefill, decode, KV cache/eviction, KV transfer
  - Tool and orchestration spans for complete-task agent workflows
  - Token-weighted ITL distribution vs. request-weighted TPOT
  - Complete error, timeout type, and deadline-miss accounting
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def linear_percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    """Standard linear-interpolation percentile."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


@dataclass
class ServerSpans:
    """Optional server-side timing spans (e.g. from vLLM engine metrics, logs, or headers)."""
    admission_ns: Optional[int] = None
    queue_entry_ns: Optional[int] = None
    queue_exit_ns: Optional[int] = None
    prefill_start_ns: Optional[int] = None
    prefill_end_ns: Optional[int] = None
    decode_start_ns: Optional[int] = None
    decode_end_ns: Optional[int] = None
    kv_transfer_start_ns: Optional[int] = None
    kv_transfer_end_ns: Optional[int] = None
    kv_eviction_event: bool = False
    prefix_cache_hit_rate: Optional[float] = None

    @property
    def queue_duration_ms(self) -> Optional[float]:
        if self.queue_entry_ns is not None and self.queue_exit_ns is not None:
            return (self.queue_exit_ns - self.queue_entry_ns) / 1e6
        return None

    @property
    def prefill_duration_ms(self) -> Optional[float]:
        if self.prefill_start_ns is not None and self.prefill_end_ns is not None:
            return (self.prefill_end_ns - self.prefill_start_ns) / 1e6
        return None

    @property
    def decode_duration_ms(self) -> Optional[float]:
        if self.decode_start_ns is not None and self.decode_end_ns is not None:
            return (self.decode_end_ns - self.decode_start_ns) / 1e6
        return None

    @property
    def kv_transfer_duration_ms(self) -> Optional[float]:
        if self.kv_transfer_start_ns is not None and self.kv_transfer_end_ns is not None:
            return (self.kv_transfer_end_ns - self.kv_transfer_start_ns) / 1e6
        return None


@dataclass
class RequestEvent:
    """Unified telemetry record for a single LLM request or agent step."""
    # Identification
    request_id: str
    run_id: str = ""
    task_id: Optional[str] = None          # Complete agent task ID
    chain_id: Optional[str] = None         # Chain identifier
    call_index: Optional[int] = None       # Sequence index within chain (1..N)
    replica_id: Optional[str] = None       # Specific GPU / container / instance ID
    model: str = ""                        # Model name alias

    # Timing milestones (nanoseconds)
    scheduled_arrival_ns: int = 0          # Scheduled dispatch time (open-loop target)
    scheduled_send_ns: int = 0             # Scheduled dispatch time alias
    actual_send_ns: int = 0                # Time socket write actually began
    first_response_byte_ns: Optional[int] = None # First byte from network stream
    first_output_token_ns: Optional[int] = None  # First output token chunk (TTFT)
    first_streamed_output_ns: Optional[int] = None # First output token chunk alias
    first_answer_token_ns: Optional[int] = None  # First visible answer token chunk (TTFAT)
    token_timestamps_ns: List[int] = field(default_factory=list) # Every token arrival event
    stream_end_ns: int = 0                 # Stream closure timestamp
    completion_or_timeout_ns: int = 0      # Stream end / timeout timestamp alias

    # Token Accounting
    input_tokens: int = 0
    actual_output_tokens: int = 0
    completion_tokens: int = 0             # Total generated tokens alias
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    finish_reason: str = "unknown"         # "stop", "length", "timeout", "error"
    usage_source: str = "stream_count"     # "server_usage_chunk", "stream_count", "approx"

    # Configuration & Metadata
    model_revision: str = ""
    serving_configuration: str = ""        # e.g., "mxfp4_chunk2048", "fp8_pdd_1p1d"
    prompt_length_bucket: str = ""         # e.g., "short_100", "long_8192"
    output_length_bucket: str = ""         # e.g., "short_64", "long_512"
    offered_load_tag: str = ""             # e.g., "qps_15", "concurrency_8"
    cache_state: str = "unknown"           # "cold_miss", "warm_hit", "partial_evicted"

    # Execution status & Failure accounting
    status: str = "completed"              # "completed", "timeout", "cancelled", "error"
    timeout_type: Optional[str] = None     # "connection", "first_byte", "idle_stream", "watchdog"
    error_message: Optional[str] = None
    retries: int = 0
    partial_output: bool = False

    # Server spans (if observable from engine logs/headers)
    server_spans: ServerSpans = field(default_factory=ServerSpans)

    # Agent Task & Orchestration Spans
    tool_start_ns: Optional[int] = None
    tool_end_ns: Optional[int] = None
    orchestration_start_ns: Optional[int] = None
    orchestration_end_ns: Optional[int] = None

    def __post_init__(self):
        if self.model and not self.model_revision:
            self.model_revision = self.model
        elif self.model_revision and not self.model:
            self.model = self.model_revision

        if self.scheduled_send_ns and not self.scheduled_arrival_ns:
            self.scheduled_arrival_ns = self.scheduled_send_ns
        elif self.scheduled_arrival_ns and not self.scheduled_send_ns:
            self.scheduled_send_ns = self.scheduled_arrival_ns

        if self.first_streamed_output_ns and not self.first_output_token_ns:
            self.first_output_token_ns = self.first_streamed_output_ns
        elif self.first_output_token_ns and not self.first_streamed_output_ns:
            self.first_streamed_output_ns = self.first_output_token_ns

        if self.completion_tokens and not self.actual_output_tokens:
            self.actual_output_tokens = self.completion_tokens
        elif self.actual_output_tokens and not self.completion_tokens:
            self.completion_tokens = self.actual_output_tokens

        if self.completion_or_timeout_ns and not self.stream_end_ns:
            self.stream_end_ns = self.completion_or_timeout_ns
        elif self.stream_end_ns and not self.completion_or_timeout_ns:
            self.completion_or_timeout_ns = self.stream_end_ns

    # -------------------------------------------------------------
    # Explicit Measurement Boundaries
    # -------------------------------------------------------------
    @property
    def generator_delay_ms(self) -> float:
        """Client load-generator send lag: actual_send - scheduled_arrival."""
        sched = self.scheduled_arrival_ns or self.scheduled_send_ns
        if sched > 0 and self.actual_send_ns >= sched:
            return (self.actual_send_ns - sched) / 1e6
        return 0.0

    @property
    def client_send_lag_ms(self) -> float:
        return self.generator_delay_ms

    @property
    def ttft_ms(self) -> Optional[float]:
        return self.actual_send_to_first_output_ms

    @property
    def ttfat_ms(self) -> Optional[float]:
        return self.actual_send_to_first_answer_ms

    @property
    def scheduled_to_first_output_ms(self) -> Optional[float]:
        """Boundary: scheduled arrival -> first output token (includes client queue lag)."""
        first = self.first_output_token_ns or self.first_streamed_output_ns
        sched = self.scheduled_arrival_ns or self.scheduled_send_ns
        if first and sched > 0:
            return (first - sched) / 1e6
        return None

    @property
    def actual_send_to_first_output_ms(self) -> Optional[float]:
        """Boundary: actual socket send -> first output token (standard client TTFT)."""
        if self.first_output_token_ns and self.actual_send_ns > 0:
            return (self.first_output_token_ns - self.actual_send_ns) / 1e6
        return None

    @property
    def actual_send_to_first_answer_ms(self) -> Optional[float]:
        """Boundary: actual socket send -> first visible answer token (TTFAT)."""
        if self.first_answer_token_ns and self.actual_send_ns > 0:
            return (self.first_answer_token_ns - self.actual_send_ns) / 1e6
        return self.actual_send_to_first_output_ms

    @property
    def server_receipt_to_first_output_ms(self) -> Optional[float]:
        """Boundary: server admission -> first output token (pure server-side TTFT)."""
        if self.server_spans.admission_ns and self.first_output_token_ns:
            return (self.first_output_token_ns - self.server_spans.admission_ns) / 1e6
        return None

    @property
    def e2e_latency_ms(self) -> float:
        """Total wall-clock duration from actual send to stream end."""
        if self.stream_end_ns and self.actual_send_ns:
            return (self.stream_end_ns - self.actual_send_ns) / 1e6
        return 0.0

    # -------------------------------------------------------------
    # Token-Level ITL & Decode Rate Mechanics
    # -------------------------------------------------------------
    @property
    def itl_intervals_ms(self) -> List[float]:
        """Distribution of individual inter-token gaps (uncompressed)."""
        if len(self.token_timestamps_ns) < 2:
            return []
        return [
            (t2 - t1) / 1e6
            for t1, t2 in zip(self.token_timestamps_ns[:-1], self.token_timestamps_ns[1:])
        ]

    @property
    def worst_itl_ms(self) -> Optional[float]:
        """Maximum single inter-token freeze during generation."""
        intervals = self.itl_intervals_ms
        return max(intervals) if intervals else None

    @property
    def per_request_tpot_ms(self) -> Optional[float]:
        """Request-average Time Per Output Token."""
        intervals = self.itl_intervals_ms
        return statistics.mean(intervals) if intervals else None

    @property
    def decode_tokens_per_second(self) -> Optional[float]:
        """
        Valid per-request output generation rate:
          (M - 1) / (t_last_output - t_first_output)
        Evaluated strictly when M > 1 and using the timestamp boundary of the
        first and last generated tokens (NOT the trailing usage chunk).
        """
        m = len(self.token_timestamps_ns)
        if m <= 1:
            return None
        t_first = self.token_timestamps_ns[0]
        t_last = self.token_timestamps_ns[-1]
        span_s = (t_last - t_first) / 1e9
        if span_s <= 0.0001:
            return None
        return (m - 1) / span_s

    @property
    def tool_duration_ms(self) -> Optional[float]:
        if self.tool_start_ns is not None and self.tool_end_ns is not None:
            return (self.tool_end_ns - self.tool_start_ns) / 1e6
        return None

    @property
    def orchestration_duration_ms(self) -> Optional[float]:
        if self.orchestration_start_ns is not None and self.orchestration_end_ns is not None:
            return (self.orchestration_end_ns - self.orchestration_start_ns) / 1e6
        return None

    def passes_slo(
        self,
        ttft_max_ms: float = 1000.0,
        worst_itl_max_ms: float = 100.0,
        e2e_max_ms: float = 10000.0,
    ) -> bool:
        """Check if request passes explicit interactive SLO thresholds."""
        if self.status != "completed":
            return False
        ttft = self.actual_send_to_first_output_ms
        if ttft is not None and ttft > ttft_max_ms:
            return False
        worst_itl = self.worst_itl_ms
        if worst_itl is not None and worst_itl > worst_itl_max_ms:
            return False
        if self.e2e_latency_ms > e2e_max_ms:
            return False
        return True


@dataclass
class CompleteAgentTaskTrace:
    """Complete multi-turn agent trajectory trace record."""
    task_id: str
    run_id: str
    workflow_name: str
    total_calls: int
    tool_calls: int
    wall_duration_ms: float
    all_calls_succeeded: bool
    status: str
    steps: List[RequestEvent] = field(default_factory=list)

    @property
    def sum_llm_duration_ms(self) -> float:
        return sum(s.e2e_latency_ms for s in self.steps)

    @property
    def sum_tool_duration_ms(self) -> float:
        return sum(s.tool_duration_ms or 0.0 for s in self.steps)

    @property
    def orchestration_overhead_ms(self) -> float:
        return max(self.wall_duration_ms - (self.sum_llm_duration_ms + self.sum_tool_duration_ms), 0.0)

    def contains_slow_call(self, single_call_threshold_ms: float) -> bool:
        return any(s.e2e_latency_ms > single_call_threshold_ms for s in self.steps)


@dataclass
class BenchmarkCorpusSummary:
    """Standardized statistical aggregation over a set of RequestEvents."""
    total_requests: int
    completed_requests: int
    timeout_requests: int
    error_requests: int
    cancelled_requests: int

    benchmark_duration_s: float

    # Boundary 1: Actual Send -> First Output Token (Request-Weighted)
    send_to_ttft_p50_ms: Optional[float]
    send_to_ttft_p95_ms: Optional[float]
    send_to_ttft_p99_ms: Optional[float]

    # Boundary 2: Scheduled Arrival -> First Output Token (Open-Loop Customer Experience)
    sched_to_ttft_p50_ms: Optional[float]
    sched_to_ttft_p95_ms: Optional[float]
    sched_to_ttft_p99_ms: Optional[float]

    # First Answer Token (TTFAT, Request-Weighted)
    ttfat_p50_ms: Optional[float]
    ttfat_p95_ms: Optional[float]
    ttfat_p99_ms: Optional[float]

    # End-to-End Latency (Request-Weighted)
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float

    # Decode Output Rate (Request-Weighted, valid M > 1)
    decode_tps_p50: Optional[float]
    decode_tps_p95: Optional[float]

    # ITL Gaps (Token-Weighted, Pooled across all token events)
    token_itl_p50_ms: Optional[float]
    token_itl_p95_ms: Optional[float]
    token_itl_p99_ms: Optional[float]
    total_token_intervals: int
    gaps_gt_50ms: int
    gaps_gt_100ms: int
    gaps_gt_1000ms: int

    # Throughput & Goodput
    offered_requests_per_s: float
    completed_requests_per_s: float
    completed_tokens_per_s: float
    slo_passed_requests: int
    slo_qualified_goodput_tokens_per_s: float
    generator_send_delay_p95_ms: float     # Load generator lag check

    @property
    def ttft_p50_ms(self) -> Optional[float]:
        return self.send_to_ttft_p50_ms

    @property
    def ttft_p95_ms(self) -> Optional[float]:
        return self.send_to_ttft_p95_ms

    @property
    def ttft_p99_ms(self) -> Optional[float]:
        return self.send_to_ttft_p99_ms

    @property
    def client_send_lag_p95_ms(self) -> float:
        return self.generator_send_delay_p95_ms

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ttft_p50_ms"] = self.ttft_p50_ms
        d["ttft_p95_ms"] = self.ttft_p95_ms
        d["ttft_p99_ms"] = self.ttft_p99_ms
        d["client_send_lag_p95_ms"] = self.client_send_lag_p95_ms
        return d


def summarize_events(
    events: List[RequestEvent],
    duration_s: float,
    ttft_slo_ms: float = 1000.0,
    worst_itl_slo_ms: float = 100.0,
    e2e_slo_ms: float = 10000.0,
) -> BenchmarkCorpusSummary:
    """Compile request events into a defensible benchmark corpus summary."""
    n = len(events)
    completed = [e for e in events if e.status == "completed"]
    timeouts = [e for e in events if e.status == "timeout"]
    errors = [e for e in events if e.status == "error"]
    cancelled = [e for e in events if e.status == "cancelled"]

    # Boundary 1: Actual Send -> TTFT
    send_ttfts = sorted([e.actual_send_to_first_output_ms for e in completed if e.actual_send_to_first_output_ms is not None])
    # Boundary 2: Scheduled Arrival -> TTFT
    sched_ttfts = sorted([e.scheduled_to_first_output_ms for e in completed if e.scheduled_to_first_output_ms is not None])
    # TTFAT
    ttfats = sorted([e.actual_send_to_first_answer_ms for e in completed if e.actual_send_to_first_answer_ms is not None])
    # E2E
    e2es = sorted([e.e2e_latency_ms for e in completed])
    # Decode rate
    tps_rates = sorted([e.decode_tokens_per_second for e in completed if e.decode_tokens_per_second is not None])

    # Token-level ITL: POOL EVERY RAW INTERVAL ACROSS ALL COMPLETED STREAMS
    all_token_itls: List[float] = []
    for e in completed:
        all_token_itls.extend(e.itl_intervals_ms)
    sorted_itls = sorted(all_token_itls)

    # Generator lag check
    lags = sorted([e.generator_delay_ms for e in events])

    # SLO Qualification
    passed = [e for e in completed if e.passes_slo(ttft_slo_ms, worst_itl_slo_ms, e2e_slo_ms)]
    passed_tokens = sum(e.actual_output_tokens for e in passed)
    total_tokens = sum(e.actual_output_tokens for e in completed)

    dur = max(duration_s, 0.001)

    return BenchmarkCorpusSummary(
        total_requests=n,
        completed_requests=len(completed),
        timeout_requests=len(timeouts),
        error_requests=len(errors),
        cancelled_requests=len(cancelled),
        benchmark_duration_s=round(dur, 2),
        send_to_ttft_p50_ms=linear_percentile(send_ttfts, 50.0),
        send_to_ttft_p95_ms=linear_percentile(send_ttfts, 95.0),
        send_to_ttft_p99_ms=linear_percentile(send_ttfts, 99.0),
        sched_to_ttft_p50_ms=linear_percentile(sched_ttfts, 50.0),
        sched_to_ttft_p95_ms=linear_percentile(sched_ttfts, 95.0),
        sched_to_ttft_p99_ms=linear_percentile(sched_ttfts, 99.0),
        ttfat_p50_ms=linear_percentile(ttfats, 50.0),
        ttfat_p95_ms=linear_percentile(ttfats, 95.0),
        ttfat_p99_ms=linear_percentile(ttfats, 99.0),
        e2e_p50_ms=linear_percentile(e2es, 50.0) or 0.0,
        e2e_p95_ms=linear_percentile(e2es, 95.0) or 0.0,
        e2e_p99_ms=linear_percentile(e2es, 99.0) or 0.0,
        decode_tps_p50=linear_percentile(tps_rates, 50.0),
        decode_tps_p95=linear_percentile(tps_rates, 95.0),
        token_itl_p50_ms=linear_percentile(sorted_itls, 50.0),
        token_itl_p95_ms=linear_percentile(sorted_itls, 95.0),
        token_itl_p99_ms=linear_percentile(sorted_itls, 99.0),
        total_token_intervals=len(sorted_itls),
        gaps_gt_50ms=sum(1 for g in sorted_itls if g > 50.0),
        gaps_gt_100ms=sum(1 for g in sorted_itls if g > 100.0),
        gaps_gt_1000ms=sum(1 for g in sorted_itls if g > 1000.0),
        offered_requests_per_s=round(n / dur, 2),
        completed_requests_per_s=round(len(completed) / dur, 2),
        completed_tokens_per_s=round(total_tokens / dur, 2),
        slo_passed_requests=len(passed),
        slo_qualified_goodput_tokens_per_s=round(passed_tokens / dur, 2),
        generator_send_delay_p95_ms=linear_percentile(lags, 95.0) or 0.0,
    )
