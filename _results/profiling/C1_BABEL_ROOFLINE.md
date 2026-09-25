# C1 control vs RVS BABEL HBM roof (25 Sep 2026)

Canonical report: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md).

**Question:** what is the practical MI350P HBM streaming limit, how close is best unprofiled C1 MXFP4 decode, and what should we optimize next?

## Results

### Best C1 (unprofiled HF control)

Same 1K/1K `ignore_eos` recipe as the campaign, GPU 0, `awq` on `:8000`:

| Metric | Value | Artifact |
|---|---:|---|
| Output tok/s | **79.53** | `_results/profiling/c1_best/bench.json` |
| Campaign 3× mean | 79.35 ± 0.57 | `_results/priority_eval/` |
| Mean ITL (Prometheus) | **12.53 ms** | `c1_best/metrics_delta.json` |
| Mean TTFT (4 timed req) | **52 ms** | same |
| Mean e2e | 12.87 s | bench.json |

This is the C1 performance point. Wrapped rocprofv3 C1 was **60.5–65.3 tok/s** (1K/128) and is **not** a campaign number.

### Practical HBM (RVS BABEL / BabelStream HIP)

`./scripts/memory_bandwidth.sh` on GPU 0, KFD id **64724**, HIP warmed, **2.0 GiB/array**, 10 kernel iterations, 3 process repeats. RVS prints `MiBytes/sec` with `mibibytes:true`; convert to decimal GB/s as \(\mathrm{MiB/s}\times 2^{20}/10^9\).

| Op | Median GB/s | Min | Max | vs 4096 GB/s spec |
|---|---:|---:|---:|---:|
| **Read** | **3793** | 3791 | 3823 | **92.6%** |
| Copy | 3283 | 3280 | 3298 | 80.2% |
| Triad | 3214 | 3193 | 3223 | 78.5% |

Size sweep (Read): 0.5 GiB ≈ 3630 GB/s, 1 GiB ≈ 3701, 2 GiB ≈ 3793–3799. Bandwidth **rises** with working set, so this is HBM, not a cache-only result.

Do **not** use 4 GiB/array or 4 GiB “total” BABEL on this ~32 GiB host: RVS hangs after “Using HIP device” (host-side alloc). Parser note: `c1_roof_read/REPORT.txt` is empty because the script originally grepped `GB/s` rather than `MiBytes/sec`; measured JSON is `_results/profiling/babel/c1_roof/summary_measured.json`.

BABEL is a **streaming microbenchmark**, not vLLM decode traffic.

### Conditional C1 roofs (17.91 GiB full-weight stream)

\(W = 17.91\,\mathrm{GiB} = 19.231\,\mathrm{GB}\).

| Roof | Formula | tok/s |
|---|---|---:|
| Spec HBM | \(4096 / 19.231\) | **213** |
| **Babel Read (practical)** | \(3793 / 19.231\) | **197** |
| Compute (order of magnitude) | \(4.6\,\mathrm{PFLOPS}/54\,\mathrm{GFLOP}\) | ~85 000 |

| Ratio | Value |
|---|---:|
| Unprofiled C1 / spec | 79.53 / 213 = **37%** |
| Unprofiled C1 / Babel Read | 79.53 / 197 = **40%** |
| Implied C1 weight stream | \(79.53 \times 19.231\) GB/s ≈ **1.53 TB/s** |
| DFLASH-3 C1 / Babel Read | 113.75 / 197 = **58%** (not a full target stream/token) |

## Bottlenecks (what this does and does not prove)

1. **The card can stream HBM.** Babel Read at 93% of `amd-smi MAX_BANDWIDTH` closes the “maybe the SKU cannot reach advertised HBM” hypothesis. Copy/Triad ~78–80% is the two- and three-array streaming bound.

2. **C1 is not PPT, thermal, or peak-MXFP4 limited.** Control decode sits ~350–410 W of a 600 W cap with ~2.19 GHz GFX. Compute roof is two orders of magnitude above 80 tok/s.

3. **C1 is not “HBM-spec starved.”** 79.53 tok/s is **40%** of the Babel Read roof under a one-full-weight-stream model. Isolated GEMM knobs that improved M=1 wall time **did not** improve serving, so that unused 60% **cannot** be assigned to pack/unpack or gate GEMM without an EngineCore timeline or calibrated serving HBM counters.

4. **Kernel ranking is still missing.** Launch-under-`rocprofv3` (`profile_enginecore_launch.sh`) on this image traces **only the API parent’s HIP init** (`hipDriverGetVersion` / `hipGetProcAddress`). `VLLM::EngineCore` still forks (`VLLM_ENABLE_V1_MULTIPROCESSING=0` does not keep HIP in PID 1). Output CSVs are ~5 KB pftrace, **no KERNEL_DISPATCH**. Attach remains blocked (`rocp-bg-attach` absent). Do not rank MXFP4 GEMM vs GDN vs LM-head from these traces.

5. **LM-head/sampling is already a weak C1 explanation.** `generation_config` sampling vs greedy was 81.98 vs 82.78 tok/s at 1K/256.

## Next steps (frozen HF control)

1. Keep `VLLM_ROCM_USE_AITER` unset and stock `M_LEQ_8`. Promotion gate = unprofiled graph-enabled C1/C8, not eager GEMM (`GEMM.md`).
2. Do not assign the unused ~60% of Babel Read to a single operator without EngineCore / serving HBM counters.
3. DFLASH-3 greedy/TTFT/ITL validation is complete (opt-in long-output only). Parser now reads RVS `MiBytes/sec`; never 4 GiB BABEL on this 32 GiB host (`scripts/parse_rvs_babel.py`).

Reproduce:

```bash
# Unprofiled C1 (server up)
python3 scripts/bench_openai_chat.py \
  --base-url http://127.0.0.1:8000/v1 --model awq \
  --input-len 1024 --output-len 1024 \
  --num-prompts 4 --concurrency 1 --timeout 900 \
  --out _results/profiling/c1_best/bench.json

# Practical HBM (stops vLLM unless --keep-server; needs sudo -n rvs)
./scripts/memory_bandwidth.sh --gpu 0 --weight-gib 17.91 \
  --out _results/profiling/babel/c1_roof
```
