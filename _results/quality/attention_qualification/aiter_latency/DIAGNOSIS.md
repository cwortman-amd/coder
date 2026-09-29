# AITER unified latency diagnosis (27 Sep 2026)

Stock `ROCM_ATTN` is the running service again. `VLLM_ROCM_USE_AITER` was
not set. No scheduler or chunk-size change was applied. The 8192-token
chunk limit is not the cause of the 8K TTFT gate failure.

## Which gate actually fails

Decode is flat. On the existing AITER C1 traces, token gaps stay near
10.0 ms from the start of a 512-token generation through the end, and near
10.5–10.8 ms at 8K and 15K context. Early and late buckets match. There is
no periodic stall and no context-growing decode tail.

| Check | Result |
|---|---|
| C1 1K ITL p95 | 10.11 ms. Passes the 11 ms ceiling. Three later repeats were 10.04–10.15 ms. |
| C4 / C8 1K ITL p95 | 11.20 / 11.80 ms. Fails the absolute 11 ms ceiling. p50 is already 11.08 / 11.71, so this is the steady step, not a tail. Both beat stock (13.34 / 13.87). |
| C1 1K TTFT | 62 ms versus stock 61 ms. Not the failure. |
| 8K TTFT, all concurrencies | Fails the frozen allowance. This is prefill/cache, not decode. |
| 15K TTFT | Passes, and beats stock. |
| Mixed versus fixed C4 | Fixed 1K ITL p95 11.30 ms. Mixed 265/1033/4105/8201 ITL p95 12.37 ms. One shape change, not a separate gate failure. |
| Large first-token gaps at C4/C8 | One admission interval while other prefills are still running. It does not set p95 ITL. |

## 8K TTFT is the uncached page remainder

The qualification client sends the same `"alpha "` prompt twice (warmup,
then measure) with prefix caching on. Mamba cache mode is `align`: a
repeated prompt reuses only complete pages and recomputes
`prompt_tokens % page_size`.

| Backend | Page | 8201-token tail | Hot TTFT | 15309-token tail | Hot TTFT |
|---|---:|---:|---:|---:|---:|
| Stock `ROCM_ATTN` | 784 | 361 | 74 ms | 413 | 108 ms |
| AITER unified | 832 | 713 | 142 ms | 333 | 99 ms |

Recompute speed is about the same. AITER looks worse at 8,201 tokens
because that length leaves a longer partial page on an 832-token grid, and
better at 15,309 tokens because the remainder is then shorter. The
qualification figures (stock 89.5 / 128 ms, AITER 155 / 114 ms) are this
hot-repeat regime.

Cold prefill, isolated with `cache_salt`, is monotonic on both backends.
AITER is slower, but not specifically at the chunk boundary:

| Prompt tokens | Stock cold | AITER cold |
|---:|---:|---:|
| 1033 | 157 ms | 68 ms |
| 8192 | 729 ms | 854 ms |
| 8201 | 729 ms | 856 ms |
| 15309 | 1570 ms | 1800 ms |

8192, 8193, and 8201 are the same cold cost on AITER (about 855 ms). A
9-token chunk remainder does not explain the gate.

## Exact page lengths miss the cache on both backends

When the prompt length is an exact multiple of that backend's page, the
immediate repeat is as slow as a cold prefill. One token past the page
drops hot TTFT to a few tens of milliseconds.

| Backend | Exact length | Cold | Hot | One token past, hot |
|---|---:|---:|---:|---:|
| AITER, page 832 | 1664, 4160, 6656, 7488 | full prefill | no speedup | 27 ms at 7489 |
| Stock, page 784 | 7056 | 610 ms | 668 ms | 39 ms at 7057 |

vLLM's `align` mode only stores a Mamba checkpoint when a scheduler step
ends on a page boundary. An exact multiple currently stores nothing usable.
This is shared serving behavior, not an AITER kernel miss. The 8,201-token
gate prompt is not an exact multiple, so fixing this would not by itself
clear that gate.

## Decision

Do not raise `max_num_batched_tokens` for this failure. Do not enable
`VLLM_ROCM_USE_AITER=1`. AITER stays a research profile.

The remaining promotion blockers are:

1. C4/C8 1K token arrival sits 0.2–0.8 ms above the frozen 11 ms ceiling.
   That is steady decode. It is already better than stock.
2. Hot 8K TTFT depends on `prompt_tokens % page_size`. A like-for-like
   remainder comparison is required before treating 8201-versus-784 as an
   attention regression. Cold 8K/15K prefill is still about 15–17% slower
   on AITER and is a separate measurement from the published gate numbers.

No focused serving change has been spent yet. The next change, if one is
made, has to target either the 11 ms steady step or matched-remainder
prefill, not the 8192 chunk cap.
