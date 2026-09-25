# MXFP4 vs DFlash greedy quality (text-only)

Campaign summary: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md).

## Claim this run can support

Speculative decoding is **expected** to preserve the Quark MXFP4 target distribution when verification and rejection sampling are correct. The **primary quality delta versus BF16/FP8 remains MXFP4 quantization**, which this run did **not** measure (no BF16/FP8 target on this box).

Native MTP was **not** quality-tested: `Qwen3_5MTP.load_weights()` still fails a shape assertion.

## Method

- Corpus: 116 text-only prompts in `_results/quality/corpus.jsonl` (short chat, code, JSON/tool, math, RAG, multilingual, reasoning, long-context 2K/8K). Vision excluded (BF16 encoder; separate lane).
- Decode: `temperature=0`, `top_p=1`, `top_k=-1`, `seed=1234`, `ignore_eos=false`, thinking off, `return_token_ids=true`.
- Sequential C1 captures. Compare complete token-ID sequences.
- Harness: `scripts/eval_greedy_tokens.py`, `scripts/compare_greedy_runs.py`.

This is **not** the `ignore_eos` throughput corpus.

## Results

| Comparison | Exact token-ID match | Text match | n |
|---|---:|---:|---:|
| MXFP4 control r1 vs r2 (restart) | **116/116 (100%)** | 100% | 116 |
| MXFP4 control r2 vs r3 (same process) | **116/116 (100%)** | 100% | 116 |
| MXFP4 control vs DFLASH-3 | **101/116 (87.1%)** | 87.1% | 116 |
| MXFP4 control vs DFLASH-7 | **101/116 (87.1%)** | 87.1% | 116 |
| MXFP4 control vs DFLASH-4 | **102/116 (87.9%)** | 87.9% | 116 |
| DFLASH-7 vs DFLASH-4 | 105/116 (90.5%) | 90.5% | 116 |
| MXFP4 C1 vs C8 concurrent subset | **21/24 (87.5%)** | — | 24 |

JSON/tool-call prompts: **15/15 exact** for DFLASH-3, DFLASH-4, and DFLASH-7 versus control.

DFLASH-3 and DFLASH-7 have the **same 15 mismatch prompt IDs** versus control, although they differ from each other on 8/116 prompts. The mismatches are real alternate greedy strings (for example regex `^-?\d+$` vs `^[-+]?\d+$`, or a more verbose 8K needle wrapper). They are **not** tokenizer/text-only mismatches: token IDs and decoded text both differ. All 15 stopped normally.

Thirteen of those IDs also mismatch under DFLASH-4. The three C8-batch diffs (`short_chat_06/11/18`) are a subset of the DFlash diffs.

## Interpretation

Do **not** report 100% greedy equality for Quark MXFP4 + vLLM 0.30 DFlash on this stack.

Do **not** treat the 12.9% mismatch rate as proof that DFlash is emitting unverified tokens. Same-process MXFP4 greedy is perfectly repeatable. Changing **batch shape** on the non-speculative control (C8 concurrent vs C1) already produces the same class of divergence vLLM documents (floating-point / batch-size numerics). DFlash verification evaluates a multi-token block, so it is in that numerical regime.

Practical gates on this corpus:

| Gate | Result |
|---|---|
| 100% greedy token equality (DFlash vs sequential C1) | **Fail** (D3/D7 87.1%; D4 87.9%) |
| JSON/tool structural exact match | **Pass** (15/15) |
| Control self-consistency | **Pass** (100% across restart and repeat) |
| MTP greedy equality | **Not run** (loader blocked) |
| MXFP4 production-sampling task score | **30/32 (93.75%)** |
| MXFP4 vs BF16/FP8 task quality | **Blocked pending exclusive-GPU FP8 run** |

The DFlash mismatch rate (12.9%) is close to non-spec C1-vs-C8 (12.5%). That does **not** prove DFlash has no independent effect. Alternate strings are semantically similar in some cases; that is not functional equivalence.

Recommended report sentence:

> On a 116-prompt text-only deterministic corpus, the fixed-shape Quark MXFP4 control was token-ID stable across restart and repeated same-process runs. DFLASH-3 and DFLASH-7 each matched 101/116 sequential-control outputs and shared the same 15 mismatch IDs, while differing from one another on 8 prompts. These results do not establish practical bit-exact losslessness for DFlash under the tested vLLM 0.30 ROCm/MXFP4 implementation. Structured JSON/tool outputs matched on the 15-prompt subset.

## Production-sampling task pass

The live, non-speculative MXFP4 control completed the 32-item C1 sequential
suite in `scripts/eval_sampling_tasks.py` with the model's checked-in generation
defaults: `temperature=1.0`, `top_p=0.95`, and `top_k=20`, seed 1234. Thinking
was disabled so short-answer tasks remain directly scoreable. This is one
sample per item, not a confidence interval.

| Category | Score | Validity |
|---|---:|---:|
| Compact GSM8K-like math | 6/8 (75%) | 8/8 |
| JSON exact value / parse validity | 8/8 (100%) | 8/8 |
| Short Python static shape / syntax validity | 8/8 (100%) | 8/8 |
| Multilingual corpus slice | 4/4 (100%) | 4/4 |
| Reasoning corpus slice | 4/4 (100%) | 4/4 |
| **Overall** | **30/32 (93.75%)** | **32/32 (100%)** |

Streaming TTFT was 43.0 ms mean and 42.2 ms median. Full row-level outputs and
latencies are in `_results/quality/sampling_mxfp4.json`. DFLASH-3 was not used.

FP8 is fully cached locally (66/66 indexed weight files, 30,866,866,928 bytes)
but cannot be served concurrently with this frozen control. BF16 full weights
are unavailable locally. The exclusive-run recipe and comparison status are in
`_results/quality/MXFP4_VS_FP8.md`.

## Next quality work

1. Repeat the 15 mismatch IDs under `--enforce-eager` C1 (isolates graph/batch numerics).
2. Repeat the production-sampling suite over multiple seeds and add executable
   code tests; the current code score is static syntax/shape only.
3. Run the fully cached FP8 target versus Quark MXFP4 during an exclusive-GPU
   window.
4. Vision QA separately.
5. MTP greedy equality only after a validated loader.
