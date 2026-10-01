---
type: Profiling Report
title: 'TRITONATTN gate 1: first-divergence logits (27 Sep 2026)'
description: C1 chat, temperature 0, topp 1, topk -1, seed 1234, enablethinking false,
  FULLANDPIECEWISE graphs. Same Quark MXFP4 checkpoint and vLLM image.
tags:
- profiling-report
- attention-ab
- divergence
- gate1
status: stable
---

# TRITON_ATTN gate 1: first-divergence logits (27 Sep 2026)

C1 chat, temperature 0, top_p 1, top_k -1, seed 1234, `enable_thinking` false,
`FULL_AND_PIECEWISE` graphs. Same Quark MXFP4 checkpoint and vLLM image.
Control log: `Overriding with ROCM_ATTN` was the process that captured the
control side. Triton log: `Using TRITON_ATTN backend (selected via
--attention-backend)`, attention block size 784. Batch shape is one request.

Thresholds fixed in `scripts/qualify_attn_divergence.py` before the Triton
capture:

- small margin: top-1 minus top-2 below 1.0 nat on both backends, and each
  argmax is inside the other backend's top-5
- large margin: top-1 minus top-2 at least 2.0 nat on either backend
- page proximity: context length within 2 tokens of a multiple of 784

Logits are chat-completion `top_logprobs` at the first stored mismatch index.
The generated prefix up to that index matched the stored shared prefix on
every replay, so both backends were scored on the same conditional tokens.
The completions API, given those same token ids, can emit a different greedy
token than chat. It was not used for the 18-prompt screen.

## Replay of the 18 stored mismatches

Seventeen still choose different tokens. `long_8k_02` now agrees: both
backends emit `needle` (id 57508). The stored control token `The` (id 760)
does not replay; on the current control it is 7.88 nats below `needle`.
Context there is 13,489 tokens, 161 tokens past a 784-token page boundary.

No live mismatch is within 2 tokens of a 784-token boundary. Context at the
split is 25–71 tokens except that long prompt. None of the chosen tokens is
EOS. `reasoning_00` has `<|im_end|>` as the control's third token, not the
argmax. Three splits are formatting characters: `def` versus a code fence,
`{` versus `{"`, and `":` versus `":"`.

| Prompt | Stage | Context | Control token (margin) | Triton token (margin) | Gap to the other token |
|---|---|---:|---|---|---|
| short_chat_04 | early | 25 | Yellow 1.25 | Blue 0.00 tie | 1.25 / 0.00 |
| short_chat_06 | early | 29 | cause 0.25 | request 1.63 | 0.25 / 1.63 |
| short_chat_11 | early | 30 | simply 0.50 | reverse 0.00 tie | 0.75 / 0.25 |
| short_chat_14 | early | 30 | symbol 0.13 | comes 0.25 | 0.13 / 0.25 |
| short_chat_18 | early | 36 | often 0.13 | important 0.00 tie | ≥1.50 / 0.00 |
| code_04 | first token | 31 | def 0.63 | ``` 0.63 | 0.63 / 0.63 |
| code_08 | early | 26 | `-` 1.00 | `[-` 1.88 | 1.00 / ≥5.63 |
| json_tool_03 | early | 33 | `":` 0.50 | `":"` 0.50 | 0.50 / 0.50 |
| json_tool_10 | first token | 33 | `{` 1.63 | `{"` 0.13 | 3.13 / 0.50 |
| math_04 | late decode | 60 | Is 0.88 | Sub 0.75 | 0.88 / 0.75 |
| math_10 | late decode | 53 | The 1.00 | `**` 0.13 | 1.88 / 0.13 |
| math_16 | late decode | 71 | `).` 1.13 | `)` 0.00 tie | 2.13 / 0.00 |
| math_18 | early | 34 | into 2.50 | to 0.13 | 2.50 / 0.13 |
| rag_01 | first token | 37 | AW 1.38 | MX 0.13 | 1.38 / 0.13 |
| reasoning_00 | first token | 44 | 5 0.63 | Let 3.75 | 0.63 / ≥4.13 |
| reasoning_04 | early | 52 | into 0.38 | it 0.50 | 0.38 / 0.50 |
| reasoning_05 | early | 32 | the 0.13 | exactly 0.75 | 0.13 / 0.75 |
| long_8k_02 | first token | 13489 | needle 7.38 | needle 3.75 | agree |

Margins are top-1 minus top-2, in nats. The gap column is how far each
backend places the other backend's argmax. `≥` means that token is outside
the returned top-5, so the gap is at least the distance to the fifth token.

Seven prompts are small-margin flips of the same pair (both margins under
1.0 nat). Six more are the same pair with a gap under 2.0 nat.
`short_chat_18` is a 0-nat tie on Triton between `important` and the control
token `often`.

Five prompts exceed a 2.0 nat gap between the two emitted tokens:

- `reasoning_00`: Triton logprob of `Let` is -0.114. The control token `5`
  is outside Triton's top-5 (fifth token is -4.24). Control's own margin is
  only 0.63, with `Let` second.
- `code_08`: Triton logprob of `[-` is -0.194. The control token `-` is
  outside Triton's top-5 (fifth token is -5.82). Control prefers `-` by 1.00 nat.
- `math_18`: control logprob of ` into` is -0.079 and of ` to` is -2.579.
  Triton flips that pair with a 0.13 nat margin.
- `json_tool_10`: control logprob of `{` is -0.217 and of Triton's `{"` is
  -3.342. Triton prefers `{"` by 0.50 nat over `{`.
- `math_16`: control logprob of `).` is -0.368 and of `)` is -2.493. Triton
  ties `)` and `).` at -0.844.

## 784-token page probes

Chat prompts fitted to lengths 783, 784, 785, 1567, 1568, 1569, 2351, 2352,
and 2353. Eight greedy tokens each.

The two backends emit the same eight tokens at 783, 784, 785, 1567, 2351,
and 2353. The three disagreements are same-pair flips:

| Context | First differing generation index | Control margin | Triton margin | Pair |
|---:|---:|---:|---:|---|
| 1568 | 1 | 0.00 | 0.63 | seems / looks |
| 1569 | 5 | 0.38 | 0.13 | sent / past |
| 2352 | 1 | 0.00 | 1.75 | seems / looks |

At 1568 and 2352 the control itself is an exact tie. That is ordinary
tie-breaking at those lengths, including exactly on a page multiple. It is
not a large-margin argmax change.

A second probe truncated one 2,353-token chat sequence and scored the next
token through the completions API. Both backends pick the same token at all
nine lengths. Lengths 783 through 1569 both emit `<|im_end|>` because the
truncation cuts the chat-token sequence. That probe checks agreement on
identical ids. It does not represent a normal user continuation.

## Decision

Keep `ROCM_ATTN` as the default and leave the Triton launcher opt-in until the
task-quality gate completes.

The 98/116 string match is not a page-boundary cluster: the live splits sit
at context 25–71, and the page probes stay inside small margins or exact
ties. Five short-context prompts still reverse the argmax across a gap of
at least 2 nats, which fails the large-margin row of the promotion table.
A real-tensor isolation is now complete for the sharpest shared reproducer,
`reasoning_00`; see `../real_capture/REASONING_00.md`. At the first full
attention layer, Q/K/V are bitwise identical across all three servers.
Triton and AITER return the same last-token output, differing from stock by
0.089% RMS, and are slightly closer than stock to CPU FP64 attention. The
model amplifies that small difference into the large final-logit flip.

This removes the evidence for an attention indexing/masking defect on this
reproducer. It does not make the task behavior equivalent: stock follows the
“Integer only” instruction with `5`, while both unified paths begin with
`Let`. Promotion therefore moves to the paired task-quality gate rather than
being blocked as a kernel-correctness defect.

Raw captures: `rocm_attn_teacher.json`, `triton_attn_teacher.json`.
