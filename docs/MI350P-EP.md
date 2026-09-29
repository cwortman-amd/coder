# Expert parallelism on two MI350P cards

Same burst as the NVIDIA forum comparison of 2× RTX PRO 6000 Blackwell on PCIe against 2× B200 on NVLink, run here on 2× Instinct MI350P.

Source post: [Expert Parallelism using 6000 Pro PCIe Gen5 vs B200 NVLink](https://forums.developer.nvidia.com/t/expert-parallelism-using-6000-pro-pcie-gen5-vs-b200-nvlink/378258) (27 July 2026). The PCIe-width question those numbers get read against is [dual PCIe 5.0 x8 vs x16](https://www.reddit.com/r/LocalLLaMA/comments/1up4d62/impact_of_dual_pcie_5_x8_vs_dual_pcie_5_x16_for/).

Date of the link reading and the eager burst: 28 September 2026. The measured pair is in section 4 and in [`_results/ep_mi350p/COMPARE.md`](../_results/ep_mi350p/COMPARE.md). The runner is `scripts/bench_ep_mi350p.sh`. Expert parallel was **0.93×** tensor parallel on output tok/s. Mean TTFT was slower (TP/EP **0.67×**). Mean TPOT was **1.07×**. Those cells are `--enforce-eager`. Peak tok/s in the table is a short window.

---

## 1. What is held constant

| Item | Forum | This host |
|---|---|---|
| Model | `Qwen/Qwen3.5-35B-A3B` | same checkpoint, 14 shards, **66.99 GiB** |
| Parallelism | TP=2, then TP=2 with `--enable-expert-parallel` | same two arms |
| Server | `--gpu-memory-utilization 0.9 --max-model-len 32768` | same |
| Client | 16 prompts, random 1000 in / 1000 out, request rate 10000, `ignore_eos` | same `vllm bench serve` flags |
| Engine | vLLM on CUDA, version not stated in the post | vLLM **0.30.0** in `vllm/vllm-openai-rocm:latest` |

The checkpoint is `Qwen3_5MoeForConditionalGeneration`: hidden size 2048, 40 layers in a 3× linear-attention + 1× full-attention repeat, **256 experts**, **8 routed experts per token**, expert intermediate size 512, plus a shared expert of the same width. Context in the config is 262144. The bench caps the server at 32768, matching the post.

66.99 GiB of BF16 weights fit in one 144 GiB MI350P. TP=2 is the comparison, chosen so each arm moves activations across the link the way the forum did. An RTX PRO 6000 is a 96 GB board; the post still forced TP=2 on both of its platforms.

---

## 2. Link on this host

Read from sysfs on 28 September 2026. Both accelerators are `1002:75a8`, gfx950.

| GPU | BDF | Negotiated | Max | NUMA |
|---|---|---|---|---|
| 0 | `0000:8b:00.0` | **32.0 GT/s, x16** | 32.0 GT/s, x16 | 0 |
| 1 | `0001:c7:00.0` | **32.0 GT/s, x16** | 32.0 GT/s, x16 | 1 |

32.0 GT/s is PCIe 5.0. Payload rate at 128b/130b encoding is **63.02 GB/s** unidirectional on x16 and **31.51 GB/s** on x8. Both cards are at the x16 width, on **different PCI domains and different sockets** (EPYC 9015, 2 sockets, 8 cores each). A copy from GPU 0 to GPU 1 crosses the inter-socket path. That path can sit under the 63 GB/s wire ceiling. The runner rewrites this table to `_results/ep_mi350p/topology.json` every time it starts.

This host is the **x16/x16** side of the dual-PCIe question. It has no x8 leg: both links trained at their maximum width, and there is no second slot pair to bifurcate. An x8-versus-x16 delta on one socket is a different machine. NVIDIA’s B200 brief lists **1.8 TB/s** of NVLink on that GPU; the forum’s second column is that fabric, and this page does not remeasure it.

The dense MXFP4 server was using GPU 0 when the link was read (`rocm-smi` VRAM used about 119 GiB on GPU 0, idle on GPU 1). Link width does not depend on that process.

---

## 3. What the two arms change

`--enable-expert-parallel` is present on this image (`EngineArgs.enable_expert_parallel`, default false). With TP=2 it is the switch the forum flipped.

| | TP=2 | TP=2 + expert parallel |
|---|---|---|
| Attention | Tensor-sharded. All-reduce of the hidden state. | Same. |
| Experts | Each expert’s weight is split across the two cards. The MoE GEMM finishes with an all-reduce. | Whole experts are placed on one card or the other (128 / 128). Tokens for a remote expert move with an all-to-all, then the result moves back. |
| Bytes that care about the link | Every MoE layer reduces a `[tokens, 2048]` BF16 activation, prefill and decode. | Only the tokens whose selected experts live on the other card. |

On a closed burst of 16 sequences the prefill is the wide transfer: 16 × 1000 tokens through 40 layers. Decode then adds one token at a time, so the collective is small next to the HBM read of the experts that actually fire. That split is what the forum measured on PCIe. Expert parallel cut RTX PRO 6000 mean TTFT from **18355 ms to 673 ms** and raised output throughput from **516 to 1186 tok/s** (2.30×). Mean TPOT stayed **12.6 ms to 12.8 ms**. On NVLink the same switch moved output throughput **1885 to 1965 tok/s** (1.04×) and mean TPOT **6.52 ms to 5.63 ms**, while mean TTFT went from **1965 ms to 2511 ms**.

Mean TTFT in this burst is the client’s time to first token with all 16 requests admitted together (`--request-rate 10000`, peak concurrent requests 16 in the post). It includes queueing behind the other prefills. It is the right column to compare with the forum. It is a different number from the single-request TTFT in [MI350P.md](MI350P.md).

The runner waits until `GET /health` returns, then starts `vllm bench serve`.

This host has 30 GiB of RAM and about 8 GiB free once the GPUs are idle. The first TP launch loaded the checkpoint (32.86 GiB of weights on each card) and finished the AOT compile of the text backbone, then Inductor sat in swap at roughly 250 MB/s with no new kernel written for 30 minutes. The saved pair therefore passes `--enforce-eager`. Eager skips that compile. Absolute tok/s on that pair sit below a graphed run and below the forum’s graphed numbers. The EP-versus-TP ratio on that pair is still one stack.

`--skip-mm-profiling` skips dummy multimodal inputs in the memory profile. It does not skip the vision tower. The 28 September log still records `Multi-modal warmup completed in 12.666s`. On vLLM 0.30, `--language-model-only` sets every modality limit to 0, and `Qwen3_5MoeForConditionalGeneration` builds `visual` inside `_mark_tower_model`, which installs a `StageMissingLayer` when those limits are 0. A default `--run` passes that flag and writes `text_tp.json` / `text_ep.json`. `tp.json` and `ep.json` stay the eager pair that still loaded the tower. `--graphs` drops `--enforce-eager` and writes `graphs_text_*.json`. It refuses to start when MemAvailable is under 24 GiB.

---

## 4. Published burst

Transcribed from the forum post. All four runs: 16 successful requests, 16000 input tokens, 16000 generated tokens.

| Metric | 6000 TP | 6000 EP | B200 TP | B200 EP |
|---|---:|---:|---:|---:|
| Output tok/s | 516.44 | 1186.12 | 1885.26 | 1965.18 |
| Total tok/s | 1032.87 | 2372.23 | 3770.52 | 3930.36 |
| Duration (s) | 30.98 | 13.49 | 8.49 | 8.14 |
| Mean TTFT (ms) | 18354.95 | 672.99 | 1965.40 | 2510.53 |
| Median TTFT (ms) | 18326.58 | 645.70 | 1935.22 | 2462.16 |
| P99 TTFT (ms) | 18486.20 | 801.48 | 2069.94 | 2661.18 |
| Mean TPOT (ms) | 12.63 | 12.82 | 6.52 | 5.63 |
| Median TPOT (ms) | 12.66 | 12.85 | 6.55 | 5.68 |
| P99 TPOT (ms) | 12.89 | 13.08 | 6.89 | 6.16 |
| Mean ITL (ms) | 12.63 | 12.82 | 6.53 | 5.63 |
| Peak output tok/s (window) | 1344 | 1328 | 2544 | 2960 |

EP / TP on output tok/s is **2.30×** on the 6000 pair and **1.04×** on the B200 pair. TP / EP on mean TTFT is **27.3×** on the 6000 pair (expert parallel is the shorter wait) and **0.78×** on the B200 pair (expert parallel is the longer wait).

MI350P cells are the eager run on 28 September 2026, both arms `--enforce-eager --skip-mm-profiling`, 16 completed and 0 failed. Scorecard: `_results/ep_mi350p/COMPARE.md`.

| Metric | MI350P TP | MI350P EP |
|---|---:|---:|
| Output tok/s | 17.54 | 16.28 |
| Total tok/s | 35.07 | 32.57 |
| Duration (s) | 912.46 | 982.54 |
| Mean TTFT (ms) | 14168.17 | 21216.38 |
| Median TTFT (ms) | 12616.68 | 20760.23 |
| P99 TTFT (ms) | 17545.49 | 23039.86 |
| Mean TPOT (ms) | 898.49 | 961.54 |
| Median TPOT (ms) | 899.93 | 961.62 |
| P99 TPOT (ms) | 900.97 | 967.46 |
| Mean ITL (ms) | 898.49 | 961.54 |
| Peak output tok/s (window) | 48 | 48 |

EP / TP on this host is **0.93×** output tok/s. Mean TTFT got longer (14.2 s to 21.2 s, TP/EP **0.67×**). Mean TPOT went from 898 ms to 962 ms (EP/TP **1.07×**). The peak of 48 tok/s is the bench client's short window. Sustained output is the 17.54 and 16.28 tok/s row, about 16 sequences at a ~900 ms inter-token time. That is the B200 direction, where expert parallel did not fix prefill, and it is far from the 6000 pair’s 2.30× output and 27× shorter TTFT. These milliseconds are an eager stack. They are not a CUDA-graph comparison with the forum.

RCCL initialized both arms as P2P/IPC across `8b000` and `1c7000` (`tp.rccl.txt`, `ep.rccl.txt`). At startup both workers held similar memory: TP0 consumed 35.95 GiB of weights and non-torch memory plus 92.42 GiB of KV cache, and TP1 consumed 35.83 GiB plus 92.54 GiB (`ep.server.log`). That burst did not save a per-GPU utilization series. Later runs write `{tag}.gpus.json` from `amd-smi metric` for the duration of `vllm bench serve`.

A useful reading of the forum columns:

- Output tok/s and mean TTFT say whether PCIe Gen5 x16 across two EPYC sockets behaves like the 6000 PCIe column (EP fixes prefill) or like the B200 column (the fabric already hid the all-reduce).
- Mean TPOT says whether decode is sitting on the collective or on HBM. The 6000 pair’s TPOT did not move when EP was enabled. MI350P HBM3E is 4 TB/s per card against the 6000’s 1,792 GB/s GDDR7, so a lower TPOT here would be a memory result, and a TPOT that matches the 6000 pair would be a launch or link result.
- RCCL’s init log (`tp.rccl.txt`, `ep.rccl.txt`) says whether the all-reduce and all-to-all used P2P or a host path. The two cards are on separate PCI domains, so that line is part of the result.

---

## 5. How to run

`--run` stops the dense server on port 8000. The checkpoint is now at `models/Qwen3.5-35B-A3B`. Host RAM is 30 GiB, so a fresh download waits until `MemAvailable` is at least 6 GiB after that server stops. The 28 September run left the MXFP4 control stopped.

```bash
cd /home/amd/workspace/coder

# Link only. Does not stop the dense server.
./scripts/bench_ep_mi350p.sh --topology

# Text-only eager pair. Stops rocm-inference-server.
# Writes text_tp.json and text_ep.json, then COMPARE.md.
./scripts/bench_ep_mi350p.sh --run

# Graphed text-only pair. Needs MemAvailable >= 24 GiB.
# Writes graphs_text_tp.json and graphs_text_ep.json.
./scripts/bench_ep_mi350p.sh --run --graphs

# One arm, after the checkpoint is local.
./scripts/bench_ep_mi350p.sh --run --arm ep

# Repeat the 28 September serve line, including the vision tower.
# Reuses tp.json and ep.json.
./scripts/bench_ep_mi350p.sh --run --with-vision

# Put the Quark MXFP4 control back.
./scripts/launch_vllm_mxfp4.sh
```

Serve flags that exist so the process starts on this ROCm image, and that the forum command does not carry: `HIP_VISIBLE_DEVICES=0,1`, `PYTORCH_ROCM_ARCH=gfx950`, `GPU_ARCHS=gfx950`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=0`. `ROCR_VISIBLE_DEVICES` stays unset. Setting it to a single index hid the second card during the [P/D bring-up](MI350P-PD.md). `HSA_OVERRIDE_GFX_VERSION` stays unset for the same reason.

The client adds `--backend openai --endpoint /v1/completions --host 127.0.0.1 --port 8000 --tokenizer` pointed at the local snapshot. Those are the vLLM 0.30 defaults the forum command relies on, pinned so a later default change does not move the burst. `--tokenizer` avoids a second download of the HF id during the bench.

---

## 6. Stack differences that survive a matched command line

- CUDA vLLM (forum) versus ROCm vLLM 0.30.0. Kernel choice for the MoE GEMM and for GDN/linear attention is ours.
- Single-root or NVLink (forum) versus two PCI domains on two NUMA nodes (this host).
- The post does not record a warmup policy. This runner starts the burst after `/health`.
- No accuracy check is part of the forum protocol. A tok/s cell is accepted when the JSON says 16 completed and 0 failed. It is not a quality result.
