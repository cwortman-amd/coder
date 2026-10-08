---
type: Technical Report
title: KV connector — MI350P 1P1D
description: This host has 2 × AMD Instinct MI350P PCIe (0x75a8, gfx950). GPU 0 is
  0000:8b:00.0. GPU 1 is 0001:c7:00.0. rocm-smi reports the link as PCIE, 3 hops,
  weight 72. The two cards are on different PCI domains.
tags:
- technical-report
- kv
- connector
status: stable
---

# KV connector — MI350P 1P1D

This host has **2 × AMD Instinct MI350P PCIe** (`0x75a8`, `gfx950`). GPU 0 is `0000:8b:00.0`. GPU 1 is `0001:c7:00.0`. `rocm-smi` reports the link as **PCIE**, **3 hops**, weight **72**. The two cards are on different PCI domains.

The serving target is `Qwen3.8-27B-Quark-AWQ-MXFP4-sharded`, served name `awq`, on `vllm/vllm-openai-rocm:latest` (vLLM **0.30.0**).

This document records the connector architecture, how to run it, and the measurements from this two-GPU host. The only measured pair is cross-NUMA. Radeon AI PRO R9700 and RTX PRO 6000 Blackwell remain unmeasured. Copy bandwidth and NIXL transfer time are separate from serving handoff time.

The native `HIP_IPC` path has completed real Qwen3.8 prefill-to-decode requests. On 29 Sep the measured placement was GPU 1 prefill and GPU 0 decode. Four serial 8192-in / 1024-out prompts through the router produced **72.49 tok/s** aggregate, with inter-token latency **10.58 ms** p50 and time-to-first-token **1.47 s** p50. Failed transfers were zero. Greedy completion token IDs still do not match a single-GPU control, and the UCX `rocm_ipc` serving waterfall was not rerun, so this is not a production-qualified P/D result.

Related phase-isolation numbers, without KV handoff, are in [MI350P-PD.md](MI350P-PD.md).

---

## 1. Architecture

```
Client
  │  POST /v1/chat/completions
  ▼
benchmark/pd_router.py          :8000
  │  prefill: max_tokens=1, return_token_ids,
  │           kv_transfer_params.do_remote_decode=true
  ▼
GPU 1  prefill                     :8100
  NixlConnector  kv_role=kv_producer
  HIP_VISIBLE_DEVICES=1,0
  side channel                    :5600
  │  response: prompt_token_ids + remote_block_ids,
  │            remote_engine_id, remote_host, remote_port
  ▼
router forwards that handshake unchanged
  ▼
GPU 0  decode                      :8300
  NixlConnector  kv_role=kv_consumer
  HIP_VISIBLE_DEVICES=0,1
  side channel                    :5602
  pull KV into decoder-owned blocks, then stream tokens
```

Copy microbenchmarks are faster GPU 0→GPU 1 (48.3 GB/s versus 21.2 GB/s). Serving is not. A connector consumer on GPU 1 stayed near 8 tok/s after its CPUs and memory were moved to NUMA 1. The same consumer on GPU 0, NUMA 0, reached about 74 tok/s on a warm 1K/256 request. Decode placement wins, so the measured pair prefills on GPU 1 and decodes on GPU 0.

`NixlConnector` in this image is the pull connector (`NixlPullConnector`). The producer keeps the blocks until the consumer reads them. The Unix/TCP side channel carries metadata. The payload is supposed to move through NIXL’s selected UCX transport.

### 1.1 What is installed

| Piece | This image | Role |
| --- | --- | --- |
| vLLM connector | `NixlConnector` registered in `KVConnectorFactory` | Prefill/decode lifecycle, block ids, completion, `kv_load_failure_policy` |
| Transfer library | `nixl-rocm` **1.4.0** (`import nixl_rocm`; `import nixl` fails) | Agent, registration, READ/WRITE |
| NIXL plugins | `libplugin_UCX.so`, `libplugin_POSIX.so` | UCX is the VRAM plugin; POSIX is host/file |
| Bundled UCX | `libuct_rocm.so` with `rocm_ipc` and `rocm_copy` | Same-host GPU transports inside UCX |
| AMD fork | [ROCm/RIXL](https://github.com/ROCm/RIXL) | Deprecated. The README points at [ai-dynamo/nixl](https://github.com/ai-dynamo/nixl). No `rixl` module is installed |
| MORI | `amd_mori` **1.0.0** from [ROCm/mori](https://github.com/ROCm/mori) | Separate library. `BackendType` is `XGMI`, `RDMA`, `TCP` |

`MoRIIOConnector` is registered in vLLM, and `mori` imports. MORI-IO’s same-host backend is the XGMI path (`hipIpcMemHandle_t` in `include/mori/application/transport/p2p/p2p.hpp`). The published MORI-IO matrix lists MI308X, MI300X, MI325X, and MI355X. It does not list MI350P or R9700, and it has no `pcie` backend. Do not report that XGMI path as a PCIe P2P KV connector.

NIXL 1.3.0’s ROCm note names gfx950 parts MI350X and MI355X. It does not name MI350P or R9700. `rocm_ipc` is present in the wheel. On this pair, automatic UCX selected `rocm_copy` with a host fragment (section 5.4). Forcing `rocm_ipc` did not create a worker.

### 1.2 Handshake

vLLM 0.30 reads `return_token_ids` and `kv_transfer_params` as **top-level** chat-completion fields. Putting them under `extra_body` drops both.

Prefill request:

```json
{
  "return_token_ids": true,
  "max_tokens": 1,
  "kv_transfer_params": {"do_remote_decode": true}
}
```

A live producer returns `kv_transfer_params` that already set `do_remote_prefill` and include `remote_block_ids`, `remote_engine_id`, `remote_request_id`, `remote_host`, `remote_port`, and `remote_num_tokens`. The router forwards that object unchanged. It does not rebuild a smaller dict.

Qwen3.8 is a hybrid attention/Mamba model. Nixl’s scheduler drops the last prompt token on the producer and has the consumer recompute that one token. Attention KV plus Mamba state both have to arrive. A normal completion string does not prove the import.

`kv_load_failure_policy` defaults to `fail`. Keep it on `fail` so a missed load cannot fall through into a silent prompt recompute.

### 1.3 Two copies, one payload

The decoder still owns local KV slots. The preferred path copies producer VRAM into those slots. That is a peer copy, not a zero-copy alias of the producer’s tensors.

| Path | Movement | When it counts |
| --- | --- | --- |
| UCX `rocm_ipc` | HIP IPC between the two GPUs | Only after a run shows this transport and low host-DRAM payload traffic |
| UCX `rocm_copy` | ROCm copy path in the same UCX library | A successful transfer that may be host-staged |
| POSIX / TCP | Host memory | Functional fallback |

`/dev/shm` is host staging. It is not a PCIe peer implementation.

### 1.4 Decode server shape

The 29 Sep “decode-biased” process used `--max-num-batched-tokens 2048` and `--max-num-seqs 16`. That cut the compile range to **2048** and CUDA-graph capture to **32**. The validated single-card MXFP4 control uses the defaults from `scripts/launch_vllm_mxfp4.sh`: compile range **8192**, capture through **512**, about **79 tok/s** at 1K/1K C1.

A connector restart should use that control command, plus the Nixl flags, on both GPUs. Do not add the 2048/16 overrides.

NixlConnector forces an 832-token attention page so the attention page is at least as large as the Mamba page. Each sequence then needs one Mamba block. An 8 GiB KV pool had 157 blocks and refused the default 1024-sequence graph capture. The measured pair uses a 32 GiB pool and `--max-num-seqs 512`, which is inside that block count and still captures graphs through batch 512. The 96 GiB single-GPU control pool is larger than this 30 GiB host can keep resident twice.

---

## 2. Usage

Stop any server already bound to 8000, 8100, or 8200. This host has about **30 GiB** of RAM. Two resident 27B engines pushed it to about **10 GiB of swap** on 29 Sep. Start the decoder alone, wait until it is healthy, then start the prefiller.

Shared arguments, from the frozen MXFP4 recipe:

```text
/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded
  --served-model-name awq
  --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len 16384
  --kv-cache-memory-bytes 103223724237
  --host 127.0.0.1
```

Environment on both containers: `PYTORCH_ROCM_ARCH=gfx950`, `GPU_ARCHS=gfx950`, `SAFETENSORS_FAST_GPU=1`. Set `HIP_VISIBLE_DEVICES` only. Leave `ROCR_VISIBLE_DEVICES` and `HSA_OVERRIDE_GFX_VERSION` unset.

Apply the NIXL plugin onto a v1.4.0 tree, then build it inside the vLLM image. The helper applies every patch under `patches/nixl/`:

```bash
scripts/apply_nixl_hip_ipc_patch.sh --check /path/to/nixl
scripts/apply_nixl_hip_ipc_patch.sh /path/to/nixl
```

Both engines need `VLLM_SSM_CONV_STATE_LAYOUT=DS`, `HSA_ENABLE_IPC_MODE_LEGACY=1`, `NIXL_PLUGIN_DIR` pointed at `libplugin_HIP_IPC.so`, and the peer GPU visible. Hiding the peer makes `hipIpcOpenMemHandle` return `invalid device context` on a GPU 1→GPU 0 import. List the local GPU first.

Prefill, GPU 1, port 8100, CPUs and memory on NUMA 1:

```text
HIP_VISIBLE_DEVICES=1,0
VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
VLLM_NIXL_SIDE_CHANNEL_PORT=5600
--max-num-seqs 512
--kv-cache-memory-bytes 34359738368
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","engine_id":"pd-prefill","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["HIP_IPC"]}}'
```

Decode, GPU 0, port 8300, CPUs and memory on NUMA 0. Give this container an 8 GiB cgroup memory reservation so the prefiller cannot swap the decoder out.

```text
HIP_VISIBLE_DEVICES=0,1
VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
VLLM_NIXL_SIDE_CHANNEL_PORT=5602
--max-num-seqs 512
--kv-cache-memory-bytes 34359738368
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","engine_id":"pd-decode","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["HIP_IPC"]}}'
```

Leave the image `/opt/rocm` in place at runtime. The plugin `.so` is built in the image; a host ROCm 7.14 mount is a different HIP than the image's PyTorch.

The side-channel ports must differ. Both engines default to 5600.

Router, after both `/health` checks succeed:

```bash
python3 benchmark/pd_router.py \
  --host 127.0.0.1 --port 8000 \
  --prefill-url http://127.0.0.1:8100/v1 \
  --decode-url http://127.0.0.1:8300/v1
```

Client traffic goes to `http://127.0.0.1:8000/v1/chat/completions`. Phase timing is on `GET /metrics/pd`. NIXL copy time is the decode engine's `vllm:nixl_xfer_time_seconds` delta, not the router's `kv_handoff_ms`.

`benchmark/pd_router.py` drops `stream_options` on the non-streaming prefill subrequest. vLLM 0.30 rejects that field when `stream` is false, which otherwise fails an 8K streaming benchmark with HTTP 400 before any KV move. Restart the router container after editing the file.

---

## 3. Test verification

Run these in order. A later gate does not replace an earlier one.

### 3.1 Device path

On this chassis, both directions:

1. `hipDeviceCanAccessPeer` for `(0,1)` and `(1,0)`.
2. Same-process `hipMemcpyPeerAsync` into destination VRAM.
3. Separate processes: `hipIpcGetMemHandle` on a producer `hipMalloc`, `hipIpcOpenMemHandle` in the consumer, then `hipMemcpyPeerAsync` into consumer-owned memory. Synchronize, compare bytes, `hipIpcCloseMemHandle`.
4. Repeat for about 1, 16, 64, and 256 MiB, then several copies at once.
5. Record whether host DRAM moves a payload-sized buffer during the copy.

If peer access is unavailable, the result is a pinned-host fallback, or a topology change. It is not a direct PCIe connector result.

The 4.3 ms figure for 268.4 MB at 63 GB/s is a PCIe 5.0 x16 serialization floor. It is not a measurement on these MI350Ps.

### 3.2 NIXL transport

Using `nixl_rocm.nixl_agent` with backend `UCX`, one process per GPU, VRAM source and VRAM destination:

| Mode | `UCX_TLS` | Question |
| --- | --- | --- |
| Automatic | unset | Which transport UCX picks |
| IPC | `rocm_ipc,sm,self` | Direct HIP IPC path |
| ROCm copy | `rocm_copy,sm,self` | Alternate ROCm copy |
| Host | POSIX or TCP | Fallback bandwidth |

For each mode record backend name, bytes, p50/p95/p99, GB/s, and whether host DRAM carries the payload. `query_xfer_backend` reports the NIXL plugin (`UCX`), so UCX logs are still required to see `rocm_ipc` versus `rocm_copy`.

### 3.3 Connector lifecycle

Restart both engines with the configs in section 2 and `kv_load_failure_policy` left at `fail`.

Pass criteria for one short greedy request through the router:

- Prefill HTTP 200 with non-empty `prompt_token_ids`.
- `kv_transfer_params.do_remote_prefill` is true and `remote_block_ids` is non-empty.
- Decode HTTP 200.
- Decoder logs show a completed NIXL receive.
- Prompt-token compute on the decoder is the hybrid tail (the last token), not the whole prompt.
- Greedy continuation token ids match a single-GPU control of the same checkpoint.

### 3.4 Serving comparison

Replay the same 8K:1K traces on 1P1D and on cache-affine DP=2. Compare completed output tok/s, TTFT, ITL, and SLO-qualified goodput. Connector GB/s alone does not decide the architecture.

---

## 4. Benchmark procedure

### 4.1 Setup used on 29 Sep 2026

The copy runs were idle-GPU microbenchmarks. `rocm-mxfp4-pd-router`, `rocm-mxfp4-pd-prefill`, and `rocm-mxfp4-pd-decode` were stopped first. After that, each GPU had about 296 MB of VRAM in use. Host RAM was about 30 GiB total. No second model server, no ACS change, and no `HIP_VISIBLE_DEVICES` filter: both processes saw both GPUs, and the benchmark chose device 0 or 1 explicitly.

Inventory, on the host:

```bash
amd-smi version
amd-smi list --json
amd-smi topology --json
amd-smi topology --access --weight --hops --link-type --dma --bi-dir
lstopo --whole-io --of console
lstopo --whole-io --of xml
```

Copies ran in `vllm/vllm-openai-rocm:latest` with the host KFD and DRM devices, video group 44, render group 993, seccomp and AppArmor unconfined, `--network host`, `--ipc host`, and `--shm-size 8g`. The workspace was mounted at `/workspace`. `HSA_OVERRIDE_GFX_VERSION` and `ROCR_VISIBLE_DEVICES` were unset. `PYTORCH_ROCM_ARCH=gfx950` was set for the NIXL run.

HIP compile, inside that image:

```bash
hipcc -O3 --offload-arch=gfx950 -std=c++17 \
  scripts/kv_xfer/hip_peer_bench.cpp \
  -o _results/kv_xfer_mi350p/hip_peer_bench
```

| Mode | Command | What it times |
| --- | --- | --- |
| Peer, both directions | `hip_peer_bench same` | `hipMemcpyPeerAsync` on one stream. HIP event on that stream |
| Both directions at once | `hip_peer_bench bidi` | Two streams. Host clock until both complete. Bytes are the sum |
| 1, 2, 4, 8 overlapping copies | `hip_peer_bench concurrent` | GPU0→GPU1 only. Host clock. Bytes are the sum. Durations are not added |
| Pinned bounce | `hip_peer_bench host` | Device→pinned host→other device, one buffer the size of the payload |
| Two processes | `ipc-producer` then `ipc-consumer` on `127.0.0.1:18766` | The socket carries `hipIpcMemHandle_t` only. The payload is `hipMemcpyPeerAsync` into consumer VRAM |

Peer and IPC sizes were 4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB, 256 MiB, and 268.4 MB, with 5 warmup copies and 200 / 100 / 50 / 30 / 20 / 12 / 12 timed copies. The source was a device-side integer pattern. The destination was zeroed, then checked with a device mismatch count after the timed loop. Bidirectional and concurrent runs used 1 / 16 / 64 / 256 MiB.

NIXL used `scripts/kv_xfer/nixl_vram_bench.py`. Two processes, `nixl_agent` backend `UCX`, `capture_telemetry=True`. Each job registers the VRAM buffer and only then exchanges `get_agent_metadata()`, because a metadata snapshot taken before `register_memory` does not publish the region. The consumer issues `READ` into its own buffer. The matrix log level was `UCX_LOG_LEVEL=warn`. A separate 16 MiB probe set `UCX_LOG_LEVEL=info` and `UCX_PROTO_INFO=y` so the selected UCX protocol was visible. `query_xfer_backend` only returns the plugin name `UCX`.

NIXL jobs on the forward path: 1 / 16 / 64 / 256 MiB with one descriptor; 64 MiB split into 16, 64, and 256 descriptors; 256 MiB split into 64 descriptors; 64 MiB with 2, 4, and 8 reads in flight. The reverse path repeated the one-descriptor size sweep. Forced `UCX_TLS=rocm_ipc,sm,self`, `rocm_ipc,tcp,sm,self`, `rocm_copy,tcp,sm,self`, and POSIX were separate probes. POSIX was asked to register VRAM and failed at registration.

Outputs: `_results/kv_xfer_mi350p/hip_*.jsonl`, `nixl_vram.json`, `probe_auto.log`, `amd-smi-topology.json`, `lstopo-pcie.txt`, `host-topology.xml`.

Standalone transfer sizes for a later serving run remain **1, 16, 64, and 256 MiB**, plus the real Qwen3.8 block geometry once a connector run exports it (attention KV, Mamba state, block metadata).

Per size:

- 3 warmup iterations, then enough timed iterations for p50/p95/p99 (30 is enough while a copy is milliseconds; fewer if a fallback is hundreds of milliseconds).
- One contiguous buffer and a batched multi-block descriptor list.
- Correctness on every size: deterministic fill, destination cleared first, full-buffer checksum after the timed loop.
- Separate the clock into pack, control/IPC setup, copy, unpack, and decode admission. The copy row is not the handoff row.

Handoff time is:

```text
T_handoff = T_pack + T_IPC/control + T_peer_copy + T_unpack + T_decode_admission
```

Telemetry: HIP events or NIXL `postDuration` / `xferDuration`, GB/s, GPU SDMA/GFX and HBM on both cards, host DRAM, and the selected UCX transport.

The single-card control to beat before quoting 1P1D decode speed is the frozen MXFP4 recipe: about **79.35 tok/s** at 1K/1K C1 and about **553.58 tok/s** at C8 (`docs/MI350P.md`). A connector decode that sits near 6 tok/s is still on the broken graph configuration or on a swapping host.

RTX PRO 6000 Blackwell, same-host PCIe, is the external reference shape: NixlConnector plus NIXL CUDA IPC, with UCCL P2P as a transport challenger. No published number of that pair is copied into this file.

---

## 5. Results

Status as of **29 September 2026**. Artifacts are in `_results/kv_xfer_mi350p/`. Tools: AMD SMI 26.5.0 (ROCm 7.14.0 on the host) and `lstopo --whole-io`. ACS was not changed. Verbose ACS bits were not readable from this user.

This chassis has two GPUs, so the directed matrix has one off-diagonal cell each way. An 8- or 16-card server still needs every prefill→decode pair, including pairs that share a switch and pairs that do not.

### 5.1 Switches on the only pair

The host is a dual-socket AMD EPYC 9015 (2 NUMA nodes). Each MI350P hangs from its own three-bridge chain. Every bridge link that exposes speed and width is **32.0 GT/s, x16**. The two chains do not meet at a PCIe switch. Their lowest common ancestor is the machine, across the two CPU packages.

```text
Socket 0, NUMA 0                              Socket 1, NUMA 1
Host bridge 0000:88:00.0  [1022:153a]         Host bridge 0001:c4:00.0  [1022:153a]
  Root port 0000:88:01.1  [1022:153e]           Root port 0001:c4:01.1  [1022:153e]
    x16 32 GT/s                                   x16 32 GT/s
    Bridge   0000:89:00.0  [1022:1500]           Bridge   0001:c5:00.0  [1022:1500]
      x16 32 GT/s                                   x16 32 GT/s
      Bridge 0000:8a:00.0  [1002:1501]            Bridge 0001:c6:00.0  [1002:1501]
        x16 32 GT/s                                   x16 32 GT/s
        GPU0 0000:8b:00.0  [1002:75a8]              GPU1 0001:c7:00.0  [1002:75a8]
        renderD128 / card1                          renderD136 / card9
```

`[1022:153a]` is the CPU host bridge. `[1022:153e]` is the root-complex PCI bridge. `[1022:1500]` and `[1002:1501]` are the next two PCI bridges in front of the GPU. `lstopo` draws the same nesting: three `PCIBridge` levels under host bridge `0000:88:00.0` for GPU 0, and three under `0001:c4:00.0` for GPU 1.

| Field | This pair, both directions |
| --- | --- |
| AMD SMI index | GPU 0 is `0000:8b:00.0`. GPU 1 is `0001:c7:00.0` |
| Class | Cross-socket / cross-NUMA. Equivalent of an NVIDIA `SYS` path, not `PIX` (one switch) or `PXB` (several switches under one root) |
| Shared switch | None. A transfer cannot stay inside one switch |
| AMD SMI | Access ENABLED both ways. Weight 72. Hops 3. Type PCIE. DMA true. Bi-directional capability false. Cache coherent NC. Atomics 32 and 64. NUMA bandwidth N/A |
| Slot | Each GPU endpoint is also 32.0 GT/s x16 |

A single Gen5 x16 link has a payload ceiling near 63 GB/s. Bytes from one GPU to the other leave that GPU’s bridge chain, cross the inter-socket link, and enter the other GPU’s bridge chain. The 63 GB/s figure is the ceiling of one of those links, not a measured rate for the pair. Hop count alone is not the transfer rate; section 5.3 is the measured copy.

### 5.2 Clocks

Unidirectional `hipMemcpyPeerAsync` duration is a HIP event on the destination copy stream. Bidirectional, concurrent, host-staged, and NIXL durations use a host monotonic clock after the copies are synchronized. GPU event clocks from the two devices are not subtracted from each other.

Bandwidth below is `bytes / latency` at that percentile, in decimal GB/s (`1e9`). For concurrent copies, `bytes` is the sum of the overlapping payloads and the clock is wall time until all of them finish.

### 5.3 Layer 1 — HIP copy on this pair

`hipDeviceCanAccessPeer` is true both ways. The test allocates no host bounce buffer. Every power-of-two size below compared equal. The non-power-of-two 268.4 MB copy GPU1→GPU0 had 305,088 mismatched words, so that row is not a bandwidth result.

Same-process peer copy, GPU0→GPU1 (`0000:8b:00.0` → `0001:c7:00.0`):

| Payload | p50 | p95 | p99 | GB/s at p50 |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB | 0.024 ms | 0.067 ms | 2.57 ms | 0.17 |
| 1 MiB | 0.046 ms | 0.066 ms | 0.075 ms | 22.8 |
| 16 MiB | 0.405 ms | 1.89 ms | 37.9 ms | 41.4 |
| 64 MiB | 1.39 ms | 6.45 ms | 43.8 ms | 48.5 |
| 256 MiB | 5.54 ms | 5.69 ms | 5.75 ms | 48.4 |

GPU1→GPU0:

| Payload | p50 | p95 | p99 | GB/s at p50 |
| --- | ---: | ---: | ---: | ---: |
| 1 MiB | 0.039 ms | 0.062 ms | 26.4 ms | 27.2 |
| 16 MiB | 0.437 ms | 0.507 ms | 6.42 ms | 38.4 |
| 64 MiB | 2.07 ms | 58.5 ms | 96.3 ms | 32.4 |
| 256 MiB | 6.68 ms | 83.7 ms | 108 ms | 40.2 |

Two-process HIP IPC, same direction GPU0→GPU1, reused mapping, byte-perfect: 256 MiB p50 **5.85 ms**, p95 31.3 ms, p99 54.6 ms, **45.9 GB/s** at p50.

Explicit pinned host bounce, GPU0→GPU1, 256 MiB: p50 **68.9 ms**, **3.90 GB/s**. The peer copy is a different and faster path than that bounce.

Simultaneous both directions, 256 MiB each way, host clock until both streams finish: p50 **61.9 ms** for 512 MiB combined, aggregate **8.68 GB/s**. AMD SMI already reported bi-directional capability false. The measured pair does not sustain two full unidirectional streams at once.

Concurrent GPU0→GPU1 copies, aggregate over wall time, all byte-perfect:

| Copies | 64 MiB each, p50 | 256 MiB each, p50 |
| --- | --- | --- |
| 1 | 1.70 ms, 39.4 GB/s | 7.44 ms, 36.1 GB/s |
| 2 | 3.60 ms, 37.4 GB/s | 15.9 ms, 33.7 GB/s |
| 4 | 5.84 ms, 46.0 GB/s | 22.6 ms, 47.6 GB/s |
| 8 | 11.2 ms, 47.8 GB/s | 183 ms, 11.7 GB/s |

Eight simultaneous 256 MiB copies fall to 11.7 GB/s. Four do not. The single-copy HIP event time (5.54 ms at 256 MiB) is the link-floor number. The 7.44 ms row is the same copy timed on the host clock.

Directed link-floor matrix at 256 MiB, same-process peer, p50:

```text
                Decode 0000:8b:00.0     Decode 0001:c7:00.0
Prefill 8b      —                       5.54 ms, 48.4 GB/s
Prefill c7      6.68 ms, 40.2 GB/s      —
```

### 5.4 Layer 2 — NIXL / UCX on the same buffers

`nixl-rocm` 1.4.0, two processes, VRAM READ into the destination GPU, byte-exact. `query_xfer_backend` returns `UCX`. With `UCX_PROTO_INFO=y`, the selected large-message protocol is **`rocm_copy` with a host fragment via `cma/memory`**, not `rocm_ipc`.

`UCX_TLS=rocm_ipc,sm,self` and `UCX_TLS=rocm_ipc,tcp,sm,self` both fail while creating the UCX worker (`Destination is unreachable`). POSIX cannot register VRAM. `UCX_TLS=rocm_copy,tcp,sm,self` completes and uses the same host-fragment `rocm_copy` protocol.

Automatic UCX, one descriptor, GPU0→GPU1:

| Payload | p50 | p95 | p99 | GB/s at p50 | Last `xferDuration` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 0.236 ms | 5.23 ms | 42.2 ms | 4.45 | 233 µs |
| 16 MiB | 4.13 ms | 28.1 ms | 48.2 ms | 4.06 | 2.96 ms |
| 64 MiB | 54.5 ms | 141 ms | 156 ms | 1.23 | 160 ms |
| 256 MiB | 77.7 ms | 119 ms | 120 ms | 3.45 | 73.8 ms |

GPU1→GPU0, 256 MiB, one descriptor: p50 **317 ms**, p99 489 ms, **0.85 GB/s**.

`postDuration` stays under 1 ms on these samples and is already inside `xferDuration`. It is not added again.

Fragmentation on the automatic path, GPU0→GPU1, does not recover the HIP rate. A 64 MiB buffer split into 1, 16, 64, or 256 descriptors stays between **1.16 and 2.03 GB/s** at p50. A 256 MiB buffer in 64 descriptors is p50 **151 ms** (1.77 GB/s).

Concurrent automatic reads of 64 MiB, aggregate bytes over wall time: 2 in flight 3.23 GB/s, 4 in flight 2.55 GB/s, 8 in flight 1.72 GB/s.

Directed NIXL matrix at 256 MiB, one descriptor, p50:

```text
                Decode 0000:8b:00.0      Decode 0001:c7:00.0
Prefill 8b      —                        77.7 ms, 3.45 GB/s
Prefill c7      317 ms, 0.85 GB/s        —
```

That is the transfer-library cost on this path. It is not the 48.4 GB/s HIP copy, and it is not `T_connector-ready`.

### 5.4.1 UCX rebuilt on host ROCm 7.14

The vLLM image’s `/opt/rocm` is **7.2.3**. The host toolchain is **ROCm 7.14.0** at `/opt/rocm-7.14.0` (HIP 7.14.60850). `/opt` is not writable, so the install prefix is `/home/amd/workspace/coder/_opt/ucx-rocm714`. Source is UCX tag `v1.22.0`, commit `8a6b06fb880accbb933a79cda893883872c68d9d`. Configure line:

```text
--prefix=/home/amd/workspace/coder/_opt/ucx-rocm714 --with-rocm=/opt/rocm-7.14.0 --enable-mt --enable-shared --disable-static
```

`ucx_info -v` reports library 1.22.0 loaded from that prefix. `libuct_rocm.so` links `libhsa-runtime64.so.1` from `/opt/rocm-7.14.0`. `ucx_info -d` lists transports `rocm_copy` and `rocm_ipc`. Kernel `6.8.0-142-generic`. Driver amdgpu `6.19.14`. This did not use ROCm 10.

A two-process `ucp_get` of `hipMalloc` memory (`scripts/kv_xfer/ucx_rocm_get.c`) with `UCX_TLS` unset selected, for messages of 128 bytes and up:

```text
remote memory read by ucp_get* into rocm/GPU1 from rocm/dev[0]
128..inf  zero-copy  rocm_ipc/rocm_ipc
```

16 MiB GPU0→GPU1 was byte-perfect, mean **0.53 ms**, **31.7 GB/s**. 256 MiB GPU0→GPU1 was byte-perfect, mean **8.35 ms** (min 6.92, max 13.1), **32.1 GB/s**. The two-process HIP IPC baseline for that 256 MiB copy is **5.85 ms / 45.9 GB/s**. The reverse 256 MiB GET was byte-perfect but mean **170 ms**, **1.58 GB/s**.

`UCX_TLS=rocm_ipc,sm,self,tcp` still fails at `ucp_worker_create` with `Destination is unreachable`. The log says there is no cross-memory-type transport with a short put among `self`, `tcp`, and `cma`. Automatic mode keeps `rocm_copy` for host lanes and uses `rocm_ipc` for the VRAM read. Forcing that TLS list removes the host lane the worker needs.

NIXL `v1.4.0` (`c0a1102b94d173049a5478c23e765ba37681e2ca`) was then built with Meson, not CMake:

```text
meson setup build-rocm714 --prefix=.../_opt/nixl-rocm714 \
  -Ducx_path=.../_opt/ucx-rocm714 -Denable_plugins=UCX -Dwheel_variant=rocm
```

`libplugin_UCX.so` loads `libucp.so.0` from `_opt/ucx-rocm714/lib`, not from `nixl_rocm.libs`. The byte-perfect `rocm_ipc` times above are from `ucx_rocm_get` on the host ROCm 7.14 stack. They are not a rerun of the image’s `nixl-rocm` wheel, which still has the private UCX 1.22.0 and the section 5.4 host-fragment results. Forcing `UCX_TLS=rocm_ipc,sm,self,tcp` remains a worker-creation failure even on this rebuild.

### 5.4.2 Matched HIP IPC and UCX timing

The earlier 5.85 ms HIP number was a 12-copy p50, while 8.35 ms UCX was an
8-copy mean. The matched run uses the same direction, 256 MiB allocation,
host `CLOCK_MONOTONIC` around posted copy through device synchronization,
two warmups, 100 reused transfers, and p50/p95/p99. Allocation, IPC open or
UCX memory registration, and the first cold transfer are reported separately.

| Path | Setup / registration | Cold copy | p50 | p95 | p99 | GB/s at p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HIP IPC, one copy | 172 ms | 5.70 ms | **5.72 ms** | 56.8 ms | 58.6 ms | **47.0** |
| UCX `rocm_ipc`, one descriptor | 125 ms | 114 ms | **9.70 ms** | 110 ms | 161 ms | **27.7** |

Both rows are byte-perfect. The comparable p50 gap is therefore **3.98 ms**,
not the previous 2.5 ms estimate. More importantly, both paths show roughly
50–160 ms outliers on this cross-socket host. The 100-transfer mean is 10.3 ms
for HIP and 30.0 ms for UCX, so optimizing only the best bulk-copy interval
would miss the tail.

The setup numbers are not directly interchangeable: HIP includes destination
allocation, stream creation, and `hipIpcOpenMemHandle`; UCX includes destination
allocation, context/worker creation, and `ucp_mem_map`. They are excluded from
steady-state copy percentiles. The first UCX transfer is much slower than its
steady p50, confirming that engines should keep contexts, endpoints, registered
KV pools, and remote keys alive.

Synthetic equal-sized fragmentation, same registered 256 MiB allocations:

| Descriptors | Cold copy | p50 | p95 | p99 | GB/s at p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 114 ms | 9.70 ms | 110 ms | 161 ms | 27.7 |
| 16 | 113 ms | 10.6 ms | 110 ms | 116 ms | 25.3 |
| 64 | 67.1 ms | 10.2 ms | 113 ms | 164 ms | 26.4 |
| 256 | 69.4 ms | **59.0 ms** | 124 ms | 216 ms | **4.55** |

Each descriptor is posted with `ucp_get_nbx`, then the batch is progressed to
completion. `UCX_TLS` is unset. `UCX_PROTO_INFO=y` confirms
`rocm_ipc/rocm_ipc` for the 256-descriptor test; the collapse is descriptor
handling, not a fallback to host staging.

This is not yet the real Qwen3.8 layout. The model has 64 layers: 16
full-attention and 48 linear-attention layers. vLLM emits one packed K/V
descriptor per attention page and separate convolution and SSM-state
descriptors for the linear-attention layers. The exact count also depends on
allocated block IDs and the hybrid page geometry. It must be read from
`vllm:nixl_num_descriptors` during a connector run rather than inferred from
the 256 MiB payload.

Artifacts: `hip_matched_256m.json`, `ucx_matched_d{1,16,64,256}.json`, and
`ucx_proto_256desc*.log`.

### 5.4.3 Native NIXL `HIP_IPC` backend

UCX is no longer the fastest same-host transport in this qualification. A
native NIXL backend was added as
`patches/nixl/0001-add-hip-ipc-backend.patch`. It keeps the vLLM
`NixlConnector` API but replaces UCX VRAM operations with:

- allocation-level `hipIpcGetMemHandle` and cached `hipIpcOpenMemHandle`;
- immutable copy plans constructed by `prepXfer`;
- merging only when source and destination ranges are both contiguous;
- `hipMemcpyAsync(..., hipMemcpyDeviceToDevice)` on a persistent pool of four
  nonblocking streams;
- reusable HIP completion events exposed through NIXL `checkXfer`;
- same-host abstract Unix datagram notifications required by NIXL remote backends; and
- explicit `HIP_IPC` backend selection, with no silent host-staging fallback.

Build against ROCm 7.14 and NIXL 1.4.0:

```bash
cd /path/to/nixl
git checkout v1.4.0
/path/to/coder/scripts/apply_nixl_hip_ipc_patch.sh .

meson setup build-hip-ipc \
  --prefix=/opt/nixl-hip-ipc \
  -Denable_plugins=HIP_IPC \
  -Dhip_path=/opt/rocm-7.14.0 \
  -Dwheel_variant=rocm \
  -Dbuild_tests=false \
  -Dbuild_examples=false
meson compile -C build-hip-ipc
meson install -C build-hip-ipc
```

Runtime selection:

```bash
export NIXL_PLUGIN_DIR=/opt/nixl-hip-ipc/lib/x86_64-linux-gnu/plugins
export LD_LIBRARY_PATH=/opt/nixl-hip-ipc/lib/x86_64-linux-gnu:/opt/rocm-7.14.0/lib
```

For vLLM, select it explicitly:

```json
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_consumer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "backends": ["HIP_IPC"]
  }
}
```

Matched NIXL backend results, GPU0→GPU1, 256 MiB, 100 reused transfers:

| Descriptors | Cold | p50 | p95 | p99 | GB/s at p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.83 ms | **5.55 ms** | 56.9 ms | 60.2 ms | **48.3** |
| 16 | 5.65 ms | **5.67 ms** | 57.1 ms | 108 ms | **47.3** |
| 64 | 6.08 ms | **5.66 ms** | 57.1 ms | 62.9 ms | **47.4** |
| 256 | 5.54 ms | **5.56 ms** | 57.2 ms | 59.1 ms | **48.3** |

All transfers were byte-perfect. Contiguous 256-descriptor input is reduced to
one prepared copy, eliminating UCX's 59.0 ms / 4.55 GB/s descriptor collapse.
The one-descriptor NIXL result is within measurement noise of raw HIP IPC
(5.72 ms) and the same-process HIP peer copy (5.54 ms).

Direction remains important for the copy itself. GPU1→GPU0 is byte-perfect but only 12.7 ms p50
(21.2 GB/s), with 175 ms p95 and 222 ms p99. That slower direction is still the serving placement: section 5.5 puts prefill on GPU 1 and decode on GPU 0 because GPU 1 decode is several times slower than the extra copy time.

The backend integration benchmark is
`scripts/kv_xfer/nixl_hip_ipc_bench.cpp`. The Qwen-shaped region record is
`reports/results/kv_xfer/nixl_hip_ipc_regions.jsonl`. The descriptor matrix
stays a local bench output under `_results/kv_xfer_mi350p/`.

Contiguous descriptors are not the Qwen3.8 layout. The live registration is
16 full-attention regions, 157 blocks, and 2,512 descriptors, with each
attention page 3,407,872 bytes. A request that touched 17 blocks is therefore
16 separate allocations times 17 pages: 926,941,184 bytes and 272 descriptors.
Adjacent pages inside one region collapse to one copy. Pages in different
allocations do not.

| Layout | Copies after coalescing | p50 | p95 | p99 | GB/s at p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 allocation, 256 MiB | 1 | 5.90 ms | 61.0 ms | 431 ms | 45.5 |
| 16 allocations, 272 descriptors, 884 MiB | 16 | 20.5 ms | 103 ms | 127 ms | 45.3 |
| 272 allocations, 272 descriptors, 884 MiB | 272 | 77.9 ms | 236 ms | 275 ms | 11.9 |

The 16-region case keeps the measured HIP bandwidth. Splitting the same bytes
into 272 independently registered allocations does not. Packing those
allocations into one buffer would add a GPU pack and unpack around the copy;
that interval has not been shown to beat the 16-copy plan. The p95 tails are
present in the one-buffer HIP path as well, so they are a scheduling and
topology limit rather than a UCX descriptor cost.

vLLM containers do not share `/tmp`. Pathname sockets are invisible across
them even with host networking. Notifications use abstract `AF_UNIX` names in
the shared network namespace. Container-local PIDs collide (both engine cores
were PID 97), so the abstract name includes the NIXL agent name and a random
suffix. One-GPU-visible processes also need `HSA_ENABLE_IPC_MODE_LEGACY=1`;
without it, `hipIpcOpenMemHandle` returns `invalid device context`. The
container image is glibc 2.35, so the plugin loaded by vLLM is the install
under `_opt/nixl-hip-ipc-container`, built inside `vllm/vllm-openai-rocm` with
`/opt/rocm-7.14.0` mounted at `/opt/rocm`. The host install requires
GLIBC_2.38 and does not load in that image.

### 5.5 Layer 3 — serving handoff

Status: experimental. A vLLM 0.30.0 pair completed Qwen3.8 requests with
`backends=["HIP_IPC"]` and `kv_load_failure_policy=fail`. Hybrid registration
requires `VLLM_SSM_CONV_STATE_LAYOUT=DS`. The live cache is 16 full-attention
regions, 2,512 registered descriptors, and attention pages of 3,407,872 bytes.
One-GPU-visible peers require `HSA_ENABLE_IPC_MODE_LEGACY=1`. Notifications
use abstract sockets because container `/tmp` is not shared, and the socket
name includes the agent id because both engine cores were PID 97.

The router’s `kv_handoff_ms` is only the HTTP gap after prefill returns. NIXL
post and completion times below are decode-engine Prometheus deltas
(`vllm:nixl_post_time_seconds`, `vllm:nixl_xfer_time_seconds`) with requests
issued serially. `scripts/kv_xfer/pd_gate.py` records them.

#### Token-ID gate

Greedy settings were `temperature=0`, `seed=1`, `ignore_eos=true`. Prompt
token IDs matched the single-GPU control. Completion IDs did not. A repeated
single-GPU run reproduced its own IDs, so the mismatch is stable.

| Prompt | Prompt tokens | Completion length | First differing completion index |
| --- | ---: | ---: | ---: |
| Fixed short instruction | 81 | 32 | 2 |
| 8K user text plus chat template | 8246 | 16 | 5 |

On every full handoff the decoder queried N external prefix-cache tokens and
hit N−1. For the 8,246-token prompt that was 8,256 hits out of 8,257 queries:
one prompt token was not imported. The short prompt prepared 208 descriptors
as 64 copies (16 attention regions plus 48 linear layers × 3 convolution
sub-projections and one SSM state). The cold 8K prompt prepared 240
descriptors as 96 copies and moved 699,203,584 bytes. Those counts show
attention, convolution, and SSM descriptors were in the plan. They do not
prove the bytes match a single-GPU cache. The single-GPU control selected a
784-token attention page; NixlConnector selected 832. Both runs used GPU 1
and `VLLM_SSM_CONV_STATE_LAYOUT=DS`.

#### 8K HIP-IPC waterfall

Eight unique cold prompts, each 8,257 prompt tokens, `max_tokens=1`, serial:

| Interval | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Prefill HTTP | 854 ms | 860 ms | 862 ms |
| NIXL post | 24.8 ms | 44.0 ms | 44.5 ms |
| NIXL transfer, post through completion | 100 ms | 235 ms | 265 ms |
| Client arrival through the one-token response | 1,189 ms | 1,443 ms | 1,447 ms |

Every sample moved 699,203,584 bytes. Seven used 240 descriptors and one used
272. Failed transfers were zero. One earlier cold sample of the same payload
completed in 21.5 ms (about 31 GB/s) with post time 5.3 ms, so the median
above is not the best observed copy.

Repeating one 8K prompt did not repeat the full transfer. Decode prefix-cache
hits kept later NIXL payloads at 208,470,016 bytes and 208 descriptors, with
transfer times from 6 ms to 152 ms. That is a warm-prefix tail, not another
8K copy.

Four concurrent unique cold 8K requests raised client p50 from 1,189 ms to
3,233 ms and prefill p50 from 854 ms to 3,012 ms. The four transfers together
moved 4 × 699,203,584 bytes in 299 ms of summed NIXL time (about 75 ms mean)
and recomputed one token each. No transfer failed.

#### 8K/1K C1 on the GPU 1→GPU 0 pair

Four serial streaming prompts, 8192 in / 1024 out, through the router. Decode Prometheus recorded zero failed transfers. Aggregate output throughput was **72.49 tok/s** over 56.5 s and 4,096 generated tokens. Inter-token latency was 10.58 ms p50 and 10.64 ms p95. Time to first token was 1.47 s p50 (mean 2.30 s). Per request the rates were 61.4, 73.4, 84.4, and 74.5 tok/s. Artifact: `reports/results/kv_xfer/hip_ipc_pd_8k1k.json`.

`python3 scripts/generate_kv_plots.py` copies the gate JSON from `_results` when a fresh run is present, writes `reports/results/kv_connector_summary.json`, and draws `reports/figures/kv/01_hip_ipc_8k1k.png` from that summary. `_results`, `_src`, and `_opt` are local scratch and are not committed.

About 11 s of a 12 s request is decode. The KV copy is about one token at this cadence. The 1.3–1.5 s time to first token is prefill on GPU 1 plus the handshake; a dedicated 8K prefill on a healthy GPU is 0.73 s (`reports/results/kv_xfer/dedicated_prefill_8k.json`, engine prefill). A same-process 1K/1K EngineCore trace attributes decode dispatch time to MXFP4 GEMM and reduction (49%), attention and KV (20%), and elementwise work (12%). That trace is the frozen single-GPU recipe, not this 1P1D process, and the profiled run itself was 57.35 tok/s.

The short-prompt probe on this same pair still fails the token gate. Prompt IDs matched the saved single-GPU control and the first completion ID did not (`optimized_peer_visible_probe.json` against `correctness_single.json`). The 72.49 tok/s figure is a transport measurement.

UCX `rocm_ipc` was not rerun through this serving waterfall. The earlier
matched microbench remains the UCX comparison: 9.70 ms p50 for one 256 MiB
descriptor and 59.0 ms p50 at 256 descriptors.

| Gate | Status | Evidence |
| --- | --- | --- |
| Topology, access, hops, type, `lstopo` route | Measured | `_results/kv_xfer_mi350p/amd-smi-topology.json`, `lstopo-pcie.txt`, `host-topology.xml` |
| HIP peer and IPC copy | Measured | `hip_same.jsonl`, `hip_ipc.jsonl` |
| NIXL VRAM READ and selected UCX protocol | Measured | `nixl_vram.json`, `probe_auto.log` |
| Forced `rocm_ipc` | Failed to initialize | `probe_ipc.log` |
| Native NIXL `HIP_IPC` READ | Measured | `nixl_hip_ipc_matrix.jsonl`; 5.55 ms p50, 48.3 GB/s |
| NixlConnector initialization | Passed with DS layout | `backends=["HIP_IPC"]`, `VLLM_SSM_CONV_STATE_LAYOUT=DS`, 2,512 descriptors |
| Qwen-shaped 16-region READ | Measured | `reports/results/kv_xfer/nixl_hip_ipc_regions.jsonl`; 20.5 ms p50, 45.3 GB/s |
| HIP-IPC Qwen3.8 handoff | Functional, experimental | Cold 8K moves 699,203,584 bytes; 240 descriptors coalesce to 96 copies |
| Single-GPU token-id match | Failed | First completion mismatch at index 2 (81 tokens) and index 5 (8,246 tokens) on the earlier GPU 0→GPU 1 pair; the GPU 1→GPU 0 short prompt also mismatches at index 0 |
| 8K HIP-IPC waterfall | Measured | Transfer p50 100 ms, p95 235 ms; client one-token p50 1,189 ms |
| 8K/1K C1 with HIP-IPC, GPU 1 prefill and GPU 0 decode | Measured, not qualified | 72.49 tok/s aggregate, ITL 10.58 ms p50, TTFT 1.47 s p50, 0 failed transfers |
| UCX serving waterfall | Not run | Microbench only |
| 1P1D versus DP=2 | Not run | |

### 5.6 Handoff attempt without a connector

The 29 Sep pair was two independent `vllm serve` processes. Neither had `--kv-transfer-config`.

Router request with `return_token_ids` and `kv_transfer_params` nested under `extra_body`:

```text
HTTP 502
Prefill engine did not return prompt_token_ids; cannot hand off KV to decode
```

The same prefill, with those fields at the top level, returned HTTP 200 and a real `prompt_token_ids` list (54 tokens on the probe sentence). `kv_transfer_params` was **null**. Token ids without a connector do not move KV. The decoder would still compute the prompt itself.

Both engines did answer a direct completion with `MI350P READY`. That only shows each process can generate.

### 5.7 Phase occupancy, same pair, no KV import

Artifacts:

- `_results/phases_pd_mxfp4_prefill/phase_profile_summary_20260929_113415.json`
- `_results/phases_pd_mxfp4_decode/phase_profile_summary_20260929_113415.json`

Prefill on GPU 0, `max_tokens=1`. Client prompt tok/s is inflated by prefix-cache hits. Use engine prefill time.

| Case | Input | Engine prefill | TTFT p50 |
| --- | ---: | ---: | ---: |
| P1 | 128 | 75 ms | 102 ms |
| P2 | 1024 | 74 ms | 80 ms |
| P3 | 4096 | 77 ms | 85 ms |
| P4 | 8192 | 74 ms | 99 ms |
| P5 | 12288 | 125 ms | 153 ms |
| P6 | 16384 | blocked | context cap is 16384 |

Decode on GPU 1, 256 output tokens, one trial, no warmup:

| Case | Prompt | Decode tok/s | TPOT p50 | ITL p95 | TTFT p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 | 128 | 6.86 | 146 ms | 323 ms | 1.82 s |
| D2 | 1024 | 5.62 | 178 ms | 372 ms | 4.52 s |
| D3 | 4096 | 4.74 | 211 ms | 423 ms | 3.83 s |
| D4 | 8192 | 3.49 | 287 ms | 489 ms | 4.97 s |

A later 64-token completion on the same decoder was **6.3 tok/s**. Engine logs during that window show about **100% GFX**, about **2.19 GHz**, **1–11% HBM activity**, and **127–202 W**. The validated control at 1K/1K C1 is about **79 tok/s**. These decode rows are not a MI350P decode floor and they are not a KV-handoff cost.

Causes recorded on that process:

- Compile range `(1, 2048)` and graph capture sizes `[1, 2, 4, 8, 16, 24, 32]`, against control range 8192 and capture through 512.
- First-request Triton JIT of `_gemm_afp4wfp4_kernel` and `_dynamic_mxfp4_quant_kernel`.
- Host swap about **10 GiB** with both engines up. Initialization was **298 s** on the decoder versus **167 s** on the earlier single-card control. Page faults did not climb during one timed 128-token request, so swap explains startup pain more clearly than the steady 6 tok/s.

### 5.8 Radeon AI PRO R9700 result placeholder

Status: **not measured**. Populate this section only from a run on two identified
R9700 cards; do not extrapolate the MI350P result.

- Host / date: TBD
- GPUs / PCI BDFs / topology: 2 × Radeon AI PRO R9700 (`gfx1201`) / TBD
- ROCm, vLLM, NIXL, and UCX versions: TBD
- Model, precision, context, and KV block geometry: TBD
- Connector and selected UCX transport: TBD
- HIP peer access, IPC correctness, and full-buffer comparison: TBD
- Transfer sizes: 1, 16, 64, and 256 MiB plus the model’s 8K-equivalent payload
- Transfer p50 / p95 / p99 and effective GB/s: TBD
- Host-staging evidence (host DRAM and GPU SDMA/GFX counters): TBD
- NixlConnector handoff, greedy token-id equality, and decoder prompt-recompute check: TBD
- 8K:1K 1P1D versus cache-affine DP=2 serving result: TBD
- Result artifacts: `_results/kv_xfer_r9700/` (reserved; not yet created)

Acceptance requires the same gates in section 3 and the same benchmark procedure
in section 4. A successful request alone is not evidence of a direct PCIe path.

### 5.9 Numbers deliberately omitted

- NixlConnector export-to-admission time (`t6 - t3`) on this host.
- R9700 pair matrix until section 5.8 is filled from that machine.
- RTX PRO 6000 Blackwell pair matrix.
- Any 4.3 ms or 63 GB/s figure used as a measured result for this pair. The 63 GB/s value is only the payload ceiling of one Gen5 x16 slot.

---

## 6. Next run

1. Find why greedy completion IDs diverge after a byte-moving HIP-IPC handoff. The 8K/1K rate is not qualified until that match exists.
2. Rerun the UCX `rocm_ipc` serving waterfall on the same prompts, with `NIXL_PLUGIN_DIR` pointed at the wheel plugin rather than the HIP-IPC install.
3. Only after the token gate passes, run the 8K:1K 1P1D versus DP=2 trace.
4. On an 8- or 16-card host, repeat the same HIP and NIXL matrix for every prefill→decode pair and classify each pair from `amd-smi topology` plus `lstopo` before quoting one latency.
