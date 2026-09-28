# TCO: 16× R9600D, 8× R9700S, 8× MI350P, 8× RTX PRO 6000

Planning comparison of one $87,000 server with 768 GB RAM. Dual-slot cards fill **eight** slots. The R9600D is single-slot, so the same chassis holds **sixteen**. Card prices and the server figure are the estimates supplied for this note, except the R9700S price noted below. They are not street quotes, and they are not a measured serving result.

| Item | Unit price | Cards in this server | GPU spend |
|---|---:|---:|---:|
| AMD Radeon AI PRO R9600D | $1,500 | 16 | $24,000 |
| AMD Radeon AI PRO R9700S | $1,299 | 8 | $10,392 |
| AMD Instinct MI350P | $15,000 | 8 | $120,000 |
| NVIDIA RTX PRO 6000 | $16,000 | 8 | $128,000 |
| Server with 768 GB RAM, no GPUs | $87,000 | 1 | $87,000 |

Date: 28 September 2026. Board specs below are vendor peak figures. Token-per-dollar is split into three shapes — 8,192/1,024, 1,024/8,192, and 1,024/1,024 — and a rate is priced only in the shape it was measured at. MI350P has two rows. **Production** is the frozen HF Quark MXFP4 recipe, `VLLM_ROCM_USE_AITER` unset. **DFlash-3** is the best non-production profile: speculative depth 3. It fails the equality gate (101/116) and the interactive latency gate, so it is not the default. R9700S rates are the R9700 MXFP4 sweep in [R9700.md](R9700.md) on the same 64 CU GPU. The R9600D has no run. The RTX PRO 6000 published median has no recorded input and output length, so it is not placed in any of the three tables. New MI350P cells are in `_results/priority_eval/tco_mi350p/`.

## Card reference

| Per card | Radeon AI PRO R9600D | Radeon AI PRO R9700S | Instinct MI350P | RTX PRO 6000 |
|---|---|---|---|---|
| Architecture | RDNA 4 | RDNA 4 | CDNA 4 | Blackwell |
| Compute units | 48 | 64 | 128 | — |
| Memory | 32 GB GDDR6 | 32 GB GDDR6 | 144 GB HBM3E | 96 GB GDDR7 ECC |
| Peak bandwidth | 640 GB/s | 640 GB/s | 4 TB/s | 1,792 GB/s |
| Board power | 150 W | 300 W | 600 W max (450 W configurable) | 600 W |
| Slots | 1, passive | 2, passive | 2, FHFL PCIe | 2 |
| Interface | PCIe 5.0 x16 | PCIe 5.0 x16 | PCIe 5.0 x16 | PCIe |
| Low-precision peak | 397 INT4 TOPS dense, 794 with sparsity | 766 INT4 TOPS dense, 1,531 with sparsity | 4.6 PFLOPS MXFP4 | 4,000 FP4 TOPS with sparsity |
| Card price | $1,500 | $1,299 | $15,000 | $16,000 |
| Cards per server | 16 | 8 | 8 | 8 |

Sources: [AMD R9600D](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9600d.html), [AMD R9700S](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700s.html), [AMD MI350P](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html), [NVIDIA RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/). The NVIDIA row uses that published workstation sheet (96 GB, 1,792 GB/s, 600 W, dual slot). The $1,500 and $16,000 figures are the supplied card prices. The $1,299 figure is AMD’s R9700 MSRP as of 1 Oct 2025; the R9700S has no separate published MSRP, and it is the same 64 CU / 32 GB / 300 W GPU. AMD’s datacenter footnote allows the R9600 series and the R9000S series, which covers both Radeon boards in this note.

Peak INT4, MXFP4, and sparse FP4 are different units. They are listed so the power and memory envelope is visible. They are not a ranking of Qwen tokens per second.

## Server fill

One chassis: eight dual-width slots, which is sixteen single-width positions. Dual-slot boards occupy two positions, so R9700S, MI350P, and RTX PRO 6000 stop at eight. The R9600D occupies one position, so the server takes sixteen. Host RAM is 768 GB either way: 48 GB per R9600D, 96 GB per dual-slot GPU.

| Server | 16× R9600D | 8× R9700S | 8× MI350P | 8× RTX PRO 6000 |
|---|---:|---:|---:|---:|
| GPU capex | $24,000 | $10,392 | $120,000 | $128,000 |
| Server | $87,000 | $87,000 | $87,000 | $87,000 |
| **System capex** | **$111,000** | **$97,392** | **$207,000** | **$215,000** |
| Aggregate VRAM | 512 GB | 256 GB | 1,152 GB | 768 GB |
| Aggregate bandwidth | 10.24 TB/s | 5.12 TB/s | 32 TB/s | 14.34 TB/s |
| GPU nameplate | 2.40 kW | 2.40 kW | 4.80 kW | 4.80 kW |
| Capex per GB of VRAM | $217 | $380 | $180 | $280 |
| Capex per GB/s | $10.84 | $19.02 | $6.47 | $15.00 |
| Host RAM per GPU | 48 GB | 96 GB | 96 GB | 96 GB |
| Capex per GPU (card + share of server) | $6,938 | $12,174 | $25,875 | $26,875 |
| Server share of capex | 78% | 89% | 42% | 40% |

Sixteen R9600D cards cost $13,608 more than eight R9700S cards and draw the same 2.4 kW. They deliver twice the VRAM and twice the aggregate bandwidth, in smaller 32 GB pools. Eight RTX PRO 6000 cards cost $8,000 more than eight MI350P cards.

## Power over three years

Electricity is GPU board power only. The server’s CPU, memory, storage, and fans are the same host in every fill. Facility PUE is not applied. Planning rate is **$0.12/kWh**. Three years is 26,298 hours. The 50% case is average board power at half of nameplate. The 100% case is the nameplate upper bound, not a measured duty cycle. MI350P can be capped at 450 W; the table uses the 600 W max.

| | 16× R9600D | 8× R9700S | 8× MI350P | 8× RTX PRO 6000 |
|---|---:|---:|---:|---:|
| Nameplate | 2.40 kW | 2.40 kW | 4.80 kW | 4.80 kW |
| 3-year energy at 50% | $3,787 | $3,787 | $7,574 | $7,574 |
| 3-year energy at 100% | $7,574 | $7,574 | $15,148 | $15,148 |

At the planning draw, three years of GPU electricity is about 3% of the R9600D system and about 4% of the other three. Moving the rate to $0.20/kWh scales those energy lines by 1.67. It does not reorder the servers.

Three-year cost of ownership at 50% GPU draw, $0.12/kWh, zero residual value:

| | 16× R9600D | 8× R9700S | 8× MI350P | 8× RTX PRO 6000 |
|---|---:|---:|---:|---:|
| Capex | $111,000 | $97,392 | $207,000 | $215,000 |
| GPU electricity | $3,787 | $3,787 | $7,574 | $7,574 |
| **3-year TCO** | **$114,787** | **$101,179** | **$214,574** | **$222,574** |
| Per year | $38,262 | $33,726 | $71,525 | $74,191 |
| Per GPU-year, including the server share | $2,391 | $4,216 | $8,941 | $9,274 |

Efficiency at nameplate: R9600D 4.7 W/GB, R9700S 9.4 W/GB, MI350P 4.2 W/GB, RTX PRO 6000 6.3 W/GB. Bandwidth per watt: R9600D 4.3 GB/s/W, R9700S 2.1 GB/s/W, MI350P 6.7 GB/s/W, RTX PRO 6000 3.0 GB/s/W.

## What the server holds

Checkpoint sizes are from this repo’s measurements, except the GPT-OSS row, which is a fit statement and not a weighed checkpoint. Usable memory is 29.8 GiB on a 32 GB card, 89.4 GiB on the RTX PRO 6000, and 134.1 GiB on the MI350P. Sixteen R9600D cards are 476.8 GiB. Eight R9700S cards are 238.4 GiB.

| Checkpoint | Size | 16× R9600D | 8× R9700S | 8× RTX PRO 6000 | 8× MI350P |
|---|---:|---|---|---|---|
| Qwen3.8-27B Quark MXFP4 | 14.2 GiB | 16 replicas, one per card | 8 replicas, one per card | 8 replicas, one per card | 8 replicas, one per card |
| Qwen3.8-27B FP8 | 27.5 GiB | 16 replicas, little KV left on each | 8 replicas, little KV left on each | 8 replicas with KV headroom | 8 replicas with KV headroom |
| openai/gpt-oss-20b native MXFP4 | fits a 32 GB card | 16 replicas, one per card | 8 replicas, one per card | 8 replicas, one per card | 8 replicas, one per card |
| Qwen3.8-Flash-Next FP8 | 172.8 GiB | 2 replicas at TP=8 (238.4 GiB each); unproven | 1 replica at TP=8 (238.4 GiB); unproven | 2 replicas at TP=4 (357.6 GiB each). TP=2 is 178.8 GiB, almost no KV room | 4 replicas at TP=2 (268.2 GiB each), the layout in [FLASH-NEXT.md](FLASH-NEXT.md); not yet loaded |

Capex per 27B replica is the per-GPU figure: R9600D $6,938, R9700S $12,174, RTX PRO 6000 $26,875, MI350P $25,875. Capex per Flash-Next replica:

| Flash-Next FP8 | 16× R9600D | 8× R9700S | 8× RTX PRO 6000 | 8× MI350P |
|---|---:|---:|---:|---:|
| Replicas per server | 2 | 1 | 2 | 4 |
| GPUs per replica | 8 | 8 | 4 | 2 |
| Capex per replica | $55,500 | $97,392 | $107,500 | $51,750 |

The R9700 tok/s does not carry to the R9600D. That sweep is 64 CUs at 300 W. The R9600D is 48 CUs at 150 W. All four cards are PCIe boards. A 27B replica on one card pays no collective. A Flash-Next replica pays it at TP=2 on MI350P, TP=4 on RTX PRO 6000, and TP=8 on R9600D or R9700S.

openai/gpt-oss-20b is native MXFP4. On a 32 GB R9700S the serve line is `VLLM_ROCM_USE_AITER=1`, `--dtype auto`, tensor parallel 1, prefix caching off. The client is 1,024 in / 1,024 out, 10 prompts, concurrency 1. Commands are in [GPT-OSS.md](GPT-OSS.md). gpt-oss-120b does not fit that card. No rate from the 20B run is in the tables below.

## Tokens per dollar

Three tables, one shape each. A cell is filled only when that card was measured at that input length, that output length, and that concurrency. Rates are not copied across shapes.

**C is concurrent sequences on each GPU.** A 16× R9600D server at C8 is serving 128 sequences. An eight-GPU server at C8 is serving 64. Dollars are the 3-year TCO above. Tokens assume that tok/s is sustained for all 26,298 hours. A server busy half the time costs twice as many dollars per token. An empty cell was not measured. It is not zero.

MI350P production is non-speculative Quark MXFP4, `ignore_eos`, one replica per GPU. MI350P DFlash-3 is the same weights plus a depth-3 drafter. The published RTX PRO 6000 figure of 167.8 tok/s is a C1 median on NVFP4 with DFlash2. The source does not record the input and output length, so that rate is not in these tables. 3-year TCO of that server is **$222,574**.

### 8,192 in / 1,024 out

| $ / million output tokens | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| 16× R9600D | — | — | — | — | — |
| 8× R9700S | $4.37 | $1.46 | $0.96 | $0.98 | — |
| 8× MI350P, production | $7.23 | $1.90 | $0.99 | $0.54 | $0.30 |
| 8× MI350P, DFlash-3 | $5.18 | $1.17 | $0.99 | $0.73 | $0.69 |
| 8× RTX PRO 6000 | — | — | — | — | — |

| Output tok/s, full server | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| 16× R9600D | — | — | — | — | — |
| 8× R9700S | 245 | 732 | 1,109 | 1,087 | — |
| 8× MI350P, production | 314 | 1,193 | 2,297 | 4,180 | 7,432 |
| 8× MI350P, DFlash-3 | 437 | 1,933 | 2,296 | 3,109 | 3,293 |
| 8× RTX PRO 6000 | — | — | — | — | — |

Production per card: C1 **39.19**, C4 **149.10**, C8 **287.08**, C16 **522.47**, C32 **929.04**. Throughput is still rising at C32. DFlash-3 per card: C1 **54.67**, C4 **241.66**, C8 **287.04**, C16 **388.67**, C32 **411.66**. DFlash-3 is cheaper at C1 and C4, tied at C8, and more expensive from C16 up because aggregate tok/s flattens.

R9700S per card, Radiance `vllm-mxfp4`: C1 **30.57**, C4 **91.46**, C8 **138.67**, C16 **135.93**. C32 was not run. C16 already lost throughput because the 32 GB card was near 99% KV capacity, so the knee is C8. The campaign drew about 200–260 W, so the 150 W planning average is low by roughly $2,000 over three years. A passive-board listing near $1,960 would raise this TCO about 5%, to $106,467. At C1 the R9700S server ($4.37) is still cheaper per token than either MI350P row. At C16 and C32 the production MI350P server is cheaper.

### 1,024 in / 8,192 out

| $ / million output tokens | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| 16× R9600D | — | — | — | — | — |
| 8× R9700S | — | — | — | — | — |
| 8× MI350P, production | $5.34 | $1.40 | $0.72 | $0.41 | $0.23 |
| 8× MI350P, DFlash-3 | $2.98 | $1.05 | $1.70 | $1.37 | $1.18 |
| 8× RTX PRO 6000 | — | — | — | — | — |

| Output tok/s, full server | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| 16× R9600D | — | — | — | — | — |
| 8× R9700S | — | — | — | — | — |
| 8× MI350P, production | 425 | 1,622 | 3,162 | 5,517 | 9,715 |
| 8× MI350P, DFlash-3 | 759 | 2,156 | 1,337 | 1,656 | 1,926 |
| 8× RTX PRO 6000 | — | — | — | — | — |

Production per card: C1 **53.08**, C4 **202.75**, C8 **395.19**, C16 **689.59**, C32 **1,214.39**. Still rising at C32. DFlash-3 per card: C1 **94.93**, C4 **269.51**, C8 **167.12**, C16 **207.03**, C32 **240.75**. On this long-output shape DFlash-3 wins at C1 and C4, then loses from C8 up. Its throughput drops from C4 to C8 and only partly recovers. The R9700S and RTX PRO 6000 have no completed run at this shape.

### 1,024 in / 1,024 out

| $ / million output tokens | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| 16× R9600D | — | — | — | — | — |
| 8× R9700S | — | — | — | — | — |
| 8× MI350P, production | $3.57 | $0.97 | $0.51 | $0.31 | $0.19 |
| 8× MI350P, DFlash-3 | $2.49 | $0.78 | $0.55 | $0.45 | $0.32 |
| 8× RTX PRO 6000 | — | — | — | — | — |

| Output tok/s, full server | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| 16× R9600D | — | — | — | — | — |
| 8× R9700S | — | — | — | — | — |
| 8× MI350P, production | 635 | 2,342 | 4,437 | 7,229 | 12,035 |
| 8× MI350P, DFlash-3 | 910 | 2,923 | 4,086 | 5,065 | 7,066 |
| 8× RTX PRO 6000 | — | — | — | — | — |

Production per card: C1 **79.43**, C4 **292.80**, C8 **554.57**, C16 **903.65**, C32 **1,504.33**. Frozen 3× means are C1 **79.35** and C8 **553.58**. The same sweep was still rising at C64 (2,017 tok/s per card, **$0.14** per million on the eight-card server). Mean request latency goes from 12.89 s at C1 to 21.77 s at C32. DFlash-3 per card: C1 **113.77**, C4 **365.43**, C8 **510.81**, C16 **633.18**, C32 **883.26**. C8 is the earlier depth-sweep cell; this run reproduced C1 at 113.77 against the published 113.75. DFlash-3 is cheaper at C1 and C4. Production is cheaper from C8 up. The R9700 has no serving sweep at this shape. An earlier MI350P FP8 sweep (51.48 / 193.28 / 371.15 tok/s at C1 / C4 / C8) is a different quant and is not in this table.

### 16× R9600D tie against production 8× MI350P

The R9600D row is empty in every table. Its 3-year TCO is $114,787, 53% of the MI350P TCO. The lines below match the **production** MI350P dollars per million tokens. A measured per-card rate above the line for that shape makes the sixteen-card server cheaper per token than production 8× MI350P.

| 16× R9600D tok/s to tie, 8,192/1,024 | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| Full server | 168 | 638 | 1,229 | 2,236 | 3,976 |
| Per card | 10 | 40 | 77 | 140 | 248 |

| 16× R9600D tok/s to tie, 1,024/8,192 | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| Full server | 227 | 868 | 1,691 | 2,951 | 5,197 |
| Per card | 14 | 54 | 106 | 184 | 325 |

| 16× R9600D tok/s to tie, 1,024/1,024 | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| Full server | 340 | 1,253 | 2,373 | 3,867 | 6,438 |
| Per card | 21 | 78 | 148 | 242 | 402 |

The R9700S rates are a 64 CU result. They are not an R9600D measurement.

## How to read the servers

- **16× R9600D** is the single-slot fill: 512 GB of GDDR6, 10.24 TB/s, 2.4 kW, 3-year TCO $114,787. Sixteen 27B replicas fit. Flash-Next needs eight cards per replica, so the server holds two. Token price is unknown at all three shapes.
- **8× R9700S** is the dual-slot 64 CU board in the same chassis: half as many 32 GB pools, same 2.4 kW, 3-year TCO $101,179. It is priced only at 8,192/1,024, where C8 is $0.96 per million tokens and C16 does not improve it.
- **8× MI350P** is the 1,152 GB / 32 TB/s server. 3-year TCO $214,574. Production keeps getting cheaper per token through C32 on every shape. The cheapest production cell is 1,024/1,024 at C32, **$0.19** per million tokens. DFlash-3 is the cheaper profile at C1 on every shape, and at C4 on every shape. It is the more expensive profile once concurrency reaches C8 on 1,024/1,024 and on 1,024/8,192, and from C16 up on 8,192/1,024. This server is the only fill here that puts Flash-Next on two cards, four replicas per server.
- **8× RTX PRO 6000** is $8,000 above the MI350P server at the same 4.8 kW, with 768 GB and 14.3 TB/s. It has no priced cell in these three tables.

Each table stays inside one shape. A blank cell means that concurrency was not measured there.
