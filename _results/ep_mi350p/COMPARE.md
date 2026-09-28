# Qwen3.5-35B-A3B expert parallel, same burst

Protocol match for the [NVIDIA developer forum post](https://forums.developer.nvidia.com/t/expert-parallelism-using-6000-pro-pcie-gen5-vs-b200-nvlink/378258): `Qwen/Qwen3.5-35B-A3B`, TP=2, `--gpu-memory-utilization 0.9`, `--max-model-len 32768`, then the same serve command with `--enable-expert-parallel`. Client: `vllm bench serve --dataset-name random --random-input-len 1000 --random-output-len 1000 --request-rate 10000 --num-prompts 16 --ignore-eos`.

NVIDIA cells are that post. MI350P cells are `tp.json` and `ep.json`: `--enforce-eager`, and the vision tower still constructed (`--skip-mm-profiling` only). An em dash means that arm has not been saved yet. Peak output tok/s is the bench client's short window. Sustained rate is the Output tok/s row.

| Metric | 6000 TP | 6000 EP | B200 TP | B200 EP | MI350P TP | MI350P EP |
|---|---:|---:|---:|---:|---:|---:|
| Output tok/s | 516.44 | 1186.12 | 1885.26 | 1965.18 | 17.54 | 16.28 |
| Total tok/s | 1032.87 | 2372.23 | 3770.52 | 3930.36 | 35.07 | 32.57 |
| Duration (s) | 30.98 | 13.49 | 8.49 | 8.14 | 912.46 | 982.54 |
| Mean TTFT (ms) | 18354.95 | 672.99 | 1965.40 | 2510.53 | 14168.17 | 21216.38 |
| Median TTFT (ms) | 18326.58 | 645.70 | 1935.22 | 2462.16 | 12616.68 | 20760.23 |
| P99 TTFT (ms) | 18486.20 | 801.48 | 2069.94 | 2661.18 | 17545.49 | 23039.86 |
| Mean TPOT (ms) | 12.63 | 12.82 | 6.52 | 5.63 | 898.49 | 961.54 |
| Median TPOT (ms) | 12.66 | 12.85 | 6.55 | 5.68 | 899.93 | 961.62 |
| P99 TPOT (ms) | 12.89 | 13.08 | 6.89 | 6.16 | 900.97 | 967.46 |
| Mean ITL (ms) | 12.63 | 12.82 | 6.53 | 5.63 | 898.49 | 961.54 |
| Peak output tok/s (window) | 1344.00 | 1328.00 | 2544.00 | 2960.00 | 48.00 | 48.00 |

## EP / TP on the same platform

| Platform | Output tok/s (EP/TP) | Mean TTFT (TP/EP) | Mean TPOT (EP/TP) |
|---|---:|---:|---:|
| RTX PRO 6000 PCIe | 2.30× | 27.27× | 1.02× |
| B200 NVLink | 1.04× | 0.78× | 0.86× |
| MI350P PCIe | 0.93× | 0.67× | 1.07× |

A TTFT ratio above 1 means expert parallel shortened time to first token. The forum columns are graphed CUDA runs. Compare those ratios with a `graphs_*.json` pair when that pair exists. The MI350P row above is the eager baseline.

## GPU use during the burst

`rocm-smi` samples every 2 s while `vllm bench serve` runs. Use is mean / max of `GPU use (%)`. VRAM is the mean of used bytes.

The eager `tp.json` / `ep.json` pair has no `*.gpus.json`. Per-GPU use during that burst was not recorded.

## Link snapshot

```json
{
  "recorded_at": "2026-09-28T14:47:17Z",
  "cpu": {
    "Model name": "AMD EPYC 9015 8-Core Processor",
    "Core(s) per socket": "8",
    "Socket(s)": "2",
    "NUMA node(s)": "2",
    "NUMA node0 CPU(s)": "0-7,16-23",
    "NUMA node1 CPU(s)": "8-15,24-31"
  },
  "gpus": [
    {
      "bdf": "0000:8b:00.0",
      "current_link_speed": "32.0 GT/s PCIe",
      "current_link_width": "16",
      "max_link_speed": "32.0 GT/s PCIe",
      "max_link_width": "16",
      "numa_node": "0"
    },
    {
      "bdf": "0001:c7:00.0",
      "current_link_speed": "32.0 GT/s PCIe",
      "current_link_width": "16",
      "max_link_speed": "32.0 GT/s PCIe",
      "max_link_width": "16",
      "numa_node": "1"
    }
  ],
  "pcie_gen5_unidirectional_GBps": {
    "x16": 63.02,
    "x8": 31.51
  },
  "note": "Payload rate is the PCIe wire ceiling. Cross-socket copies can land below it."
}
```
