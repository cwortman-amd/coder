# GPU metrics

Socket power, UMC memory-bandwidth estimate, and amd-smi PCIe bandwidth from each published throughput run. UMC GB/s is UMC percent times catalog peak and is not a calibrated HBM counter. PCIe MB/s is measured traffic (PCIE_BANDWIDTH), not the link peak. Sample traces are not published.

| Run | Shape | Power % TDP | Memory % peak | PCIe % of Gen5 x16 | Power (W) | UMC GB/s | PCIe MB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261002_050257_active | 1024:1024 | 52.59 | 21.25 | 0.693 | 315.55 | 870.41 | 436.74 |
| 20261002_050257_active | 1024:8192 | 47.58 | 17.87 | 0.584 | 285.49 | 732.1 | 368.26 |
| 20261002_050257_active | 8192:1024 | 44.77 | 14.99 | 0.496 | 268.64 | 613.79 | 312.47 |
