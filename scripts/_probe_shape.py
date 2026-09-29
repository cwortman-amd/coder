from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config

cases = [
    ("gate_up", 1, 34816, 2560),
    ("down", 1, 5120, 8704),
    ("out", 1, 5120, 3072),
    ("qkvz", 1, 16384, 2560),
    ("prefill_gate", 1024, 34816, 2560),
]
for name, M, N, K in cases:
    cfg, tuned = _get_config(M, N, K)
    print(
        f"{name:12} tuned={str(tuned):5} "
        f"BM={cfg['BLOCK_SIZE_M']} BN={cfg['BLOCK_SIZE_N']} "
        f"warps={cfg['num_warps']} ksplit={cfg['NUM_KSPLIT']}"
    )
