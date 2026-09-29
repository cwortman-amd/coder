from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config

cases = [
    ("down", 1, 5120, 8704),
    ("gate_up", 1, 34816, 2560),
    ("out", 1, 5120, 3072),
    ("qkvz", 1, 16384, 2560),
]
for name, M, N, K in cases:
    cfg, tuned = _get_config(M, N, K)
    print(
        f"{name:8} tuned={str(tuned):5} warps={cfg['num_warps']} ksplit={cfg['NUM_KSPLIT']} "
        f"BM={cfg['BLOCK_SIZE_M']}"
    )
