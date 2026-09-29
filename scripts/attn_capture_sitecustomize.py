"""Opt-in real-tensor capture for one attention-backend prefill.

Loaded through PYTHONPATH only when REAL_ATTN_CAPTURE_DIR is set by the
launcher. Captures the 16 full-attention layers for one prompt of the requested
token length. It is intentionally a diagnostic launcher path, never default.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading


CAPTURE_DIR = os.environ.get("REAL_ATTN_CAPTURE_DIR")
TARGET_TOKENS = int(os.environ.get("REAL_ATTN_CAPTURE_TOKENS", "44"))

# Importing vLLM from every interpreter also affects build helper subprocesses
# spawned during startup. Only the multiprocessing worker becomes EngineCore.
if CAPTURE_DIR and "--multiprocessing-fork" in sys.argv:
    import torch

    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionImpl,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    _capture_lock = threading.Lock()
    _capture_index = 0

    def _cpu(tensor):
        if tensor is None:
            return None
        return tensor.detach().to("cpu", non_blocking=False).clone()

    def _make_capture_forward(original_forward, backend_name):
        def _capture_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            global _capture_index
            result = original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
            if attn_metadata is None:
                return result
            num_actual = int(attn_metadata.num_actual_tokens)
            max_query = int(attn_metadata.max_query_len)
            if num_actual != TARGET_TOKENS or max_query != TARGET_TOKENS:
                return result

            with _capture_lock:
                index = _capture_index
                if index >= 16:
                    return result
                _capture_index += 1

            target = Path(CAPTURE_DIR)
            target.mkdir(parents=True, exist_ok=True)
            payload = {
                "backend": backend_name,
                "capture_index": index,
                "model_layer_index": 4 * index + 3,
                "query": _cpu(query[:num_actual]),
                "key": _cpu(key[:num_actual]),
                "value": _cpu(value[:num_actual]),
                "backend_output": _cpu(output[:num_actual]),
                "stock_output": (
                    _cpu(output[:num_actual]) if backend_name == "rocm_attn" else None
                ),
                "query_start_loc": _cpu(attn_metadata.query_start_loc),
                "seq_lens": _cpu(attn_metadata.seq_lens),
                "block_table": _cpu(attn_metadata.block_table),
                "num_actual_tokens": num_actual,
                "max_query_len": max_query,
                "max_seq_len": int(attn_metadata.max_seq_len),
                "causal": bool(attn_metadata.causal),
                "scale": float(self.scale),
                "sliding_window": tuple(self.sliding_window),
                "logits_soft_cap": float(self.logits_soft_cap),
                "num_heads": int(self.num_heads),
                "num_kv_heads": int(self.num_kv_heads),
                "head_size": int(self.head_size),
                "kv_cache_dtype": str(self.kv_cache_dtype),
                "query_stride": tuple(query.stride()),
                "key_stride": tuple(key.stride()),
                "value_stride": tuple(value.stride()),
                "output_stride": tuple(output.stride()),
                "kv_cache_shape": tuple(kv_cache.shape),
                "kv_cache_stride": tuple(kv_cache.stride()),
                "layer_type": type(layer).__name__,
                "k_scale": _cpu(getattr(layer, "_k_scale", None)),
                "v_scale": _cpu(getattr(layer, "_v_scale", None)),
            }
            path = target / f"layer_{index:02d}_model_{4 * index + 3:02d}.pt"
            torch.save(payload, path)
            print(
                f"REAL_ATTN_CAPTURE backend={backend_name} saved {path}",
                flush=True,
            )
            return result

        return _capture_forward

    RocmAttentionImpl.forward = _make_capture_forward(
        RocmAttentionImpl.forward, "rocm_attn"
    )
    TritonAttentionImpl.forward = _make_capture_forward(
        TritonAttentionImpl.forward, "triton_attn"
    )
    RocmAiterUnifiedAttentionImpl.forward = _make_capture_forward(
        RocmAiterUnifiedAttentionImpl.forward, "rocm_aiter_unified_attn"
    )
