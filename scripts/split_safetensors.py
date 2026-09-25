#!/usr/bin/env python3
"""Split a single-file safetensors checkpoint into HF sharded files (stdlib only)."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return n, hdr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--max-mb", type=int, default=400)
    args = p.parse_args()

    src_file = args.src / "model.safetensors"
    if not src_file.is_file():
        raise SystemExit(f"missing {src_file}")
    header_n, hdr = read_header(src_file)
    meta = hdr.pop("__metadata__", {"format": "pt"})
    tensors = [(k, v) for k, v in hdr.items() if isinstance(v, dict) and "data_offsets" in v]
    tensors.sort(key=lambda kv: kv[1]["data_offsets"][0])
    max_bytes = args.max_mb * 1024 * 1024

    shards: list[list[tuple[str, dict]]] = []
    cur: list[tuple[str, dict]] = []
    cur_sz = 0
    for name, info in tensors:
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        if cur and cur_sz + nbytes > max_bytes:
            shards.append(cur)
            cur, cur_sz = [], 0
        cur.append((name, info))
        cur_sz += nbytes
    if cur:
        shards.append(cur)

    args.dst.mkdir(parents=True, exist_ok=True)
    nshard = len(shards)
    weight_map = {}
    total = 0
    payload_base = 8 + header_n

    with src_file.open("rb") as src:
        for i, group in enumerate(shards, start=1):
            fname = f"model-{i:05d}-of-{nshard:05d}.safetensors"
            offset = 0
            new_hdr: dict = {"__metadata__": meta}
            blobs = []
            for name, info in group:
                start, end = info["data_offsets"]
                nbytes = end - start
                src.seek(payload_base + start)
                blobs.append(src.read(nbytes))
                new_hdr[name] = {
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "data_offsets": [offset, offset + nbytes],
                }
                weight_map[name] = fname
                offset += nbytes
                total += nbytes
            hdr_bytes = json.dumps(new_hdr, separators=(",", ":")).encode()
            out = args.dst / fname
            with out.open("wb") as dst:
                dst.write(struct.pack("<Q", len(hdr_bytes)))
                dst.write(hdr_bytes)
                for blob in blobs:
                    dst.write(blob)
            print(f"wrote {fname} tensors={len(group)} bytes={offset}", flush=True)
            del blobs

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (args.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    skip = {"model.safetensors", "model.safetensors.index.json"}
    for item in args.src.iterdir():
        if item.name in skip or item.name.startswith("model-"):
            continue
        dest = args.dst / item.name
        if item.is_file():
            shutil.copy2(item, dest)
    print(f"done shards={nshard} total_gb={total/1024**3:.2f} dst={args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
