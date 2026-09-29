#!/usr/bin/env python3
"""Two-process nixl-rocm VRAM READ benchmark.

The consumer pulls bytes from the producer GPU into its own VRAM.
Bandwidth for one handoff is payload_bytes / transfer_wall_time.
Concurrent jobs report total_bytes / wall_time across the overlapping posts.
Per-transfer NIXL telemetry is recorded separately and is not used as link bandwidth.
"""

import json
import os
import socket
import struct
import subprocess
import sys
import time

import torch
from nixl_rocm import nixl_agent, nixl_agent_config


def send_blob(conn, blob: bytes) -> None:
    conn.sendall(struct.pack("!Q", len(blob)) + blob)


def recv_blob(conn) -> bytes:
    hdr = recv_exact(conn, 8)
    (n,) = struct.unpack("!Q", hdr)
    return recv_exact(conn, n)


def recv_exact(conn, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)


def checksum(buf: torch.Tensor) -> int:
    return int(buf.view(torch.uint8).to(torch.int64).sum().item())


def make_agent(name: str, backends: list[str], port: int):
    cfg = nixl_agent_config(
        backends=backends,
        capture_telemetry=True,
        listen_port=port,
        enable_listen_thread=False,
    )
    return nixl_agent(name, cfg)


def telemetry(agent, handle) -> dict:
    try:
        tel = agent.get_xfer_telemetry(handle)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    out = {"repr": repr(tel)}
    for key in (
        "startTime",
        "postDuration",
        "xferDuration",
        "totalBytes",
        "descCount",
        "start_time",
        "post_duration",
        "xfer_duration",
        "total_bytes",
        "desc_count",
    ):
        if hasattr(tel, key):
            out[key] = getattr(tel, key)
    return out


def pct(values):
    if not values:
        return {}
    xs = sorted(values)
    def at(p):
        idx = (p / 100.0) * (len(xs) - 1)
        i = int(idx)
        f = idx - i
        if i + 1 >= len(xs):
            return xs[-1]
        return xs[i] * (1 - f) + xs[i + 1] * f
    return {
        "p50": at(50),
        "p95": at(95),
        "p99": at(99),
        "min": xs[0],
        "max": xs[-1],
        "mean": sum(xs) / len(xs),
    }


def gbps(nbytes, ms):
    sec = ms / 1e3
    return (nbytes / sec) / 1e9 if sec > 0 else 0.0


def split_descs(ptr, nbytes, dev, parts):
    parts = max(1, parts)
    base = nbytes // parts
    descs = []
    off = 0
    for i in range(parts):
        length = nbytes - off if i == parts - 1 else base
        if length <= 0:
            break
        descs.append((ptr + off, length, dev))
        off += length
    return descs


def run_producer(port: int, src_dev: int, backends: list[str]) -> None:
    torch.cuda.set_device(src_dev)
    agent = make_agent("producer", backends, port + 10)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    # Metadata is exchanged after each registration. A snapshot taken before
    # register_memory does not publish the VRAM region to the other agent.
    while True:
        msg = json.loads(recv_blob(conn))
        if msg.get("bye"):
            break
        nbytes = int(msg["bytes"])
        pattern = int(msg["pattern"])
        buf = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{src_dev}")
        buf.fill_(pattern)
        torch.cuda.synchronize(src_dev)
        agent.register_memory([buf], "VRAM", backends)
        send_blob(
            conn,
            json.dumps(
                {"ptr": int(buf.data_ptr()), "dev": int(buf.get_device()), "bytes": nbytes}
            ).encode(),
        )
        send_blob(conn, agent.get_agent_metadata())
        recv_blob(conn)
        del buf
    conn.close()
    srv.close()


def one_xfer(agent, handle):
    status = agent.transfer(handle)
    spins = 0
    while status == "PROC":
        status = agent.check_xfer_state(handle)
        spins += 1
        if spins > 10000000:
            return "TIMEOUT"
    return status


def run_consumer(port: int, src_dev: int, dst_dev: int, backends: list[str], jobs: list[dict]) -> None:
    torch.cuda.set_device(dst_dev)
    agent = make_agent("consumer", backends, port + 11)
    conn = socket.socket()
    for _ in range(50):
        try:
            conn.connect(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
            time.sleep(0.1)
    else:
        raise SystemExit("producer did not listen")
    for job in jobs:
        nbytes = int(job["bytes"])
        descs_n = int(job["descs"])
        inflight = int(job["inflight"])
        iters = int(job["iters"])
        pattern = int(job["pattern"])
        send_blob(conn, json.dumps({"bytes": nbytes, "pattern": pattern}).encode())
        remote = json.loads(recv_blob(conn))
        remote_name = agent.add_remote_agent(recv_blob(conn))
        locals_ = []
        handles = []
        for i in range(inflight):
            local = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")
            local.fill_(0)
            agent.register_memory([local], "VRAM", backends)
            local_descs = agent.get_xfer_descs(
                split_descs(int(local.data_ptr()), nbytes, int(local.get_device()), descs_n),
                "VRAM",
            )
            remote_descs = agent.get_xfer_descs(
                split_descs(int(remote["ptr"]), nbytes, int(remote["dev"]), descs_n),
                "VRAM",
            )
            handle = agent.initialize_xfer(
                "READ", local_descs, remote_descs, remote_name, backends=backends
            )
            locals_.append(local)
            handles.append(handle)
        torch.cuda.synchronize(dst_dev)
        # Warmup is outside the recorded sample set.
        for _ in range(2):
            for handle in handles:
                status = one_xfer(agent, handle)
                if status != "DONE":
                    raise RuntimeError(f"warmup status {status}")
        samples = []
        post_notes = []
        for _ in range(iters):
            for local in locals_:
                local.fill_(0)
            torch.cuda.synchronize(dst_dev)
            t0 = time.perf_counter()
            statuses = [agent.transfer(h) for h in handles]
            guard = 0
            while any(s == "PROC" for s in statuses):
                statuses = [
                    agent.check_xfer_state(h) if s == "PROC" else s
                    for h, s in zip(handles, statuses)
                ]
                guard += 1
                if guard > 10000000:
                    statuses = ["TIMEOUT"]
                    break
            torch.cuda.synchronize(dst_dev)
            t1 = time.perf_counter()
            if any(s != "DONE" for s in statuses):
                raise RuntimeError(f"transfer status {statuses}")
            samples.append((t1 - t0) * 1e3)
        expected = pattern * nbytes
        mismatches = 0
        for local in locals_:
            got = checksum(local)
            if got != expected:
                mismatches += 1
        tel = telemetry(agent, handles[-1])
        try:
            backend = agent.query_xfer_backend(handles[-1])
        except Exception as exc:  # noqa: BLE001
            backend = f"error:{exc}"
        ms = pct(samples)
        total_bytes = nbytes * inflight
        bw = {k: gbps(total_bytes, v) for k, v in ms.items()}
        row = {
            "mode": "nixl_read",
            "backends": backends,
            "ucx_tls": os.environ.get("UCX_TLS", ""),
            "label": job["label"],
            "bytes_each": nbytes,
            "total_bytes": total_bytes,
            "descriptors_each": descs_n,
            "inflight": inflight,
            "src_dev": src_dev,
            "dst_dev": dst_dev,
            "direction": f"{src_dev}to{dst_dev}",
            "ok": mismatches == 0,
            "mismatched_buffers": mismatches,
            "iters": len(samples),
            "clock": "host_perf_counter_plus_cuda_synchronize",
            "latency_ms": ms,
            "bandwidth_GBps_aggregate": bw,
            "nixl_backend": backend,
            "telemetry_last": tel,
        }
        print(json.dumps(row), flush=True)
        send_blob(conn, b"done")
        del locals_, handles
    send_blob(conn, json.dumps({"bye": True}).encode())
    conn.close()


def jobs_for(mode: str) -> list[dict]:
    sizes = [
        (1 << 20, "1MiB", 20),
        (16 << 20, "16MiB", 12),
        (64 << 20, "64MiB", 8),
        (256 << 20, "256MiB", 6),
    ]
    jobs = []
    if mode == "posix":
        jobs.append(
            {
                "bytes": 1 << 20,
                "descs": 1,
                "inflight": 1,
                "iters": 5,
                "pattern": 0x5A,
                "label": "1MiB_d1",
            }
        )
        return jobs
    for nbytes, label, iters in sizes:
        jobs.append(
            {
                "bytes": nbytes,
                "descs": 1,
                "inflight": 1,
                "iters": iters,
                "pattern": 0xA5,
                "label": f"{label}_d1",
            }
        )
    for descs in (16, 64, 256):
        jobs.append(
            {
                "bytes": 64 << 20,
                "descs": descs,
                "inflight": 1,
                "iters": 6,
                "pattern": 0x3C,
                "label": f"64MiB_d{descs}",
            }
        )
    jobs.append(
        {
            "bytes": 256 << 20,
            "descs": 64,
            "inflight": 1,
            "iters": 4,
            "pattern": 0x3C,
            "label": "256MiB_d64",
        }
    )
    for inflight in (2, 4, 8):
        jobs.append(
            {
                "bytes": 64 << 20,
                "descs": 1,
                "inflight": inflight,
                "iters": 4,
                "pattern": 0xA5,
                "label": f"64MiB_x{inflight}",
            }
        )
    return jobs


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "orchestrate"
    if role == "producer":
        port = int(sys.argv[2])
        src = int(sys.argv[3])
        backends = json.loads(sys.argv[4])
        run_producer(port, src, backends)
        return
    if role == "consumer":
        port = int(sys.argv[2])
        src = int(sys.argv[3])
        dst = int(sys.argv[4])
        backends = json.loads(sys.argv[5])
        jobs = json.loads(sys.argv[6])
        run_consumer(port, src, dst, backends, jobs)
        return

    out_path = sys.argv[2]
    directions = [(0, 1), (1, 0)]
    modes = [
        ("ucx_auto", None, ["UCX"], "both"),
        ("ucx_rocm_ipc", "rocm_ipc,sm,self", ["UCX"], "both"),
        ("ucx_rocm_copy", "rocm_copy,sm,self", ["UCX"], "forward"),
        ("posix", None, ["POSIX"], "forward"),
    ]
    rows = []
    port = 18770
    for name, tls, backends, which in modes:
        dirs = directions if which == "both" else directions[:1]
        # Fragmentation and concurrency are measured on the forward cross-NUMA
        # path. The reverse direction repeats the single-descriptor size sweep.
        for src, dst in dirs:
            full = jobs_for(name if src == 0 else "size_only")
            if src != 0 and name != "posix":
                full = [j for j in jobs_for(name) if j["descs"] == 1 and j["inflight"] == 1]
            env = os.environ.copy()
            if tls:
                env["UCX_TLS"] = tls
            else:
                env.pop("UCX_TLS", None)
            env["UCX_LOG_LEVEL"] = "warn"
            log_path = out_path + f".{name}.{src}to{dst}.log"
            port += 1
            producer = subprocess.Popen(
                [sys.executable, __file__, "producer", str(port), str(src), json.dumps(backends)],
                env=env,
                stderr=open(log_path, "w"),
            )
            consumer = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "consumer",
                    str(port),
                    str(src),
                    str(dst),
                    json.dumps(backends),
                    json.dumps(full),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert consumer.stdout is not None
            for line in consumer.stdout:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    row["transport_mode"] = name
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                else:
                    print(line, file=sys.stderr, flush=True)
            rc_c = consumer.wait()
            rc_p = producer.wait()
            if rc_c != 0 or rc_p != 0:
                rows.append(
                    {
                        "mode": "nixl_read",
                        "transport_mode": name,
                        "direction": f"{src}to{dst}",
                        "ok": False,
                        "returncode_consumer": rc_c,
                        "returncode_producer": rc_p,
                        "log": log_path,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
