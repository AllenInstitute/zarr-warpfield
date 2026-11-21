"""Encode/decode throughput: Rust backend vs numpy reference vs zfp, chunk by chunk.

    uv run --with "zfpy==1.0.1.post0" python scripts/throughput.py [--scale full]

MB/s are of raw input (float32) per second. Single-threaded unless marked;
the threaded rows decode/encode all chunks of a preset on a thread pool (the
Rust backend releases the GIL).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import MIN_DET
from warpfield import Config, JacobianConfig, _backend, chunked, core, synth


def chunks_of(p, u):
    out = []
    for sl in chunked.chunk_slices(u.shape, p.chunks):
        b = np.zeros(p.chunks, dtype=u.dtype)
        b[tuple(slice(0, s.stop - s.start) for s in sl)] = u[sl]
        out.append(b)
    return out


def timed(f, items):
    t0 = time.perf_counter()
    res = [f(x) for x in items]
    return res, time.perf_counter() - t0


JS = Path(__file__).resolve().parents[1] / "rust" / "warpfield-wasm" / "js"


def bench_preset(name, scale, threads, node):
    rows = []
    rs = _backend.rs
    wasm = JS / "warpfield.wasm"
    p = synth.preset(name, scale)
    u, _ = p.full()
    blocks = chunks_of(p, u)
    mb = u.nbytes / 1e6  # real data, not the zero padding of edge chunks
    h = (float(p.spacing),) * p.ks
    cfg = Config(p.axis_roles, 0.25, jacobian=JacobianConfig(h, MIN_DET))
    cj = json.dumps(cfg.to_dict())
    shp = p.chunks

    def row(label, te, td, nbytes, **kw):
        r = dict(
            preset=name,
            codec=label,
            enc_MBps=mb / te if te else None,
            dec_MBps=mb / td if td else None,
            ratio=mb * 1e6 / nbytes if nbytes else None,
            **kw,
        )
        rows.append(r)
        enc = f"{r['enc_MBps']:8.1f}" if te else "       -"
        dec = f"{r['dec_MBps']:8.1f}" if td else "       -"
        rat = f"{r['ratio']:6.1f}" if r["ratio"] else "     -"
        print(
            f"{name:14s} {label:24s} enc {enc} MB/s  dec {dec} MB/s  ratio {rat}",
            flush=True,
        )

    # Rust, single thread, zstd levels
    for lvl in (19, 9, 3):
        bufs, te = timed(lambda b, lvl=lvl: rs.encode(b, cj, zstd_level=lvl)[0], blocks)
        _, td = timed(lambda buf: rs.decode(buf, list(shp), "float32", cj), bufs)
        row(f"rust (zstd {lvl})", te, td, sum(map(len, bufs)))
    # Rust, thread pool
    with ThreadPoolExecutor(threads) as ex:
        t0 = time.perf_counter()
        bufs = list(ex.map(lambda b: rs.encode(b, cj, zstd_level=3)[0], blocks))
        te = time.perf_counter() - t0
        t0 = time.perf_counter()
        list(ex.map(lambda buf: rs.decode(buf, list(shp), "float32", cj), bufs))
        td = time.perf_counter() - t0
    row(f"rust (zstd 3) x{threads} threads", te, td, sum(map(len, bufs)))
    # wasm under node (same chunks, zstd-native encoded with level 19)
    if node and wasm.exists():
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            b19 = [rs.encode(b, cj)[0] for b in blocks]
            for i, (b, buf) in enumerate(zip(blocks, b19, strict=True)):
                (d / f"{name}.{i}.encoded").write_bytes(buf)
                (d / f"{name}.{i}.raw").write_bytes(b.tobytes())
            (d / f"{name}.json").write_text(
                json.dumps(
                    {
                        "shape": list(shp),
                        "dtype": "float32",
                        "config": cfg.to_dict(),
                        "n": len(blocks),
                    }
                )
            )
            r = subprocess.run(
                [node, str(JS / "test" / "bench.ts"), str(d), str(wasm)],
                capture_output=True,
                text=True,
                check=True,
            )
            w = json.loads(r.stdout.strip().splitlines()[-1])
            row("wasm (node)", w["encode_s"], w["decode_s"], 0)
    # numpy reference
    bufs, te = timed(lambda b: core.encode(b, cfg, backend="python"), blocks)
    _, td = timed(lambda buf: core.decode(buf, shp, np.float32, cfg, backend="python"), bufs)
    row("numpy reference", te, td, sum(map(len, bufs)))
    # zfp fixed accuracy, matched max error (tolerance chosen per preset from the sweep)
    try:
        import zfpy

        for tol in (0.25, 1.0):

            def enc(b, tol=tol):
                out = []
                for c in range(b.shape[-1]):
                    a = np.ascontiguousarray(b[..., c])
                    lead = a.shape[:-4] if a.ndim > 4 else ()
                    for idx in np.ndindex(*lead) if lead else [()]:
                        out.append(
                            zfpy.compress_numpy(np.ascontiguousarray(a[idx]), tolerance=tol)
                        )
                return out

            zb, te = timed(enc, blocks)
            _, td = timed(lambda bs: [zfpy.decompress_numpy(x) for x in bs], zb)
            row(f"zfp (tol {tol})", te, td, sum(len(x) for bs in zb for x in bs))
    except ImportError:
        pass
    return rows


def run(scale, out, threads):
    rows = []
    node = os.environ.get("NODE") or shutil.which("node")
    for name in synth.PRESET_NAMES:
        rows += bench_preset(name, scale, threads, node)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", type=Path, default=Path("out/throughput.json"))
    a = ap.parse_args()
    run(a.scale, a.out, a.threads)
