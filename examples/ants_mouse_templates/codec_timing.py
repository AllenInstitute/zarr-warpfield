"""Codec-only encode / decode timing on the full ANTs fields.

Reads each field chunk by chunk (64^3 x 3) from the lossless store written by
verify.py, so only one chunk is in memory at a time. For each error bound it
then times, with the Rust backend:

- single-thread encode and decode of every chunk (no I/O inside the timers);
- decoding all chunks on N threads, as a reader would;
- an end-to-end parallel encode of the whole field with `warpfield.tiles`,
  including reading the zstd source.

    uv run python codec_timing.py OUT [--threads 14] [--schedule ALPHA BETA]

With --schedule every chunk uses that level schedule instead of tuning one
(results go to OUT/codec_timing_schedule-ALPHA-BETA.json).

OUT is verify.py's output directory (with <field>_zstd19-lossless.zarr and
<field>_warpfield-eps*.zarr). Results go to OUT/codec_timing.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from warpfield import Config, core
from warpfield.tiles import _padded, encode_tiles, iter_tiles, tile_slices

EPS = ("1um", "2.5um", "6.25um")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--fields", nargs="+", default=["forward", "inverse"])
    ap.add_argument(
        "--schedule",
        nargs=2,
        type=float,
        metavar=("ALPHA", "BETA"),
        help="fixed level schedule instead of per-chunk tuning",
    )
    a = ap.parse_args()
    sched = tuple(a.schedule) if a.schedule else None

    results = {"threads": a.threads, "backend": "rust", "schedule": sched, "fields": {}}
    for field in a.fields:
        src = zarr.open_array(a.out / f"{field}_zstd19-lossless.zarr", mode="r")
        shape, tile = src.shape, src.chunks
        cfgs = {}
        for e in EPS:
            meta = json.loads(
                (a.out / f"{field}_warpfield-eps{e}.zarr" / "zarr.json").read_text()
            )
            codec = next(c for c in meta["codecs"] if c["name"] == "warpfield")
            cfgs[e] = Config.from_dict(codec["configuration"])
        n_chunks = 0
        t_read = 0.0
        enc_s = dict.fromkeys(EPS, 0.0)
        dec_s = dict.fromkeys(EPS, 0.0)
        encoded = {e: [] for e in EPS}
        for index in iter_tiles(shape, tile):
            t0 = time.perf_counter()
            block = np.asarray(src[tile_slices(index, shape, tile)])
            t_read += time.perf_counter() - t0
            chunk = _padded(block, tile, 0.0)
            n_chunks += 1
            for e, cfg in cfgs.items():
                t0 = time.perf_counter()
                buf = core.encode(chunk, cfg, backend="rust", schedule=sched)
                enc_s[e] += time.perf_counter() - t0
                t0 = time.perf_counter()
                core.decode(buf, tile, chunk.dtype, cfg, backend="rust")
                dec_s[e] += time.perf_counter() - t0
                encoded[e].append(buf)
        chunk_bytes = int(np.prod(tile)) * 4
        total = n_chunks * chunk_bytes
        raw = int(np.prod(shape)) * 4
        r = {
            "shape": list(shape),
            "chunks": n_chunks,
            "raw_bytes": raw,
            "chunk_bytes_total": total,
            "source_read_s": t_read,
            "eps": {},
        }
        for e, cfg in cfgs.items():
            with ThreadPoolExecutor(a.threads) as pool:
                t0 = time.perf_counter()
                list(
                    pool.map(
                        lambda b, cfg=cfg, tile=tile: core.decode(
                            b, tile, np.float32, cfg, backend="rust"
                        ),
                        encoded[e],
                    )
                )
                dec_par = time.perf_counter() - t0
            t0 = time.perf_counter()
            encode_tiles(src, tile, cfg, workers=a.threads, backend="rust", schedule=sched)
            enc_par = time.perf_counter() - t0
            size = sum(len(b) for b in encoded[e])
            r["eps"][e] = {
                "bytes": size,
                "ratio_vs_raw": raw / size,
                "encode_1t_s": enc_s[e],
                "decode_1t_s": dec_s[e],
                "encode_1t_MBps": total / enc_s[e] / 1e6,
                "decode_1t_MBps": total / dec_s[e] / 1e6,
                "decode_nt_s": dec_par,
                "decode_nt_MBps": total / dec_par / 1e6,
                "encode_nt_end_to_end_s": enc_par,
            }
            x = r["eps"][e]
            print(
                f"{field:8s} eps {e:7s} {size / 1e6:6.2f} MB ({x['ratio_vs_raw']:5.0f}x) | "
                f"1 thread: encode {enc_s[e]:6.1f}s ({x['encode_1t_MBps']:5.1f} MB/s), "
                f"decode {dec_s[e]:5.2f}s ({x['decode_1t_MBps']:4.0f} MB/s) | "
                f"{a.threads} threads: decode {dec_par:5.2f}s ({x['decode_nt_MBps']:5.0f} MB/s), "
                f"encode+read {enc_par:5.1f}s",
                flush=True,
            )
        results["fields"][field] = r
    name = (
        "codec_timing.json"
        if sched is None
        else "codec_timing_schedule-{:g}-{:g}.json".format(*sched)
    )
    (a.out / name).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
