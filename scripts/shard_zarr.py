"""Repack a zarr v3 store (an array, a field, or a whole scene) into sharded
arrays: each array's chunks are gathered into `sharding_indexed` shards of
N x N x N chunks. The chunk bytes are moved, not decoded: the inner codec chain
is the array's original one (warpfield, blosc, ...), so every inner chunk is
byte-for-byte the chunk it was, and the array decodes to exactly the same
values. Group metadata and everything else is copied as it is.

A shard is the chunks' bytes back to back followed by the index (offset and
length per inner chunk, uint64 little endian, C order, then crc32c): the
zarr-python default layout. Chunks that do not exist (outside the array, or
never written) are marked missing in the index.

    uv run python scripts/shard_zarr.py SRC DST [--chunks-per-shard N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from pathlib import Path

import google_crc32c
import numpy as np

MISSING = np.uint64(2**64 - 1)
INDEX_CODECS = [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "crc32c"},
]


def chunk_key(enc: dict, coords) -> str:
    sep = enc.get("configuration", {}).get("separator", "/")
    if enc.get("name", "default") == "v2":
        return sep.join(map(str, coords)) or "0"
    return sep.join(["c", *map(str, coords)])


def shard_array(src: Path, dst: Path, per_shard: int) -> tuple[int, int]:
    meta = json.loads((src / "zarr.json").read_text())
    shape = meta["shape"]
    chunk = meta["chunk_grid"]["configuration"]["chunk_shape"]
    if any(c["name"] == "sharding_indexed" for c in meta["codecs"]):
        raise ValueError(f"{src} is already sharded")
    grid = [math.ceil(n / c) for n, c in zip(shape, chunk, strict=True)]
    # per dimension: as many chunks per shard as asked, but no more than exist
    k = [min(per_shard, g) for g in grid]
    shard = [c * m for c, m in zip(chunk, k, strict=True)]
    enc = meta["chunk_key_encoding"]

    dst.mkdir(parents=True, exist_ok=True)
    out = dict(meta)
    out["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": shard}}
    out["codecs"] = [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": chunk,
                "codecs": meta["codecs"],
                "index_codecs": INDEX_CODECS,
                "index_location": "end",
            },
        }
    ]
    (dst / "zarr.json").write_text(json.dumps(out, indent=2))

    n_shards = 0
    n_chunks = 0
    shard_grid = [math.ceil(g / m) for g, m in zip(grid, k, strict=True)]
    for s in itertools.product(*map(range, shard_grid)):
        index = np.full((*k, 2), MISSING, dtype="<u8")
        parts = []
        offset = 0
        for local in itertools.product(*map(range, k)):
            g = [si * m + li for si, m, li in zip(s, k, local, strict=True)]
            if any(gi >= n for gi, n in zip(g, grid, strict=True)):
                continue
            f = src / chunk_key(enc, g)
            if not f.exists():
                continue
            b = f.read_bytes()
            index[local] = (offset, len(b))
            parts.append(b)
            offset += len(b)
        if not parts:
            continue
        idx = index.tobytes()
        crc = np.uint32(google_crc32c.value(idx)).astype("<u4").tobytes()
        p = dst / chunk_key(enc, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            for b in parts:
                fh.write(b)
            fh.write(idx)
            fh.write(crc)
        n_shards += 1
        n_chunks += len(parts)
    return n_chunks, n_shards


def walk(src: Path, dst: Path, per_shard: int, report: list):
    meta = json.loads((src / "zarr.json").read_text())
    if meta.get("node_type") == "array":
        n_chunks, n_shards = shard_array(src, dst, per_shard)
        report.append((dst, n_chunks, n_shards))
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "zarr.json", dst / "zarr.json")
    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "zarr.json").exists():
            walk(child, dst / child.name, per_shard, report)


def files(p: Path) -> list[Path]:
    return [f for f in p.rglob("*") if f.is_file()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--chunks-per-shard", type=int, default=4)
    a = ap.parse_args()
    report: list = []
    walk(a.src, a.dst, a.chunks_per_shard, report)
    for d, n_chunks, n_shards in report:
        print(f"{d.relative_to(a.dst)}: {n_chunks} chunks -> {n_shards} shards")
    fs, fd = files(a.src), files(a.dst)
    ss, sd = sum(f.stat().st_size for f in fs), sum(f.stat().st_size for f in fd)
    print(f"total: {len(fs)} files, {ss / 1e6:.2f} MB -> {len(fd)} files, {sd / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
