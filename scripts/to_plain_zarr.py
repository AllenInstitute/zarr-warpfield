"""Copy a warpfield OME-Zarr store (a field or a whole scene) to a plain
OME-Zarr: every array that uses the warpfield codec is decoded and written
again with an ordinary lossless codec; everything else (group metadata, other
arrays, the scene's bijection) is copied as it is. Chunk by chunk, never the
whole array.

The result needs no warpfield decoder: any zarr v3 reader, and Neuroglancer's
"warp" option, read it as a regular float32 array. It holds the decoded values
(within the store's error bound of the original field), losslessly.

    uv run python scripts/to_plain_zarr.py SRC DST [--codec blosc|zstd|gzip] [--level N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec, GzipCodec, ZstdCodec


def plain_codecs(name: str, level: int | None, itemsize: int):
    if name == "blosc":
        blosc = BloscCodec(
            cname="zstd", clevel=level or 5, shuffle="shuffle", typesize=itemsize
        )
        return [BytesCodec(), blosc]
    if name == "zstd":
        return [BytesCodec(), ZstdCodec(level=level or 19)]
    if name == "gzip":
        return [BytesCodec(), GzipCodec(level=level or 9)]
    raise ValueError(name)


def copy_array(src: Path, dst: Path, codec: str, level: int | None) -> None:
    meta = json.loads((src / "zarr.json").read_text())
    names = [c["name"] for c in meta.get("codecs", [])]
    if "warpfield" not in names:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    a = zarr.open_array(str(src), mode="r")
    serializer, *compressors = plain_codecs(codec, level, a.dtype.itemsize)
    b = zarr.create_array(
        str(dst),
        shape=a.shape,
        chunks=a.chunks,
        dtype=a.dtype,
        fill_value=a.fill_value,
        dimension_names=meta.get("dimension_names"),
        attributes=meta.get("attributes", {}),
        serializer=serializer,
        compressors=compressors,
        overwrite=True,
    )
    grid = [range(0, n, c) for n, c in zip(a.shape, a.chunks, strict=True)]
    for lo in itertools.product(*grid):
        sl = tuple(
            slice(o, min(o + c, n)) for o, c, n in zip(lo, a.chunks, a.shape, strict=True)
        )
        b[sl] = np.asarray(a[sl])


def size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def walk(src: Path, dst: Path, codec: str, level: int | None):
    """Copy the node at src: arrays transcoded, groups recursed."""
    meta = json.loads((src / "zarr.json").read_text())
    if meta.get("node_type") == "array":
        copy_array(src, dst, codec, level)
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "zarr.json", dst / "zarr.json")
    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "zarr.json").exists():
            walk(child, dst / child.name, codec, level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--codec", choices=["blosc", "zstd", "gzip"], default="blosc")
    ap.add_argument("--level", type=int)
    a = ap.parse_args()
    walk(a.src, a.dst, a.codec, a.level)
    for p in sorted(a.dst.rglob("zarr.json")):
        d = p.parent
        rel = d.relative_to(a.dst)
        is_array = json.loads(p.read_text()).get("node_type") == "array"
        if is_array and (a.src / rel).exists():
            s, t = size(a.src / rel), size(d)
            if s != t:
                print(f"{rel}: {s / 1e6:.1f} MB -> {t / 1e6:.1f} MB ({t / max(s, 1):.1f}x)")
    print(f"total: {size(a.src) / 1e6:.1f} MB -> {size(a.dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
