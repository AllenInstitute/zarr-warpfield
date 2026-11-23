"""Encode and decode a large array tile by tile, without zarr.

A field too large to code as one array is split into tiles of `tile_shape`
(C order of tile index). Each tile is one `warpfield` buffer, so every tile gets
the format's guarantees on its own, including the tighter error bound on its
faces that keeps errors from jumping across tile boundaries.

Tiles are padded at the upper edges with `fill_value`, exactly as zarr pads
edge chunks. With the same configuration, a tile's bytes are therefore the bytes
of the zarr chunk with the same index, and `zarr_chunk_key` gives zarr v3's
default chunk key: tiles written to a directory under those keys, next to a
matching `zarr.json`, form a zarr array.

The source only needs `.shape`, `.dtype` and numpy-style slicing (a numpy
array, a memory map, an h5py dataset, a zarr array, ...): it is read one tile at
a time, so memory stays at a few tiles whatever the size of the field. Encoded
tiles go to any mutable mapping, as they are produced.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from warpfield import halo
from warpfield.core import Config, WarpFieldSpecError, decode, encode

TileIndex = tuple[int, ...]


def tile_grid(shape: tuple[int, ...], tile_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Number of tiles along each axis."""
    if len(shape) != len(tile_shape):
        raise WarpFieldSpecError(
            f"tile_shape {tuple(tile_shape)} does not match array rank {len(shape)}"
        )
    if any(t < 1 for t in tile_shape):
        raise WarpFieldSpecError("tile extents must be at least 1")
    return tuple(-(-s // t) for s, t in zip(shape, tile_shape, strict=True))


def tile_slices(
    index: TileIndex, shape: tuple[int, ...], tile_shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """The part of the array covered by tile `index` (clipped at the upper edges)."""
    return tuple(
        slice(i * t, min((i + 1) * t, s))
        for i, t, s in zip(index, tile_shape, shape, strict=True)
    )


def iter_tiles(shape: tuple[int, ...], tile_shape: tuple[int, ...]) -> Iterator[TileIndex]:
    """All tile indices, in C order."""
    return itertools.product(*(range(n) for n in tile_grid(shape, tile_shape)))


def zarr_chunk_key(index: TileIndex, separator: str = "/") -> str:
    """zarr v3's default chunk key for a tile: "c/0/1/2"."""
    return "c" + "".join(f"{separator}{i}" for i in index)


def _check(cfg: Config, shape: tuple[int, ...], tile_shape: tuple[int, ...], dtype) -> None:
    roles = cfg.axis_roles
    if len(roles) != len(shape):
        raise WarpFieldSpecError(
            f"array rank {len(shape)} does not match {len(roles)} axis_roles"
        )
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise WarpFieldSpecError(f"data type must be float32 or float64, not {dtype}")
    tile_grid(shape, tile_shape)
    if "component" in roles:
        a = roles.index("component")
        if tile_shape[a] != shape[a]:
            raise WarpFieldSpecError(
                "tiles must cover the `component` axis in full "
                f"(tile {tile_shape[a]} vs array {shape[a]})"
            )


def _padded(block: np.ndarray, tile_shape: tuple[int, ...], fill_value: float) -> np.ndarray:
    if block.shape == tuple(tile_shape):
        return np.ascontiguousarray(block)
    out = np.full(tile_shape, fill_value, dtype=block.dtype)
    out[tuple(slice(0, n) for n in block.shape)] = block
    return out


def encode_tiles(
    source: Any,
    tile_shape: tuple[int, ...],
    cfg: Config,
    *,
    fill_value: float = 0.0,
    out: MutableMapping[Hashable, bytes] | None = None,
    key: Callable[[TileIndex], Hashable] = lambda index: index,
    skip_fill: bool = False,
    workers: int = 1,
    backend: str | None = None,
    schedule: tuple[float, float] | None = None,
    jacobian_halo: str | None = None,
    effort: str = "fast",
) -> MutableMapping[Hashable, bytes]:
    """Encode `source` tile by tile into `out` (a new dict by default).

    `jacobian_halo` ("tight" or "verbatim") extends the Jacobian guarantee to
    cells that straddle tiles: each tile is read with one sample of halo and
    the corners of straddling cells that a worst-case face error could break
    are coded tighter or stored exactly (see `warpfield.halo`). Decoding is
    unchanged.

    `schedule` fixes the level schedule (alpha, beta) for every tile; None
    tunes it per tile. `effort` is "fast" (default) or "max" (see
    `warpfield.encode`).

    `key` maps a tile index to the key stored in `out` (the index itself by
    default; `zarr_chunk_key` for zarr's layout). With `skip_fill`, tiles that
    hold only `fill_value` are not stored, as zarr does by default; decoding
    fills missing tiles with `fill_value`. With `workers` > 1, tiles are encoded
    in parallel threads (the Rust backend releases the GIL); at most twice that
    many tiles are held in memory.
    """
    shape = tuple(int(s) for s in source.shape)
    tile_shape = tuple(int(t) for t in tile_shape)
    dtype = np.dtype(source.dtype)
    _check(cfg, shape, tile_shape, dtype)
    out = {} if out is None else out

    def code(index: TileIndex) -> tuple[TileIndex, bytes | None]:
        sl = tile_slices(index, shape, tile_shape)
        min_classes = None
        if jacobian_halo is not None:
            ext = halo.read_extended(source, sl, tile_shape, cfg, fill_value)
            inner = tuple(
                slice(None) if r == "component" else slice(1, t + 1)
                for r, t in zip(cfg.axis_roles, tile_shape, strict=True)
            )
            tile = np.ascontiguousarray(ext[inner].astype(dtype))
            min_classes = halo.straddle_classes(ext, cfg, jacobian_halo)
        else:
            block = np.asarray(source[sl], dtype=dtype)
            tile = _padded(block, tile_shape, fill_value)
        if skip_fill and np.array_equal(tile, np.full_like(tile, fill_value), equal_nan=True):
            return index, None
        return index, encode(
            tile,
            cfg,
            backend=backend,
            schedule=schedule,
            min_classes=min_classes,
            effort=effort,
        )

    def store(result: tuple[TileIndex, bytes | None]) -> None:
        index, buf = result
        if buf is not None:
            out[key(index)] = buf

    if workers <= 1:
        for index in iter_tiles(shape, tile_shape):
            store(code(index))
        return out
    with ThreadPoolExecutor(workers) as pool:
        pending = []
        for index in iter_tiles(shape, tile_shape):
            pending.append(pool.submit(code, index))
            if len(pending) >= 2 * workers:
                store(pending.pop(0).result())
        for f in pending:
            store(f.result())
    return out


def decode_tiles(
    tiles: Mapping[Hashable, bytes],
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    dtype: Any,
    cfg: Config,
    *,
    fill_value: float = 0.0,
    region: tuple[slice, ...] | None = None,
    key: Callable[[TileIndex], Hashable] = lambda index: index,
    out: np.ndarray | None = None,
    workers: int = 1,
    backend: str | None = None,
) -> np.ndarray:
    """Decode the array (or `region` of it: one step-1 slice per axis) from `tiles`.

    Only the tiles that overlap the region are read and decoded; missing tiles
    are `fill_value`. The result has the region's shape (or is written into
    `out`, which must have that shape).
    """
    shape = tuple(int(s) for s in shape)
    tile_shape = tuple(int(t) for t in tile_shape)
    dtype = np.dtype(dtype)
    _check(cfg, shape, tile_shape, dtype)
    if region is None:
        region = tuple(slice(0, s) for s in shape)
    normalized = []
    for r, s in zip(region, shape, strict=True):
        start, stop, step = r.indices(s)
        if step != 1:
            raise ValueError("region slices must have step 1")
        normalized.append(slice(start, max(stop, start)))
    region = tuple(normalized)
    rshape = tuple(max(r.stop - r.start, 0) for r in region)
    if out is None:
        out = np.empty(rshape, dtype=dtype)
    elif out.shape != rshape:
        raise ValueError(f"out has shape {out.shape}, the region {rshape}")
    if 0 in rshape:
        return out
    ranges = [
        range(r.start // t, (r.stop - 1) // t + 1)
        for r, t in zip(region, tile_shape, strict=True)
    ]

    def code(index: TileIndex) -> tuple[TileIndex, np.ndarray | None]:
        buf = tiles.get(key(index))
        if buf is None:
            return index, None
        return index, decode(buf, tile_shape, dtype, cfg, backend=backend)

    def place(result: tuple[TileIndex, np.ndarray | None]) -> None:
        index, tile = result
        src, dst = [], []
        for i, t, r in zip(index, tile_shape, region, strict=True):
            lo = max(i * t, r.start)
            hi = min((i + 1) * t, r.stop)
            src.append(slice(lo - i * t, hi - i * t))
            dst.append(slice(lo - r.start, hi - r.start))
        out[tuple(dst)] = fill_value if tile is None else tile[tuple(src)]

    indices = itertools.product(*ranges)
    if workers <= 1:
        for index in indices:
            place(code(index))
    else:
        with ThreadPoolExecutor(workers) as pool:
            for result in pool.map(code, indices):
                place(result)
    return out
