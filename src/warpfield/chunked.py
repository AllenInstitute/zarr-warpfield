"""Encode and decode a whole array chunk by chunk, without zarr.

Edge chunks are padded the way zarr pads them (with `fill_value`), so sizes and
errors match what a zarr array with this codec stores.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from warpfield.core import Config, decode, encode


@dataclass
class ChunkedResult:
    decoded: np.ndarray
    classes: np.ndarray  # per-sample error class (0 interior, 1 face, 2 tight, 3 verbatim)
    nbytes: int
    n_chunks: int
    jacobian_rounds: int
    encode_seconds: float = 0.0
    decode_seconds: float = 0.0

    def ratio(self, original: np.ndarray) -> float:
        return original.nbytes / self.nbytes


def chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    for idx in itertools.product(
        *[range(-(-s // c)) for s, c in zip(shape, chunks, strict=True)]
    ):
        yield tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunks, shape, strict=True)
        )


def roundtrip(
    arr: np.ndarray,
    chunks: tuple[int, ...],
    cfg: Config,
    fill_value: float = 0.0,
    schedule: tuple[float, float] | None = None,
    effort: str = "fast",
):
    out = np.empty_like(arr)
    cls = np.empty(arr.shape, dtype=np.uint8)
    total = 0
    n = 0
    rounds = 0
    te = td = 0.0
    for sl in chunk_slices(arr.shape, chunks):
        block = np.full(chunks, fill_value, dtype=arr.dtype)
        inner = tuple(slice(0, s.stop - s.start) for s in sl)
        block[inner] = arr[sl]
        t0 = time.perf_counter()
        buf, st, c = encode(block, cfg, stats=True, schedule=schedule, effort=effort)
        t1 = time.perf_counter()
        dec = decode(buf, chunks, arr.dtype, cfg)
        te += t1 - t0
        td += time.perf_counter() - t1
        out[sl] = dec[inner]
        cls[sl] = c[inner]
        total += len(buf)
        n += 1
        rounds = max(rounds, st.jacobian_rounds)
    return ChunkedResult(out, cls, total, n, rounds, te, td)
