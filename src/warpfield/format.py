"""Byte layout of one encoded chunk, and the integer stream helpers it uses.

    offset  size  field
    0       4     magic  b"WFLD"
    4       1     format version (2; version 1 is still read)
    5       1     rank R of the canonical array (batch axis + predicted axes)
    6       1     code coder: 0 zigzag LEB128 varints, 1 canonical Huffman (v2)
    7       1     reserved, zero
    8       8*R   canonical shape, little-endian uint64
    ...     2*8   level schedule alpha, beta, little-endian float64 (v2 only)
    ...     4*4   byte lengths of the four sections, little-endian uint32
    ...     3*8   counts: quantization codes, tight indices, verbatim indices (uint64)
    ...           section 0: quantization codes (coder above), zstd frame
                  section 1: tight-class sample indices, delta + LEB128, zstd frame
                  section 2: verbatim sample indices, delta + LEB128, zstd frame
                  section 3: verbatim values, raw little-endian data_type, zstd frame

The encoder writes section 0 both ways and keeps the smaller. Version 1 chunks
(varints, uniform schedule) decode unchanged.

Sample indices are linear (C-order) indices into the canonical array; see
`core.layout`.
"""

from __future__ import annotations

import struct
import threading

import numpy as np
import zstandard

MAGIC = b"WFLD"
VERSION = 2
CODER_VARINT, CODER_HUFFMAN = 0, 1
UNIFORM = (1.0, 1.0)
ZSTD_LEVEL = 19

# zstandard contexts are not thread-safe; zarr codes chunks on a thread pool
_local = threading.local()


def _ctx() -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    c = getattr(_local, "ctx", None)
    if c is None:
        c = _local.ctx = (
            zstandard.ZstdCompressor(level=ZSTD_LEVEL),
            zstandard.ZstdDecompressor(),
        )
    return c


class WarpFieldFormatError(ValueError):
    pass


def zigzag(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.int64, copy=False)
    return ((q << 1) ^ (q >> 63)).astype(np.uint64)


def unzigzag(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.uint64, copy=False)
    return ((z >> np.uint64(1)).astype(np.int64)) ^ (-(z & np.uint64(1)).astype(np.int64))


def varint_encode(v: np.ndarray) -> bytes:
    """LEB128-encode unsigned 64-bit integers, vectorized."""
    v = np.ascontiguousarray(v, dtype=np.uint64).ravel()
    if v.size == 0:
        return b""
    nb = np.ones(v.size, dtype=np.int64)
    for b in range(1, 10):
        nb += v >= (np.uint64(1) << np.uint64(7 * b))
    off = np.zeros(v.size, dtype=np.int64)
    np.cumsum(nb[:-1], out=off[1:])
    out = np.zeros(int(nb.sum()), dtype=np.uint8)
    for b in range(int(nb.max())):
        sel = nb > b
        byte = (v[sel] >> np.uint64(7 * b)) & np.uint64(0x7F)
        cont = (nb[sel] > b + 1).astype(np.uint64) << np.uint64(7)
        out[off[sel] + b] = (byte | cont).astype(np.uint8)
    return out.tobytes()


def varint_decode(buf: bytes, count: int) -> np.ndarray:
    a = np.frombuffer(buf, dtype=np.uint8)
    if count == 0:
        return np.zeros(0, dtype=np.uint64)
    ends = np.flatnonzero(a < 0x80)
    if ends.size != count:
        raise WarpFieldFormatError(f"expected {count} varints, found {ends.size}")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lens = ends - starts + 1
    owner_start = np.repeat(starts, lens)
    shift = ((np.arange(a.size) - owner_start) * 7).astype(np.uint64)
    contrib = (a.astype(np.uint64) & np.uint64(0x7F)) << shift
    return np.add.reduceat(contrib, starts)


def pack_indices(idx: np.ndarray) -> bytes:
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    d = np.diff(idx, prepend=0).astype(np.uint64)
    return varint_encode(d)


def unpack_indices(buf: bytes, count: int) -> np.ndarray:
    return np.cumsum(varint_decode(buf, count).astype(np.int64))


def write_chunk(
    shape: tuple[int, ...],
    codes: np.ndarray,
    tight: np.ndarray,
    verbatim: np.ndarray,
    verbatim_values: np.ndarray,
    schedule: tuple[float, float] = UNIFORM,
) -> bytes:
    from warpfield import huffman

    cctx = _ctx()[0]
    var = cctx.compress(varint_encode(zigzag(codes)))
    huf = cctx.compress(huffman.encode(codes))
    coder, sec0 = (CODER_HUFFMAN, huf) if len(huf) < len(var) else (CODER_VARINT, var)
    sections = [
        sec0,
        cctx.compress(pack_indices(tight)),
        cctx.compress(pack_indices(verbatim)),
        cctx.compress(
            np.ascontiguousarray(verbatim_values)
            .astype(verbatim_values.dtype.newbyteorder("<"), copy=False)
            .tobytes()
        ),
    ]
    head = MAGIC + struct.pack("<BBBB", VERSION, len(shape), coder, 0)
    head += struct.pack(f"<{len(shape)}Q", *shape)
    head += struct.pack("<2d", *(float(v) for v in schedule))
    head += struct.pack("<4I", *(len(s) for s in sections))
    # counts are needed to split the varint streams; keep them in the header too
    head += struct.pack("<3Q", codes.size, len(tight), len(verbatim))
    return head + b"".join(sections)


def read_header(buf: bytes):
    """(version, coder, canonical shape, schedule (alpha, beta), offset of the lengths)."""
    if buf[:4] != MAGIC:
        raise WarpFieldFormatError("not a warpfield chunk (bad magic)")
    try:
        version, rank, coder, _ = struct.unpack_from("<BBBB", buf, 4)
        if version not in (1, 2):
            raise WarpFieldFormatError(f"unsupported warpfield format version {version}")
        if (version == 1 and coder != CODER_VARINT) or coder > CODER_HUFFMAN:
            raise WarpFieldFormatError(f"unknown code coder {coder}")
        o = 8
        shape = struct.unpack_from(f"<{rank}Q", buf, o)
        o += 8 * rank
        schedule = UNIFORM
        if version == 2:
            schedule = struct.unpack_from("<2d", buf, o)
            o += 16
            a, b = schedule
            if not (np.isfinite(a) and np.isfinite(b) and a >= 1 and b >= 1):
                raise WarpFieldFormatError(
                    "level schedule alpha and beta must be finite and >= 1"
                )
    except struct.error:
        raise WarpFieldFormatError("truncated chunk") from None
    return version, coder, tuple(int(s) for s in shape), schedule, o


def read_chunk(buf: bytes, dtype: np.dtype):
    from warpfield import huffman

    _, coder, shape, _, o = read_header(buf)
    lens = struct.unpack_from("<4I", buf, o)
    o += 16
    n_codes, n_tight, n_verb = struct.unpack_from("<3Q", buf, o)
    o += 24
    dctx = _ctx()[1]
    raw = []
    for n in lens:
        # streaming decompression: frames without a content size (e.g. from the
        # pure-Rust/wasm encoder) are valid too
        raw.append(dctx.decompressobj().decompress(buf[o : o + n]))
        o += n
    if coder == CODER_HUFFMAN:
        codes = huffman.decode(raw[0], n_codes)
    else:
        codes = unzigzag(varint_decode(raw[0], n_codes))
    tight = unpack_indices(raw[1], n_tight)
    verb = unpack_indices(raw[2], n_verb)
    vals = np.frombuffer(raw[3], dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype)
    if vals.size != n_verb:
        raise WarpFieldFormatError("verbatim value count mismatch")
    return shape, codes, tight, verb, vals
