"""Canonical Huffman coding of the quantization codes (format version 2, coder 1).

The entropy stage of SZ3 (Di and Cappello, IPDPS 2016; Zhao et al., ICDE
2021), followed by zstd as SZ3 does. Same byte layout as the Rust crate::

    u16      S: alphabet size; symbol s < S - 1 is zigzag(q) = s, symbol S - 1 is
             the escape (only when some zigzag(q) >= ESCAPE)
    S bytes  code length of each symbol (0 = unused, at most MAX_LEN)
    u64      byte length B of the bit stream
    B bytes  bit stream: canonical codes, most significant bit first
    rest     zigzag LEB128 varints of the escaped codes, in order

The encoder is vectorized; the decoder is a plain loop over a lookup table,
adequate for a reference implementation (the Rust decoder is the fast path).
"""

from __future__ import annotations

import struct

import numpy as np

from warpfield.format import (
    WarpFieldFormatError,
    unzigzag,
    varint_decode,
    varint_encode,
    zigzag,
)

ESCAPE = 4095
MAX_LEN = 24
LUT_BITS = 12


def _unlimited(freq: np.ndarray) -> np.ndarray:
    """Two-queue Huffman construction; ties prefer leaves, then lower symbols."""
    syms = np.flatnonzero(freq)
    length = np.zeros(freq.size, dtype=np.int64)
    if syms.size == 1:
        length[syms[0]] = 1
        return length
    order = sorted(syms.tolist(), key=lambda s: (int(freq[s]), s))
    nl = len(order)
    weight = [int(freq[s]) for s in order]
    parent = [0] * (2 * nl - 1)
    li, ii = 0, nl
    for _ in range(nl - 1):
        picked = []
        for _ in range(2):
            if li < nl and (ii >= len(weight) or weight[li] <= weight[ii]):
                picked.append(li)
                li += 1
            else:
                picked.append(ii)
                ii += 1
        n = len(weight)
        weight.append(weight[picked[0]] + weight[picked[1]])
        parent[picked[0]] = parent[picked[1]] = n
    total = 2 * nl - 1
    depth = [0] * total
    for n in range(total - 2, -1, -1):
        depth[n] = depth[parent[n]] + 1
    for k, s in enumerate(order):
        length[s] = depth[k]
    return length


def _lengths(freq: np.ndarray) -> np.ndarray:
    f = freq.astype(np.int64)
    while True:
        length = _unlimited(f)
        if length.max(initial=0) <= MAX_LEN:
            return length
        f = np.where(f > 0, (f + 1) // 2, 0)


def _canonical(length: np.ndarray) -> tuple[np.ndarray, list[int]]:
    order = sorted((int(length[s]), s) for s in np.flatnonzero(length))
    code = np.zeros(length.size, dtype=np.uint64)
    c = prev = 0
    for k, (ln, s) in enumerate(order):
        if k:
            c += 1
        c <<= ln - prev
        prev = ln
        code[s] = c
    return code, [s for _, s in order]


def encode(codes: np.ndarray) -> bytes:
    z = zigzag(np.asarray(codes, dtype=np.int64))
    sym = np.minimum(z, np.uint64(ESCAPE)).astype(np.int64)
    nsym = int(sym.max()) + 1 if sym.size else 0
    freq = np.bincount(sym, minlength=nsym)
    length = _lengths(freq) if nsym else np.zeros(0, dtype=np.int64)
    code, _ = _canonical(length)
    head = struct.pack("<H", nsym) + length.astype(np.uint8).tobytes()
    if not sym.size:
        return head + struct.pack("<Q", 0)
    ln = length[sym]
    cv = code[sym]
    pos = np.zeros(sym.size, dtype=np.int64)
    np.cumsum(ln[:-1], out=pos[1:])
    bits = np.zeros(int(ln.sum()), dtype=np.uint8)
    for b in range(int(ln.max())):
        m = ln > b
        shift = (ln[m] - 1 - b).astype(np.uint64)
        bits[pos[m] + b] = ((cv[m] >> shift) & np.uint64(1)).astype(np.uint8)
    packed = np.packbits(bits).tobytes()
    escapes = varint_encode(z[z >= np.uint64(ESCAPE)])
    return head + struct.pack("<Q", len(packed)) + packed + escapes


def decode(buf: bytes, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        (nsym,) = struct.unpack_from("<H", buf, 0)
        if len(buf) < 2 + nsym:
            raise struct.error
        length = np.frombuffer(buf, dtype=np.uint8, count=nsym, offset=2).astype(np.int64)
        (nbytes,) = struct.unpack_from("<Q", buf, 2 + nsym)
    except struct.error:
        raise WarpFieldFormatError("truncated Huffman section") from None
    o = 2 + nsym + 8
    if o + nbytes > len(buf):
        raise WarpFieldFormatError("truncated Huffman section")
    if length.max(initial=0) > MAX_LEN:
        raise WarpFieldFormatError("Huffman code length exceeds the limit")
    if not length.any():
        raise WarpFieldFormatError("empty Huffman code")
    if sum(2.0 ** -int(ln) for ln in length[length > 0]) > 1.0:
        raise WarpFieldFormatError("over-subscribed Huffman code")
    code, order = _canonical(length)
    # (length, code) -> symbol, and a LUT for short codes
    lut_sym = [0] * (1 << LUT_BITS)
    lut_len = [0] * (1 << LUT_BITS)
    long_codes: dict[tuple[int, int], int] = {}
    for s in order:
        ln, c = int(length[s]), int(code[s])
        if ln <= LUT_BITS:
            base = c << (LUT_BITS - ln)
            for e in range(base, base + (1 << (LUT_BITS - ln))):
                lut_sym[e] = s
                lut_len[e] = ln
        else:
            long_codes[(ln, c)] = s
    escape = nsym - 1 if nsym - 1 == ESCAPE else None
    esc_buf = buf[o + nbytes :]
    n_esc = sum(1 for b in esc_buf if b < 0x80)
    escaped = varint_decode(esc_buf, n_esc).tolist() if n_esc else []

    data = buf[o : o + nbytes]
    total = 8 * nbytes
    window = 0  # the next `have` bits of the stream, as an integer
    have = 0
    nxt = 0  # next byte of `data` to load
    used = 0
    out = np.empty(count, dtype=np.uint64)
    ei = 0
    mask_lut = (1 << LUT_BITS) - 1
    for k in range(count):
        while have < MAX_LEN:
            window = (window << 8) | (data[nxt] if nxt < nbytes else 0)
            nxt += 1
            have += 8
        top = (window >> (have - LUT_BITS)) & mask_lut
        ln = lut_len[top]
        if ln:
            s = lut_sym[top]
        else:
            for ln in range(LUT_BITS + 1, MAX_LEN + 1):
                s = long_codes.get((ln, (window >> (have - ln)) & ((1 << ln) - 1)))
                if s is not None:
                    break
            else:
                raise WarpFieldFormatError("invalid Huffman code")
        have -= ln
        window &= (1 << have) - 1
        used += ln
        if used > total:
            raise WarpFieldFormatError("truncated Huffman bit stream")
        if s == escape:
            if ei >= len(escaped):
                raise WarpFieldFormatError("missing escaped code")
            out[k] = escaped[ei]
            ei += 1
        else:
            out[k] = s
    if ei != len(escaped):
        raise WarpFieldFormatError("unused escaped codes")
    return unzigzag(out)
