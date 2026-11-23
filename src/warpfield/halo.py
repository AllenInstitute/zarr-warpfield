"""The Jacobian guarantee across chunk faces, for writers (one-sample halo).

A chunk's encoder sees only its own samples, so it cannot check a Jacobian
cell whose corners lie in two or more chunks: the cell's original determinant
depends on the neighbour's values. A writer that reads one extra sample
around each chunk can. Every corner of such a cell is a face sample of its
own chunk, coded to at most `face_error_bound`. This module computes, from
original values only, whether a worst-case error at every corner could break
the cell (a new determinant at or below `min_det`, or a sign change), and if
so forces the corners in this chunk to a tighter class.

Each chunk that owns a corner of the cell sees the same original values
(through its halo) and makes the same decision, so the chunks agree without
communicating, in any order and in parallel. The bound is worst case, so the
guarantee holds whatever the neighbour's quantizer does. Decoding is
unchanged: the stored chunks are ordinary warpfield chunks.

Worst case: in cell x, entry (a, b) of the Jacobian is a difference of
samples x and x + e_b along axis b, so its error is at most
g_b = (e_x + e_{x+e_b}) / h_b. With E the error matrix (|E_ab| <= g_b),
det(J + E) - det(J) is a sum over non-empty sets S of columns of the
determinant with the columns in S taken from E. By Hadamard's inequality each
term is at most prod_{b in S} sqrt(k) g_b * prod_{b not in S} |J_b|.

Strategies for a cell that a face-bound error could break:

- "verbatim": store its corners exactly.
- "tight": code its corners at the loosest tight tier (`tight_error_bound`, a
  number or a ladder of decreasing bounds) that is enough by the same worst
  case, and exactly when none is. A corner shared by several cells takes the
  tightest of their choices.
"""

from __future__ import annotations

import itertools

import numpy as np

from warpfield.core import VERBATIM, Config, WarpFieldSpecError

STRATEGIES = ("verbatim", "tight")
# slack on the worst case, for the float64 rounding of det itself
_REL = 1e-9


def _check(cfg: Config) -> tuple[int, int]:
    """(number of spatial axes, component axis) for the supported layouts."""
    roles = cfg.axis_roles
    if cfg.jacobian is None:
        raise WarpFieldSpecError("the halo check needs a `jacobian` configuration")
    if any(r not in ("spatial", "component") for r in roles) or roles.count("component") != 1:
        raise WarpFieldSpecError(
            "the halo check supports `spatial` axes and one `component` axis only"
        )
    return len(roles) - 1, roles.index("component")


def _det_parts(u: np.ndarray, spacing: tuple[float, ...]):
    """det J and the column norms |J_b| per cell; u is (*P, k), float64."""
    k = u.shape[-1]
    cells = tuple(n - 1 for n in u.shape[:-1])
    base = tuple(slice(0, n) for n in cells)
    J = [[None] * k for _ in range(k)]
    for b in range(k):
        up = list(base)
        up[b] = slice(1, cells[b] + 1)
        d = (u[tuple(up)] - u[base]) / spacing[b]
        for a in range(k):
            J[a][b] = d[..., a] + (1.0 if a == b else 0.0)
    if k == 1:
        det = J[0][0]
    elif k == 2:
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    else:
        det = (
            J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])
        )
    cols = [np.sqrt(sum(J[a][b] ** 2 for a in range(k))) for b in range(k)]
    return det, cols


def _worst(cols, e: float, spacing) -> np.ndarray:
    k = len(cols)
    g = [np.sqrt(k) * 2.0 * e / h for h in spacing]
    total = np.zeros_like(cols[0])
    for S in itertools.product((0, 1), repeat=k):
        if any(S):
            t = np.ones_like(cols[0])
            for b in range(k):
                t = t * (g[b] if S[b] else cols[b])
            total += t
    return total


def _corners(level: np.ndarray, k: int) -> np.ndarray:
    """Per sample, the largest `level` of the cells it is a corner of (x and
    x + e_b of cell x), in sample shape."""
    out = np.zeros(tuple(n + 1 for n in level.shape), dtype=level.dtype)
    base = tuple(slice(0, n) for n in level.shape)
    np.maximum(out[base], level, out=out[base])
    for b in range(k):
        sl = list(base)
        sl[b] = slice(1, level.shape[b] + 1)
        np.maximum(out[tuple(sl)], level, out=out[tuple(sl)])
    return out


def straddle_classes(ext: np.ndarray, cfg: Config, strategy: str = "tight") -> np.ndarray:
    """Minimum classes for one chunk from its halo-extended values.

    `ext` is the chunk with one extra sample on both sides of every spatial
    axis, NaN where there is no neighbour (outside the array). Returns uint8
    classes of the chunk's shape: a tight tier class or VERBATIM for corners
    of face-crossing cells that need it, 0 elsewhere.
    """
    if strategy not in STRATEGIES:
        raise WarpFieldSpecError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
    k, ca = _check(cfg)
    j = cfg.jacobian
    u = np.moveaxis(np.asarray(ext, dtype=np.float64), ca, -1)
    n = tuple(s - 2 for s in u.shape[:-1])
    det, cols = _det_parts(u, tuple(j.grid_spacing))
    cells = det.shape  # n + 1 per axis

    # cells whose corners are partly inside the chunk, partly in the halo
    inside = np.ones(u.shape[:-1], dtype=bool)
    for b in range(k):
        idx = np.arange(u.shape[b])
        sh = [1] * k
        sh[b] = -1
        inside &= ((idx >= 1) & (idx <= n[b])).reshape(sh)
    base = tuple(slice(0, c) for c in cells)
    any_in = inside[base].copy()
    all_in = inside[base].copy()
    for b in range(k):
        up = list(base)
        up[b] = slice(1, cells[b] + 1)
        any_in |= inside[tuple(up)]
        all_in &= inside[tuple(up)]
    crossing = any_in & ~all_in & np.isfinite(det)

    min_det = j.min_det
    margin = np.where(det > min_det, det - min_det, np.abs(det))
    margin = np.where(det == 0, np.inf, margin)

    def unsafe(e: float) -> np.ndarray:
        w = _worst(cols, e, j.grid_spacing)
        return crossing & ~(w * (1 + _REL) + 1e-300 < margin)

    # level per cell: 0 nothing, 1 + t tight tier t, 1 + len(tiers) verbatim
    tiers = [
        (c, e)
        for c, e in zip(cfg.tight_classes, cfg.tight_bounds, strict=True)
        if strategy == "tight" and e < cfg.eps_face
    ]
    open_ = unsafe(cfg.eps_face)
    level = np.zeros(open_.shape, dtype=np.uint8)
    for t, (_, e) in enumerate(tiers):
        ok = open_ & ~unsafe(e)
        level[ok] = t + 1
        open_ &= ~ok
    level[open_] = len(tiers) + 1
    lut = np.array([0, *(c for c, _ in tiers), VERBATIM], dtype=np.uint8)
    out = lut[_corners(level, k)]
    out = out[tuple(slice(1, m + 1) for m in n)]
    out = np.broadcast_to(out[..., None], (*n, u.shape[-1]))
    return np.ascontiguousarray(np.moveaxis(out, -1, ca))


def read_extended(source, sl: tuple[slice, ...], tile_shape, cfg: Config, fill_value: float):
    """The tile at `sl` with one sample of halo around every spatial axis, as
    the chunks hold it: array values inside the array, `fill_value` where a
    chunk is padded past the array's edge (as zarr pads edge chunks), NaN
    where there is no chunk at all."""
    _, ca = _check(cfg)
    shape = tuple(int(s) for s in source.shape)
    ext_shape = tuple(t if a == ca else t + 2 for a, t in enumerate(tile_shape))
    ext = np.full(ext_shape, np.nan, dtype=np.float64)
    grid, inside, src, dst = [], [], [], []
    for a, s in enumerate(sl):
        if a == ca:
            for lst in (grid, inside, src, dst):
                lst.append(slice(None))
            continue
        t = tile_shape[a]
        padded = -(-shape[a] // t) * t  # extent of the chunk grid
        g0 = s.start - 1  # global index of ext[0]
        lo, hi = max(g0, 0), min(g0 + t + 2, padded)
        grid.append(slice(lo - g0, hi - g0))
        lo2, hi2 = max(g0, 0), min(g0 + t + 2, shape[a])
        src.append(slice(lo2, hi2))
        dst.append(slice(lo2 - g0, hi2 - g0))
    ext[tuple(grid)] = fill_value
    ext[tuple(dst)] = np.asarray(source[tuple(src)], dtype=np.float64)
    return ext
