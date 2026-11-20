"""Jacobian determinant of the map x -> x + u(x) on a sampled displacement field.

The field is given as an array of shape (..., C, *P) — leading batch axes, a
component axis of extent C, then the predicted axes P. `spatial` lists which of
the P axes are spatial; component c is the displacement along spatial[c], in
the same units as `spacing[c]` (typically image voxels per field sample).

Derivatives are forward differences over each cell, so det has extent P-1
along spatial axes. Cell `x` uses samples x and x + e_b for every spatial
axis b.
"""

from __future__ import annotations

import numpy as np


def _shift(a: np.ndarray, axis: int, lo: int) -> np.ndarray:
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lo, a.shape[axis] - 1 + lo)
    return a[tuple(sl)]


def det_jacobian(
    u: np.ndarray, comp_axis: int, spatial: tuple[int, ...], spacing: tuple[float, ...]
) -> np.ndarray:
    """det(I + du/dx) per cell. `spatial` are axis indices into `u`."""
    k = len(spatial)
    if u.shape[comp_axis] != k:
        raise ValueError(f"need {k} components for {k} spatial axes, got {u.shape[comp_axis]}")
    u = u.astype(np.float64, copy=False)
    comps = [np.take(u, c, axis=comp_axis) for c in range(k)]
    # axis indices shift by one after removing comp_axis
    sp = [a - (a > comp_axis) for a in spatial]

    def trim(a: np.ndarray, skip: int | None) -> np.ndarray:
        # restrict to cells: drop the last sample along every spatial axis
        for ax in sp:
            if ax != skip:
                a = _shift(a, ax, 0)
        return a

    J = [[None] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            d = (_shift(comps[a], sp[b], 1) - _shift(comps[a], sp[b], 0)) / spacing[b]
            d = trim(d, sp[b])
            J[a][b] = d + (1.0 if a == b else 0.0)
    if k == 1:
        return J[0][0]
    if k == 2:
        return J[0][0] * J[1][1] - J[0][1] * J[1][0]
    if k == 3:
        return (
            J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])
        )
    raise ValueError("at most 3 spatial axes are supported")


def violations(d0: np.ndarray, d1: np.ndarray, min_det: float) -> np.ndarray:
    """Cells where decoding broke the Jacobian guarantee.

    A cell violates if the original was above `min_det` and the decoded is
    not, or if the sign of the determinant changed (this covers folds,
    where the original is already negative).
    """
    return ((d0 > min_det) & (d1 <= min_det)) | ((d0 > 0) & (d1 <= 0)) | ((d0 < 0) & (d1 >= 0))


def cell_samples(cells: np.ndarray, spatial: tuple[int, ...]) -> np.ndarray:
    """Boolean mask over samples: every corner of every flagged cell.

    `cells` has the cell shape (extent P-1 along spatial axes); the result has
    one more sample along each spatial axis.
    """
    shape = list(cells.shape)
    for ax in spatial:
        shape[ax] += 1
    out = np.zeros(shape, dtype=bool)
    k = len(spatial)
    for corner in range(1 << k):
        sl = [slice(None)] * cells.ndim
        for i, ax in enumerate(spatial):
            lo = (corner >> i) & 1
            sl[ax] = slice(lo, lo + cells.shape[ax])
        out[tuple(sl)] |= cells
    return out
