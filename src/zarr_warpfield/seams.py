"""Chunk-seam metrics for a decoded field.

A seam is a systematic jump in the reconstruction error e = decoded - original
between the last sample of one chunk and the first sample of the next. The
seam ratio compares the mean |e[i+1] - e[i]| over sample pairs that straddle
a chunk boundary with the mean over pairs inside chunks. A ratio near 1 means
the boundary is statistically invisible.
"""

from __future__ import annotations

import numpy as np


def _pairs(err: np.ndarray, axis: int, chunk: int, comp_axis: int | None):
    d = np.abs(np.diff(err.astype(np.float64), axis=axis))
    if comp_axis is not None:
        d = d.max(axis=comp_axis)
        ax = axis - (axis > comp_axis)
    else:
        ax = axis
    i = np.arange(d.shape[ax])
    straddle = (i + 1) % chunk == 0
    db = np.compress(straddle, d, axis=ax)
    di = np.compress(~straddle, d, axis=ax)
    return db, di


def seam_stats(
    err: np.ndarray,
    chunks: tuple[int, ...],
    axes: tuple[int, ...],
    comp_axis: int | None = None,
) -> dict:
    """Per-axis and overall seam statistics of an error field."""
    out = {}
    allb, alli = [], []
    for a in axes:
        if err.shape[a] <= chunks[a]:
            continue
        db, di = _pairs(err, a, chunks[a], comp_axis)
        out[a] = dict(
            boundary_mean=float(db.mean()),
            interior_mean=float(di.mean()),
            boundary_max=float(db.max()),
            ratio=float(db.mean() / max(di.mean(), 1e-30)),
        )
        allb.append(db.ravel())
        alli.append(di.ravel())
    if allb:
        b, i = np.concatenate(allb), np.concatenate(alli)
        out["all"] = dict(
            boundary_mean=float(b.mean()),
            interior_mean=float(i.mean()),
            boundary_max=float(b.max()),
            ratio=float(b.mean() / max(i.mean(), 1e-30)),
        )
    return out
