"""Multilevel interpolation schedule and stencils.

Along one predicted axis of extent N, the samples known at stride `s` are

    clamped:      {0, s, 2s, ...} ∪ {N-1}      (default; nothing is extrapolated)
    extrapolate:  {0, s, 2s, ...}              (SZ3-like baseline; the tail past
                                                the last anchor is extrapolated)

The top stride S is the smallest power of two with S >= N_max - 1, so the
coarsest known set is the chunk's corners. Refinement from stride s = 2h to h
visits the predicted axes in order; along axis j the new samples are
known(h) \\ known(s), the axes before j are already at known(h), and the axes
after j are still at known(s). A new sample is predicted from the samples of
known(s) on its own axis line only, so a sample on a chunk face is predicted
from samples on that same face.

Every stencil is described by four slots (l2, l1, r1, r2): the two nearest
known positions on each side. Weights are Lagrange weights for the actual
(possibly non-uniform) positions, evaluated in float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PREDICTORS = ("linear", "cubic", "eno")
BOUNDARIES = ("clamped", "extrapolate")

# ENO switches from the cubic stencil to a one-sided quadratic when one side's
# second divided difference exceeds the other's by this factor.
ENO_RATIO = 3.0


def top_stride(shape: tuple[int, ...]) -> int:
    m = max([n - 1 for n in shape] + [1])
    s = 1
    while s < m:
        s *= 2
    return s


def known(s: int, n: int, boundary: str) -> np.ndarray:
    k = np.arange(0, n, s, dtype=np.int64)
    if boundary == "clamped" and k[-1] != n - 1:
        k = np.append(k, n - 1)
    return k


def new_points(h: int, n: int, boundary: str) -> np.ndarray:
    a = known(h, n, boundary)
    b = known(2 * h, n, boundary)
    return np.setdiff1d(a, b, assume_unique=True)


@dataclass(frozen=True)
class Stencil:
    """Per-target stencil description along one axis (all arrays length m)."""

    targets: np.ndarray  # (m,) target positions
    slots: np.ndarray  # (4, m) positions of l2, l1, r1, r2 (missing -> l1 or r1)
    have: np.ndarray  # (4, m) bool availability of each slot
    pos: np.ndarray  # (4, m) float64 positions (for divided differences)
    w_lin: np.ndarray  # (4, m) weights
    w_qL: np.ndarray
    w_qR: np.ndarray
    w_cub: np.ndarray
    w_ext: np.ndarray  # one-sided extrapolation (only when no right neighbour)


def _lagrange(xs: list[np.ndarray], t: np.ndarray) -> list[np.ndarray]:
    out = []
    for i, xi in enumerate(xs):
        w = np.ones_like(t, dtype=np.float64)
        for j, xj in enumerate(xs):
            if i != j:
                w = w * (t - xj) / (xi - xj)
        out.append(w)
    return out


def stencil(targets: np.ndarray, anchors: np.ndarray) -> Stencil:
    m = targets.size
    a = anchors
    p = np.searchsorted(a, targets)  # index of first anchor > target (targets ∉ anchors)
    have = np.zeros((4, m), dtype=bool)
    slots = np.zeros((4, m), dtype=np.int64)
    # l1, l2
    have[1] = p - 1 >= 0
    have[0] = p - 2 >= 0
    have[2] = p < a.size
    have[3] = p + 1 < a.size
    l1 = a[np.clip(p - 1, 0, a.size - 1)]
    slots[1] = l1
    slots[0] = np.where(have[0], a[np.clip(p - 2, 0, a.size - 1)], l1)
    r1 = np.where(have[2], a[np.clip(p, 0, a.size - 1)], l1)
    slots[2] = r1
    slots[3] = np.where(have[3], a[np.clip(p + 1, 0, a.size - 1)], r1)
    pos = slots.astype(np.float64)
    t = targets.astype(np.float64)

    def weights(idx: list[int], ok: np.ndarray) -> np.ndarray:
        w = np.zeros((4, m), dtype=np.float64)
        xs = [np.where(ok, pos[i], pos[i] + 1.0 + k) for k, i in enumerate(idx)]
        for i, wi in zip(idx, _lagrange(xs, t), strict=True):
            w[i] = np.where(ok, wi, 0.0)
        return w

    ok_lin = have[1] & have[2]
    ok_qL = ok_lin & have[0]
    ok_qR = ok_lin & have[3]
    ok_cub = ok_qL & have[3]
    w_lin = weights([1, 2], ok_lin)
    w_qL = weights([0, 1, 2], ok_qL)
    w_qR = weights([1, 2, 3], ok_qR)
    w_cub = weights([0, 1, 2, 3], ok_cub)
    # Extrapolation (baseline boundary mode only): linear from l2, l1, else constant.
    w_ext = np.zeros((4, m), dtype=np.float64)
    no_r = ~have[2]
    lin_ext = no_r & have[0]
    xs = [np.where(lin_ext, pos[0], 0.0), np.where(lin_ext, pos[1], 1.0)]
    e0, e1 = _lagrange(xs, t)
    w_ext[0] = np.where(lin_ext, e0, 0.0)
    w_ext[1] = np.where(lin_ext, e1, np.where(no_r, 1.0, 0.0))
    return Stencil(targets, slots, have, pos, w_lin, w_qL, w_qR, w_cub, w_ext)


def _apply(w: np.ndarray, v: list[np.ndarray], shp: tuple[int, ...]) -> np.ndarray:
    # Fixed summation order l2, l1, r1, r2 (bit-exact reproducibility).
    acc = w[0].reshape(shp) * v[0]
    for i in range(1, 4):
        acc = acc + w[i].reshape(shp) * v[i]
    return acc


def _dd2(xa, xb, xc, fa, fb, fc):
    return ((fc - fb) / (xc - xb) - (fb - fa) / (xb - xa)) / (xc - xa)


def predict(
    st: Stencil, v: list[np.ndarray], axis: int, ndim: int, predictor: str, tau: float
) -> np.ndarray:
    """Predict targets from the gathered slot values `v` (4 arrays, target axis = `axis`).

    `tau` is the ENO tie tolerance on second divided differences.
    """
    shp = [1] * ndim
    shp[axis] = st.targets.size
    shp = tuple(shp)
    have = [h.reshape(shp) for h in st.have]
    ok_lin = have[1] & have[2]
    ok_qL = ok_lin & have[0]
    ok_qR = ok_lin & have[3]
    ok_cub = ok_qL & have[3]

    p_lin = _apply(st.w_lin, v, shp)
    p_ext = _apply(st.w_ext, v, shp)
    base = np.where(ok_lin, p_lin, p_ext)
    if predictor == "linear":
        return base
    p_qL = _apply(st.w_qL, v, shp)
    p_qR = _apply(st.w_qR, v, shp)
    p_cub = _apply(st.w_cub, v, shp)
    if predictor == "cubic":
        out = np.where(ok_qR, p_qR, base)
        out = np.where(ok_qL, p_qL, out)
        return np.where(ok_cub, p_cub, out)
    # eno
    x = [p.reshape(shp) for p in st.pos]
    with np.errstate(divide="ignore", invalid="ignore"):
        dL = np.abs(_dd2(x[0], x[1], x[2], v[0], v[1], v[2]))
        dR = np.abs(_dd2(x[1], x[2], x[3], v[1], v[2], v[3]))
    smooth = (dL <= ENO_RATIO * dR + tau) & (dR <= ENO_RATIO * dL + tau)
    one_sided = np.where(dL <= dR, p_qL, p_qR)
    out = np.where(ok_qR, p_qR, base)
    out = np.where(ok_qL, p_qL, out)
    return np.where(ok_cub, np.where(smooth, p_cub, one_sided), out)
