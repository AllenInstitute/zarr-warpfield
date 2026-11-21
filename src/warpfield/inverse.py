"""Inverting a decoded displacement field, and round-trip error measurement.

Conventions: `u` has shape (*P, k) over k spatial axes (no stack axis), with
component c the displacement along axis c in image voxels; field sample i
sits at image coordinate i * h. The forward map is phi(p) = p + U(p), where U
is the multilinear interpolant of u. Multilinear interpolation uses convex
weights, so |U - U_hat| <= max |u - u_hat| <= error_bound everywhere, not just
at samples.

Requires scipy.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def interp(u: np.ndarray, p: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Multilinear interpolation of u (*P, k) at image coordinates p (..., k)."""
    k = u.shape[-1]
    idx = [p[..., a] / h[a] for a in range(p.shape[-1])]
    return np.stack(
        [ndimage.map_coordinates(u[..., c], idx, order=1, mode="nearest") for c in range(k)],
        axis=-1,
    )


def grid_points(shape: tuple[int, ...], h: np.ndarray) -> np.ndarray:
    g = np.meshgrid(
        *[np.arange(n, dtype=np.float64) * h[i] for i, n in enumerate(shape)], indexing="ij"
    )
    return np.stack(g, axis=-1)


def jacobian_at(u: np.ndarray, p: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Exact Jacobian dU_c/dp_b of the multilinear interpolant at points p, (..., k, k).

    Along axis b the interpolant is linear inside a cell, so its derivative is
    the forward difference at floor(p_b), interpolated linearly along the
    other axes.
    """
    k = u.shape[-1]
    shape = u.shape[:-1]
    idx = [p[..., a] / h[a] for a in range(k)]
    J = np.empty((*p.shape[:-1], k, k))
    for b in range(k):
        D = np.diff(u.astype(np.float64), axis=b) / h[b]
        D = np.concatenate([D, D.take([-1], axis=b)], axis=b)  # pad to the grid shape
        cell = np.clip(np.floor(idx[b]), 0, shape[b] - 2)
        coords = [idx[a] if a != b else cell for a in range(k)]
        for c in range(k):
            J[..., c, b] = ndimage.map_coordinates(D[..., c], coords, order=1, mode="nearest")
    return J


def invert(
    u: np.ndarray, h, iters: int = 60, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement v of the inverse map psi(y) = y + v(y), on the same grid.

    Damped Newton on phi(y + v) = y using the exact piecewise Jacobian of the
    multilinear interpolant. Returns (v, residual norm per sample); the
    residual stays large only where no inverse exists (gaps, folds).
    """
    h = np.asarray(h, dtype=np.float64)
    k = u.shape[-1]
    y = grid_points(u.shape[:-1], h)
    v = -u.astype(np.float64)
    eye = np.eye(k)
    for _ in range(iters):
        p = y + v
        r = v + interp(u, p, h)  # phi(y+v) - y
        if np.abs(r).max() < tol:
            break
        J = eye + jacobian_at(u, p, h)
        det = np.linalg.det(J)
        good = det > 0.05
        step = np.where(
            good[..., None],
            np.linalg.solve(np.where(good[..., None, None], J, eye), r[..., None])[..., 0],
            0.5 * r,
        )
        # limit each step to one grid cell so a bad step cannot jump across a crack
        lim = h.min()
        step = np.clip(step, -lim, lim)
        v = v - step
    res = np.linalg.norm(v + interp(u, y + v, h), axis=-1)
    return v.astype(np.float32), res


def roundtrip(u_fwd: np.ndarray, v_inv: np.ndarray, h) -> np.ndarray:
    """psi then phi at every grid point y: returns |phi(psi(y)) - y| (max over components)."""
    h = np.asarray(h, dtype=np.float64)
    y = grid_points(u_fwd.shape[:-1], h)
    p = y + v_inv
    return np.abs(v_inv + interp(u_fwd, p, h)).max(axis=-1)


def roundtrip_fg(u_fwd: np.ndarray, v_inv: np.ndarray, h) -> np.ndarray:
    """phi then psi at every grid point x: |psi(phi(x)) - x| (max over components)."""
    h = np.asarray(h, dtype=np.float64)
    x = grid_points(u_fwd.shape[:-1], h)
    p = x + u_fwd
    return np.abs(u_fwd + interp(v_inv, p, h)).max(axis=-1)


def lipschitz_inf(v: np.ndarray, h) -> float:
    """Infinity-norm Lipschitz constant of psi(y) = y + V(y) (multilinear V).

    max over components c of sum over axes b of max |d psi_c / d p_b|, using
    the cell differences that bound a multilinear interpolant.
    """
    h = np.asarray(h, dtype=np.float64)
    k = v.shape[-1]
    worst = 0.0
    for c in range(k):
        s = 0.0
        for b in range(k):
            d = np.diff(v[..., c].astype(np.float64), axis=b) / h[b]
            s += np.abs(d + (1.0 if b == c else 0.0)).max()
        worst = max(worst, s)
    return worst
