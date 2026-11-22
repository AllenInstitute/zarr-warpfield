"""Evaluate an em_coregistration transform (AllenInstitute/em_coregistration).

The MICrONS EM <-> in vivo coregistration is a staged `TransformList` of
`PolynomialModel` (order 1-3), `ChunkedModel` (piecewise polynomial along one
axis) and `SplineModel` (affine + |r| radial kernels at control points: the 3-D
biharmonic spline). This reimplements their `tform` exactly, vectorized, with
the radial terms computed in blocks of points so memory stays bounded for
millions of points.

    T = load_transform("phase3_staged_transform_solution.json")
    dst = T(src)          # src, dst: (n, 3)

In the phase3 forward solution, src is EM voxels at 4 x 4 x 40 nm and dst is
in vivo stack coordinates in mm.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

BLOCK = 1024  # points per block of radial-kernel evaluation


def polynomial_kernel(src: np.ndarray, order: int) -> np.ndarray:
    """[1, x, y, z, then the order-2, order-3 monomials] in em_coregistration order."""
    cols = [np.ones(len(src))]
    for o in range(1, order + 1):
        for combo in itertools.combinations_with_replacement(range(3), o):
            c = np.ones(len(src))
            for i in combo:
                c = c * src[:, i]
            cols.append(c)
    return np.stack(cols, axis=1)


class Polynomial:
    def __init__(self, j: dict):
        self.order = j["order"]
        self.p = np.asarray(j["parameters"], float)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        return polynomial_kernel(src, self.order) @ self.p


class Chunked:
    def __init__(self, j: dict):
        self.axis = j["axis"]
        self.ranges = np.asarray(j["ranges"], float)
        self.parts = [Polynomial(t) for t in j["transforms"]]

    def __call__(self, src: np.ndarray) -> np.ndarray:
        which = np.searchsorted(self.ranges, src[:, self.axis])
        dst = np.zeros_like(src)
        for i, part in enumerate(self.parts):
            sel = which == i
            if sel.any():
                dst[sel] = part(src[sel])
        return dst


class Spline:
    def __init__(self, j: dict):
        self.cp = np.asarray(j["control_pts"], float)
        p = np.asarray(j["parameters"], float)
        self.affine = p[:4]  # offset, then x, y, z rows
        self.radial = p[4:]

    def __call__(self, src: np.ndarray) -> np.ndarray:
        dst = self.affine[0] + src @ self.affine[1:4]
        for s in range(0, len(src), BLOCK):
            b = src[s : s + BLOCK]
            # Differences, not the expanded square: coordinates are large (EM
            # voxels ~ 1e5) and the expansion would lose precision.
            r = np.sqrt(sum((b[:, a, None] - self.cp[None, :, a]) ** 2 for a in range(3)))
            dst[s : s + BLOCK] += r @ self.radial
        return dst


STAGES = {"PolynomialModel": Polynomial, "ChunkedModel": Chunked, "SplineModel": Spline}


class TransformList:
    def __init__(self, j: dict):
        if j["name"] != "TransformList":
            raise ValueError(f"expected a TransformList, got {j['name']}")
        self.stages = [STAGES[t["name"]](t) for t in j["transforms"]]

    def __call__(self, src: np.ndarray) -> np.ndarray:
        dst = np.asarray(src, float)
        for stage in self.stages:
            dst = stage(dst)
        return dst


def load_transform(path: str | Path) -> TransformList:
    return TransformList(json.loads(Path(path).read_text()))
