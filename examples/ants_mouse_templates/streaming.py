"""Chunked access to ANTs displacement fields (NIfTI or warpfield-zarr) and
slab-by-slab warping, so nothing is ever loaded whole.

Conventions (ANTs / ITK):
- physical space is LPS, mm;
- a displacement field u stored as a vector NIfTI (intent "vector") holds LPS
  components; resampling a moving image M into the field's fixed space gives
  W(x) = M(x + u(x)); u is linearly interpolated, with nearest-neighbour
  extrapolation outside its grid; M is linearly interpolated and 0 outside.

Both field sources expose the same view: an array (Nk, Nj, Ni, 3) in C order of
the ITK index (k, j, i), components *index-aligned* (component c = displacement
in mm along +array axis c), with `vox_to_lps` (array axes -> LPS) and `R`
(index-aligned -> LPS, a signed permutation).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import zarr
from scipy import ndimage

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


class Grid:
    """Array axes (C order) -> LPS mm."""

    def __init__(self, shape, vox_to_lps):
        self.shape = tuple(int(s) for s in shape)
        A = np.asarray(vox_to_lps, float)
        self.A = A
        self.M = A[:3, :3]
        self.t = A[:3, 3]
        self.Minv = np.linalg.inv(self.M)

    def to_index(self, p):  # (n, 3) LPS -> (3, n) continuous array index
        return self.Minv @ (p - self.t).T

    def to_lps(self, idx):  # (3, n) -> (n, 3)
        return (self.M @ idx).T + self.t

    def axis_dirs(self):
        return self.M / np.linalg.norm(self.M, axis=0)


def signed_permutation(grid: Grid) -> np.ndarray:
    D = grid.axis_dirs()
    R = np.round(D)
    if not (
        np.allclose(D, R, atol=1e-6)
        and (np.abs(R).sum(0) == 1).all()
        and (np.abs(R).sum(1) == 1).all()
    ):
        raise ValueError(
            "grid is not axis-aligned; index-aligned components would not be exact"
        )
    return R


class Field:
    grid: Grid
    R: np.ndarray

    def read(self, k0, k1, j0=0, j1=None, i0=0, i1=None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def meta(self) -> dict:
        return {
            "vox_to_lps": self.grid.A.tolist(),
            "components_to_lps": self.R.tolist(),
            "units": "mm",
        }


class NiftiField(Field):
    """An ANTs vector NIfTI, read k-slab by k-slab (contiguous in the file)."""

    def __init__(self, path):
        self.img = nib.load(str(path), keep_file_open=True)
        sh = self.img.shape  # (Ni, Nj, Nk, 1, 3)
        assert len(sh) == 5 and sh[3] == 1 and sh[4] == 3, sh
        A = RAS_TO_LPS @ self.img.affine  # NIfTI index (i, j, k) -> LPS
        B = np.eye(4)
        B[:3, :3] = A[:3, :3][:, ::-1]  # array axes (k, j, i)
        B[:3, 3] = A[:3, 3]
        self.grid = Grid(sh[:3][::-1], B)
        self.R = signed_permutation(self.grid)
        # index-aligned component c = sum_l R[l, c] u_lps[l]: (source component, sign)
        self._perm = [
            (int(np.flatnonzero(self.R[:, c])[0]), float(self.R[:, c].sum())) for c in range(3)
        ]

    def read(self, k0, k1, j0=0, j1=None, i0=0, i1=None):
        Nk, Nj, Ni = self.grid.shape
        j1 = Nj if j1 is None else j1
        i1 = Ni if i1 is None else i1
        raw = np.asarray(self.img.dataobj[i0:i1, j0:j1, k0:k1, 0, :], dtype=np.float32)
        raw = raw.transpose(2, 1, 0, 3)  # (k, j, i, c_lps)
        out = np.empty(raw.shape, np.float32)
        for c, (l, s) in enumerate(self._perm):
            out[..., c] = raw[..., l] if s > 0 else -raw[..., l]
        return out


class ZarrField(Field):
    def __init__(self, path):
        self.z = zarr.open_array(str(path), mode="r")
        m = self.z.attrs["ants_displacement_field"]
        self.grid = Grid(self.z.shape[:3], m["vox_to_lps"])
        self.R = np.asarray(m["components_to_lps"], float)

    def read(self, k0, k1, j0=0, j1=None, i0=0, i1=None):
        return self.z[k0:k1, j0:j1, i0:i1, :]


def _region(idx: np.ndarray, shape, margin=1):
    lo = np.maximum(np.floor(idx.min(axis=1)).astype(int) - margin, 0)
    hi = np.minimum(np.ceil(idx.max(axis=1)).astype(int) + margin + 1, np.array(shape))
    return lo, np.maximum(hi, lo + 1)


def sample_field(field: Field, pts: np.ndarray, slab: int = 64) -> np.ndarray:
    """u (LPS, mm) at LPS points: linear inside, nearest-neighbour outside (ITK).

    Points are grouped by k-slab so each part of the field is read once.
    """
    g = field.grid
    idx = g.to_index(pts)
    for a in range(3):  # nearest-neighbour extrapolation == clamping the index
        np.clip(idx[a], 0, g.shape[a] - 1, out=idx[a])
    out = np.empty((pts.shape[0], 3))
    sid = np.minimum(idx[0] // slab, (g.shape[0] - 1) // slab).astype(int)
    for s in np.unique(sid):
        sel = np.flatnonzero(sid == s)
        sub = idx[:, sel]
        lo, hi = _region(sub, g.shape)
        reg = field.read(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        loc = sub - lo[:, None]
        u = np.stack(
            [
                ndimage.map_coordinates(reg[..., c], loc, order=1, mode="nearest")
                for c in range(3)
            ],
            axis=1,
        )
        out[sel] = u @ field.R.T
    return out


def sample_image(img: np.ndarray, grid: Grid, pts: np.ndarray) -> np.ndarray:
    """ITK linear interpolation of a (memory-mapped) image at LPS points; 0 outside."""
    idx = grid.to_index(pts)
    inside = np.all((idx >= -0.5) & (idx <= (np.array(grid.shape) - 0.5)[:, None]), axis=0)
    out = np.zeros(pts.shape[0], np.float32)
    if not inside.any():
        return out
    sub = idx[:, inside]
    lo, hi = _region(sub, grid.shape)
    reg = np.asarray(img[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], dtype=np.float32)
    out[inside] = ndimage.map_coordinates(reg, sub - lo[:, None], order=1, mode="nearest")
    return out


def slab_axis(fixed: Grid, field: Grid) -> int:
    """The fixed-grid axis most parallel to the field's k axis (sequential NIfTI reads)."""
    return int(np.argmax(np.abs(fixed.axis_dirs().T @ field.axis_dirs()[:, 0])))


def fixed_slabs(fixed: Grid, axis: int, step: int):
    """Yield (slice tuple into the fixed array, (n, 3) LPS points of its voxels)."""
    for a0 in range(0, fixed.shape[axis], step):
        a1 = min(a0 + step, fixed.shape[axis])
        rng = [np.arange(n) for n in fixed.shape]
        rng[axis] = np.arange(a0, a1)
        g = np.meshgrid(*rng, indexing="ij")
        idx = np.stack([x.ravel() for x in g]).astype(np.float64)
        sl = [slice(None)] * 3
        sl[axis] = slice(a0, a1)
        yield tuple(sl), tuple(len(r) for r in rng), fixed.to_lps(idx)


def warp_points(moving: np.ndarray, mgrid: Grid, field: Field, pts: np.ndarray) -> np.ndarray:
    return sample_image(moving, mgrid, pts + sample_field(field, pts))
