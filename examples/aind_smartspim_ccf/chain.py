"""The SmartSPIM brain <-> STPT 2020 (CCF) transform chain, as point maps.

Two registrations are composed:

1. Brain -> SPIM LCA template (2024-05), from the AIND SmartSPIM CCF pipeline
   (aind-smartspim-ccf-registration 0.0.37, ANTs SyN; fixed = template,
   moving = the brain): `ls_to_template_SyN_0GenericAffine.mat` (A) and
   `ls_to_template_SyN_1Warp.nii.gz` / `_1InverseWarp.nii.gz` (W, IW), both on
   the template's 25 um grid. The rigid / affine .mat files only initialised
   this registration; A already contains them.
2. SPIM LCA template <-> STPT template 2020 (stereotaxic CCF frame): the
   displacement fields `transform-SPIM.nii.gz` (on the STPT grid, STPT point
   -> SPIM point) and `inverse-transform-SPIM.nii.gz` (on the SPIM grid, SPIM
   point -> STPT point).

ANTs conventions (physical space LPS, mm; vectors LPS): a registration with
fixed F and moving M maps a fixed point x to the moving point A(x + W(x)), and
a moving point y to the fixed point z + IW(z), z = A^-1(y). Fields are
linearly interpolated, with nearest-sample extrapolation outside their grid.

The brain frame. The pipeline registered level 3 of the fused OME-Zarr
(16 x 14.4 x 14.4 um, z y x) after reordering its axes into the template's
(right->left, anterior->posterior, superior->inferior): the acquisition is
z superior->inferior, y anterior->posterior, x left->right, so ITK index
(a, b, c) = (W3 - 1 - x3, y3, z3), and it took the template's origin and
direction (`prep_zarr_img.nii.gz` has exactly that header). Neuroglancer shows
the OME-Zarr in micrometres (z, y, x) with level-0 voxel j covering
[1.8 j, 1.8 (j + 1)); level-3 voxel n is the 8^3 block mean of level 0, centred
at 14.4 n + 7.2 (x, y) and 16 n + 8 (z).

All brain coordinates here are "brain um" (z, y, x) as Neuroglancer shows the
image; STPT coordinates are "STPT 2020 mm RAS" (z, y, x) = (S, A, R), the frame
of the Allen OME-Zarr templates.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io

# level 3 of the fused OME-Zarr: (x, y, z) shape and voxel size (um)
LEVEL3_SHAPE_XYZ = np.array([929, 1281, 486])
LEVEL3_UM_XYZ = np.array([14.4, 14.4, 16.0])
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


def nifti_header_affine(url: str) -> tuple[tuple[int, ...], np.ndarray]:
    """(shape, RAS affine) of a remote .nii.gz from its first 64 KB."""
    import io
    import urllib.request

    req = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
    raw = gzip.GzipFile(fileobj=io.BytesIO(urllib.request.urlopen(req).read())).read(352)
    h = nib.Nifti1Header.from_fileobj(io.BytesIO(raw))
    return tuple(int(n) for n in h.get_data_shape()), h.get_best_affine()


def unpack(gz: Path, cache: Path) -> Path:
    """Decompress a .nii.gz once (streamed) so it can be memory-mapped."""
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / gz.name.removesuffix(".gz")
    if not out.exists() or out.stat().st_mtime < gz.stat().st_mtime:
        tmp = out.with_suffix(".part")
        with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 << 20)
        tmp.rename(out)
    return out


def lps_affine(img) -> np.ndarray:
    """4x4 voxel index -> LPS mm of a NIfTI image."""
    a = np.eye(4)
    a[:3] = RAS_TO_LPS @ img.affine[:3]
    return a


class Field:
    """An ITK displacement field (.nii, vector intent), memory-mapped."""

    def __init__(self, path: Path, cache: Path):
        img = nib.load(unpack(path, cache), mmap=True)
        self.u = np.asanyarray(img.dataobj)  # (X, Y, Z, 1, 3), Fortran order, LPS
        if self.u.ndim != 5 or self.u.shape[3:] != (1, 3):
            raise ValueError(f"{path}: not a 3-D vector field, shape {self.u.shape}")
        self.shape = np.array(self.u.shape[:3])
        self.to_lps = lps_affine(img)
        self.to_index = np.linalg.inv(self.to_lps)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        """Displacement (LPS mm) at LPS points p (N, 3): trilinear, nearest
        sample outside the grid (as ITK)."""
        idx = p @ self.to_index[:3, :3].T + self.to_index[:3, 3]
        idx = np.clip(idx, 0, self.shape - 1)
        i0 = np.minimum(np.floor(idx).astype(np.int64), self.shape - 2)
        f = idx - i0
        out = np.zeros((len(p), 3))
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    v = self.u[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, 0, :]
                    out += (wx * wy * wz)[:, None] * v
        return out


class Affine:
    """An ITK AffineTransform .mat: x -> M (x - c) + c + t (LPS mm)."""

    def __init__(self, path: Path):
        m = scipy.io.loadmat(path)
        key = next(k for k in m if k.startswith("AffineTransform"))
        p = m[key].ravel().astype(float)
        self.M = p[:9].reshape(3, 3)
        self.t = p[9:12]
        self.c = m["fixed"].ravel().astype(float)
        self.Minv = np.linalg.inv(self.M)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.c) @ self.M.T + self.c + self.t

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (y - self.c - self.t) @ self.Minv.T + self.c


def lps_to_ras_zyx(p: np.ndarray) -> np.ndarray:
    """LPS (x, y, z) -> RAS (z, y, x) = (S, A, R)."""
    return np.stack([p[:, 2], -p[:, 1], -p[:, 0]], axis=1)


def ras_zyx_to_lps(q: np.ndarray) -> np.ndarray:
    return np.stack([-q[:, 2], -q[:, 1], q[:, 0]], axis=1)


class Chain:
    """brain um (z, y, x) <-> STPT 2020 mm RAS (z, y, x)."""

    def __init__(
        self,
        level3_affine: np.ndarray,  # 4x4 RAS affine of the reoriented level-3 grid (prep_zarr_img.nii.gz)
        affine: Path,  # ls_to_template_SyN_0GenericAffine.mat
        warp: Path,  # ls_to_template_SyN_1Warp.nii.gz (template -> brain)
        inverse_warp: Path,  # ls_to_template_SyN_1InverseWarp.nii.gz (brain -> template)
        stpt_to_spim: Path,  # transform-SPIM.nii.gz (on the STPT grid)
        spim_to_stpt: Path,  # inverse-transform-SPIM.nii.gz (on the SPIM grid)
        cache: Path,
    ):
        self.abc_to_lps = np.eye(4)
        self.abc_to_lps[:3] = RAS_TO_LPS @ np.asarray(level3_affine, float)[:3]
        self.lps_to_abc = np.linalg.inv(self.abc_to_lps)
        self.A = Affine(affine)
        self.W = Field(warp, cache) if warp else None
        self.IW = Field(inverse_warp, cache) if inverse_warp else None
        self.U = Field(stpt_to_spim, cache) if stpt_to_spim else None
        self.V = Field(spim_to_stpt, cache) if spim_to_stpt else None

    # brain um (z, y, x) <-> reoriented ITK index (a, b, c) <-> LPS
    def brain_to_lps(self, d: np.ndarray) -> np.ndarray:
        xyz3 = d[:, ::-1] / LEVEL3_UM_XYZ - 0.5  # continuous level-3 index, voxel centres
        abc = np.stack([LEVEL3_SHAPE_XYZ[0] - 1 - xyz3[:, 0], xyz3[:, 1], xyz3[:, 2]], axis=1)
        return abc @ self.abc_to_lps[:3, :3].T + self.abc_to_lps[:3, 3]

    def lps_to_brain(self, p: np.ndarray) -> np.ndarray:
        abc = p @ self.lps_to_abc[:3, :3].T + self.lps_to_abc[:3, 3]
        xyz3 = np.stack([LEVEL3_SHAPE_XYZ[0] - 1 - abc[:, 0], abc[:, 1], abc[:, 2]], axis=1)
        return ((xyz3 + 0.5) * LEVEL3_UM_XYZ)[:, ::-1]

    # the two registrations, in LPS
    def brain_lps_to_template(self, y: np.ndarray) -> np.ndarray:
        z = self.A.inverse(y)
        return z + self.IW(z)

    def template_to_brain_lps(self, x: np.ndarray) -> np.ndarray:
        return self.A(x + self.W(x))

    def template_to_stpt(self, t: np.ndarray) -> np.ndarray:
        return t + self.V(t)

    def stpt_to_template(self, s: np.ndarray) -> np.ndarray:
        return s + self.U(s)

    # composites
    def brain_to_template_lps(self, d: np.ndarray) -> np.ndarray:
        return self.brain_lps_to_template(self.brain_to_lps(d))

    def brain_to_spim(self, d: np.ndarray) -> np.ndarray:
        """brain um (z, y, x) -> SPIM template mm RAS (z, y, x)."""
        return lps_to_ras_zyx(self.brain_to_template_lps(d))

    def spim_to_brain(self, q: np.ndarray) -> np.ndarray:
        """SPIM template mm RAS (z, y, x) -> brain um (z, y, x)."""
        return self.lps_to_brain(self.template_to_brain_lps(ras_zyx_to_lps(q)))

    def stpt_to_spim(self, q: np.ndarray) -> np.ndarray:
        """STPT 2020 mm RAS (z, y, x) -> SPIM template mm RAS (z, y, x)."""
        return lps_to_ras_zyx(self.stpt_to_template(ras_zyx_to_lps(q)))

    def spim_to_stpt(self, q: np.ndarray) -> np.ndarray:
        """SPIM template mm RAS (z, y, x) -> STPT 2020 mm RAS (z, y, x)."""
        return lps_to_ras_zyx(self.template_to_stpt(ras_zyx_to_lps(q)))

    def pull(self, d: np.ndarray) -> np.ndarray:
        """brain um (z, y, x) -> STPT 2020 mm RAS (z, y, x)."""
        return lps_to_ras_zyx(self.template_to_stpt(self.brain_to_template_lps(d)))

    def push(self, q: np.ndarray) -> np.ndarray:
        """STPT 2020 mm RAS (z, y, x) -> brain um (z, y, x)."""
        t = self.stpt_to_template(ras_zyx_to_lps(q))
        return self.lps_to_brain(self.template_to_brain_lps(t))


def in_chunks(fn, pts: np.ndarray, n: int = 1 << 21) -> np.ndarray:
    return np.concatenate([fn(pts[i : i + n]) for i in range(0, len(pts), n)]) if len(pts) else pts
