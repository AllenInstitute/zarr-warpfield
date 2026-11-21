"""Convert an ANTs displacement field (vector NIfTI) to a warpfield-compressed
zarr v3 array and back, slab by slab: peak memory is one slab of 64 planes.

Storage layout
--------------
- Array axes are the field grid in C order, (k, j, i) of the ITK/NIfTI index,
  plus a trailing component axis: shape (Nk, Nj, Ni, 3), chunks 64^3 x 3.
- Components are *index-aligned*: component c is the displacement, in mm, along
  the positive direction of array axis c. For axis-aligned grids (direction a
  signed permutation, as here) this is an exact relabelling of ANTs' LPS
  components, so the conversion itself is lossless, and the codec's Jacobian
  check sees the true Jacobian.
- `attributes["ants_displacement_field"]`: `vox_to_lps` (4x4, array axes ->
  LPS mm) and `components_to_lps` (3x3), enough to rebuild the NIfTI exactly.

    convert.py to-zarr  transform.nii.gz transform.zarr [--error-bound 0.0025] [--chunks 64]
    convert.py to-nifti transform.zarr transform.nii      (uncompressed; gzip it if you like)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr
from streaming import RAS_TO_LPS, NiftiField, ZarrField

from zarr_warpfield import WarpFieldCodec

APPLIES = (
    "ANTs/ITK displacement field: resampling a moving image M into this field's space gives "
    "W(x) = M(x + u(x)) at every physical (LPS, mm) point x of the fixed space; u is linearly "
    "interpolated on this grid, with nearest-neighbour extrapolation outside it."
)


def to_zarr(
    nifti: Path, out: Path, error_bound: float | None, chunks: int = 64, min_det: float = 0.1
) -> zarr.Array:
    """error_bound None writes a lossless zstd-19 array instead."""
    src = NiftiField(nifti)
    spacing = [float(s) for s in np.linalg.norm(src.grid.M, axis=0)]
    kw = dict(
        shape=(*src.grid.shape, 3),
        chunks=(chunks, chunks, chunks, 3),
        dtype="float32",
        fill_value=0.0,
        overwrite=True,
    )
    if error_bound is None:
        z = zarr.create_array(out, compressors=zarr.codecs.ZstdCodec(level=19), **kw)
    else:
        codec = WarpFieldCodec(
            axis_roles=["spatial", "spatial", "spatial", "component"],
            error_bound=error_bound,
            jacobian={"grid_spacing": spacing, "min_det": min_det},
        )
        z = zarr.create_array(out, serializer=codec, compressors=None, **kw)
    z.attrs["ants_displacement_field"] = src.meta | {"applies": APPLIES, "source": nifti.name}
    for k0 in range(0, src.grid.shape[0], chunks):
        slab = src.read(k0, min(k0 + chunks, src.grid.shape[0]))
        z[k0 : k0 + slab.shape[0]] = slab
    return z


def to_nifti(store: Path, out: Path, chunks: int = 64):
    """Write an ANTs-readable vector NIfTI (uncompressed, LPS components)."""
    src = ZarrField(store)
    Nk, Nj, Ni = src.grid.shape
    A = np.eye(4)  # NIfTI index (i, j, k) -> LPS
    A[:3, :3] = src.grid.M[:, ::-1]
    A[:3, 3] = src.grid.t
    ras = np.linalg.inv(RAS_TO_LPS) @ A
    hdr = nib.Nifti1Header()
    hdr.set_data_shape((Ni, Nj, Nk, 1, 3))
    hdr.set_data_dtype(np.float32)
    hdr.set_qform(ras, code=1)
    hdr.set_sform(ras, code=1)
    hdr.set_intent("vector")
    hdr.set_xyzt_units("mm", "sec")
    with open(out, "wb") as f:
        hdr.write_to(f)
    off = int(hdr["vox_offset"])
    data = np.memmap(
        out, dtype=np.float32, mode="r+", offset=off, shape=(Ni, Nj, Nk, 1, 3), order="F"
    )
    perm = [(int(np.flatnonzero(src.R[:, c])[0]), float(src.R[:, c].sum())) for c in range(3)]
    for k0 in range(0, Nk, chunks):
        k1 = min(k0 + chunks, Nk)
        slab = src.read(k0, k1)  # (k, j, i, c) index-aligned
        for c, (l, s) in enumerate(perm):
            data[:, :, k0:k1, 0, l] = (slab[..., c] * s).transpose(2, 1, 0)
    data.flush()
    del data


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-zarr")
    a.add_argument("nifti", type=Path)
    a.add_argument("zarr", type=Path)
    a.add_argument("--error-bound", type=float, default=0.0025, help="mm; 0 = lossless zstd")
    a.add_argument("--chunks", type=int, default=64)
    b = sub.add_parser("to-nifti")
    b.add_argument("zarr", type=Path)
    b.add_argument("nifti", type=Path)
    args = ap.parse_args()
    if args.cmd == "to-zarr":
        to_zarr(args.nifti, args.zarr, args.error_bound or None, args.chunks)
    else:
        to_nifti(args.zarr, args.nifti)


if __name__ == "__main__":
    main()
