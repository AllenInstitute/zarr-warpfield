"""Write an ANTs displacement field as an OME-Zarr 0.6 multiscale image: the
field array of an RFC-5 `displacements` transformation, displayable as an image.

    ome_field.py transform.nii.gz forward/ [--error-bound 0.0025] [--output-space "mm RAS"]

Put the forward and inverse fields in a scene that declares the transformation
between the two template spaces with `ome_scene.py`.

Layout
------
forward/
  zarr.json      OME-Zarr 0.6 multiscale image, axes (c, k, j, i). Axis c has
                 "type": "displacement" (RFC-5): component c is the displacement,
                 in mm, along array axis c + 1 of the coordinate system
                 "intrinsic" (k, j, i in mm; "index-aligned", an exact signed
                 permutation of ANTs' LPS). An affine maps "intrinsic" to the
                 output space ("mm RAS", axes z, y, x: the coordinate system of
                 the Allen OME-Zarr templates).
                 attributes.warpfield: what OME-Zarr has no place for (below).
  s0/            float32 (3, Nk, Nj, Ni), chunks (3, 64, 64, 64), warpfield codec.
  s1/, s2/, ...  2x2x2 block means of the level below, same codec and error bound.
  bounds/        float32 (6, nbk, nbj, nbi): per block of 16^3 field samples
                 (plus the first sample of the next block), the min (rows 0-2)
                 and max (rows 3-5) over the samples of x + u(x), in output mm,
                 output axis order. Lets a viewer fetch the moving-image chunks
                 a region needs before (or without) decoding the field.
  jacobian/      (only with --jacobian-image) OME-Zarr 0.6 image: det(I + du/dx)
                 per sample of the decoded field, central differences. Not
                 needed for display: neuroglancer/field_shader.glsl computes
                 det J in the shader from neighbouring voxels.

attributes.warpfield (version 0.2):
  error_bound             the codec's error bound, in field units (mm)
  levels                  per multiscale level: interpolation_error, the max / p99
                          deviation from s0 when linearly interpolated, in field
                          units (sampled), so viewers can pick a coarse level safely
  bounds                  {path, block_shape}
  extrapolation           "nearest" outside the grid (as ANTs / ITK)
  det_jacobian_min, source

Resampling a moving image M (in the same output space) into the field's space:
W(x) = M(A(A^-1 x + u(A^-1 x))), A the affine "intrinsic" -> output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

from zarr_warpfield import WarpFieldCodec

# LPS (x, y, z) -> output (z, y, x) of "mm RAS" = (S, A, R)
LPS_TO_OUT = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]], float)
BLOCK = 16


def ome(
    name,
    intrinsic_axes,
    levels,
    index_to_output_mm,
    spacing,
    channel,
    space_name,
    unit="millimeter",
    output_axes=("z", "y", "x"),
):
    """OME-Zarr 0.6 multiscale; levels: [(path, factor)], factor 2**L per spatial axis.

    Level L sample n sits at the center of the 2**L level-0 samples it averages.
    """
    sp = [{"name": a, "type": "space", "unit": unit} for a in intrinsic_axes]
    out = [{"name": a, "type": "space", "unit": unit} for a in output_axes]
    # intrinsic (mm along the array axes) -> output: linear part / spacing
    lin = index_to_output_mm[:, :3] / np.asarray(spacing)[None, :]
    aff = np.zeros((3, 4))
    aff[:, :3] = lin
    aff[:, 3] = index_to_output_mm[:, 3]
    if channel:
        # RFC-5: the vector axis of a displacement field
        ch = [{"name": "c", "type": "displacement", "discrete": True}]
        sp, out = ch + sp, ch + out
        a4 = np.zeros((4, 5))
        a4[0, 0] = 1
        a4[1:, 1:] = aff
        aff = a4
    return {
        "version": "0.6",
        "multiscales": [
            {
                "name": name,
                "coordinateSystems": [
                    {"name": "intrinsic", "axes": sp},
                    {"name": space_name, "axes": out},
                ],
                "datasets": [
                    {
                        "path": path,
                        "coordinateTransformations": [
                            {
                                "type": "sequence",
                                "input": {"path": path},
                                "output": {"name": "intrinsic"},
                                "transformations": [
                                    {
                                        "type": "scale",
                                        "scale": ([1.0] if channel else [])
                                        + [float(s * f) for s in spacing],
                                    },
                                    {
                                        "type": "translation",
                                        "translation": ([0.0] if channel else [])
                                        + [float(s * (f - 1) / 2) for s in spacing],
                                    },
                                ],
                            }
                        ],
                    }
                    for path, f in levels
                ],
                "coordinateTransformations": [
                    {
                        "type": "affine",
                        "affine": aff.tolist(),
                        "input": {"name": "intrinsic"},
                        "output": {"name": space_name},
                    }
                ],
            }
        ],
    }


def det_slab(u, spacing):
    """det(I + du/dx) per sample, central differences; u (3, k, j, i) index-aligned."""
    g = [list(np.gradient(u[c], *spacing, axis=(0, 1, 2))) for c in range(3)]
    for a in range(3):
        g[a][a] += 1
    (a, b, c), (d, e, f), (h, i, j) = g
    return a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h)


def level_errors(root, n_samples=400_000, slab=64, seed=0):
    """Max and p99 error (field units) of each coarse level, linearly interpolated
    at random level-0 samples, against level 0 -- so a viewer can pick the coarsest
    level that is accurate enough for the resolution it renders at."""
    from scipy import ndimage

    s0 = root["s0"]
    N = np.array(s0.shape[1:])
    pts = np.floor(np.random.default_rng(seed).random((n_samples, 3)) * N).astype(int)
    ref = np.empty((n_samples, 3), np.float32)
    for k0 in range(0, N[0], slab):
        sel = np.flatnonzero((pts[:, 0] >= k0) & (pts[:, 0] < k0 + slab))
        if len(sel):
            ref[sel] = s0[:, k0 : k0 + slab][:, pts[sel, 0] - k0, pts[sel, 1], pts[sel, 2]].T
    errors = [{"max": 0.0, "p99": 0.0}]
    level = 1
    while f"s{level}" in root:
        a = root[f"s{level}"][:]  # coarse levels are small
        f = 2**level
        idx = (pts - (f - 1) / 2) / f
        for d in range(3):
            np.clip(idx[:, d], 0, a.shape[1 + d] - 1, out=idx[:, d])
        v = np.stack(
            [ndimage.map_coordinates(a[c], idx.T, order=1, mode="nearest") for c in range(3)], 1
        )
        e = np.abs(v - ref).max(1)
        errors.append({"max": float(e.max()), "p99": float(np.percentile(e, 99))})
        level += 1
    return errors


def write(
    nifti: Path,
    out: Path,
    error_bound: float,
    chunks=64,
    min_det=0.1,
    space="mm RAS",
    jacobian_image=False,
    min_level_extent=16,
):
    """An ANTs displacement field (NIfTI) as a field store."""
    from streaming import NiftiField  # needs nibabel; only this path does

    src = NiftiField(nifti)
    idx_to_out = np.zeros((3, 4))
    idx_to_out[:, :3] = LPS_TO_OUT @ src.grid.M
    idx_to_out[:, 3] = LPS_TO_OUT @ src.grid.t
    return write_field_store(
        out,
        src.grid.shape,
        src.read,
        idx_to_out,
        LPS_TO_OUT @ src.R,
        error_bound,
        chunks=chunks,
        min_det=min_det,
        space=space,
        jacobian_image=jacobian_image,
        min_level_extent=min_level_extent,
        source=nifti.name,
    )


def write_field_store(
    out: Path,
    shape,
    read_slab,
    idx_to_out,
    vec_to_out,
    error_bound: float,
    chunks=64,
    min_det=0.1,
    space="mm RAS",
    unit="millimeter",
    output_axes=("z", "y", "x"),
    jacobian_image=False,
    min_level_extent=16,
    source=None,
    extra_attributes=None,
):
    """Write a displacement field store (layout in the module docstring).

    shape: (Nk, Nj, Ni) samples. read_slab(k0, k1) -> (k1 - k0, Nj, Ni, 3)
    displacement components, index-aligned (component c along array axis c).
    idx_to_out: 3x4, array index (sample n at n) -> output coordinates, whose
    axes are `output_axes` in `unit`; vec_to_out: 3x3, components -> output
    (equal to idx_to_out's linear part divided by the spacing).
    """
    Nk, Nj, Ni = (int(n) for n in shape)
    idx_to_out = np.asarray(idx_to_out, float)
    vec_to_out = np.asarray(vec_to_out, float)
    spacing = [float(s) for s in np.linalg.norm(idx_to_out[:, :3], axis=0)]

    root = zarr.open_group(out, mode="w", zarr_format=3)
    codec = WarpFieldCodec(
        axis_roles=["component", "spatial", "spatial", "spatial"],
        error_bound=error_bound,
        jacobian={"grid_spacing": spacing, "min_det": min_det},
    )
    field = root.create_array(
        "s0",
        shape=(3, Nk, Nj, Ni),
        chunks=(3, chunks, chunks, chunks),
        dtype="float32",
        fill_value=0.0,
        serializer=codec,
        compressors=None,
        dimension_names=["c", "k", "j", "i"],
    )
    for k0 in range(0, Nk, chunks):
        slab = read_slab(k0, min(k0 + chunks, Nk))  # (k, j, i, 3)
        field[:, k0 : k0 + slab.shape[0]] = np.moveaxis(slab, -1, 0)

    nb = [-(-n // BLOCK) for n in (Nk, Nj, Ni)]
    bounds = root.create_array(
        "bounds",
        shape=(6, *nb),
        chunks=(6, 32, 32, 32),
        dtype="float32",
        fill_value=float("nan"),
        compressors=zarr.codecs.ZstdCodec(level=19),
        dimension_names=["minmax", "bk", "bj", "bi"],
    )
    jac = det = None
    if jacobian_image:
        jac = zarr.open_group(out / "jacobian", mode="w", zarr_format=3)
        det = jac.create_array(
            "s0",
            shape=(Nk, Nj, Ni),
            chunks=(chunks, chunks, chunks),
            dtype="float32",
            fill_value=1.0,
            compressors=zarr.codecs.ZstdCodec(level=19),
            dimension_names=["k", "j", "i"],
        )
    jj, ii = np.meshgrid(
        np.arange(Nj, dtype=np.float32), np.arange(Ni, dtype=np.float32), indexing="ij"
    )
    det_min = np.inf
    for k0 in range(0, Nk, BLOCK):  # slabs of the *decoded* field, one plane of overlap
        k1 = min(k0 + BLOCK, Nk)
        r0, r1 = max(k0 - 1, 0), min(k1 + 1, Nk)
        u = field[:, r0:r1]
        d = det_slab(u, spacing)[k0 - r0 : k0 - r0 + (k1 - k0)]
        det_min = min(det_min, float(d.min()))
        # display only: 1/1024 steps compress ~10x better than float32 noise
        if det is not None:
            det[k0:k1] = np.round(d * 1024) / 1024
        # x + u(x) in output mm, for the block bounds
        kk = np.arange(r0, r1, dtype=np.float32)[:, None, None]
        pos = np.empty((3, r1 - r0, Nj, Ni), np.float32)
        for o in range(3):
            pos[o] = (
                idx_to_out[o, 0] * kk
                + idx_to_out[o, 1] * jj
                + idx_to_out[o, 2] * ii
                + idx_to_out[o, 3]
                + sum(vec_to_out[o, c] * u[c] for c in range(3))
            )
        out_b = np.full((6, -(-(k1 - k0) // BLOCK), nb[1], nb[2]), np.nan, np.float32)
        for bk in range(out_b.shape[1]):
            ka, kb = k0 + bk * BLOCK, min(k0 + (bk + 1) * BLOCK, Nk - 1)
            sl = pos[:, ka - r0 : kb - r0 + 1]
            for bj in range(nb[1]):
                ja, jb = bj * BLOCK, min((bj + 1) * BLOCK, Nj - 1)
                for bi in range(nb[2]):
                    ia, ib = bi * BLOCK, min((bi + 1) * BLOCK, Ni - 1)
                    blk = sl[:, :, ja : jb + 1, ia : ib + 1].reshape(3, -1)
                    out_b[:3, bk, bj, bi] = blk.min(1)
                    out_b[3:, bk, bj, bi] = blk.max(1)
        bounds[:, k0 // BLOCK : k0 // BLOCK + out_b.shape[1]] = out_b

    # Coarser levels: 2x2x2 block means of the decoded level below, same absolute
    # error bound, so coarse views of a warp need only a few small field chunks.
    levels = [("s0", 1)]
    scales_meta = [{"path": "s0", "index_to_output": idx_to_out.tolist()}]
    prev, shape = field, (Nk, Nj, Ni)
    while min(shape) > 2 * min_level_extent:
        f = 2 ** len(levels)
        shape = tuple(-(-n // 2) for n in shape)
        lvl_spacing = [x * f for x in spacing]
        arr = root.create_array(
            f"s{len(levels)}",
            shape=(3, *shape),
            chunks=(3, chunks, chunks, chunks),
            dtype="float32",
            fill_value=0.0,
            serializer=WarpFieldCodec(
                axis_roles=["component", "spatial", "spatial", "spatial"],
                error_bound=error_bound,
                jacobian={"grid_spacing": lvl_spacing, "min_det": min_det},
            ),
            compressors=None,
            dimension_names=["c", "k", "j", "i"],
        )
        for k0 in range(0, shape[0], chunks):
            k1 = min(k0 + chunks, shape[0])
            # one whole output chunk row per write (lossy chunks are never
            # rewritten), filled 16 output planes at a time
            buf = np.empty((3, k1 - k0, *shape[1:]), np.float32)
            for s0 in range(k0, k1, BLOCK):
                s1 = min(s0 + BLOCK, k1)
                u = prev[:, 2 * s0 : 2 * s1]
                # replicate the last plane / row / column of odd extents
                pad = [(0, 0)] + [
                    (0, 2 * n - m)
                    for n, m in zip((s1 - s0, *shape[1:]), u.shape[1:], strict=True)
                ]
                u = np.pad(u, pad, mode="edge")
                buf[:, s0 - k0 : s1 - k0] = u.reshape(
                    3, s1 - s0, 2, shape[1], 2, shape[2], 2
                ).mean(axis=(2, 4, 6), dtype=np.float64)
            arr[:, k0:k1] = buf
        lvl = idx_to_out.copy()
        lvl[:, :3] *= f
        lvl[:, 3] = idx_to_out[:, 3] + idx_to_out[:, :3] @ np.full(3, (f - 1) / 2)
        levels.append((f"s{len(levels)}", f))
        scales_meta.append({"path": levels[-1][0], "index_to_output": lvl.tolist()})
        prev = arr

    for meta, err in zip(scales_meta, level_errors(root), strict=True):
        meta["interpolation_error"] = err
    root.attrs["ome"] = ome(
        out.name,
        ["k", "j", "i"],
        levels,
        idx_to_out,
        spacing,
        True,
        space,
        unit,
        output_axes,
    )
    # RFC-5: the vectors are expressed in the displacements transformation's
    # coordinate system, here "intrinsic", whose affine to the output is idx_to_out
    # divided by the spacing.
    if not np.allclose(vec_to_out, idx_to_out[:, :3] / np.asarray(spacing)[None, :]):
        raise ValueError("the field's vectors are not aligned with its array axes")
    root.attrs["warpfield"] = {
        "version": "0.2",
        "error_bound": error_bound,
        "levels": [
            {"path": m["path"], "interpolation_error": m["interpolation_error"]}
            for m in scales_meta
        ],
        "bounds": {"path": "bounds", "block_shape": [BLOCK] * 3},
        "extrapolation": "nearest",
        "det_jacobian_min": det_min,
        "source": source,
        **(extra_attributes or {}),
    }
    if jac is not None:
        jac.attrs["ome"] = ome(
            out.name + " det J",
            ["k", "j", "i"],
            [("s0", 1)],
            idx_to_out,
            spacing,
            False,
            space,
            unit,
            output_axes,
        )
    return det_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nifti", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--error-bound", type=float, default=0.0025)
    ap.add_argument("--output-space", default="mm RAS")
    ap.add_argument(
        "--jacobian-image",
        action="store_true",
        help="also write det J as an image (large; the field shader computes it on the fly)",
    )
    a = ap.parse_args()
    d = write(
        a.nifti,
        a.out,
        a.error_bound,
        space=a.output_space,
        jacobian_image=a.jacobian_image,
    )
    print(a.out, "min det J", d)


if __name__ == "__main__":
    main()
