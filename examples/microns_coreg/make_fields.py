"""MICrONS in vivo stack <-> EM warp fields from the em_coregistration transform.

    make_fields.py OUT --forward phase3_staged_transform_solution.json \\
        --inverse inverse_phase3_staged_transform_solution.json \\
        --landmarks 17797_2Pfix_EMmoving_Phase3.csv [--spacing auto]

The em_coregistration forward solution T maps EM voxels (4 x 4 x 40 nm) to in
vivo stack coordinates (mm); the inverse solution maps back. Everything here is
in EM physical micrometres, axes (z, y, x), so that the fields line up with the
EM layers in Neuroglancer.

- Placement L (stack um -> EM um): the inverse of the least-squares affine
  fit of T over minnie65. Every moving layer (the stack, the 2P fields) is
  placed in EM space with L, so the fields only hold what is not affine.
- Forward field, on a grid over minnie65 + minnie35: u(x) = L(T(x)) - x. A
  viewer resamples a moving layer placed by L at x + u(x).
- Inverse field, on a grid over L(stack): u(v) = T^-1(L^-1 v) - v, for moving
  points (meshes, annotations) directly.
- OUT/microns_em_2p.zarr: both, in an OME-Zarr 0.6 scene (ome_scene.py).
- OUT/placement.json: L and the conventions, for building viewer states.

The landmarks only constrain minnie65; over minnie35 T extrapolates, which is
recorded in attributes.warpfield.support and reported separately.

The phase3 transform is not continuous: its ChunkedModel stages are independent
affines over slabs along x, so T jumps (median 7 um, up to 38 um) on 22
surfaces roughly perpendicular to EM z (the slabs are along the stack's x, which L maps to EM z). A sampled field turns each jump into a
ramp one grid cell wide; elsewhere 5 um sampling reproduces T to ~0.003 um
(median). Hence the default spacing of 5 um.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ants_mouse_templates"))

from coreg_transform import load_transform  # noqa: E402
from ome_field import write_field_store  # noqa: E402
from ome_scene import write as write_scene  # noqa: E402

# EM voxel (4, 4, 40 nm) -> um, per (x, y, z)
EM_VOXEL_UM = np.array([0.004, 0.004, 0.04])
# EM volume bounds in um (x, y, z), from the precomputed info files (8 x 8 x 40 nm)
MINNIE65 = np.array([[13824, 13824, 14816], [226816, 194048, 27904]]) * [0.008, 0.008, 0.04]
MINNIE35 = np.array([[13824, 13824, 7904], [226816, 194048, 14848]]) * [0.008, 0.008, 0.04]
# in vivo stack (x, y, z): 706 x 661 x 335 voxels of 2 um
STACK_UM = np.array([706, 661, 335]) * 2.0
OUTPUT_AXES = ("z", "y", "x")
EM_SPACE = "MICrONS EM, um"
STACK_SPACE = "MICrONS in vivo stack placed in EM, um"


class Coreg:
    """The em_coregistration transforms in micrometres.

    forward(p): EM output um (z, y, x) -> stack um (x, y, z).
    inverse(s): stack um (x, y, z) -> EM output um (z, y, x).
    `stack_offset` (um, x y z) is added to the stack coordinates of the
    landmark files to get the stack's viewer coordinates (half a voxel if the
    landmarks are voxel centres at n * 2 um).
    """

    def __init__(self, forward_json, inverse_json, stack_offset):
        self.T = load_transform(forward_json)
        self.Tinv = load_transform(inverse_json)
        self.offset = np.asarray(stack_offset, float)

    def forward(self, p_zyx: np.ndarray) -> np.ndarray:
        em_vox = p_zyx[:, ::-1] / EM_VOXEL_UM
        return self.T(em_vox) * 1000.0 + self.offset

    def inverse(self, s_xyz: np.ndarray) -> np.ndarray:
        em_vox = self.Tinv((s_xyz - self.offset) / 1000.0)
        return (em_vox * EM_VOXEL_UM)[:, ::-1]


def affine_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 least-squares affine dst ~ A [src, 1]."""
    X = np.c_[src, np.ones(len(src))]
    A = np.linalg.lstsq(X, dst, rcond=None)[0].T
    return np.vstack([A, [0, 0, 0, 1]])


def apply(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ A[:3, :3].T + A[:3, 3]


def box_points(lo_xyz, hi_xyz, n, rng) -> np.ndarray:
    """n random points (z, y, x) in an (x, y, z) box."""
    return (lo_xyz + (hi_xyz - lo_xyz) * rng.random((n, 3)))[:, ::-1]


def interpolation_error(f, h: float, origin, pts) -> np.ndarray:
    """|trilinear interpolation of f on a lattice of spacing h - f| at pts."""
    q = (pts - origin) / h
    q0 = np.floor(q)
    t = q - q0
    acc = np.zeros((len(pts), 3))
    for c in np.ndindex(2, 2, 2):
        c = np.asarray(c)
        w = np.prod(np.where(c, t, 1 - t), axis=1)
        acc += w[:, None] * f(origin + (q0 + c) * h)
    return np.linalg.norm(acc - f(pts), axis=1)


def grid_reader(origin, h, shape, fn):
    """read_slab(k0, k1) for write_field_store: fn at grid samples, (k, j, i, 3)."""
    _, nj, ni = shape

    def read_slab(k0, k1):
        k, j, i = np.meshgrid(np.arange(k0, k1), np.arange(nj), np.arange(ni), indexing="ij")
        p = origin + np.stack([k, j, i], -1).reshape(-1, 3) * h
        return fn(p).reshape(k1 - k0, nj, ni, 3).astype(np.float32)

    return read_slab


def lattice(lo_zyx, hi_zyx, h):
    origin = np.floor(np.asarray(lo_zyx) / h) * h
    shape = tuple(int(np.ceil((hi - o) / h)) + 1 for o, hi in zip(origin, hi_zyx, strict=True))
    return origin, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--inverse", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument(
        "--spacing",
        default="5",
        help="field spacing in um, or auto (the coarsest within --max-interp-error; "
        "never met near the steps of a ChunkedModel stage)",
    )
    ap.add_argument("--max-interp-error", type=float, default=0.25, help="um, for auto")
    ap.add_argument("--error-bound", type=float, default=0.1, help="codec bound, um")
    ap.add_argument("--margin", type=float, default=20.0, help="um around the EM")
    ap.add_argument(
        "--stack-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="um, x y z"
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    co = Coreg(a.forward, a.inverse, a.stack_offset)
    rng = np.random.default_rng(0)

    rows = [r for r in csv.reader(open(a.landmarks)) if r[1] == "TRUE"]
    lm_em = np.array([[float(r[i]) for i in (2, 3, 4)] for r in rows]) * EM_VOXEL_UM
    lm_em_zyx = lm_em[:, ::-1]
    lm_stack = np.array([[float(r[i]) for i in (5, 6, 7)] for r in rows]) * 1000 + co.offset
    res = np.linalg.norm(co.forward(lm_em_zyx) - lm_stack, axis=1)
    res_inv = np.linalg.norm(co.inverse(lm_stack) - lm_em_zyx, axis=1)
    rt = np.linalg.norm(co.inverse(co.forward(lm_em_zyx)) - lm_em_zyx, axis=1)
    report = {
        "landmarks": len(rows),
        "fit_residual_forward_um": dict(
            median=float(np.median(res)),
            p95=float(np.percentile(res, 95)),
            max=float(res.max()),
        ),
        "fit_residual_inverse_um": dict(
            median=float(np.median(res_inv)),
            p95=float(np.percentile(res_inv, 95)),
            max=float(res_inv.max()),
        ),
        "round_trip_forward_inverse_um": dict(
            median=float(np.median(rt)), p95=float(np.percentile(rt, 95)), max=float(rt.max())
        ),
    }
    print(json.dumps(report, indent=1), flush=True)

    # Placement L: inverse of the affine fit of T over minnie65.
    fit_pts = box_points(*MINNIE65, 50_000, rng)
    A = affine_fit(fit_pts, co.forward(fit_pts))  # EM (z, y, x) -> stack (x, y, z)
    L = np.linalg.inv(A)  # stack (x, y, z) -> EM (z, y, x)
    Linv = A

    def u_forward(p):
        return apply(L, co.forward(p)) - p

    def u_inverse(v):
        return co.inverse(apply(Linv, v)) - v

    # Spacing: the coarsest whose interpolation error stays within the target
    # inside minnie65.
    test65 = box_points(*MINNIE65, 20_000, rng)
    candidates = [40.0, 20.0, 10.0, 5.0, 2.5]
    if a.spacing == "auto":
        for h in candidates:
            e = interpolation_error(co.forward, h, np.zeros(3), test65)
            print(
                f"spacing {h:5.1f} um: interpolation error max {e.max():.3f} um, "
                f"p99 {np.percentile(e, 99):.3f} um",
                flush=True,
            )
            if e.max() <= a.max_interp_error:
                break
        spacing = h
    else:
        spacing = float(a.spacing)
    report["spacing_um"] = spacing

    # Forward field over minnie65 + minnie35.
    lo = np.minimum(MINNIE65[0], MINNIE35[0]) - a.margin
    hi = np.maximum(MINNIE65[1], MINNIE35[1]) + a.margin
    origin, shape = lattice(lo[::-1], hi[::-1], spacing)
    print(f"forward field: {shape} samples at {spacing} um", flush=True)
    support = {
        "description": "region constrained by the coregistration landmarks; "
        "elsewhere (minnie35) the transform extrapolates",
        "minnie65_z_um": [float(MINNIE65[0, 2]), float(MINNIE65[1, 2])],
        "landmark_bbox_zyx_um": [lm_em_zyx.min(0).tolist(), lm_em_zyx.max(0).tolist()],
    }
    idx_to_out = np.c_[np.eye(3) * spacing, origin]
    tmp = a.out / "fields"
    tmp.mkdir(exist_ok=True)
    det_fwd = write_field_store(
        tmp / "forward",
        shape,
        grid_reader(origin, spacing, shape, u_forward),
        idx_to_out,
        np.eye(3),
        a.error_bound,
        space=EM_SPACE,
        unit="micrometer",
        output_axes=OUTPUT_AXES,
        source=a.forward.name,
        extra_attributes={"support": support},
    )

    # Inverse field over L(stack).
    corners = np.array(list(np.ndindex(2, 2, 2))) * STACK_UM
    v = apply(L, corners)
    origin_i, shape_i = lattice(v.min(0) - a.margin, v.max(0) + a.margin, spacing)
    print(f"inverse field: {shape_i} samples at {spacing} um", flush=True)
    det_inv = write_field_store(
        tmp / "inverse",
        shape_i,
        grid_reader(origin_i, spacing, shape_i, u_inverse),
        np.c_[np.eye(3) * spacing, origin_i],
        np.eye(3),
        a.error_bound,
        space=STACK_SPACE,
        unit="micrometer",
        output_axes=OUTPUT_AXES,
        source=a.inverse.name,
    )
    report["det_jacobian_min"] = {"forward": det_fwd, "inverse": det_inv}

    scene = a.out / "microns_em_2p.zarr"
    write_scene(
        scene, tmp / "forward", tmp / "inverse", EM_SPACE, STACK_SPACE, unit="micrometer"
    )
    shutil.rmtree(tmp)

    # Decoded field vs the exact transform, inside minnie65 and over minnie35.
    report["decoded_vs_exact_um"] = check_decoded(
        scene / "coordinateTransformations" / "forward", origin, spacing, co, L, rng
    )
    placement = {
        "stack_to_em_um": L.tolist(),
        "em_to_stack_um": Linv.tolist(),
        "axes": {"stack": ["x", "y", "z"], "em": list(OUTPUT_AXES)},
        "stack_offset_um": list(co.offset),
        "spacing_um": spacing,
        "forward_field": "microns_em_2p.zarr/coordinateTransformations/forward",
        "inverse_field": "microns_em_2p.zarr/coordinateTransformations/inverse",
        "report": report,
    }
    (a.out / "placement.json").write_text(json.dumps(placement, indent=1))
    print(json.dumps(report, indent=1))


def check_decoded(field_path, origin, h, co, L, rng):
    """Percentiles of |L(T(x)) - (x + u_decoded(x))| at random points, by region.

    The phase3 transform is discontinuous (its ChunkedModel stages are
    independent affines along x): within one grid cell of those surfaces the
    field has a ramp where T has a step, which sets the max.
    """
    import zarr
    from scipy import ndimage

    import zarr_warpfield  # noqa: F401

    u = zarr.open_array(field_path / "s0", mode="r")[...]  # (3, k, j, i), small
    out = {}
    for name, box in (("minnie65", MINNIE65), ("minnie35", MINNIE35)):
        p = box_points(*box, 50_000, rng)
        idx = ((p - origin) / h).T
        ud = np.stack([ndimage.map_coordinates(u[c], idx, order=1) for c in range(3)], 1)
        e = np.linalg.norm(p + ud - apply(L, co.forward(p)), axis=1)
        out[name] = {f"p{q}": float(np.percentile(e, q)) for q in (50, 90, 99)} | {
            "max": float(e.max())
        }
        # how far T moves points there, beyond the affine placement
        out[name]["nonaffine_displacement_max_um"] = float(np.linalg.norm(ud, axis=1).max())
    return out


if __name__ == "__main__":
    main()
