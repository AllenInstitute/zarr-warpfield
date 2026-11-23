"""SmartSPIM brain <-> STPT 2020 (CCF) warp fields from two composed registrations.

    make_fields.py OUT --src SRC --fields ~/Downloads [--spacing 25] [--error-bound 1]

See `chain.py` for the two registrations and their conventions. Everything is
in "brain um", axes (z, y, x), the frame Neuroglancer shows the fused OME-Zarr
channels in, so the fields line up with them.

- Placement L (STPT 2020 mm RAS (z, y, x) -> brain um): the least-squares
  affine fit of the composed transform over the brain. Every layer in STPT
  space (CCF annotation, templates, meshes) is placed in brain space with L, so
  the fields only hold what is not affine.
- Forward field, on a lattice over the brain: u(d) = L(pull(d)) - d. A viewer
  resamples a layer placed by L at d + u(d): CCF data in the brain's space.
- Inverse field, on a lattice over L(STPT template): u(v) = push(L^-1 v) - v.
  Moves points (meshes, annotations) directly, and resamples the brain into
  "STPT placed in brain" space.
- OUT/smartspim_894847_ccf.zarr: both, in an OME-Zarr 0.6 scene (ome_scene.py).
- The SPIM LCA template gets its own placement L_spim (SPIM mm RAS -> brain
  um) and two more scenes: OUT/smartspim_894847_spim.zarr (brain <-> SPIM
  template: the pipeline's registration alone) and OUT/stpt_spim_placed.zarr
  (STPT placed by L <-> SPIM placed by L_spim: the template fields alone), so
  the SPIM template can be shown, warped, in brain space and in CCF space.
- OUT/placement.json: L, its inverse, conventions and the accuracy report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ants_mouse_templates"))
sys.path.insert(0, str(HERE.parent / "microns_coreg"))

from chain import (  # noqa: E402
    LEVEL3_SHAPE_XYZ,
    LEVEL3_UM_XYZ,
    Chain,
    in_chunks,
    lps_affine,
    nifti_header_affine,
)
from make_fields import affine_fit, apply, grid_reader, lattice  # noqa: E402
from ome_field import write_field_store  # noqa: E402
from ome_scene import write as write_scene  # noqa: E402

ALIGN = "image_atlas_alignment/Ex_639_Em_667"
BUCKET = (
    "https://aind-open-data.s3.us-west-2.amazonaws.com/"
    "894847_2026-11-20_11-27-00_stitched_2026-11-21_13-35-21/"
)
BRAIN_SPACE = "SmartSPIM 894847, um"
STPT_SPACE = "STPT template 2020 placed in SmartSPIM 894847, um"
OUTPUT_AXES = ("z", "y", "x")
# STPT template 2020, 25 um: (z, y, x) RAS extent in mm of its voxel centres
# (template_25.nii.gz: 528 x 320 x 456, A = -0.025 i + 5.35, S = -0.025 j + 5.22,
# R = 0.025 k - 5.695)
STPT_RAS_ZYX = np.array([[5.22 - 0.025 * 319, 5.35 - 0.025 * 527, -5.695], [5.22, 5.35, -5.695 + 0.025 * 455]])
BRAIN_UM_ZYX = (LEVEL3_SHAPE_XYZ * LEVEL3_UM_XYZ)[::-1]
# SPIM LCA template 2024-05, 25 um: (z, y, x) RAS extent in mm of its voxel centres
# (template_25.nii.gz: 576 x 648 x 440, R = -0.025 i + 1.51139, A = -0.025 j + 1.5,
# S = -0.025 k + 1.5)
SPIM_RAS_ZYX = np.array([[1.5 - 0.025 * 439, 1.5 - 0.025 * 647, 1.51139 - 0.025 * 575], [1.5, 1.5, 1.51139]])
SPIM_SPACE = "SPIM LCA template 2024-05 placed in SmartSPIM 894847, um"


def brain_points(src: Path, stride: int) -> np.ndarray:
    """Brain um (z, y, x) of labelled voxels of the pipeline's CCF-on-brain
    volume (25 um reoriented grid): where the fit and the checks look."""
    img = nib.load(src / ALIGN / "metadata/registration_metadata/moved_ccf_anno_to_ls.nii.gz")
    lab = np.asarray(img.dataobj[::stride, ::stride, ::stride]) > 0
    abc = np.argwhere(lab) * stride
    return abc @ lps_affine(img)[:3, :3].T + lps_affine(img)[:3, 3]  # LPS


def box_corners(box_zyx: np.ndarray) -> np.ndarray:
    return np.array([[box_zyx[c[i], i] for i in range(3)] for c in np.ndindex(2, 2, 2)])


def build_scene(out, name, h, margin, error_bound, fwd, fwd_domain, inv, inv_domain, spaces, sources, rng, pts):
    """One scene: forward field u(p) = fwd(p) - p on a lattice over `fwd_domain`
    (corner points, um), inverse field u(q) = inv(q) - q over `inv_domain`;
    decoded vs exact at `pts` (forward) and fwd(pts) (inverse)."""
    tmp = out / "fields"
    tmp.mkdir(exist_ok=True)
    report, lat = {}, {}
    for key, f, dom, space, src in (
        ("forward", fwd, fwd_domain, spaces[0], sources[0]),
        ("inverse", inv, inv_domain, spaces[1], sources[1]),
    ):
        origin, shape = lattice(dom.min(0) - margin, dom.max(0) + margin, h)
        lat[key] = origin
        print(f"{name} {key} field: {shape} samples at {h} um", flush=True)
        report[f"{key}_det_jacobian_min"] = write_field_store(
            tmp / key, shape, grid_reader(origin, h, shape, lambda p, f=f: in_chunks(lambda q: f(q) - q, p)),
            np.c_[np.eye(3) * h, origin], np.eye(3), error_bound,
            space=space, unit="micrometer", output_axes=OUTPUT_AXES, source=src,
        )
    scene = out / f"{name}.zarr"
    write_scene(scene, tmp / "forward", tmp / "inverse", spaces[0], spaces[1], unit="micrometer")
    shutil.rmtree(tmp)
    report["forward_decoded_vs_exact_um"] = check(scene, "forward", lat["forward"], h, pts, fwd)
    report["inverse_decoded_vs_exact_um"] = check(scene, "inverse", lat["inverse"], h, in_chunks(fwd, pts), inv)
    return report


def fit_placement(target_pts, brain_pts):
    """Affine A with brain ~ A(target), and its residual (um)."""
    A = affine_fit(target_pts, brain_pts)
    res = np.linalg.norm(apply(A, target_pts) - brain_pts, axis=1)
    return A, {"median": float(np.median(res)), "p99": float(np.percentile(res, 99)), "max": float(res.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, required=True, help="local copy of the dataset folder")
    ap.add_argument("--fields", type=Path, default=Path.home() / "Downloads",
                    help="folder with transform-SPIM.nii.gz and inverse-transform-SPIM.nii.gz")
    ap.add_argument("--cache", type=Path, help="where to unpack the NIfTI fields (default OUT/raw)")
    ap.add_argument("--spacing", type=float, default=25.0, help="field lattice spacing, um")
    ap.add_argument("--error-bound", type=float, default=1.0, help="codec error bound, um")
    ap.add_argument("--margin", type=float, default=250.0, help="um around the domains")
    ap.add_argument("--scenes", nargs="+", default=["ccf", "spim", "stpt_spim"],
                    choices=["ccf", "spim", "stpt_spim"],
                    help="which scenes to (re)build; the others' placements are read from placement.json")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    al = a.src / ALIGN
    shape3, level3 = nifti_header_affine(BUCKET + ALIGN + "/metadata/registration_metadata/prep_zarr_img.nii.gz")
    assert shape3[:3] == tuple(LEVEL3_SHAPE_XYZ), shape3
    ch = Chain(
        level3,
        al / "ls_to_template_SyN_0GenericAffine.mat",
        al / "ls_to_template_SyN_1Warp.nii.gz",
        al / "ls_to_template_SyN_1InverseWarp.nii.gz",
        a.fields / "transform-SPIM.nii.gz",
        a.fields / "inverse-transform-SPIM.nii.gz",
        a.cache or a.out / "raw",
    )
    ppath = a.out / "placement.json"
    placement = json.loads(ppath.read_text()) if ppath.exists() else {"report": {}}
    report = placement.setdefault("report", {})
    h = a.spacing

    # where the fit and the checks look: the brain
    d_fit = ch.lps_to_brain(brain_points(a.src, 4))
    pts = d_fit[rng.choice(len(d_fit), min(100_000, len(d_fit)), replace=False)]
    pts = pts + rng.uniform(-h / 2, h / 2, pts.shape)
    brain_box = np.array([np.zeros(3), BRAIN_UM_ZYX])

    # Placement L: STPT mm RAS (z, y, x) -> brain um; L_spim: SPIM template mm RAS -> brain um
    if "ccf" in a.scenes or "stpt_spim" in a.scenes and "stpt_ras_mm_to_brain_um" not in placement:
        s_fit = in_chunks(ch.pull, d_fit)
        L, report["placement_fit_residual_um"] = fit_placement(s_fit, d_fit)
        back = in_chunks(ch.push, s_fit[:: max(1, len(s_fit) // 200_000)])
        rt = np.linalg.norm(back - d_fit[:: max(1, len(s_fit) // 200_000)], axis=1)
        report["push_of_pull_vs_identity_um"] = {
            "median": float(np.median(rt)), "p99": float(np.percentile(rt, 99)), "max": float(rt.max())
        }
    else:
        L = np.asarray(placement["stpt_ras_mm_to_brain_um"])
    Linv = np.linalg.inv(L)
    if "spim" in a.scenes or "stpt_spim" in a.scenes and "spim_ras_mm_to_brain_um" not in placement:
        Ls, report["spim_placement_fit_residual_um"] = fit_placement(in_chunks(ch.brain_to_spim, d_fit), d_fit)
    else:
        Ls = np.asarray(placement["spim_ras_mm_to_brain_um"])
    Lsinv = np.linalg.inv(Ls)
    print(json.dumps(report, indent=1), flush=True)

    stpt_placed = apply(L, box_corners(STPT_RAS_ZYX))
    spim_placed = apply(Ls, box_corners(SPIM_RAS_ZYX))
    if "ccf" in a.scenes:
        report["ccf"] = build_scene(
            a.out, "smartspim_894847_ccf", h, a.margin, a.error_bound,
            lambda q: apply(L, ch.pull(q)), brain_box,
            lambda q: ch.push(apply(Linv, q)), stpt_placed,
            (BRAIN_SPACE, STPT_SPACE),
            ("ls_to_template_SyN (AIND) + inverse-transform-SPIM.nii.gz",
             "transform-SPIM.nii.gz + ls_to_template_SyN (AIND)"),
            rng, pts,
        )
    if "spim" in a.scenes:
        report["spim"] = build_scene(
            a.out, "smartspim_894847_spim", h, a.margin, a.error_bound,
            lambda q: apply(Ls, ch.brain_to_spim(q)), brain_box,
            lambda q: ch.spim_to_brain(apply(Lsinv, q)), spim_placed,
            (BRAIN_SPACE, SPIM_SPACE),
            ("ls_to_template_SyN_1InverseWarp + 0GenericAffine (AIND)",
             "ls_to_template_SyN_1Warp + 0GenericAffine (AIND)"),
            rng, pts,
        )
    if "stpt_spim" in a.scenes:
        # checks at the brain's points, as placed in STPT space
        report["stpt_spim"] = build_scene(
            a.out, "stpt_spim_placed", h, a.margin, a.error_bound,
            lambda q: apply(Ls, ch.stpt_to_spim(apply(Linv, q))), stpt_placed,
            lambda q: apply(L, ch.spim_to_stpt(apply(Lsinv, q))), spim_placed,
            (STPT_SPACE, SPIM_SPACE),
            ("transform-SPIM.nii.gz", "inverse-transform-SPIM.nii.gz"),
            rng, apply(L, in_chunks(ch.pull, pts)),
        )
    placement.update({
        "stpt_ras_mm_to_brain_um": L.tolist(),
        "brain_um_to_stpt_ras_mm": Linv.tolist(),
        "spim_ras_mm_to_brain_um": Ls.tolist(),
        "brain_um_to_spim_ras_mm": Lsinv.tolist(),
        "axes": {"brain": list(OUTPUT_AXES), "stpt": ["z (S)", "y (A)", "x (R)"], "spim": ["z (S)", "y (A)", "x (R)"]},
        "spacing_um": h,
        "error_bound_um": a.error_bound,
        "forward_field": "smartspim_894847_ccf.zarr/coordinateTransformations/forward",
        "inverse_field": "smartspim_894847_ccf.zarr/coordinateTransformations/inverse",
        "spim_forward_field": "smartspim_894847_spim.zarr/coordinateTransformations/forward",
        "spim_inverse_field": "smartspim_894847_spim.zarr/coordinateTransformations/inverse",
        "stpt_to_spim_field": "stpt_spim_placed.zarr/coordinateTransformations/forward",
        "spim_to_stpt_field": "stpt_spim_placed.zarr/coordinateTransformations/inverse",
    })
    ppath.write_text(json.dumps(placement, indent=1))
    print(json.dumps(report, indent=1))


def check(scene, name, origin, h, pts, exact, boxes=8, half=32, seed=1):
    """|exact(p) - (p + u_decoded(p))| (um) at the points that fall in `boxes`
    regions of (2 half)^3 field samples around random points: reads only those
    regions of the decoded s0, trilinear."""
    from scipy.ndimage import map_coordinates

    arr = zarr.open_array(scene / "coordinateTransformations" / name / "s0", mode="r")
    q = (pts - origin) / h
    rng = np.random.default_rng(seed)
    errs = []
    for c in q[rng.choice(len(q), boxes, replace=False)]:
        lo = np.maximum(np.floor(c).astype(int) - half, 0)
        hi = np.minimum(lo + 2 * half, arr.shape[1:])
        inside = np.all((q >= lo) & (q <= hi - 1), axis=1)
        if not inside.any():
            continue
        u = arr[:, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float64)
        qq = (q[inside] - lo).T
        got = pts[inside] + np.stack([map_coordinates(u[k], qq, order=1, mode="nearest") for k in range(3)], 1)
        errs.append(np.linalg.norm(got - in_chunks(exact, pts[inside]), axis=1))
    err = np.concatenate(errs)
    return {"n": int(err.size), "median": float(np.median(err)), "p99": float(np.percentile(err, 99)), "max": float(err.max())}


if __name__ == "__main__":
    main()
