"""Check a region of the `brain_in_ccf` warp against an independent ITK
implementation of the ANTs transforms.

The Neuroglancer view: display points v are "STPT 2020 placed in the brain by
L" (um, z y x). The brain is drawn at the brain point v + u_inv(v), where
u_inv is the decoded `smartspim_894847_ccf.zarr` inverse field. That field was
sampled from `chain.Chain.push(L^-1 v)`.

Here SimpleITK does the same job without any of this example's code. ITK reads
the ANTs files itself: the .mat through ReadTransform, the two displacement
fields through its NIfTI reader, and `prep_zarr_img.nii.gz`, the pipeline's
level-3 image with its own header. It composes them as antsApplyTransforms
would (`-t A -t W -t transform-SPIM`, last applied first) after the display ->
STPT affine L^-1, and it resamples the brain onto a display-space grid around a
point. Two outputs:

- an OME-Zarr of the ITK resample, axis-aligned in display um, to overlay on
  the Neuroglancer warp (`itk_check/<name>.ome.zarr`, plus a state);
- three mappings compared at every grid point, in level-3 index units converted
  to um: ITK, chain.push (numpy), and the decoded field that Neuroglancer uses.
  The output also has det J of each registration along the path, to tell which
  one makes any visible distortion.

    uv run --project ../.. --with nibabel --with scipy --with SimpleITK python itk_check.py OUT \
        --src OUT/src --point 3495 9266 1837 --name x3495_y9266_z1837
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain import LEVEL3_SHAPE_XYZ, LEVEL3_UM_XYZ, Chain, in_chunks  # noqa: E402

ALIGN = "image_atlas_alignment/Ex_639_Em_667"
RAS_ZYX_TO_LPS_XYZ = np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]], float)  # (S, A, R) -> (L, P, S)
XYZ_ZYX = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)


def cindex(im: sitk.Image, p: np.ndarray) -> np.ndarray:
    """Continuous ITK index of physical points p (N, 3) in image im."""
    m = np.array(im.GetDirection()).reshape(3, 3) @ np.diag(im.GetSpacing())
    return (p - np.array(im.GetOrigin())) @ np.linalg.inv(m).T


def read_crop(path: Path, p: np.ndarray, margin: int = 4) -> tuple[sitk.Image, dict]:
    """Read an image with ITK and keep only the index box around points p
    (plus a margin), so that every point lies well inside the buffer. Outside
    its buffer an ITK displacement field is zero, which differs from ANTs'
    nearest-sample edge, so the margin rules that difference out."""
    im = sitk.ReadImage(str(path))
    idx = cindex(im, p)
    size = np.array(im.GetSize())
    lo = np.floor(idx.min(0)).astype(int) - margin
    hi = np.ceil(idx.max(0)).astype(int) + margin + 1
    info = {"file": path.name, "size": size.tolist(), "index_min": idx.min(0).round(2).tolist(),
            "index_max": idx.max(0).round(2).tolist(), "inside": bool((lo >= 0).all() and (hi <= size).all())}
    lo, hi = np.maximum(lo, 0), np.minimum(hi, size)
    crop = im[int(lo[0]) : int(hi[0]), int(lo[1]) : int(hi[1]), int(lo[2]) : int(hi[2])]
    del im
    return crop, info


def field_det(img: sitk.Image) -> sitk.Image:
    """det(I + du/dp) of an ITK displacement field in physical space (the
    direction matrix included, which DisplacementFieldJacobianDeterminant
    leaves out), as a scalar image on the field's grid."""
    u = sitk.GetArrayFromImage(img).astype(np.float64)  # (k, j, i, 3), vectors in physical axes
    g = np.stack(np.gradient(u, axis=(2, 1, 0)), -1)  # d u / d (i, j, k)
    m = np.array(img.GetDirection()).reshape(3, 3) @ np.diag(img.GetSpacing())
    det = sitk.GetImageFromArray(np.linalg.det(np.eye(3) + g @ np.linalg.inv(m)))
    det.CopyInformation(img)
    return det


def mapped(tx: sitk.Transform, grid: dict) -> np.ndarray:
    """tx applied to every grid point (ITK x fastest), as (N, 3) (x, y, z)."""
    d = sitk.TransformToDisplacementField(tx, sitk.sitkVectorFloat64, grid["size"], grid["origin"], grid["spacing"])
    return grid["points"] + sitk.GetArrayViewFromImage(d).reshape(-1, 3)


def sample_scalar(im: sitk.Image, p: np.ndarray) -> np.ndarray:
    """Trilinear samples of a scalar ITK image at physical points p (N, 3)."""
    from scipy.ndimage import map_coordinates

    a = sitk.GetArrayViewFromImage(im)  # (z, y, x)
    q = cindex(im, p)[:, ::-1].T
    return map_coordinates(np.asarray(a, np.float64), q, order=1, mode="nearest")


def stats(x: np.ndarray) -> dict:
    return {"median": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--fields", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--point", type=float, nargs=3, required=True, help="display x y z (um), as Neuroglancer shows it")
    ap.add_argument("--half", type=float, nargs=3, default=[2500, 100, 2500], help="half extent x y z (um)")
    ap.add_argument("--spacing", type=float, default=10.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fields-url", default="http://127.0.0.1:9079/")
    a = ap.parse_args()
    out = a.out / "itk_check"
    out.mkdir(exist_ok=True)
    placement = json.loads((a.out / "placement.json").read_text())
    L = np.asarray(placement["stpt_ras_mm_to_brain_um"])
    Linv = np.linalg.inv(L)  # display um (z, y, x) -> STPT mm RAS (z, y, x)

    # the display grid, ITK (x, y, z) order
    h = a.spacing
    centre, half = np.array(a.point), np.array(a.half)
    size = (2 * np.round(half / h) + 1).astype(int)
    origin = centre - (size - 1) / 2 * h
    ii = np.stack(np.meshgrid(*[np.arange(n) for n in size[::-1]], indexing="ij")[::-1], -1).reshape(-1, 3)
    pts = origin + ii * h
    grid = {"size": [int(n) for n in size], "origin": origin.tolist(), "spacing": [h] * 3, "points": pts}
    print(f"grid {size.tolist()} at {h} um, {len(pts)} points", flush=True)

    # T0: display (x, y, z) um -> STPT 2020 LPS (x, y, z) mm
    T0 = sitk.AffineTransform(3)
    T0.SetMatrix((RAS_ZYX_TO_LPS_XYZ @ Linv[:3, :3] @ XYZ_ZYX).ravel().tolist())
    T0.SetTranslation((RAS_ZYX_TO_LPS_XYZ @ Linv[:3, 3]).tolist())

    al = a.src / ALIGN
    report = {"point_xyz_um": a.point, "grid": {k: v for k, v in grid.items() if k != "points"}, "crops": []}

    # transform-SPIM: STPT point -> SPIM template point
    p1 = mapped(T0, grid)
    Uimg, info = read_crop(a.out / "raw" / "transform-SPIM.nii", p1)
    report["crops"].append(info)
    detU = field_det(Uimg)
    U = sitk.DisplacementFieldTransform(sitk.Cast(Uimg, sitk.sitkVectorFloat64))
    del Uimg
    c1 = sitk.CompositeTransform([U, T0])
    p2 = mapped(c1, grid)

    # SyN warp W (template -> brain, before A)
    Wimg, info = read_crop(a.out / "raw" / "ls_to_template_SyN_1Warp.nii", p2)
    report["crops"].append(info)
    detW = field_det(Wimg)
    W = sitk.DisplacementFieldTransform(sitk.Cast(Wimg, sitk.sitkVectorFloat64))
    del Wimg
    A = sitk.ReadTransform(str(al / "ls_to_template_SyN_0GenericAffine.mat"))
    full = sitk.CompositeTransform([A, W, U, T0])  # T0 first, A last (ITK applies the last added first)
    p4 = mapped(full, grid)

    # the composite order, checked one transform at a time at the centre
    k = len(pts) // 2
    step = A.TransformPoint(W.TransformPoint(U.TransformPoint(T0.TransformPoint(pts[k].tolist()))))
    report["composite_order_check_mm"] = float(np.linalg.norm(np.array(step) - p4[k]))

    # the brain: the pipeline's level-3 image, its geometry from its own header
    brain, info = read_crop(a.src / ALIGN / "metadata/registration_metadata/prep_zarr_img.nii.gz", p4, margin=2)
    report["crops"].append(info)
    res = sitk.Resample(brain, grid["size"], full, sitk.sitkLinear, grid["origin"], grid["spacing"],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1], 0.0, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(res)  # (z, y, x)

    # det J along the path
    shape_zyx = tuple(size[::-1])
    jU = sample_scalar(detU, p1)
    jW = sample_scalar(detW, p2)
    jA = float(np.linalg.det(np.array(A.GetParameters()[:9]).reshape(3, 3)))
    g = p4.reshape(*shape_zyx, 3)
    J = np.stack([np.stack(np.gradient(g[..., c], h, axis=(2, 1, 0)), -1) for c in range(3)], -2)  # d p4 / d (x, y, z)
    detJ = np.linalg.det(J).ravel() * 1e9  # brain mm^3 per display um^3 -> brain um^3 per display um^3
    sign = np.sign(np.median(detJ))
    report["det_J"] = {
        "display -> brain (1 = as the affine placement)": stats(sign * detJ) | {
            "min": float((sign * detJ).min()), "folded_fraction": float(np.mean(sign * detJ <= 0))},
        "transform-SPIM (STPT -> SPIM)": {"min": float(jU.min()), "median": float(np.median(jU)), "max": float(jU.max())},
        "SyN 1Warp (template -> brain)": {"min": float(jW.min()), "median": float(np.median(jW)), "max": float(jW.max())},
        "SyN affine": jA,
    }
    c = np.argmin(np.linalg.norm(pts - centre, axis=1))
    near = np.linalg.norm(pts - centre, axis=1) <= 300
    for key, j in (("display -> brain (1 = as the affine placement)", sign * detJ), ("transform-SPIM (STPT -> SPIM)", jU),
                   ("SyN 1Warp (template -> brain)", jW)):
        report["det_J"][key] |= {"at_point": float(j[c]), "within_300um": [float(j[near].min()), float(j[near].max())],
                                 "argmin_xyz_um": pts[np.argmin(j)].tolist(), "argmax_xyz_um": pts[np.argmax(j)].tolist()}
    del J, g

    # the same points by this example's code: chain.push (numpy), and the decoded field
    import nibabel as nib

    ch = Chain(nib.load(a.src / ALIGN / "metadata/registration_metadata/prep_zarr_img.nii.gz").affine,
               al / "ls_to_template_SyN_0GenericAffine.mat", al / "ls_to_template_SyN_1Warp.nii.gz", None,
               a.fields / "transform-SPIM.nii.gz", None, a.out / "raw")
    v = pts[:, ::-1]  # display (z, y, x) um
    q = v @ Linv[:3, :3].T + Linv[:3, 3]
    d_chain = in_chunks(ch.push, q)  # brain um (z, y, x)

    from scipy.ndimage import map_coordinates

    store = a.out / placement["inverse_field"]
    attrs = json.loads((store / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]
    f_origin = np.array(attrs["coordinateTransformations"][0]["affine"])[1:, 4]
    f_h = attrs["datasets"][0]["coordinateTransformations"][0]["transformations"][0]["scale"][1]
    arr = zarr.open_array(store / "s0", mode="r")
    fi = (v - f_origin) / f_h
    lo = np.maximum(np.floor(fi.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(fi.max(0)).astype(int) + 2, arr.shape[1:])
    u = np.asarray(arr[:, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], np.float64)
    d_field = v + np.stack([map_coordinates(u[c], (fi - lo).T, order=1, mode="nearest") for c in range(3)], 1)

    # compare in the pipeline's level-3 index (from its header), in um
    def brain_um_to_abc(d):
        xyz3 = d[:, ::-1] / LEVEL3_UM_XYZ - 0.5
        return np.stack([LEVEL3_SHAPE_XYZ[0] - 1 - xyz3[:, 0], xyz3[:, 1], xyz3[:, 2]], 1)

    hdr = sitk.ImageFileReader()
    hdr.SetFileName(str(a.src / ALIGN / "metadata/registration_metadata/prep_zarr_img.nii.gz"))
    hdr.ReadImageInformation()
    full_origin = np.array(hdr.GetOrigin())
    full_m = np.array(hdr.GetDirection()).reshape(3, 3) @ np.diag(hdr.GetSpacing())
    abc_itk = (p4 - full_origin) @ np.linalg.inv(full_m).T  # ITK index in the full prep_zarr_img
    um = LEVEL3_UM_XYZ  # a <-> x3, b <-> y3, c <-> z3
    for key, d in (("chain.push (numpy)", d_chain), ("decoded field (Neuroglancer)", d_field)):
        report.setdefault("vs_itk_um", {})[key] = stats(np.linalg.norm((brain_um_to_abc(d) - abc_itk) * um, axis=1))
    e = np.linalg.norm(d_field - d_chain, axis=1)
    report["field_vs_chain_um"] = stats(e) | {"argmax_xyz_um": pts[np.argmax(e)].tolist(),
                                              "within_300um_max": float(e[near].max())}
    report["prep_header"] = {"origin": full_origin.tolist(), "spacing": list(hdr.GetSpacing()),
                             "direction": list(hdr.GetDirection()), "size": list(hdr.GetSize())}

    # the ITK resample, for Neuroglancer (voxel k centred on its grid point)
    zpath = out / f"{a.name}.ome.zarr"
    root = zarr.open_group(str(zpath), mode="w", zarr_format=2)
    root.attrs["multiscales"] = [{
        "version": "0.4", "name": f"ITK resample {a.name}",
        "axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in ("z", "y", "x")],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [h] * 3},
            {"type": "translation", "translation": (origin[::-1] - h / 2).tolist()}]}],
    }]
    root.create_array("s0", data=np.clip(np.round(img), 0, 65535).astype(np.uint16), chunks=(64, 64, 64),
                      compressors=None)
    report["itk_image_percentiles_1_50_99"] = np.percentile(img[img > 0], [1, 50, 99]).tolist() if (img > 0).any() else []
    (out / f"{a.name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    # a state: the Neuroglancer warp (magenta) and the ITK resample (green)
    st = json.loads((a.out / "states" / "brain_vs_spim_template_in_ccf_coronal.json").read_text())
    for lyr in st["layers"]:
        if "SPIM" in lyr["name"]:
            lyr["visible"] = False
    base = a.fields_url.rstrip("/")
    st["layers"].append({
        "type": "image", "source": f"zarr2://{base}/itk_check/{a.name}.ome.zarr/", "name": "ITK resample (prep_zarr_img)",
        "blend": "additive", "opacity": 1.0,
        "shader": "#uicontrol invlerp normalized\nvoid main() { emitRGB(vec3(0.0, normalized(), 0.0)); }",
        "shaderControls": {"normalized": {"range": [0, float(np.percentile(img[img > 0], 99)) if (img > 0).any() else 1000]}},
    })
    st["position"] = [a.point[0], a.point[1], a.point[2], 0.5]
    st["crossSectionScale"] = 4
    st["selectedLayer"] = {"layer": "ITK resample (prep_zarr_img)", "visible": True}
    (a.out / "states" / f"itk_check_{a.name}.json").write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
