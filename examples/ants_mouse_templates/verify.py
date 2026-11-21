"""Convert the two ANTs fields to warpfield-zarr and verify them, chunk by chunk.

    verify.py ATLAS_DIR FORWARD.nii.gz INVERSE.nii.gz OUTDIR [--eps 0.001 0.0025 0.00625]

Nothing is loaded whole: the NIfTI fields are read in k-slabs (with indexed_gzip
seeking), zarr stores region by region, templates through memory maps.

1. Convert each field once, slab by slab, into a lossless zstd-19 zarr and one
   warpfield zarr per error bound (mm).
2. Field check per k-slab (+1 plane overlap): max / RMS error, det J of every
   cell, Jacobian violations (min_det 0.1 or a sign change).
3. Warp check per slab of the fixed template: the moving template warped with
   the original NIfTI field and with each zarr; NCC to the fixed template,
   max / mean difference between the two warps.
4. Forward / inverse round trip x -> y = x + u_f(x) -> y + u_i(y) at STPT brain
   points, with the original and with the 2.5 um fields.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import zarr
from streaming import (
    RAS_TO_LPS,
    Grid,
    NiftiField,
    ZarrField,
    fixed_slabs,
    sample_field,
    sample_image,
    slab_axis,
)

from warpfield import jacobian
from zarr_warpfield import WarpFieldCodec

MIN_DET = 0.1
SLAB = 64


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def variant_name(eps):
    return "zstd19-lossless" if eps is None else f"warpfield-eps{eps * 1000:g}um"


def convert(src: NiftiField, out: Path, key: str, eps_list, chunks=64):
    shape = (*src.grid.shape, 3)
    spacing = [float(s) for s in np.linalg.norm(src.grid.M, axis=0)]
    stores, timing = {}, {}
    for eps in [None, *eps_list]:
        kw = dict(
            shape=shape,
            chunks=(chunks, chunks, chunks, 3),
            dtype="float32",
            fill_value=0.0,
            overwrite=True,
        )
        path = out / f"{key}_{variant_name(eps)}.zarr"
        if eps is None:
            z = zarr.create_array(path, compressors=zarr.codecs.ZstdCodec(level=19), **kw)
        else:
            codec = WarpFieldCodec(
                axis_roles=["spatial", "spatial", "spatial", "component"],
                error_bound=eps,
                jacobian={"grid_spacing": spacing, "min_det": MIN_DET},
            )
            z = zarr.create_array(path, serializer=codec, compressors=None, **kw)
        z.attrs["ants_displacement_field"] = src.meta | {"source": key}
        stores[eps] = (path, z)
        timing[eps] = 0.0
    t_read = 0.0
    for k0 in range(0, shape[0], chunks):
        t0 = time.perf_counter()
        slab = src.read(k0, min(k0 + chunks, shape[0]))
        t_read += time.perf_counter() - t0
        for eps, (_, z) in stores.items():
            t0 = time.perf_counter()
            z[k0 : k0 + slab.shape[0]] = slab
            timing[eps] += time.perf_counter() - t0
    return {eps: (p, timing[eps]) for eps, (p, _) in stores.items()}, t_read


def field_check(src: NiftiField, z: ZarrField, eps):
    N = src.grid.shape[0]
    spacing = tuple(np.linalg.norm(src.grid.M, axis=0))
    mx = ss = n = 0.0
    viol = neg0 = neg1 = over = 0
    dmin0 = dmin1 = np.inf
    t_dec = 0.0
    for k0 in range(0, N, SLAB):
        k1 = min(k0 + SLAB + 1, N)  # +1 plane: cells across slab boundaries are included
        a = src.read(k0, k1)
        t0 = time.perf_counter()
        b = z.read(k0, k1)
        t_dec += time.perf_counter() - t0
        own = slice(0, min(SLAB, k1 - k0))  # error over this slab's own planes
        e = np.abs(b[own].astype(np.float64) - a[own])
        mx = max(mx, float(e.max()))
        ss += float((e**2).sum())
        n += e.size
        if eps:
            over += int((e > eps * (1 + 1e-9)).sum())
        d0 = jacobian.det_jacobian(a, 3, (0, 1, 2), spacing)
        d1 = jacobian.det_jacobian(b, 3, (0, 1, 2), spacing)
        viol += int(jacobian.violations(d0, d1, MIN_DET).sum())
        neg0 += int((d0 <= 0).sum())
        neg1 += int((d1 <= 0).sum())
        dmin0 = min(dmin0, float(d0.min()))
        dmin1 = min(dmin1, float(d1.min()))
    return dict(
        max_err_um=mx * 1e3,
        rms_err_um=np.sqrt(ss / n) * 1e3,
        samples_over_eps=over,
        jacobian_violations=viol,
        det_le0_cells_orig=neg0,
        det_le0_cells_dec=neg1,
        det_min_orig=dmin0,
        det_min_dec=dmin1,
        read_s=t_dec,
    )


class Acc:
    """Running NCC statistics over masked voxels."""

    def __init__(self):
        self.s = dict(n=0, x=0.0, y=0.0, xx=0.0, yy=0.0, xy=0.0)

    def add(self, x, y):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        s = self.s
        s["n"] += x.size
        s["x"] += x.sum()
        s["y"] += y.sum()
        s["xx"] += (x * x).sum()
        s["yy"] += (y * y).sum()
        s["xy"] += (x * y).sum()

    def ncc(self):
        s = self.s
        n = s["n"]
        cov = s["xy"] - s["x"] * s["y"] / n
        return cov / np.sqrt((s["xx"] - s["x"] ** 2 / n) * (s["yy"] - s["y"] ** 2 / n))


def warp_check(fixed, fgrid, moving, mgrid, ref: NiftiField, decoded: dict, step: int):
    axis = slab_axis(fgrid, ref.grid)
    msub = np.asarray(moving[::4, ::4, ::4])
    lo, hi = np.percentile(msub[msub > 0], [0.5, 99.5])
    span = float(hi - lo)
    fsub = np.asarray(fixed[::4, ::4, ::4])
    thr = np.percentile(fsub[fsub > 0], 5)
    acc_ref = Acc()
    acc = {k: (Acc(), Acc()) for k in decoded}
    dmax = {k: 0.0 for k in decoded}
    dsum = {k: 0.0 for k in decoded}
    nmask = 0
    for sl, _, pts in fixed_slabs(fgrid, axis, step):
        F = np.asarray(fixed[sl], np.float32).ravel()
        m = F > thr
        if not m.any():
            continue
        p = pts[m]
        w_ref = sample_image(moving, mgrid, p + sample_field(ref, p))
        acc_ref.add(F[m], w_ref)
        nmask += int(m.sum())
        for k, zf in decoded.items():
            w = sample_image(moving, mgrid, p + sample_field(zf, p))
            acc[k][0].add(F[m], w)
            acc[k][1].add(w_ref, w)
            d = np.abs(w - w_ref)
            dmax[k] = max(dmax[k], float(d.max()))
            dsum[k] += float(d.sum())
    out = {"ncc_fixed_vs_original_warp": acc_ref.ncc(), "intensity_span": span, "voxels": nmask}
    for k in decoded:
        out[k] = dict(
            ncc_fixed_vs_warp=acc[k][0].ncc(),
            ncc_vs_original_warp=acc[k][1].ncc(),
            max_abs_diff_frac=dmax[k] / span,
            mean_abs_diff_frac=dsum[k] / nmask / span,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atlas", type=Path)
    ap.add_argument("forward", type=Path)
    ap.add_argument("inverse", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.001, 0.0025, 0.00625])
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    tmpl = {}
    for n in ("spim", "stpt"):
        img = np.load(a.atlas / f"{n}_s1.npy", mmap_mode="r")
        tmpl[n] = (img, Grid(img.shape, RAS_TO_LPS @ np.load(a.atlas / f"{n}_s1_affine.npy")))
    results = {"eps_mm": a.eps, "fields": {}}
    fields = {}
    for key, path, fixed, moving in (
        ("forward", a.forward, "stpt", "spim"),
        ("inverse", a.inverse, "spim", "stpt"),
    ):
        src = NiftiField(path)
        raw = int(np.prod(src.grid.shape)) * 12
        print(
            f"\n=== {key}: {path.name}, grid {src.grid.shape}, {moving} -> {fixed} space; "
            f"raw {raw / 1e9:.2f} GB, nii.gz {path.stat().st_size / 1e9:.2f} GB",
            flush=True,
        )
        stores, t_read = convert(src, a.outdir, key, a.eps)
        print(
            f"  converted in one streaming pass (NIfTI read {t_read:.1f}s), "
            f"peak RSS {rss_gb():.2f} GB",
            flush=True,
        )
        r = {"raw_bytes": raw, "nifti_gz_bytes": path.stat().st_size, "variants": {}}
        decoded = {}
        for eps, (zp, te) in stores.items():
            name = variant_name(eps)
            zf = ZarrField(zp)
            v = {"bytes": du(zp), "encode_s": te}
            v["ratio_vs_raw"] = raw / v["bytes"]
            v["ratio_vs_nifti_gz"] = path.stat().st_size / v["bytes"]
            v |= field_check(src, zf, eps)
            r["variants"][name] = v
            print(
                f"  {name:22s} {v['bytes'] / 1e6:8.1f} MB  {v['ratio_vs_raw']:6.1f}x raw "
                f"{v['ratio_vs_nifti_gz']:6.1f}x nii.gz | encode {te:5.1f}s | max err "
                f"{v['max_err_um']:.3f} um rms {v['rms_err_um']:.3f} um, over eps "
                f"{v['samples_over_eps']} | jacobian violations {v['jacobian_violations']}, "
                f"det<=0 cells {v['det_le0_cells_orig']} -> {v['det_le0_cells_dec']} | "
                f"peak RSS {rss_gb():.2f} GB",
                flush=True,
            )
            if eps is not None:
                decoded[name] = zf
        t0 = time.perf_counter()
        w = warp_check(*tmpl[fixed], *tmpl[moving], src, decoded, step=16)
        r["warp"] = w
        print(
            f"  warp check ({w['voxels']} {fixed} brain voxels, {time.perf_counter() - t0:.0f}s): "
            f"NCC(fixed, warped by NIfTI) {w['ncc_fixed_vs_original_warp']:.5f}",
            flush=True,
        )
        for name in decoded:
            x = w[name]
            print(
                f"    {name:22s} NCC(fixed, warped) {x['ncc_fixed_vs_warp']:.5f}  "
                f"NCC(warped, NIfTI-warped) {x['ncc_vs_original_warp']:.7f}  image diff max "
                f"{x['max_abs_diff_frac']:.2e} mean {x['mean_abs_diff_frac']:.2e} of range "
                f"| peak RSS {rss_gb():.2f} GB",
                flush=True,
            )
        results["fields"][key] = r
        fields[key] = src
        if 0.0025 in stores:
            fields[key + "_2.5"] = ZarrField(stores[0.0025][0])
    # round trip at STPT brain points
    img, g = tmpl["stpt"]
    sub = np.asarray(img[::3, ::3, ::3])
    vox = np.argwhere(sub > np.percentile(sub[sub > 0], 5)) * 3
    rng = np.random.default_rng(0)
    vox = vox[rng.choice(len(vox), min(200_000, len(vox)), replace=False)]
    pts = g.to_lps(vox.T.astype(float))
    rt = {}
    for label, f, i in (
        ("original NIfTI", "forward", "inverse"),
        ("warpfield 2.5 um", "forward_2.5", "inverse_2.5"),
    ):
        if f not in fields or i not in fields:
            continue
        y = pts + sample_field(fields[f], pts)
        z = y + sample_field(fields[i], y)
        e = np.linalg.norm(z - pts, axis=1) * 1e3
        rt[label] = dict(
            median_um=float(np.median(e)),
            p99_um=float(np.percentile(e, 99)),
            max_um=float(e.max()),
        )
        print(
            f"round trip forward then inverse, {label}: median {rt[label]['median_um']:.3f} um, "
            f"p99 {rt[label]['p99_um']:.3f} um, max {rt[label]['max_um']:.2f} um",
            flush=True,
        )
    results["roundtrip"] = rt
    results["peak_rss_gb"] = rss_gb()
    print(f"\npeak RSS {rss_gb():.2f} GB")
    (a.outdir / "results.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
