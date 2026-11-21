"""Figure: registration check and original-vs-compressed comparison, from slices only.

    figures.py ATLAS_DIR FORWARD.nii.gz INVERSE.nii.gz STORE_DIR OUT.png [--eps-um 2.5]

For three orthogonal mid-slices of each fixed template: the fixed template,
magenta/green overlays of the fixed template with the moving template warped by
the original NIfTI field and by the warpfield zarr, the difference of the two
warps, and the field difference |u_zarr - u_nifti| at those points.
Only the field slabs the slices pass through are read.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from streaming import RAS_TO_LPS, Grid, NiftiField, ZarrField, sample_field, sample_image


def norm(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atlas")
    ap.add_argument("forward")
    ap.add_argument("inverse")
    ap.add_argument("stores")
    ap.add_argument("out")
    ap.add_argument("--eps-um", default="2.5")
    a = ap.parse_args()
    tm = {}
    for n in ("spim", "stpt"):
        img = np.load(f"{a.atlas}/{n}_s1.npy", mmap_mode="r")
        tm[n] = (img, Grid(img.shape, RAS_TO_LPS @ np.load(f"{a.atlas}/{n}_s1_affine.npy")))
    eps = float(a.eps_um)
    cols = [
        "fixed",
        "overlay: NIfTI warp",
        f"overlay: zarr warp (ε={eps:g} µm)",
        "zarr − NIfTI warp (±2% range)",
        "|u_zarr − u_NIfTI| (µm)",
    ]
    fig, axs = plt.subplots(6, 5, figsize=(20, 22))
    for r, (key, path, fixed, moving) in enumerate(
        (("forward", a.forward, "stpt", "spim"), ("inverse", a.inverse, "spim", "stpt"))
    ):
        nf = NiftiField(path)
        zf = ZarrField(f"{a.stores}/{key}_warpfield-eps{a.eps_um}um.zarr")
        F, fg = tm[fixed]
        M, mg = tm[moving]
        msub = np.asarray(M[::4, ::4, ::4])
        mlo, mhi = np.percentile(msub[msub > 0], [0.5, 99.5])
        fsub = np.asarray(F[::4, ::4, ::4])
        flo, fhi = np.percentile(fsub[fsub > 0], [0.5, 99.5])
        for s in range(3):
            rng = [np.arange(n) for n in fg.shape]
            rng[s] = np.array([fg.shape[s] // 2])
            g = np.meshgrid(*rng, indexing="ij")
            shp = tuple(len(x) for i, x in enumerate(rng) if i != s)
            pts = fg.to_lps(np.stack([x.ravel() for x in g]).astype(float))
            u0 = sample_field(nf, pts)
            u1 = sample_field(zf, pts)
            fix = np.asarray(
                F[tuple(rng[i] if i == s else slice(None) for i in range(3))]
            ).reshape(shp)
            w0 = sample_image(M, mg, pts + u0).reshape(shp)
            w1 = sample_image(M, mg, pts + u1).reshape(shp)
            err = (np.abs(u1 - u0).max(1) * 1e3).reshape(shp)
            fn, w0n, w1n = norm(fix, flo, fhi), norm(w0, mlo, mhi), norm(w1, mlo, mhi)
            panels = [
                (fn, "gray", (0, 1)),
                (np.stack([fn, w0n, fn], -1), None, (0, 1)),
                (np.stack([fn, w1n, fn], -1), None, (0, 1)),
                ((w1 - w0) / (mhi - mlo), "RdBu_r", (-0.02, 0.02)),
                (err, "viridis", (0, eps)),
            ]
            row = axs[3 * r + s]
            for c, (ax, (img, cmap, (vmin, vmax))) in enumerate(zip(row, panels, strict=True)):
                h = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if s == 0:
                    title = cols[c] if c else f"fixed ({fixed.upper()})"
                    ax.set_title(title, fontsize=13)
                    if c >= 3:
                        fig.colorbar(h, ax=ax, fraction=0.04)
            row[0].set_ylabel(
                f"{key}: {moving.upper()} → {fixed.upper()}\naxis {s} mid-slice", fontsize=12
            )
    fig.suptitle(
        "Allen mouse templates: ANTs fields as original NIfTI vs warpfield zarr, applied "
        "ANTs/ITK-style (LPS, linear)\noverlay: fixed = magenta, warped moving = green",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(a.out, dpi=60)
    print(a.out)


if __name__ == "__main__":
    main()
