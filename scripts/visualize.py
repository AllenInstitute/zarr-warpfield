"""Image gallery: test images pushed through raw and compressed alignment fields.

    uv run python scripts/visualize.py [--scale full|small] [--out out/gallery] [--eps 0.25]

For every preset and a few representative 2D views (slices through the stack /
time / z axes), writes PNGs to <out>/<preset>/ and a local index.html:

  warp_<view>.png     test image; warped through the raw field; through warpfield;
                      through the naive chunked baseline; through zfp. Zoomed crops at a
                      chunk corner near a crack, and difference images (codec - raw) on one
                      colour scale with the chunk grid overlaid.
  field_<view>.png    displacement as HSV (hue = direction, value = magnitude), magnitude
                      with quiver, ground-truth discontinuities with the samples the
                      encoder promoted (tight / verbatim), det J and |grad u|.
  error_<view>.png    |decoded - raw| per codec with eps / eps_face scale and chunk grid,
                      plus an error profile along a line crossing chunk faces and a crack.
  stack_<view>.png    (stacked presets) the field across the stack/time axis.
  roundtrip.png       (2D presets) image warped forward through the decoded field and back
                      through the inverse computed from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from warpfield import Config, JacobianConfig, chunked, inverse, jacobian, synth  # noqa: E402

LABEL_COLORS = {
    1: (0.90, 0.10, 0.10),  # slip crack
    2: (1.00, 0.55, 0.00),  # opening crack
    3: (0.00, 0.75, 0.85),  # steep band
    4: (0.35, 0.35, 1.00),  # over-compression (continuous, det J < 0)
    5: (1.00, 0.90, 0.00),  # misaligned section
    6: (0.85, 0.00, 0.85),  # fold (overlap jump)
}
DPI = 80


# -- codecs -----------------------------------------------------------------------
def zfp_roundtrip(u, chunks, eps):
    try:
        import zfpy
    except ImportError:
        return None
    out = np.empty_like(u)
    total = 0
    for sl in chunked.chunk_slices(u.shape, chunks):
        blk = u[sl]
        dec = np.empty_like(blk)
        for c in range(blk.shape[-1]):
            a = np.ascontiguousarray(blk[..., c])
            lead = a.shape[:-4] if a.ndim > 4 else ()
            for idx in np.ndindex(*lead) if lead else [()]:
                b = zfpy.compress_numpy(np.ascontiguousarray(a[idx]), tolerance=eps)
                dec[idx + (..., c)] = zfpy.decompress_numpy(b)
                total += len(b)
        out[sl] = dec
    return out, u.nbytes / total


def run_codecs(p, u, eps):
    h = (p.spacing,) * p.ks
    cfgs = {
        "warpfield": Config(p.axis_roles, eps, jacobian=JacobianConfig(h, 0.1)),
        "baseline": Config(p.axis_roles, eps, face_error_bound=eps, boundary="extrapolate"),
    }
    res = {}
    for k, c in cfgs.items():
        r = chunked.roundtrip(u, p.chunks, c)
        res[k] = dict(decoded=r.decoded, classes=r.classes, ratio=r.ratio(u), cfg=c)
    z = zfp_roundtrip(u, p.chunks, eps)
    if z is not None:
        res["zfp"] = dict(decoded=z[0], classes=None, ratio=z[1], cfg=None)
    return res


# -- views ------------------------------------------------------------------------
def choose_views(p, lab):
    """(name, title, index tuple over the non-component axes fixing all but the last 2)."""
    nd = lab.ndim
    views = []
    lead = nd - 2  # axes fixed by a view (all but the last two spatial axes)

    def best(sub_axes_lab, counts_fn):
        return int(np.argmax(counts_fn(sub_axes_lab)))

    if lead == 0:
        return [("plane", "the whole 2D field", ())]
    sa = p.stack_axis
    if p.ks == 2:  # 2d_stack: (Z, Y, X)
        crack_counts = np.isin(lab, synth.CRACK_LABELS).sum(axis=(1, 2))
        z1 = int(np.argmax(crack_counts))
        views.append((f"z{z1}", f"section {z1} (cracked)", (z1,)))
        if p.misaligned:
            z2 = p.misaligned[0]
            views.append((f"z{z2}", f"section {z2} (misaligned)", (z2,)))
        zc = [z for z in range(lab.shape[0]) if not crack_counts[z] and z not in p.misaligned]
        if zc:
            views.append((f"z{zc[0]}", f"section {zc[0]} (no crack)", (zc[0],)))
        return views
    # 3D spatial; optional leading stack/time axis
    if sa is None:
        cnt = np.isin(lab, (*synth.CRACK_LABELS, 4)).sum(axis=(1, 2))
        z = int(np.argmax(cnt))
        views.append((f"z{z}", f"z-slice {z}", (z,)))
        cnt3 = (lab == 3).sum(axis=(1, 2))
        z3 = int(np.argmax(cnt3))
        if abs(z3 - z) > 4:
            views.append((f"z{z3}", f"z-slice {z3} (steep band)", (z3,)))
        return views
    tname = "t" if p.axis_roles[sa] in ("smooth", "independent") else "s"
    ts = [1, lab.shape[0] - 2] if lab.shape[0] > 3 else [lab.shape[0] // 2]
    if p.misaligned:
        ts = [ts[0], p.misaligned[0]]
    ts = list(dict.fromkeys(ts))
    for t in ts:
        cnt = np.isin(lab[t], (*synth.CRACK_LABELS, 4)).sum(axis=(1, 2))
        z = int(np.argmax(cnt)) if cnt.max() > 0 else lab.shape[1] // 2
        what = " (misaligned)" if t in p.misaligned else ""
        views.append((f"{tname}{t}_z{z}", f"{tname}={t}{what}, z-slice {z}", (t, z)))
    return views


def spatial_volume(p, arr, idx):
    """The spatial sub-array (all spatial axes, components) containing view `idx`."""
    sa = p.stack_axis
    if sa is None:
        return arr, idx
    return arr[idx[0]], idx[1:]


# -- warping -----------------------------------------------------------------------
def warp(field, fixed, h, step=1.0, period=32.0):
    """Image of the test pattern pulled back through `field` on the plane of the last two axes.

    field: (*S, k) spatial field; fixed: sample indices for the leading spatial axes.
    Returns (warped image, plane image coordinates).
    """
    S = field.shape[:-1]
    k = field.shape[-1]
    ny = int((S[-2] - 1) * h / step) + 1
    nx = int((S[-1] - 1) * h / step) + 1
    yy, xx = np.meshgrid(np.arange(ny) * step, np.arange(nx) * step, indexing="ij")
    P = [np.full_like(yy, f * h) for f in fixed] + [yy, xx]
    U = [
        ndimage.map_coordinates(field[..., c], [q / h for q in P], order=1, mode="nearest")
        for c in range(k)
    ]
    W = synth.test_image([P[i] + U[i] for i in range(k)], period)
    return W, P


def chunk_grid(ax, p, scale, plane_axes, **kw):
    kw = {"color": "white", "lw": 0.6, "alpha": 0.7, **kw}
    ay, axx = plane_axes
    for c in range(p.chunks[axx], p.shape[axx], p.chunks[axx]):
        ax.axvline((c - 0.5) * scale, **kw)
    for c in range(p.chunks[ay], p.shape[ay], p.chunks[ay]):
        ax.axhline((c - 0.5) * scale, **kw)


def image_period(p):
    """Grid period (voxels) giving about 20 grid cells across the view."""
    ext = (max(p.shape[a] for a in p.spatial_axes[-2:]) - 1) * p.spacing
    return float(max(32, 16 * round(ext / 20 / 16)))


def chunk_ticks(ax, p, scale, plane_axes, color="0.2"):
    """Mark chunk faces with ticks on the frame (lines across an image look like seams)."""
    ay, axx = plane_axes
    ax.set_xticks(
        [(c - 0.5) * scale for c in range(p.chunks[axx], p.shape[axx], p.chunks[axx])]
    )
    ax.set_yticks([(c - 0.5) * scale for c in range(p.chunks[ay], p.shape[ay], p.chunks[ay])])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=7, width=1.5, color=color, top=True, right=True, direction="inout")


def plane_axes_of(p):
    sp = p.spatial_axes
    return sp[-2], sp[-1]


def zoom_center(p, lab2d):
    """A crack sample that lies close to a chunk face (field sample coords)."""
    ay, ax = plane_axes_of(p)
    ys = np.arange(p.chunks[ay], p.shape[ay], p.chunks[ay]) - 0.5
    xs = np.arange(p.chunks[ax], p.shape[ax], p.chunks[ax]) - 0.5
    m = np.argwhere((lab2d == 1) | (lab2d == 2))
    if m.size == 0 or (ys.size == 0 and xs.size == 0):
        return lab2d.shape[0] // 2, lab2d.shape[1] // 2
    dy = np.abs(m[:, :1] - ys[None]).min(1) if ys.size else np.full(len(m), np.inf)
    dx = np.abs(m[:, 1:] - xs[None]).min(1) if xs.size else np.full(len(m), np.inf)
    # prefer a crack point near both faces (a chunk corner), else near one
    score = np.minimum(dy, dx) + 0.25 * np.maximum(dy, dx)
    return tuple(m[int(np.argmin(score))])


# -- figures -------------------------------------------------------------------------
def fig_warp(p, view, u, res, lab2d, outdir):
    name, title, idx = view
    h = p.spacing
    vol_raw, fixed = spatial_volume(p, u, idx)
    period = image_period(p)
    imgs = {"raw field": warp(vol_raw, fixed, h, period=period)[0]}
    for k, r in res.items():
        vol, _ = spatial_volume(p, r["decoded"], idx)
        imgs[k] = warp(vol, fixed, h, period=period)[0]
    ny, nx = imgs["raw field"].shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    P0 = [np.full_like(yy, f * h, dtype=np.float64) for f in fixed] + [yy, xx]
    original = synth.test_image(P0, period)
    cy, cx = zoom_center(p, lab2d)
    cy, cx = int(cy * h), int(cx * h)
    half = int(1.6 * period)
    zs = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
    cols = ["undistorted", "raw field", *res.keys()]
    allimgs = {"undistorted": original, **imgs}
    ncol = len(cols)
    fig, axs = plt.subplots(3, ncol, figsize=(3.3 * ncol, 10.4))
    diffs = {k: imgs[k] - imgs["raw field"] for k in res}
    vmax = max(np.percentile(np.abs(d), 99.5) for d in diffs.values())
    vmax = max(vmax, 1e-3)
    pax = plane_axes_of(p)
    for j, c in enumerate(cols):
        a = axs[0, j]
        a.imshow(allimgs[c], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        a.add_patch(
            plt.Rectangle(
                (zs[1].start, zs[0].start), 2 * half, 2 * half, fill=False, ec="yellow", lw=1
            )
        )
        sub = ""
        if c in res and res[c]["ratio"]:
            sub = f"\n{res[c]['ratio']:.0f}x"
        a.set_title(f"{c}{sub}", fontsize=10)
        a = axs[1, j]
        a.imshow(allimgs[c][zs], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        chunk_grid(a, p, h, pax, color="yellow", lw=0.8)
        a.set_xlim(-0.5, 2 * half - 0.5)
        a.set_ylim(2 * half - 0.5, -0.5)
        # shift grid lines into crop coordinates
        for ln in a.lines:
            if ln.get_xdata()[0] == ln.get_xdata()[-1]:
                ln.set_xdata([ln.get_xdata()[0] - zs[1].start] * 2)
            else:
                ln.set_ydata([ln.get_ydata()[0] - zs[0].start] * 2)
        a.set_title("zoom: a crack crossing a chunk face", fontsize=8)
        a = axs[2, j]
        if c in diffs:
            im = a.imshow(diffs[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            chunk_ticks(a, p, h, pax)
            a.set_title(f"{c} − raw  (rms {np.sqrt((diffs[c] ** 2).mean()):.4f})", fontsize=9)
        elif c == "raw field":
            g = np.repeat(imgs["raw field"][..., None], 3, -1) * 0.8
            labu = np.kron(lab2d, np.ones((int(h), int(h)), dtype=lab2d.dtype))[:ny, :nx]
            for kk, col in LABEL_COLORS.items():
                g[labu == kk] = col
            a.imshow(g)
            a.set_title("discontinuities on the raw-warped image", fontsize=9)
        else:
            a.axis("off")
    for a in list(axs[0]) + list(axs[1]) + list(axs[2, :2]):
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=axs[2, -1], fraction=0.046, label="intensity difference")
    fig.suptitle(
        f"{p.name}: {title} — image pushed through each field (ε = "
        f"{res['warpfield']['cfg'].error_bound} voxel)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"warp_{name}.png", dpi=DPI)
    plt.close(fig)


def _slice2d(p, arr, idx):
    """2D plane (Y, X[, C]) of an array over the full axes at view idx."""
    return arr[idx] if idx else arr


def fig_field(p, view, u, lab, res, d0, d1, outdir):
    name, title, idx = view
    f2 = _slice2d(p, u, idx)  # (Y, X, C)
    l2 = _slice2d(p, lab, idx)
    cls = _slice2d(p, res["warpfield"]["classes"], idx).max(axis=-1)
    uy, ux = f2[..., -2], f2[..., -1]
    mag = np.hypot(uy, ux)
    hue = (np.arctan2(uy, ux) / (2 * np.pi)) % 1.0
    hsv = np.stack([hue, np.full_like(hue, 0.85), mag / mag.max()], -1)
    rgb = mcolors.hsv_to_rgb(hsv)
    pax = plane_axes_of(p)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    a = axs[0, 0]
    a.imshow(rgb, interpolation="nearest")
    a.set_title("displacement: hue = direction, brightness = |u|")
    chunk_grid(a, p, 1, pax)
    a = axs[0, 1]
    im = a.imshow(mag, cmap="viridis")
    st = max(f2.shape[0] // 24, 1)
    yy, xx = np.mgrid[0 : f2.shape[0] : st, 0 : f2.shape[1] : st]
    a.quiver(
        xx,
        yy,
        ux[::st, ::st],
        -uy[::st, ::st],
        color="white",
        scale_units="xy",
        angles="xy",
        scale=1.0 / max(st / 8, 0.5),
        width=0.003,
    )
    fig.colorbar(im, ax=a, fraction=0.046, label="|u| (voxels)")
    a.set_title("|u| with displacement vectors (in-plane)")
    a = axs[0, 2]
    base = 0.25 + 0.5 * (mag - mag.min()) / (np.ptp(mag) + 1e-9)
    over = np.repeat(base[..., None], 3, -1)
    for k, col in LABEL_COLORS.items():
        over[l2 == k] = col
    a.imshow(over, interpolation="nearest")
    ty, tx = np.nonzero(cls == 2)
    vy, vx = np.nonzero(cls == 3)
    a.scatter(tx, ty, s=2, c="white", marker="s", lw=0, label=f"tight ({tx.size})")
    a.scatter(vx, vy, s=3, c="black", marker="s", lw=0, label=f"verbatim ({vx.size})")
    chunk_grid(a, p, 1, pax, color="0.3")
    handles = [
        plt.Line2D([], [], color=c, lw=6, label=synth.LABELS[k])
        for k, c in LABEL_COLORS.items()
        if (l2 == k).any()
    ]
    a.legend(
        handles=handles + a.get_legend_handles_labels()[0][:2],
        fontsize=7,
        loc="lower left",
        framealpha=0.85,
    )
    a.set_title("discontinuities (truth) and samples the encoder promoted")
    # det J of the original on this plane
    d0p = _det_plane(p, d0, idx)
    d1p = _det_plane(p, d1, idx)
    a = axs[1, 0]
    im = a.imshow(d0p, cmap="RdBu", vmin=0, vmax=2, interpolation="nearest")
    a.contour(d0p <= 0, levels=[0.5], colors="black", linewidths=0.8)
    fig.colorbar(im, ax=a, fraction=0.046, label="det J (raw)")
    a.set_title("det J of the raw field (black: det ≤ 0 — fold jumps, over-compression)")
    a = axs[1, 1]
    dd = d1p - d0p
    lim = max(np.percentile(np.abs(dd), 99.5), 1e-3)
    im = a.imshow(dd, cmap="PuOr", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=a, fraction=0.046, label="det J change")
    bad = jacobian.violations(d0p, d1p, 0.1)
    by, bx = np.nonzero(bad)
    a.scatter(
        bx, by, s=12, facecolors="none", edgecolors="red", label=f"violations ({bx.size})"
    )
    a.legend(fontsize=7, loc="lower left")
    chunk_grid(a, p, 1, pax, color="0.4")
    a.set_title("det J(warpfield) − det J(raw)")
    a = axs[1, 2]
    gy = np.gradient(f2[..., -1], axis=0) / p.spacing
    gx = np.gradient(f2[..., -1], axis=1) / p.spacing
    gy2 = np.gradient(f2[..., -2], axis=0) / p.spacing
    gx2 = np.gradient(f2[..., -2], axis=1) / p.spacing
    g = np.sqrt(gx**2 + gy**2 + gx2**2 + gy2**2)
    im = a.imshow(
        g,
        cmap="magma",
        norm=mcolors.LogNorm(vmin=1e-3, vmax=max(g.max(), 1e-2)),
        interpolation="nearest",
    )
    fig.colorbar(im, ax=a, fraction=0.046, label="|∇u| (per voxel)")
    a.set_title("|∇u|: smooth field vs. sharp features")
    for a in axs.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"{p.name}: {title} — the field and its discontinuities (chunk grid shown)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(outdir / f"field_{name}.png", dpi=DPI)
    plt.close(fig)


def _det_plane(p, d, idx):
    # det arrays have the same leading axes as the field minus the component axis,
    # with spatial axes one shorter; view indices along spatial axes stay valid if < n-1.
    idx2 = list(idx)
    for i, a in enumerate(range(len(idx))):
        if a in p.spatial_axes:
            idx2[i] = min(idx2[i], d.shape[a] - 1)
    return d[tuple(idx2)] if idx2 else d


def fig_error(p, view, u, lab, res, outdir):
    name, title, idx = view
    eps = res["warpfield"]["cfg"].error_bound
    epsf = res["warpfield"]["cfg"].eps_face
    pax = plane_axes_of(p)
    keys = list(res)
    fig = plt.figure(figsize=(5 * len(keys), 13.5))
    gs = fig.add_gridspec(3, len(keys), height_ratios=[1.2, 0.8, 0.8])
    errs = {
        k: np.abs(_slice2d(p, r["decoded"], idx).astype(np.float64) - _slice2d(p, u, idx)).max(
            -1
        )
        for k, r in res.items()
    }
    for j, k in enumerate(keys):
        a = fig.add_subplot(gs[0, j])
        im = a.imshow(errs[k], cmap="magma", vmin=0, vmax=eps, interpolation="nearest")
        chunk_ticks(a, p, 1, pax)
        a.set_title(f"|error| {k} (max {errs[k].max():.3f})", fontsize=10)
    cb = fig.colorbar(im, ax=fig.axes[: len(keys)], fraction=0.02)
    cb.set_label("|decoded − raw| (voxels), scale = ε")
    cb.ax.axhline(epsf, color="cyan", lw=2)
    # profile: along the row (in the plane) that crosses the most cracks
    l2 = _slice2d(p, lab, idx)
    row = int(np.argmax(((l2 == 1) | (l2 == 2)).sum(axis=1)))
    a = fig.add_subplot(gs[1, :])
    x = np.arange(l2.shape[1])
    for k in keys:
        e = _slice2d(p, res[k]["decoded"], idx).astype(np.float64) - _slice2d(p, u, idx)
        a.plot(x, e[row, :, -1], lw=1, label=k, drawstyle="steps-mid")
    for s in (eps, -eps):
        a.axhline(s, color="k", ls="--", lw=0.8)
    for s in (epsf, -epsf):
        a.axhline(s, color="c", ls=":", lw=0.8)
    for c in range(p.chunks[pax[1]], p.shape[pax[1]], p.chunks[pax[1]]):
        a.axvspan(c - 1, c, color="0.85", lw=0)
    disc = np.nonzero(l2[row] > 0)[0]
    a.plot(
        disc, np.full(disc.size, -1.15 * eps), "|", color="red", ms=10, label="discontinuity"
    )
    a.set_ylim(-1.25 * eps, 1.25 * eps)
    a.set_xlim(0, l2.shape[1] - 1)
    a.set_xlabel(f"sample along row {row} (grey bands: chunk faces)")
    a.set_ylabel("error in x-displacement (voxels)")
    a.legend(fontsize=8, ncol=5, loc="upper right")
    a.set_title(f"error profile — dashed ±ε = {eps}, dotted ±ε_face = {epsf}")
    # seam detector: mean |e[i+1] - e[i]| over the WHOLE array, per position along x
    a = fig.add_subplot(gs[2, :])
    xa = pax[1]
    for k in keys:
        e = res[k]["decoded"].astype(np.float64) - u
        d = np.abs(np.diff(e, axis=xa)).max(axis=-1)
        other = tuple(i for i in range(d.ndim) if i != xa)
        a.plot(np.arange(d.shape[xa]) + 0.5, d.mean(axis=other), lw=1.2, label=k)
    for c in range(p.chunks[xa], p.shape[xa], p.chunks[xa]):
        a.axvline(c - 0.5, color="0.6", lw=0.8, ls="--")
    a.set_xlim(0, p.shape[xa] - 1)
    a.set_ylim(bottom=0)
    a.set_xlabel("position along x (dashed: chunk faces)")
    a.set_ylabel("mean |e[i+1] − e[i]|")
    a.legend(fontsize=8, ncol=4, loc="upper right")
    a.set_title(
        "seam detector: error jump between neighbours, averaged over the whole array — "
        "a systematic seam is a spike at every dashed line"
    )
    fig.suptitle(f"{p.name}: {title} — reconstruction error", fontsize=12)
    fig.savefig(outdir / f"error_{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_stack(p, u, lab, res, outdir):
    sa = p.stack_axis
    if sa is None:
        return None
    # resliced view: stack axis vertical, last spatial axis horizontal, middle of the rest
    sl = [slice(None)] * (u.ndim - 1)
    for a in range(u.ndim - 1):
        if a not in (sa, plane_axes_of(p)[1]):
            sl[a] = u.shape[a] // 2
    sl = tuple(sl)
    raw = u[sl][..., -1]
    dec = res["warpfield"]["decoded"][sl][..., -1]
    l2 = lab[sl]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    im = axs[0].imshow(raw, cmap="viridis", aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=axs[0], label="x-displacement")
    axs[0].set_title("raw field across the stack axis")
    over = np.repeat(((raw - raw.min()) / (np.ptp(raw) + 1e-9))[..., None] * 0.6 + 0.2, 3, -1)
    for k, col in LABEL_COLORS.items():
        over[l2 == k] = col
    axs[1].imshow(over, aspect="auto", interpolation="nearest")
    axs[1].set_title("discontinuities (yellow rows: misaligned)")
    e = dec - raw
    im = axs[2].imshow(
        e, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto", interpolation="nearest"
    )
    fig.colorbar(im, ax=axs[2], label="error")
    axs[2].set_title("warpfield error")
    role = p.axis_roles[sa]
    for a in axs:
        a.set_ylabel(f"{'time' if p.ks == 3 else 'section'} index ({role})")
        a.set_xlabel("x sample")
        for c in range(p.chunks[sa], p.shape[sa], p.chunks[sa]):
            a.axhline(c - 0.5, color="white", lw=0.6)
    fig.suptitle(f"{p.name}: the field along the stack/time axis", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "stack.png", dpi=DPI)
    plt.close(fig)
    return "stack.png"


def fig_roundtrip(p, u, lab, res, outdir, idx=()):
    if p.ks != 2:
        return None
    h = np.array([p.spacing] * 2)
    f_raw = (u[idx] if idx else u).astype(np.float64)
    f_dec = res["warpfield"]["decoded"][idx] if idx else res["warpfield"]["decoded"]
    f_dec = f_dec.astype(np.float64)
    v_hat, _ = inverse.invert(f_dec, h)
    v_hat = v_hat.astype(np.float64)
    ny, nx = (np.array(f_raw.shape[:2]) - 1) * int(p.spacing) + 1
    yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    Y = np.stack([yy, xx], -1)
    V = inverse.interp(v_hat, Y, h)
    Q = Y + V  # psi_hat(y)
    Pd = Q + inverse.interp(f_dec, Q, h)  # phi_hat(psi_hat(y))
    Pr = Q + inverse.interp(f_raw, Q, h)  # phi(psi_hat(y))
    per = image_period(p)
    orig = synth.test_image([yy, xx], per)
    rt = synth.test_image([Pd[..., 0], Pd[..., 1]], per)
    rt_err = np.abs(Pd - Y).max(-1)
    rt_true = np.abs(Pr - Y).max(-1)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    axs[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("original image")
    axs[1].imshow(rt, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("forward through decoded field, back through its inverse")
    im = axs[2].imshow(rt_err, cmap="magma", vmin=0, vmax=0.25)
    fig.colorbar(im, ax=axs[2], fraction=0.046, label="voxels")
    axs[2].set_title("|φ̂(ψ̂(y)) − y|  (bright: no inverse —\ncrack gaps, fold overlaps)")
    im = axs[3].imshow(rt_true, cmap="magma", vmin=0, vmax=0.5)
    fig.colorbar(im, ax=axs[3], fraction=0.046, label="voxels")
    axs[3].set_title("|φ(ψ̂(y)) − y|  with the raw forward field\n(≤ the left map + ε)")
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    smooth = ndimage.zoom((lab[idx] if idx else lab) == 0, p.spacing, order=0)[:ny, :nx]
    fig.suptitle(
        f"{p.name}: round trip with the inverse computed from the decoded field — "
        f"median {np.median(rt_err[smooth]):.1e}, 99th pct with raw φ "
        f"{np.percentile(rt_true[smooth], 99):.3f} voxel (away from discontinuities)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "roundtrip.png", dpi=DPI)
    plt.close(fig)
    return "roundtrip.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full")
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("out/gallery"))
    ap.add_argument("--presets", nargs="*", default=list(synth.PRESET_NAMES))
    a = ap.parse_args()
    manifest = []
    for name in a.presets:
        p = synth.preset(name, a.scale)
        outdir = a.out / name
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"{name}: generating and encoding", flush=True)
        u, lab = p.full()
        res = run_codecs(p, u, a.eps)
        ca = u.ndim - 1
        hh = (p.spacing,) * p.ks
        d0 = jacobian.det_jacobian(u, ca, p.spatial_axes, hh)
        d1 = jacobian.det_jacobian(res["warpfield"]["decoded"], ca, p.spatial_axes, hh)
        entry = dict(
            preset=name,
            description=p.description,
            shape=list(u.shape),
            chunks=list(p.chunks),
            axis_roles=list(p.axis_roles),
            ratios={k: r["ratio"] for k, r in res.items()},
            views=[],
        )
        for view in choose_views(p, lab):
            print(f"  view {view[0]}", flush=True)
            fig_warp(p, view, u, res, _slice2d(p, lab, view[2]), outdir)
            fig_field(p, view, u, lab, res, d0, d1, outdir)
            fig_error(p, view, u, lab, res, outdir)
            entry["views"].append(
                dict(
                    name=view[0],
                    title=view[1],
                    files=[
                        f"warp_{view[0]}.png",
                        f"field_{view[0]}.png",
                        f"error_{view[0]}.png",
                    ],
                )
            )
        extra = [f for f in (fig_stack(p, u, lab, res, outdir),) if f]
        rt_idx = ()
        if p.ks == 2 and p.stack_axis is not None:
            rt_idx = choose_views(p, lab)[0][2]
        rt = fig_roundtrip(p, u, lab, res, outdir, rt_idx)
        if rt:
            extra.append(rt)
        entry["extra"] = extra
        manifest.append(entry)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    html = [
        "<!doctype html><meta charset=utf-8><title>warpfield gallery</title>",
        "<style>body{font-family:sans-serif;max-width:1400px;margin:auto}"
        "img{max-width:100%;border:1px solid #ccc;margin:6px 0}</style>",
        "<h1>warpfield gallery</h1>",
    ]
    for e in manifest:
        html.append(
            f"<h2>{e['preset']} — {e['description']}</h2><p>shape {e['shape']}, "
            f"chunks {e['chunks']}, roles {e['axis_roles']}, ratios "
            + ", ".join(f"{k}: {v:.0f}x" for k, v in e["ratios"].items() if v)
            + "</p>"
        )
        for v in e["views"]:
            html.append(f"<h3>{v['title']}</h3>")
            html += [f"<img src='{e['preset']}/{f}'>" for f in v["files"]]
        html += [f"<img src='{e['preset']}/{f}'>" for f in e["extra"]]
    (a.out / "index.html").write_text("\n".join(html))
    print(f"wrote {a.out / 'index.html'}")


if __name__ == "__main__":
    main()
