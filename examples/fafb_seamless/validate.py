"""Check the field store against the published v14.1 render, independently of
Neuroglancer: warp FAFB v14 (fafb_v14_orig, 16 nm) with the decoded field,
v14.1(p) = v14(p + u(p)), and compare with FlyWire's v14.1 image at 16 nm.

Reports, per section, the normalized cross-correlation of v14.1 with v14
unwarped and warped (against v14.1 sections z-1, z, z+1, which checks the
section pairing), and the residual shift between warped v14 and v14.1 by phase
correlation (which checks direction, units and pixel-center conventions).

    uv run python examples/fafb_seamless/validate.py [--sections 4334 4384 4433]
"""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from make_field import OUT, fetch
from PIL import Image
from scipy import ndimage

import zarr_warpfield  # noqa: F401  (registers the codec)

V14 = "https://storage.googleapis.com/neuroglancer-fafb-data/fafb_v14/fafb_v14_orig/16_16_40"
V14_CHUNK, V14_SIZE = 64, (62208, 33536, 7063)  # jpeg, offset 0
V141 = (
    "https://storage.googleapis.com/microns-seunglab/drosophila_v0/alignment/"
    "image_rechunked/16_16_40"
)
V141_CHUNK, V141_OFFSET = (256, 256, 16), (4096, 0, 1)  # raw


def read_precomputed(base, chunk, offset, lo, hi, decode, threads=16):
    """uint8 [z, y, x] of the box [lo, hi) (x, y, z) of a precomputed scale."""
    out = np.zeros([hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]], np.uint8)
    ranges = [
        range((lo[a] - offset[a]) // chunk[a], -(-(hi[a] - offset[a]) // chunk[a]))
        for a in range(3)
    ]
    keys = [(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]]

    def get(ijk):
        a0 = [offset[a] + ijk[a] * chunk[a] for a in range(3)]
        a1 = [a0[a] + chunk[a] for a in range(3)]
        body, _ = fetch(f"{base}/{a0[0]}-{a1[0]}_{a0[1]}-{a1[1]}_{a0[2]}-{a1[2]}")
        return a0, body

    with ThreadPoolExecutor(threads) as pool:
        for a0, body in pool.map(get, keys):
            if body is None:
                continue
            arr = decode(body, chunk)  # [z, y, x]
            s = [max(lo[a], a0[a]) for a in range(3)]
            e = [min(hi[a], a0[a] + arr.shape[2 - a]) for a in range(3)]
            if any(e[a] <= s[a] for a in range(3)):
                continue
            out[
                s[2] - lo[2] : e[2] - lo[2],
                s[1] - lo[1] : e[1] - lo[1],
                s[0] - lo[0] : e[0] - lo[0],
            ] = arr[
                s[2] - a0[2] : e[2] - a0[2],
                s[1] - a0[1] : e[1] - a0[1],
                s[0] - a0[0] : e[0] - a0[0],
            ]
    return out


def jpeg(body, chunk):
    im = np.asarray(Image.open(io.BytesIO(body)).convert("L"))
    nx = im.shape[1]
    ny = chunk[1] if im.shape[0] >= chunk[1] else im.shape[0]
    return im.reshape(-1, ny, nx)


def raw(body, chunk):
    return np.frombuffer(body, np.uint8).reshape(chunk[2], chunk[1], chunk[0])


def ncc(a, b, mask):
    a, b = a[mask].astype(float), b[mask].astype(float)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def shift(a, b, mask):
    """Sub-pixel (dy, dx) such that b(p) ~ a(p + d): phase correlation peak."""
    a = (
        np.where(mask, a - a[mask].mean(), 0)
        * np.hanning(a.shape[0])[:, None]
        * np.hanning(a.shape[1])
    )
    b = (
        np.where(mask, b - b[mask].mean(), 0)
        * np.hanning(b.shape[0])[:, None]
        * np.hanning(b.shape[1])
    )
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r = np.fft.fftshift(np.real(np.fft.ifft2(F / (np.abs(F) + 1e-9))))
    p = np.unravel_index(np.argmax(r), r.shape)
    d = []
    for ax in range(2):
        m = list(p)
        m[ax] -= 1
        l = r[tuple(m)]
        m[ax] += 2
        h = r[tuple(m)]
        c = r[p]
        d.append(p[ax] - r.shape[ax] // 2 + 0.5 * (l - h) / (l - 2 * c + h))
    return [float(v) for v in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fafb_v14_to_v14.1")
    ap.add_argument("--sections", type=int, nargs="+", default=[4334, 4384, 4433])
    ap.add_argument(
        "--center",
        type=float,
        nargs="+",
        default=[74067, 42119],
        help="x y (4 nm pixels); several pairs check several places",
    )
    ap.add_argument("--half", type=int, default=256, help="16 nm pixels")
    a = ap.parse_args()
    root = zarr.open_group(OUT / f"{a.name}.zarr", mode="r")
    src = root.attrs["warpfield"]["source"]["region_16nm"]
    x0, y0, z0 = src["x"][0], src["y"][0], src["z"][0]
    report = {}
    centers = list(zip(a.center[::2], a.center[1::2], strict=True))
    for (px, py), z in ((c, z) for c in centers for z in a.sections):
        cx, cy = int(px / 4), int(py / 4)
        X0, X1, Y0, Y1 = cx - a.half, cx + a.half, cy - a.half, cy + a.half
        k = z - z0
        u = root["s0"][:, k, Y0 - y0 : Y1 - y0, X0 - x0 : X1 - x0]  # (dz, dy, dx) nm
        pad = int(np.ceil(np.abs(u).max() / 16)) + 8  # > max |u|, 16 nm px
        v14 = read_precomputed(
            V14,
            (V14_CHUNK,) * 3,
            (0, 0, 0),
            (X0 - pad, Y0 - pad, z),
            (X1 + pad, Y1 + pad, z + 1),
            jpeg,
        )[0].astype(np.float32)
        v141 = read_precomputed(
            V141, V141_CHUNK, V141_OFFSET, (X0, Y0, z - 1), (X1, Y1, z + 2), raw
        )
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        # pixel centers in both: v14 index = absolute 16 nm index + u / 16 nm
        sy = yy + u[1] / 16 - (Y0 - pad)
        sx = xx + u[2] / 16 - (X0 - pad)
        warped = ndimage.map_coordinates(v14, [sy, sx], order=1)
        unwarped = v14[pad:-pad, pad:-pad]
        ref = v141[1].astype(np.float32)
        mask = ref > 0
        key = str(z) if len(centers) == 1 else f"{z} @ ({px:.0f}, {py:.0f})"
        report[key] = {
            "ncc_unwarped": ncc(unwarped, ref, mask),
            "ncc_warped": {
                f"v14.1 z{d:+d}": ncc(warped, v141[1 + d].astype(np.float32), mask)
                for d in (-1, 0, 1)
            },
            "residual_shift_px16_dy_dx": shift(warped, ref, mask),
            "shift_unwarped_px16_dy_dx": shift(unwarped, ref, mask),
            "mean_u_nm_dy_dx": [float(u[1].mean()), float(u[2].mean())],
        }
        print(key, json.dumps(report[key]), flush=True)
    (OUT / f"{a.name}.validate.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
