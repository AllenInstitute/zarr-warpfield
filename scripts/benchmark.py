"""Compression ratio, error, seams, Jacobian violations and speed on every preset.

    uv run python scripts/benchmark.py [--scale full|small] [--out out/benchmark.json]

Codecs compared, all chunked with the preset's chunk shape:

- warpfield      this codec (face bound eps/8, ENO, Jacobian guarantee min_det 0.1)
- no-jacobian    same without the Jacobian pass
- baseline       same predictor/quantizer, but SZ3-style extrapolating boundary and
                 no face tightening: what a plain chunked error-bounded codec does
- zfp            zfpy fixed-accuracy (tolerance = eps), per chunk and component
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from warpfield import Config, JacobianConfig, chunked, jacobian, seams, synth

MIN_DET = 0.1


def zfp_roundtrip(u: np.ndarray, chunks, eps: float):
    import zfpy

    out = np.empty_like(u)
    total = 0
    te = td = 0.0
    for sl in chunked.chunk_slices(u.shape, chunks):
        blk = u[sl]
        dec = np.empty_like(blk)
        for c in range(blk.shape[-1]):
            a = np.ascontiguousarray(blk[..., c])
            lead = a.shape[:-4] if a.ndim > 4 else ()
            for idx in np.ndindex(*lead) if lead else [()]:
                t0 = time.perf_counter()
                b = zfpy.compress_numpy(np.ascontiguousarray(a[idx]), tolerance=eps)
                t1 = time.perf_counter()
                dec[idx + (..., c)] = zfpy.decompress_numpy(b)
                te += t1 - t0
                td += time.perf_counter() - t1
                total += len(b)
        out[sl] = dec
    return out, total, te, td


def cross_chunk(bad: np.ndarray, p) -> int:
    straddle = np.zeros(bad.shape, dtype=bool)
    for a in p.spatial_axes:
        i = np.arange(bad.shape[a])
        shp = [1] * bad.ndim
        shp[a] = -1
        straddle |= ((i + 1) % p.chunks[a] == 0).reshape(shp)
    return int((bad & straddle).sum())


def run(scale: str, eps_list, out_path: Path | None):
    rows = []
    for name in synth.PRESET_NAMES:
        p = synth.preset(name, scale)
        u, lab = p.full()
        ca = u.ndim - 1
        h = (p.spacing,) * p.ks
        axes = tuple(i for i, r in enumerate(p.axis_roles) if r in ("spatial", "smooth"))
        d0 = jacobian.det_jacobian(u, ca, p.spatial_axes, h)
        mb = u.nbytes / 1e6
        for eps in eps_list:
            cands = {
                "warpfield": Config(p.axis_roles, eps, jacobian=JacobianConfig(h, MIN_DET)),
                "no-jacobian": Config(p.axis_roles, eps),
                "baseline": Config(
                    p.axis_roles, eps, face_error_bound=eps, boundary="extrapolate"
                ),
            }
            results = {}
            for label, cfg in cands.items():
                r = chunked.roundtrip(u, p.chunks, cfg)
                results[label] = (
                    r.decoded,
                    r.nbytes,
                    r.encode_seconds,
                    r.decode_seconds,
                    int((r.classes == 3).sum()),
                    int((r.classes == 2).sum()),
                )
            try:
                d, n, te, td = zfp_roundtrip(u, p.chunks, eps)
                results["zfp"] = (d, n, te, td, 0, 0)
            except ImportError:
                pass
            for label, (dec, nbytes, te, td, nverb, ntight) in results.items():
                err = dec.astype(np.float64) - u
                s = seams.seam_stats(err, p.chunks, axes, ca)["all"]
                d1 = jacobian.det_jacobian(dec, ca, p.spatial_axes, h)
                bad = jacobian.violations(d0, d1, MIN_DET)
                row = dict(
                    preset=name,
                    shape=list(u.shape),
                    chunks=list(p.chunks),
                    eps=eps,
                    codec=label,
                    ratio=u.nbytes / nbytes,
                    bits_per_value=8 * nbytes / u.size,
                    max_err=float(np.abs(err).max()),
                    max_err_on_discontinuities=float(np.abs(err[lab > 0]).max()),
                    seam_ratio=s["ratio"],
                    seam_boundary_max=s["boundary_max"],
                    jac_violations=int(bad.sum()),
                    jac_violations_cross_chunk=cross_chunk(bad, p),
                    verbatim=nverb,
                    tight=ntight,
                    encode_MBps=mb / max(te, 1e-9),
                    decode_MBps=mb / max(td, 1e-9),
                )
                rows.append(row)
                print(
                    f"{name:14s} eps={eps:<5} {label:12s} ratio={row['ratio']:7.1f} "
                    f"maxerr={row['max_err']:.3f} seam={row['seam_ratio']:.2f} "
                    f"jac={row['jac_violations']}({row['jac_violations_cross_chunk']} x-chunk) "
                    f"enc={row['encode_MBps']:.1f}MB/s dec={row['decode_MBps']:.1f}MB/s",
                    flush=True,
                )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full")
    ap.add_argument("--eps", type=float, nargs="*", default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--out", type=Path, default=Path("out/benchmark.json"))
    a = ap.parse_args()
    run(a.scale, a.eps, a.out)
