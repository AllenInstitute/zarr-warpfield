"""Rate-distortion sweep: zfp fixed-accuracy at loosened tolerances vs warpfield.

zfp's `tolerance` is a loose upper bound; its realized error is usually far below it.
This sweeps the tolerance upward and reports the *realized* error so the codecs can
be compared at matched max error (the guarantee) and matched RMS error.

    uv run --with "zfpy==1.0.1.post0" python scripts/zfp_sweep.py [--scale full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import MIN_DET, cross_chunk, zfp_roundtrip
from zarr_warpfield import Config, JacobianConfig, chunked, jacobian, seams, synth

ZFP_TOL = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
WF_EPS = [0.1, 0.25, 0.5, 1.0]


def metrics(name, p, u, lab, d0, dec, nbytes, axes, h):
    ca = u.ndim - 1
    err = dec.astype(np.float64) - u
    a = np.abs(err)
    s = seams.seam_stats(err, p.chunks, axes, ca)["all"]
    bad = jacobian.violations(d0, jacobian.det_jacobian(dec, ca, p.spatial_axes, h), MIN_DET)
    return dict(
        ratio=u.nbytes / nbytes,
        max_err=float(a.max()),
        p999_err=float(np.quantile(a, 0.999)),
        rms_err=float(np.sqrt((err**2).mean())),
        max_err_disc=float(a[lab > 0].max()) if (lab > 0).any() else 0.0,
        seam_ratio=s["ratio"],
        seam_boundary_max=s["boundary_max"],
        jac=int(bad.sum()),
        jac_x=cross_chunk(bad, p),
    )


def run(scale, out):
    rows = []
    for name in synth.PRESET_NAMES:
        p = synth.preset(name, scale)
        u, lab = p.full()
        h = (p.spacing,) * p.ks
        axes = tuple(i for i, r in enumerate(p.axis_roles) if r in ("spatial", "smooth"))
        d0 = jacobian.det_jacobian(u, u.ndim - 1, p.spatial_axes, h)
        todo = [("warpfield", e) for e in WF_EPS] + [("zfp", t) for t in ZFP_TOL]
        for codec, knob in todo:
            if codec == "zfp":
                dec, nbytes, _, _ = zfp_roundtrip(u, p.chunks, knob)
            else:
                cfg = Config(p.axis_roles, knob, jacobian=JacobianConfig(h, MIN_DET))
                r = chunked.roundtrip(u, p.chunks, cfg)
                dec, nbytes = r.decoded, r.nbytes
            m = metrics(name, p, u, lab, d0, dec, nbytes, axes, h)
            rows.append(dict(preset=name, codec=codec, knob=knob, **m))
            print(
                f"{name:14s} {codec:9s} {knob:<5} ratio={m['ratio']:7.1f} "
                f"max={m['max_err']:.3f} p99.9={m['p999_err']:.3f} rms={m['rms_err']:.4f} "
                f"disc={m['max_err_disc']:.3f} seam={m['seam_ratio']:.2f} "
                f"bmax={m['seam_boundary_max']:.3f} jac={m['jac']}({m['jac_x']}x)",
                flush=True,
            )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full")
    ap.add_argument("--out", type=Path, default=Path("out/zfp_sweep.json"))
    a = ap.parse_args()
    run(a.scale, a.out)
