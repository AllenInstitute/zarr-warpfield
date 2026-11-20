"""Write every synthetic preset as a zarr v3 group, region by region.

    uv run python scripts/make_synthetic.py [--scale full|small] [--out out/synthetic]
                                            [--eps 0.25] [--presets 2d 3d ...]

Each <preset>.zarr group holds

    raw        float32 displacement field (zstd), the ground truth
    warpfield  the same field through the warpfield codec (Jacobian guarantee on)
    labels     uint8 discontinuity labels (see attributes["labels"])

The generator is evaluated one write-region at a time from global
coordinates, so the same code writes arbitrarily large arrays without holding
them in memory: pass --shape to override a preset's extent, e.g.
`--presets 3d --shape 1024 1024 1024 3`.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from zarr_warpfield import WarpFieldCodec, synth


def regions(shape, step):
    for idx in itertools.product(
        *[range(-(-s // c)) for s, c in zip(shape, step, strict=True)]
    ):
        yield tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, step, shape, strict=True)
        )


def write(p: synth.Preset, out: Path, eps: float) -> None:
    root = zarr.open_group(out / f"{p.name}.zarr", mode="w")
    axes = {"spatial": ["z", "y", "x"][-p.ks :], "component": ["c"]}
    names = []
    sp = iter(axes["spatial"])
    for r in p.axis_roles:
        names.append("c" if r == "component" else next(sp) if r == "spatial" else "t")
    root.attrs.update(
        description=p.description,
        axis_roles=list(p.axis_roles),
        dimension_names=names,
        field_spacing_voxels=p.spacing,
        labels={str(k): v for k, v in synth.LABELS.items()},
        displacement_units="image voxels; component c is along the c-th spatial axis",
    )
    codec = WarpFieldCodec(
        axis_roles=p.axis_roles,
        error_bound=eps,
        jacobian={"grid_spacing": [p.spacing] * p.ks, "min_det": 0.1},
    )
    kw = dict(
        shape=p.shape, chunks=p.chunks, dtype="float32", fill_value=0.0, dimension_names=names
    )
    raw = root.create_array("raw", compressors=ZstdCodec(level=5), **kw)
    wf = root.create_array("warpfield", serializer=codec, compressors=None, **kw)
    lab = root.create_array(
        "labels",
        shape=p.shape[:-1],
        chunks=p.chunks[:-1],
        dtype="uint8",
        fill_value=0,
        compressors=ZstdCodec(level=5),
        dimension_names=names[:-1],
    )
    # write in regions of whole chunks (2 chunks per axis) so nothing large is held in memory
    step = tuple(2 * c for c in p.chunks[:-1])
    for sl in regions(p.shape[:-1], step):
        u, labels = p.region(sl)
        raw[sl] = u
        wf[sl] = u
        lab[sl] = labels
    g = out / f"{p.name}.zarr"
    size = {
        k: sum(f.stat().st_size for f in (g / k).rglob("*") if f.is_file())
        for k in ("raw", "warpfield")
    }
    n = 4 * int(np.prod(p.shape))
    print(
        f"{p.name}: wrote {g}  shape {p.shape} chunks {p.chunks}  "
        f"raw+zstd {n / size['raw']:.1f}x  warpfield {n / size['warpfield']:.1f}x"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full")
    ap.add_argument("--out", type=Path, default=Path("out/synthetic"))
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--presets", nargs="*", default=list(synth.PRESET_NAMES))
    ap.add_argument("--shape", type=int, nargs="*", help="override the array shape")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.presets:
        p = synth.preset(name, a.scale)
        if a.shape:
            p = replace(p, shape=tuple(a.shape))
        write(p, a.out, a.eps)


if __name__ == "__main__":
    main()
