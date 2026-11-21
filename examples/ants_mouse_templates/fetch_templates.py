"""Download a resolution level of the two Allen mouse templates (OME-Zarr 0.6) to .npy,
with each array's voxel -> RAS (mm) affine derived from its OME coordinate transforms.

    uv run --with fsspec --with aiohttp python examples/ants_mouse_templates/fetch_templates.py OUTDIR [--level s1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

BASE = "https://allen-atlas-assets.s3.us-west-2.amazonaws.com/templates"
TEMPLATES = {
    "spim": f"{BASE}/allen-adult-mouse-spim-lca-template/2024-05/template.ome.zarr",
    "stpt": f"{BASE}/allen-adult-mouse-stpt-template/2020/template.ome.zarr",
}


def apply_ct(t: dict, p: np.ndarray) -> np.ndarray:
    """Apply one OME 0.6 (RFC-5) coordinate transformation to points p (..., n)."""
    k = t["type"]
    if k == "sequence":
        for s in t["transformations"]:
            p = apply_ct(s, p)
        return p
    if k == "scale":
        return p * np.asarray(t["scale"])
    if k == "translation":
        return p + np.asarray(t["translation"])
    if k == "rotation":
        return p @ np.asarray(t["rotation"]).T
    if k == "affine":
        a = np.asarray(t["affine"])
        return p @ a[:, :-1].T + a[:, -1]
    if k == "identity":
        return p
    raise NotImplementedError(k)


def voxel_to_ras(meta: dict, level: str) -> np.ndarray:
    """4x4 affine: array index (axis order) -> RAS mm (x, y, z)."""
    ms = meta["attributes"]["ome"]["multiscales"][0]
    ds = next(d for d in ms["datasets"] if d["path"] == level)
    to_intrinsic = ds["coordinateTransformations"][0]
    to_ras = next(t for t in ms["coordinateTransformations"] if t["output"]["name"] == "mm RAS")
    pts = np.vstack([np.zeros(3), np.eye(3)])
    out = apply_ct(to_ras, apply_ct(to_intrinsic, pts))  # "mm RAS" axes order: (z, y, x)
    out = out[:, ::-1]  # -> (x=R, y=A, z=S)
    A = np.eye(4)
    A[:3, 3] = out[0]
    A[:3, :3] = (out[1:] - out[0]).T
    return A


def main(outdir: Path, level: str):
    outdir.mkdir(parents=True, exist_ok=True)
    import fsspec

    for name, url in TEMPLATES.items():
        meta = json.loads(fsspec.open(f"{url}/zarr.json").open().read())
        A = voxel_to_ras(meta, level)
        arr = zarr.open_array(
            store=zarr.storage.FsspecStore.from_url(f"{url}/{level}"), mode="r"
        )
        print(name, level, arr.shape, arr.dtype, "\nvoxel->RAS\n", np.round(A, 5), flush=True)
        data = arr[...]
        np.save(outdir / f"{name}_{level}.npy", data)
        np.save(outdir / f"{name}_{level}_affine.npy", A)
        print(f"  saved, range {data.min():.3g}..{data.max():.3g}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--level", default="s1")
    a = ap.parse_args()
    sys.exit(main(a.outdir, a.level))
