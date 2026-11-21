"""Which SPIM chunks does a view need: native (rigid shift) vs warped (visible pixels
only) vs warped computed in 64^3 chunks. Offline, from the field store and the
SPIM OME metadata."""

import json
import urllib.request
from pathlib import Path

import numpy as np
import zarr
from fetch_templates import voxel_to_ras
from ome_scene import field_geometry
from scipy import ndimage

SPIM = "https://allen-atlas-assets.s3.us-west-2.amazonaws.com/templates/allen-adult-mouse-spim-lca-template/2024-05/template.ome.zarr"
meta = json.load(urllib.request.urlopen(SPIM + "/zarr.json"))
FIELD = (
    Path(__file__).parents[2] / "out/ants/spim_stpt_ants.zarr/coordinateTransformations/forward"
)
g = zarr.open_group(FIELD, mode="r")
levels, V = field_geometry(FIELD)
F = g["s1"][:].astype(np.float32)  # field level s1 (50 um), (3, k, j, i)
I2O = levels["s1"]
O2I = np.linalg.inv(np.vstack([I2O, [0, 0, 0, 1]]))
s0 = levels["s0"]
N0 = np.array(g["s0"].shape[1:])
corners = np.array(
    [
        [a, b, c]
        for a in (-0.5, N0[0] - 0.5)
        for b in (-0.5, N0[1] - 0.5)
        for c in (-0.5, N0[2] - 0.5)
    ]
)
dom = corners @ s0[:, :3].T + s0[:, 3]
LOWER = dom.min(0)  # field domain, output (z, y, x) mm
SHIFT = -np.array([5.2192, 5.2394, 5.6828])  # native SPIM rigid shift, output mm


def u_out(p):  # p (n, 3) output zyx mm -> displacement (n, 3)
    idx = (p @ O2I[:3, :3].T + O2I[:3, 3]).T
    for a in range(3):
        np.clip(idx[a], 0, F.shape[a + 1] - 1, out=idx[a])
    u = np.stack(
        [ndimage.map_coordinates(F[c], idx, order=1, mode="nearest") for c in range(3)], 1
    )
    return u @ V.T


def spim_chunks(q, level, chunk=128):  # q (n, 3) output zyx mm -> set of chunk ids
    A = voxel_to_ras(meta, level)
    shape = (
        np.array(
            zarr.open_array(
                store=zarr.storage.FsspecStore.from_url(f"{SPIM}/{level}"), mode="r"
            ).shape
        )
        if level not in _shapes
        else _shapes[level]
    )
    _shapes[level] = shape
    ras = q[:, ::-1]
    idx = (ras - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T
    inside = np.all((idx >= -0.5) & (idx <= shape - 0.5), 1)
    idx = idx[inside]
    f = np.floor(idx).astype(int)
    out = set()
    for d in np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]):
        k = np.clip(f + d, 0, shape - 1) // chunk
        out |= set(map(tuple, np.unique(k, axis=0)))
    return out


_shapes = {}


def plane_points(pos, axis, w, h, step):  # pos (z,y,x) mm; plane normal = output axis `axis`
    hz, vt = (
        (2, 1) if axis == 0 else (2, 0) if axis == 1 else (0, 1)
    )  # xy: x,y; xz: x,z; yz: z,y
    a = np.arange(-w / 2, w / 2, step)
    b = np.arange(-h / 2, h / 2, step)
    A, B = np.meshgrid(a, b, indexing="ij")
    p = np.tile(pos, (A.size, 1)).astype(float)
    p[:, hz] += A.ravel()
    p[:, vt] += B.ravel()
    return p


def warped_chunk_points(pos, axis, w, h, vs, cs=(64, 64, 64), stride=4):
    """All voxel centres (subsampled) of the warped chunks the visible rectangle touches."""
    cs = np.array(cs)
    rect = plane_points(pos, axis, w, h, vs * cs.min() / 2)
    rect = np.vstack([rect, plane_points(pos, axis, w, h, vs)[::997]])
    cids = np.unique(np.floor((rect - LOWER) / (vs * cs)).astype(int), axis=0)
    pts = []
    ax = [np.append(np.arange(0, c, min(stride, c), dtype=float), c - 1) for c in cs]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    for c in cids:
        pts.append(LOWER + (c * cs + G + 0.5) * vs)
    return np.vstack(pts), len(cids)


views = {
    "zoom": dict(pos=np.array([1.0, -1.5, 0.0]), w=4.19, h=2.53, vs=0.010, level="s0"),
    "overview": dict(pos=np.array([1.23, -1.24, 0.005]), w=15.4, h=9.3, vs=0.025, level="s1"),
}
layouts = {
    "64^3": lambda axis: (64, 64, 64),
    "32^3": lambda axis: (32, 32, 32),
    "16^3": lambda axis: (16, 16, 16),
    "thin 64x64x8 per orientation": lambda axis: tuple(
        8 if a == axis else 64 for a in range(3)
    ),
}
for name, v in views.items():
    nat, ideal = set(), set()
    for axis in range(3):
        p = plane_points(v["pos"], axis, v["w"], v["h"], v["vs"])
        nat |= spim_chunks(p + SHIFT, v["level"])
        ideal |= spim_chunks(p + u_out(p), v["level"])
    print(f"{name} ({v['level']}): native {len(nat)}, warp visible pixels {len(ideal)}")
    for lname, lay in layouts.items():
        full = set()
        n = 0
        vox = 0
        for axis in range(3):
            pw, nc = warped_chunk_points(v["pos"], axis, v["w"], v["h"], v["vs"], cs=lay(axis))
            n += nc
            vox += nc * int(np.prod(lay(axis)))
            full |= spim_chunks(pw + u_out(pw), v["level"])
        print(
            f"   {lname:30s} SPIM chunks {len(full):3d}   warped chunks {n:4d}   voxels computed {vox / 1e6:6.1f} M"
        )
