"""Neuroglancer states for the FAFB v14 -> v14.1 field cutout.

v14_warped.json   FAFB v14 (CLAHE) warped into v14.1 by the field, next to
                  FlyWire's v14.1 render and unwarped v14 (toggle layers).
v14_overlay.json  warped v14 (green) over v14.1 (magenta): grey where they
                  agree.
For another store (--name ..._eb4) the files are eb4_v14_warped.json etc.;
--compare NAME adds eb4_vs_NAME.json, v14 warped by both stores.

  uv run python examples/fafb_seamless/make_states.py [--base-url http://127.0.0.1:9080]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import zarr
from make_field import OUT

import zarr_warpfield  # noqa: F401

V14_CLAHE = "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe"
V14_ORIG = "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig"
V141 = "precomputed://gs://microns-seunglab/drosophila_v0/alignment/image_rechunked"

COLOR = """#uicontrol vec3 color color(default="{c}")
void main() {{ emitRGB(color * toNormalized(getDataValue())); }}
"""


def layer(name, url, warp=None, visible=True, shader=None, blend=None):
    source = {"url": url} if warp is None else {"url": url, "transformSource": warp, "prependTransform": True}
    out = {"type": "image", "source": source, "name": name, "tab": "source"}
    if not visible:
        out["visible"] = False
    if shader:
        out["shader"] = shader
    if blend:
        out["blend"] = blend
    return out


def state(layers, position, zoom=1.0):
    return {
        "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
        "position": position,
        "crossSectionScale": zoom,
        "projectionScale": 8192,
        "layers": layers,
        "selectedLayer": {"layer": layers[-1]["name"], "visible": True},
        "layout": "xy",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fafb_v14_to_v14.1")
    ap.add_argument("--base-url", default="http://127.0.0.1:9080")
    ap.add_argument("--center", type=float, nargs=3, default=[74067, 42119, 4384])
    ap.add_argument(
        "--compare",
        help="another field store: also write <name>_vs_<compare>.json, v14 warped "
        "by each (green: this one, magenta: the other)",
    )
    ap.add_argument(
        "--out-dir",
        help="where to write the states (default out/fafb_seamless/states)",
    )
    ap.add_argument(
        "--zoom", type=float, default=1.0, help="crossSectionScale (4 nm pixels per screen px)"
    )
    a = ap.parse_args()
    warp = f"zarr3://{a.base_url}/{a.name}.zarr/"
    # the center is in v14; the warped layers live in v14.1: q = p - u(p), roughly
    root = zarr.open_group(OUT / f"{a.name}.zarr", mode="r")
    src = root.attrs["warpfield"]["source"]["region_16nm"]
    k = int(a.center[2]) - src["z"][0]
    j, i = int(a.center[1] / 4) - src["y"][0], int(a.center[0] / 4) - src["x"][0]
    u = root["s0"][:, k, j, i]  # (dz, dy, dx) nm
    position = [a.center[0] - u[2] / 4, a.center[1] - u[1] / 4, a.center[2] + 0.5]
    position = [round(float(p), 1) for p in position]
    states = Path(a.out_dir) if a.out_dir else OUT / "states"
    states.mkdir(parents=True, exist_ok=True)
    warped = state(
        [
            layer("v14 (unwarped)", V14_CLAHE, visible=False),
            layer("v14.1 (FlyWire render)", V141, visible=False),
            layer("v14 warped to v14.1 (warpfield)", V14_CLAHE, warp=warp),
        ],
        position,
        a.zoom,
    )
    overlay = state(
        [
            layer("v14.1 (magenta)", V141, shader=COLOR.format(c="#ff00ff")),
            layer(
                "v14 warped (green)",
                V14_ORIG,
                warp=warp,
                shader=COLOR.format(c="#00ff00"),
                blend="additive",
            ),
        ],
        position,
        a.zoom,
    )
    out = [("v14_warped", warped), ("v14_overlay", overlay)]
    if a.compare:
        other = f"zarr3://{a.base_url}/{a.compare}.zarr/"
        compare = state(
            [
                layer(
                    f"v14 warped by {a.compare} (magenta)",
                    V14_ORIG,
                    warp=other,
                    shader=COLOR.format(c="#ff00ff"),
                ),
                layer(
                    f"v14 warped by {a.name} (green)",
                    V14_ORIG,
                    warp=warp,
                    shader=COLOR.format(c="#00ff00"),
                    blend="additive",
                ),
            ],
            position,
            a.zoom,
        )
        out.append((f"vs_{a.compare}", compare))
    # the default store keeps the plain names
    prefix = (
        "" if a.name == "fafb_v14_to_v14.1" else a.name.removeprefix("fafb_v14_to_v14.1_") + "_"
    )
    for name, s in out:
        path = states / f"{prefix}{name}.json"
        path.write_text(json.dumps(s, indent=2))
        print(path, position)


if __name__ == "__main__":
    main()
