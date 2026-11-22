"""Neuroglancer states: the MICrONS in vivo stack and 2P fields, warped onto the EM.

    make_states.py OUT --fields-url http://127.0.0.1:9078 \\
        [--coreg-state all_fields_coreg.json]

Reads OUT/placement.json (make_fields.py). Every moving layer is placed in EM
space with the affine L, composed with its own transform into the stack (the 2P
fields' affines from the coregistration state, or the stack's voxel size), and
warped by the forward field. Writes:

- OUT/states/warped.json: EM (minnie65 + 35), blood vessels, segmentation, the
  stack (vessels red, cells green) and all 2P fields, warped onto the EM.
- OUT/states/affine.json: the same placed by L only (no warp), for comparison.
- OUT/states/field.json: the forward field over the EM, with det J.

The states need a Neuroglancer build with warp support (warpfield-transform).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

COREG_STATE = "https://storage.googleapis.com/microns-static-links/mm3/all_fields_coreg.json"
FIELD_SHADER = (
    Path(__file__).resolve().parent.parent
    / "ants_mouse_templates/neuroglancer/field_shader.glsl"
)

EM_LAYERS = [
    {
        "type": "image",
        "source": [
            {
                "url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie35/em",
                "subsources": {"default": True, "bounds": True},
                "enableDefaultSubsources": False,
            },
            "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
        ],
        "name": "em",
    },
    {
        "type": "segmentation",
        "source": {
            "url": "precomputed://s3://bossdb-open-data/wei2024/minnie/bv",
            "subsources": {"default": True, "mesh": True},
            "enableDefaultSubsources": False,
        },
        "pick": False,
        "segments": ["1"],
        "segmentDefaultColor": "#ff0000",
        "segmentColors": {"1": "#ab2b2b"},
        "name": "bv",
    },
    {
        "type": "segmentation",
        "source": [
            "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300",
            "precomputed://gs://iarpa_microns/minnie/minnie35/seg",
        ],
        "segments": [],
        "visible": False,
        "name": "seg",
    },
]

VIEW = {
    "dimensions": {"x": [2.56e-7, "m"], "y": [2.56e-7, "m"], "z": [1.28e-6, "m"]},
    "position": [3483.0, 2922.2, 464.2],
    "crossSectionScale": 4.0,
    "projectionScale": 5423.6,
    "layout": "4panel",
}

UM = [1e-6, "m"]


def em_matrix(placement) -> np.ndarray:
    """4x4: stack um (x, y, z) -> EM um (x, y, z)."""
    L = np.asarray(placement["stack_to_em_um"])  # rows (z, y, x)
    P = np.eye(4)[[2, 1, 0, 3]]  # (z, y, x) -> (x, y, z)
    return P @ L


def compose(M: np.ndarray, matrix) -> list:
    """M (4x4) after a 3x4 Neuroglancer matrix, as a 3x4 list."""
    F = np.vstack([np.asarray(matrix, float), [0, 0, 0, 1]])
    return (M @ F)[:3].round(9).tolist()


def stack_layer(coreg_state, M, warp_url):
    layer = copy.deepcopy(next(lyr for lyr in coreg_state["layers"] if lyr["name"] == "stack"))
    url = layer["source"] if isinstance(layer["source"], str) else layer["source"]["url"]
    # inputs (c^, x, y, z). Neuroglancer transform matrices are physical:
    # the stack's own (identity) transform already maps its 2 um voxels to um.
    spatial = compose(M, np.c_[np.eye(3), np.zeros(3)])
    matrix = [[1, 0, 0, 0, 0]] + [[0, *row] for row in spatial]
    source = {
        "url": url,
        "transform": {
            "matrix": matrix,
            "outputDimensions": {"c^": [1, ""], "x": UM, "y": UM, "z": UM},
        },
    }
    if warp_url:
        source["warp"] = warp_url
    layer["source"] = source
    layer["name"] = "in vivo stack"
    layer["opacity"] = 0.8
    layer["blend"] = "additive"
    layer.pop("tab", None)
    return layer


def scan_layers(coreg_state, M, warp_url):
    out = []
    for layer in coreg_state["layers"]:
        if layer["name"] == "stack":
            continue
        layer = copy.deepcopy(layer)
        sources = layer["source"] if isinstance(layer["source"], list) else [layer["source"]]
        new = []
        for s in sources:
            t = s["transform"]
            src = {
                "url": s["url"],
                "transform": {
                    "matrix": compose(M, t["matrix"]),
                    "outputDimensions": {"x": UM, "y": UM, "z": UM},
                },
            }
            if warp_url:
                src["warp"] = warp_url
            new.append(src)
        layer["source"] = new
        layer["visible"] = False
        layer.pop("tab", None)
        out.append(layer)
    return out


def field_layer(field_url, spacing_um):
    # The ANTs example's shader has slider ranges for mm fields; this field is
    # in um and holds only what is not affine (offsets 0).
    shader = re.sub(
        r"#uicontrol float spacing_mm slider\([^)]*\)",
        f"#uicontrol float spacing_mm slider(min=0.5, max=50, default={spacing_um})",
        FIELD_SHADER.read_text(),
    )
    shader = re.sub(
        r"#uicontrol float range_mm slider\([^)]*\)",
        "#uicontrol float range_mm slider(min=1, max=400, default=50)",
        shader,
    )
    shader = re.sub(
        r"(#uicontrol float offset\d slider\()[^)]*\)",
        r"\1min=-400, max=400, default=0)",
        shader,
    )
    return {
        "type": "image",
        "name": "forward field: det J",
        "source": {
            "url": f"zarr3://{field_url}",
            "transform": {"outputDimensions": {"c^": [1, ""], "z": UM, "y": UM, "x": UM}},
        },
        "shader": shader,
        # the shader's "mm" controls are just the field's own units (um here)
        "shaderControls": {"spacing_mm": spacing_um},
        "opacity": 0.6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--fields-url", required=True, help="URL of OUT, as served")
    ap.add_argument("--coreg-state", default=COREG_STATE)
    a = ap.parse_args()
    placement = json.loads((a.out / "placement.json").read_text())
    src = a.coreg_state
    if src.startswith("http"):
        with urllib.request.urlopen(src) as r:
            coreg_state = json.load(r)
    else:
        coreg_state = json.loads(Path(src).read_text())
    M = em_matrix(placement)
    warp = f"{a.fields_url.rstrip('/')}/{placement['forward_field']}/"
    states = a.out / "states"
    states.mkdir(exist_ok=True)
    for name, warp_url in (("warped", warp), ("affine", None)):
        state = dict(VIEW)
        state["layers"] = (
            copy.deepcopy(EM_LAYERS)
            + [stack_layer(coreg_state, M, warp_url)]
            + scan_layers(coreg_state, M, warp_url)
        )
        state["selectedLayer"] = {"layer": "in vivo stack", "visible": True}
        (states / f"{name}.json").write_text(json.dumps(state, indent=1))
    state = dict(VIEW)
    state["layers"] = [copy.deepcopy(EM_LAYERS[0]), field_layer(warp, placement["spacing_um"])]
    (states / "field.json").write_text(json.dumps(state, indent=1))
    print("wrote", sorted(p.name for p in states.iterdir()))


if __name__ == "__main__":
    main()
