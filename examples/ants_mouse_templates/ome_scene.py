"""Put a forward and an inverse displacement field in an OME-Zarr 0.6 scene that
declares the transformation between two template spaces (RFC-5).

    ome_scene.py SCENE.zarr --forward forward/ --inverse inverse/ \\
        --input-space "STPT template 2020, mm RAS" \\
        --output-space "SPIM template LCA 2024-05, mm RAS"

The fields are written by ome_field.py (fields written by its earlier version,
with attributes.displacement_field, are converted). They are copied into the
scene; their chunks are not re-encoded.

Layout
------
SCENE.zarr/
  zarr.json                       attributes.ome.scene: the two coordinate systems
                                  and one "bijection" between them
  coordinateTransformations/
    forward/                      field on the input space's grid
    inverse/                      field on the output space's grid

The bijection's forward transformation maps a point x of the input space to the
output space. In ANTs terms that is the transform that resamples the output-space
image into the input space: W(x) = M(forward(x)). It is a sequence: the field's
affine from the space to its array-aligned coordinate system "intrinsic", the
`displacements`, and the affine back. The inverse transformation is the same with
the inverse field.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SPACE_AXES = [{"name": a, "type": "space", "unit": "millimeter"} for a in ("z", "y", "x")]


def field_affine(field: Path) -> np.ndarray:
    """The field's 3x4 affine from "intrinsic" to its output space (spatial part)."""
    ome = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]
    (ms,) = ome["multiscales"]
    (t,) = [x for x in ms["coordinateTransformations"] if x["input"]["name"] == "intrinsic"]
    a = np.asarray(t["affine"], float)
    axes = next(c for c in ms["coordinateSystems"] if c["name"] == "intrinsic")["axes"]
    spatial = [i for i, ax in enumerate(axes) if ax.get("type") == "space"]
    return np.concatenate([a[np.ix_(spatial, spatial)], a[spatial, -1:]], axis=1)


def field_geometry(field: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per level, the 3x4 map from spatial array index (sample n at n) to the
    output space; and the 3x3 map from vectors to output displacements."""
    ome = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]
    (ms,) = ome["multiscales"]
    a = field_affine(field)
    axes = next(c for c in ms["coordinateSystems"] if c["name"] == "intrinsic")["axes"]
    spatial = [i for i, ax in enumerate(axes) if ax.get("type") == "space"]
    levels = {}
    for d in ms["datasets"]:
        scale, trans = np.ones(len(axes)), np.zeros(len(axes))
        for t in d["coordinateTransformations"]:
            for x in t["transformations"] if t["type"] == "sequence" else [t]:
                if x["type"] == "scale":
                    scale *= x["scale"]
                    trans *= x["scale"]
                elif x["type"] == "translation":
                    trans += x["translation"]
        s, t = scale[spatial], trans[spatial]
        levels[d["path"]] = np.concatenate(
            [a[:, :3] * s[None, :], (a[:, :3] @ t + a[:, 3])[:, None]], axis=1
        )
    return levels, a[:, :3]


def inverse_affine(a: np.ndarray) -> np.ndarray:
    lin = np.linalg.inv(a[:, :3])
    return np.concatenate([lin, -lin @ a[:, 3:]], axis=1) + 0.0  # no -0.0


def upgrade_field(field: Path) -> None:
    """Convert a field group written with attributes.displacement_field (in place)."""
    zj = field / "zarr.json"
    meta = json.loads(zj.read_text())
    attrs = meta["attributes"]
    old = attrs.pop("displacement_field", None)
    if old is None:
        return
    for cs in attrs["ome"]["multiscales"][0]["coordinateSystems"]:
        for ax in cs["axes"]:
            if ax.get("type") == "channel":
                ax.clear()
                ax.update({"name": "c", "type": "displacement", "discrete": True})
    attrs["warpfield"] = {
        "version": "0.2",
        "error_bound": old["error_bound"],
        "levels": [
            {"path": s["path"], "interpolation_error": s["interpolation_error"]}
            for s in old["scales"]
        ],
        "bounds": {"path": old["bounds"]["path"], "block_shape": old["bounds"]["block_shape"]},
        "extrapolation": old.get("extrapolation", "nearest"),
        "det_jacobian_min": attrs.pop("det_jacobian_min", None),
        "source": old.get("source"),
    }
    # the vectors must be in "intrinsic" (RFC-5): check against the old metadata
    a = field_affine(field)
    if not np.allclose(np.asarray(old["vector_to_output"]), a[:, :3]):
        raise ValueError(f"{field}: vectors are not aligned with the array axes")
    zj.write_text(json.dumps(meta, indent=2))


def displacements(field_path: str, affine: np.ndarray) -> dict:
    return {
        "type": "sequence",
        "transformations": [
            {"type": "affine", "affine": inverse_affine(affine).tolist()},
            {"type": "displacements", "path": field_path, "interpolation": "linear"},
            {"type": "affine", "affine": affine.tolist()},
        ],
    }


def write(scene: Path, forward: Path, inverse: Path, input_space: str, output_space: str):
    if scene.exists():
        shutil.rmtree(scene)
    paths = {}
    for name, src in (("forward", forward), ("inverse", inverse)):
        dst = scene / "coordinateTransformations" / name
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store"))
        upgrade_field(dst)
        paths[name] = f"coordinateTransformations/{name}"
    (scene / "coordinateTransformations" / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}, indent=2)
    )
    root = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.6",
                "scene": {
                    "coordinateSystems": [
                        {"name": input_space, "axes": SPACE_AXES},
                        {"name": output_space, "axes": SPACE_AXES},
                    ],
                    "coordinateTransformations": [
                        {
                            "type": "bijection",
                            "name": f"{input_space} -> {output_space}",
                            "input": {"name": input_space},
                            "output": {"name": output_space},
                            "forward": displacements(
                                paths["forward"],
                                field_affine(scene / paths["forward"]),
                            ),
                            "inverse": displacements(
                                paths["inverse"],
                                field_affine(scene / paths["inverse"]),
                            ),
                        }
                    ],
                },
            }
        },
    }
    (scene / "zarr.json").write_text(json.dumps(root, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--inverse", type=Path, required=True)
    ap.add_argument("--input-space", required=True)
    ap.add_argument("--output-space", required=True)
    a = ap.parse_args()
    write(a.scene, a.forward, a.inverse, a.input_space, a.output_space)
    print(a.scene)


if __name__ == "__main__":
    main()
