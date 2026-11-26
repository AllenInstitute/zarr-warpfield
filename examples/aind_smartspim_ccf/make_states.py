"""Neuroglancer states for the SmartSPIM 894847 <-> CCF warp fields.

    make_states.py OUT --fields-url http://127.0.0.1:9079/

Reads OUT/placement.json (make_fields.py) and the dataset's own state
(neuroglancer_config.json) for the channel layers. Writes OUT/states/:

- ccf_on_brain.json: the three fused channels in their own space, with the
  CCF 2017 annotation (volume and meshes) and the STPT 2020 template placed by
  L and warped by the forward field; the pipeline's static annotation
  (ccf_annotation_precomputed) alongside, for comparison.
- ccf_on_brain_affine.json: the same, placed by L only (no warp).
- brain_in_ccf.json: the channels warped into "STPT placed in brain" space by
  the inverse field; CCF annotation and STPT template placed by L.
- The SPIM LCA template is in every state, warped by the field that maps the
  display space to it directly: in brain space the pipeline's registration
  alone (smartspim_894847_spim.zarr), in CCF space the template fields alone
  (stpt_spim_placed.zarr); placed by L_spim only in the affine state.
- brain_vs_{template,spim_template}_in_{brain,ccf}_{coronal,horizontal}.json:
  the brain (magenta) over a template (green).
- Every state also shows the field(s) of its display space as det J images
  (white = volume kept, red = expansion, blue = compression, yellow = det J <=
  min_det; field_layer), the ones not drawing a visible layer hidden.

Neuroglancer source transform matrices act on physical coordinates in the
output units (here um), so a source whose own coordinates are mm keeps its
linear part and has its translation scaled by 1000.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

DATASET = "894847_2026-11-20_11-27-00_stitched_2026-11-21_13-35-21"
DATASET_STATE = f"https://aind-open-data.s3.us-west-2.amazonaws.com/{DATASET}/neuroglancer_config.json"
STATIC_ANNOTATION = f"precomputed://s3://aind-open-data/{DATASET}/image_atlas_alignment/ccf_annotation_precomputed"
ATLAS = "https://allen-atlas-assets.s3.us-west-2.amazonaws.com"
CCF_2017 = f"precomputed://{ATLAS}/annotation-sets/allen-adult-mouse-annotation/2017/annotations.precomputed"
STPT_2020_ZARR = f"{ATLAS}/templates/allen-adult-mouse-stpt-template/2020/template.ome.zarr/"
SPIM_LCA_ZARR = f"{ATLAS}/templates/allen-adult-mouse-spim-lca-template/2024-05/template.ome.zarr/"
UM = [1e-6, "m"]
# CCF 2017 precomputed (x, y, z, mm) -> STPT 2020 mm RAS (z, y, x): voxel centres
# of the 10 um grid at n * 10 um in the 2015 frame, 2015 -> 2020 translation
# from the template_25 headers (examples/ants_mouse_templates)
CCF_TO_STPT_MM = np.array(
    [[0.0, -1.0, 0.0, 5.225], [0.0, 0.0, -1.0, 5.355], [1.0, 0.0, 0.0, -5.7], [0, 0, 0, 1]]
)


def intrinsic_to_ras_mm(zarr_url: str) -> np.ndarray:
    """4x4 intrinsic (z, y, x) mm -> "mm RAS" (z, y, x) of an Allen template
    OME-Zarr 0.6, composed from its multiscale transform sequence. A source
    "transform" in Neuroglancer replaces this default placement, so a template's
    matrix must include it."""
    with urllib.request.urlopen(zarr_url + "zarr.json") as r:
        ms = json.load(r)["attributes"]["ome"]["multiscales"][0]
    (t,) = [t for t in ms["coordinateTransformations"] if t["output"]["name"] == "mm RAS"]
    M = np.eye(4)
    for step in t["transformations"] if t["type"] == "sequence" else [t]:
        S = np.eye(4)
        if step["type"] == "affine":
            S[:3] = np.asarray(step["affine"], float)
        elif step["type"] == "rotation":
            S[:3, :3] = np.asarray(step["rotation"], float)
        elif step["type"] == "translation":
            S[:3, 3] = np.asarray(step["translation"], float)
        elif step["type"] == "scale":
            S[:3, :3] = np.diag(step["scale"])
        else:
            raise ValueError(f"unsupported transformation {step['type']}")
        M = S @ M
    return M


FIELD_SHADER = Path(__file__).resolve().parent.parent / "ants_mouse_templates/neuroglancer/field_shader.glsl"


def field_layer(out: Path, base_url: str, store: str, name: str, visible: bool = True) -> dict:
    """A displacement field as an image in the display space it lives in,
    coloured by det J of x -> x + u(x): white = volume preserved, red =
    expansion, blue = compression, yellow = det J <= min_det.

    The shader differences neighbouring samples at a fixed spacing, so
    Neuroglancer must not switch to the coarser levels when zoomed out: the
    layer reads OUT/fields_s0/<store>, the field's own OME metadata with only
    its s0 level (a symlink to the field's s0 chunks). (The bare s0 array with
    an explicit source transform renders a constant in this Neuroglancer.)"""
    src = out / store
    zj = json.loads((src / "zarr.json").read_text())
    ms = zj["attributes"]["ome"]["multiscales"][0]
    ms["datasets"] = ms["datasets"][:1]
    zj["attributes"].get("warpfield", {}).pop("levels", None)
    h = ms["datasets"][0]["coordinateTransformations"][0]["transformations"][0]["scale"][1]
    rel = Path("fields_s0") / store.replace("/coordinateTransformations/", "_")
    dst = out / rel
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "zarr.json").write_text(json.dumps(zj, indent=1))
    link = dst / ms["datasets"][0]["path"]
    if not link.is_symlink():
        link.symlink_to(src.resolve() / ms["datasets"][0]["path"])
    shader = FIELD_SHADER.read_text()
    for control, spec in (("spacing_mm", f"min=1, max=100, default={h}"), ("range_mm", "min=1, max=1000, default=100")):
        shader = re.sub(rf"#uicontrol float {control} slider\([^)]*\)", f"#uicontrol float {control} slider({spec})", shader)
    shader = re.sub(r"(#uicontrol float offset\d slider\()[^)]*\)", r"\1min=-1000, max=1000, default=0)", shader)
    return {
        "type": "image",
        "name": name,
        "source": {
            "url": f"zarr3://{base_url}/{rel.as_posix()}/",
            "transform": {"outputDimensions": {"c^": [1, ""], "z": UM, "y": UM, "x": UM}},
        },
        "shader": shader,  # the shader's "mm" controls are the field's own units, um here
        "shaderControls": {"log2_range": 2.0},
        "opacity": 0.5,
        "visible": visible,
    }


SEGMENTS = ["672", "382", "795", "56", "733", "961", "1089", "1097"]


def um_input(m_mm: np.ndarray) -> np.ndarray:
    """A 4x4 map whose input and output are mm, as one acting on um."""
    m = m_mm.copy()
    m[:3, 3] *= 1000.0
    return m


def matrix(m: np.ndarray) -> list:
    return m[:3].round(9).tolist()


def ccf_layer(name, M, warp_url, meshes=True):
    src = {"url": CCF_2017, "transform": {"matrix": matrix(M), "outputDimensions": {"z": UM, "y": UM, "x": UM}}}
    if warp_url:
        src["transformSource"] = warp_url
        src["prependTransform"] = True
    layer = {
        "type": "segmentation",
        "source": src,
        "name": name,
        "selectedAlpha": 0.35,
        "notSelectedAlpha": 0.08,
        "objectAlpha": 0.85,
        "segments": SEGMENTS if meshes else [],
    }
    return layer


def template_layer(zarr_url, name, colour, value_range, M_mm_to_brain, warp_url, visible=False):
    """An Allen template OME-Zarr placed in brain um by M (its mm RAS -> brain um)."""
    # template intrinsic (um) -> mm RAS (as um) -> brain um (M acting on um)
    M_um = M_mm_to_brain.copy()
    M_um[:3, :3] /= 1000.0
    m = M_um @ um_input(intrinsic_to_ras_mm(zarr_url))
    src = {"url": f"zarr3://{zarr_url}", "transform": {"matrix": matrix(m), "outputDimensions": {"z": UM, "y": UM, "x": UM}}}
    if warp_url:
        src["transformSource"] = warp_url
        src["prependTransform"] = True
    rgb = {"green": "0.0, normalized(), 0.0", "cyan": "0.0, normalized(), normalized()"}[colour]
    return {
        "type": "image",
        "source": src,
        "name": name,
        "visible": visible,
        "opacity": 0.6,
        "blend": "additive",
        "shader": f"#uicontrol invlerp normalized(range=[{value_range[0]}, {value_range[1]}])\n"
        f"void main() {{ emitRGB(vec3({rgb})); }}",
    }


def stpt_layer(L, warp_url, visible=False):
    return template_layer(STPT_2020_ZARR, "STPT template 2020", "green", (0, 500), L, warp_url, visible)


def spim_layer(Ls, warp_url, visible=False):
    return template_layer(SPIM_LCA_ZARR, "SPIM LCA template 2024-05", "cyan", (0, 1.1), Ls, warp_url, visible)


def channel_layers(dataset_state, warp_url):
    out = []
    for layer in dataset_state["layers"]:
        layer = copy.deepcopy(layer)
        if warp_url:
            url = layer["source"] if isinstance(layer["source"], str) else layer["source"]["url"]
            layer["source"] = {"url": url, "transformSource": warp_url, "prependTransform": True}
        layer.pop("tab", None)
        out.append(layer)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--fields-url", required=True, help="URL of OUT, as served")
    ap.add_argument("--dataset-state", default=DATASET_STATE)
    a = ap.parse_args()
    placement = json.loads((a.out / "placement.json").read_text())
    L = np.asarray(placement["stpt_ras_mm_to_brain_um"])  # mm (z, y, x) -> um (z, y, x)
    Ls = np.asarray(placement["spim_ras_mm_to_brain_um"])
    base = a.fields_url.rstrip("/")
    fwd = f"{base}/{placement['forward_field']}/"
    inv = f"{base}/{placement['inverse_field']}/"
    spim_fwd = f"{base}/{placement['spim_forward_field']}/"  # brain -> SPIM placed by Ls
    stpt_to_spim = f"{base}/{placement['stpt_to_spim_field']}/"  # STPT placed by L -> SPIM placed by Ls
    src = a.dataset_state
    if src.startswith("http"):
        with urllib.request.urlopen(src) as r:
            dataset_state = json.load(r)
    else:
        dataset_state = json.loads(Path(src).read_text())

    # CCF precomputed (um input) -> brain um: L after the CCF placement
    L_um = L.copy()
    L_um[:3, :3] /= 1000.0  # L acting on um input
    M_ccf = L_um @ um_input(CCF_TO_STPT_MM)
    centre = (np.array([486 * 16.0, 1281 * 14.4, 929 * 14.4]) / 2).round(1).tolist()  # z, y, x
    view = {
        "dimensions": {"x": UM, "y": UM, "z": UM, "t": [0.001, "s"]},
        "position": [centre[2], centre[1], centre[0], 0.5],
        "crossSectionScale": 20,
        "projectionScale": 25000,
        "layout": "4panel",
    }
    states = a.out / "states"
    states.mkdir(exist_ok=True)
    # the fields as det J images, each in the display space it lives in
    F = {
        "fwd": lambda v=True: field_layer(a.out, base, placement["forward_field"], "field det J: brain -> STPT", v),
        "inv": lambda v=True: field_layer(a.out, base, placement["inverse_field"], "field det J: STPT -> brain", v),
        "spim": lambda v=True: field_layer(a.out, base, placement["spim_forward_field"], "field det J: brain -> SPIM template", v),
        "stpt_spim": lambda v=True: field_layer(a.out, base, placement["stpt_to_spim_field"], "field det J: STPT -> SPIM template", v),
    }
    static = {"type": "segmentation", "source": STATIC_ANNOTATION, "name": "pipeline CCF annotation (static)",
              "visible": False, "selectedAlpha": 0.35, "notSelectedAlpha": 0.08}
    for name, warp in (("ccf_on_brain", fwd), ("ccf_on_brain_affine", None)):
        st = copy.deepcopy(view)
        st["layers"] = (
            channel_layers(dataset_state, None)
            + [stpt_layer(L, warp), spim_layer(Ls, spim_fwd if warp else None), copy.deepcopy(static),
               ccf_layer("CCF 2017 annotation" + (" (warped)" if warp else " (affine)"), M_ccf, warp),
               F["fwd"](), F["spim"](False)]
        )
        st["selectedLayer"] = {"layer": st["layers"][-3]["name"], "visible": True}
        (states / f"{name}.json").write_text(json.dumps(st, indent=1))
    st = copy.deepcopy(view)
    st["layers"] = (
        channel_layers(dataset_state, inv)
        + [stpt_layer(L, None, visible=True), spim_layer(Ls, stpt_to_spim),
           ccf_layer("CCF 2017 annotation", M_ccf, None), F["inv"](), F["stpt_spim"](False)]
    )
    (states / "brain_in_ccf.json").write_text(json.dumps(st, indent=1))

    # Side-by-side checks: one coronal panel, the 639 channel in grayscale, all
    # CCF regions at once (same colours by ID in every layer), one overlay each.
    gray = next(copy.deepcopy(lyr) for lyr in dataset_state["layers"] if "639" in lyr["name"])
    gray.pop("tab", None)
    gray["shader"] = "#uicontrol invlerp normalized\nvoid main() { emitGrayscale(normalized()); }"
    for name, overlay in (
        ("compare_warped", ccf_layer("CCF 2017 annotation (warped)", M_ccf, fwd, meshes=False)),
        ("compare_affine", ccf_layer("CCF 2017 annotation (affine)", M_ccf, None, meshes=False)),
        ("compare_static", dict(static, visible=True)),
    ):
        overlay = copy.deepcopy(overlay)
        overlay["segments"] = []
        overlay["notSelectedAlpha"] = 0.3
        overlay["selectedAlpha"] = 0.3
        st = {
            "dimensions": view["dimensions"],
            "position": view["position"],
            "crossSectionScale": 9,
            "layout": "xz",
            "layers": [gray, overlay, F["fwd"]()],
        }
        (states / f"{name}.json").write_text(json.dumps(st, indent=1))
    # warped CCF meshes with a coronal slice of the brain (639, grayscale)
    st = {
        "dimensions": view["dimensions"],
        "position": view["position"],
        "projectionScale": 16000,
        "projectionOrientation": [-0.35, 0.25, 0.1, 0.9],
        "layout": "3d",
        "showSlices": True,
        "layers": [copy.deepcopy(gray), ccf_layer("CCF 2017 meshes (warped)", M_ccf, fwd), F["fwd"]()],
    }
    st["layers"][1]["notSelectedAlpha"] = 0.0
    (states / "meshes_on_brain.json").write_text(json.dumps(st, indent=1))

    # the brain (639, magenta) over the STPT template (green), in each space
    magenta = copy.deepcopy(gray)
    magenta["blend"] = "additive"
    magenta["shader"] = "#uicontrol invlerp normalized\nvoid main() { emitRGB(vec3(normalized(), 0.0, normalized())); }"
    for name, brain_warp, tmpl, fields in (
        ("brain_vs_template_in_ccf", inv, stpt_layer(L, None, visible=True), [F["inv"]()]),
        ("brain_vs_template_in_brain", None, stpt_layer(L, fwd, visible=True), [F["fwd"]()]),
        ("brain_vs_spim_template_in_brain", None, spim_layer(Ls, spim_fwd, visible=True), [F["spim"]()]),
        ("brain_vs_spim_template_in_ccf", inv, spim_layer(Ls, stpt_to_spim, visible=True),
         [F["inv"](), F["stpt_spim"](False)]),
    ):
        b = copy.deepcopy(magenta)
        if brain_warp:
            b["source"] = {"url": b["source"] if isinstance(b["source"], str) else b["source"]["url"], "transformSource": brain_warp, "prependTransform": True}
        t = copy.deepcopy(tmpl)
        t["opacity"] = 1.0
        if "spim" in name:  # green, like the STPT comparisons
            t["shader"] = t["shader"].replace("vec3(0.0, normalized(), normalized())", "vec3(0.0, normalized(), 0.0)")
        for plane, layout in (("coronal", "xz"), ("horizontal", "xy")):
            st = {"dimensions": view["dimensions"], "position": view["position"], "crossSectionScale": 9,
                  "layout": layout, "layers": [t, b] + copy.deepcopy(fields)}
            (states / f"{name}_{plane}.json").write_text(json.dumps(st, indent=1))
    print("wrote", sorted(p.name for p in states.iterdir()))


if __name__ == "__main__":
    main()
