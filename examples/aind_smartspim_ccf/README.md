# SmartSPIM brain ↔ CCF: composing two registrations into warp fields

A whole-brain SmartSPIM light-sheet dataset from the AIND pipeline
(`s3://aind-open-data/894847_2026-11-20_11-27-00_stitched_2026-11-21_13-35-21`,
three fused channels, 1.8 × 1.8 × 2 µm), brought into register with the Allen
CCF by composing:

1. **brain → SPIM LCA template (2024-05)**: the pipeline's ANTs SyN registration
   (`image_atlas_alignment/Ex_639_Em_667/ls_to_template_SyN_0GenericAffine.mat`,
   `…_1Warp.nii.gz`, `…_1InverseWarp.nii.gz`);
2. **SPIM LCA template ↔ STPT template 2020 (stereotaxic CCF frame)**: the
   template-to-template fields of `examples/ants_mouse_templates`
   (`transform-SPIM.nii.gz`, `inverse-transform-SPIM.nii.gz`), which register
   the templates better than the template → CCF transform the pipeline used.

## Which transforms do what

The pipeline (`aind-smartspim-ccf-registration` 0.0.37, `processing.json`)
registers **level 3** of the fused OME-Zarr (16 × 14.4 × 14.4 µm, z y x) to the
SPIM LCA template, then maps to CCF with a separate template → CCF transform
from its own data asset (`lightsheet_template_ccf_registration`).

- `ls_to_template_SyN_0GenericAffine.mat` + `…_1Warp` / `…_1InverseWarp`: the
  final brain ↔ template registration. Fixed = template, moving = brain. A
  template point x maps to the brain point A(x + W(x)); a brain point y maps to
  the template point z + IW(z), z = A⁻¹(y).
- `ls_to_template_rigid_0GenericAffine.mat`, `…_affine_0GenericAffine.mat`:
  only the initialisations of that registration; A already contains them.
- `ccf_annotation_precomputed` (the static overlay) is a CCF annotation that
  had been moved into template space, resampled onto the brain's level-3 grid
  with `[A⁻¹, IW]`.

The brain frame: the pipeline reorders the level-3 array into the template's
axis order (right→left, anterior→posterior, superior→inferior; the acquisition
is z superior→inferior, y anterior→posterior, x left→right, so x is flipped)
and gives it the template's origin and direction. `chain.py` documents the
exact index conventions. The SPIM template frame of the pipeline, of
allen-atlas-assets and of `inverse-transform-SPIM.nii.gz` (padded 32 voxels)
are identical.

## Steps

```sh
# inputs: a local copy of the dataset's small files and the two SyN warps
# (1.4 GB each), the two template fields in ~/Downloads, two 25 um annotations
uv run --with nibabel --with scipy python validate.py SRC ATLAS CACHE --stride 3
uv run --with nibabel --with scipy python make_fields.py OUT --src SRC --spacing 25 --error-bound 1
python make_states.py OUT --fields-url http://127.0.0.1:9079/
uv run --with nibabel --with scipy --with SimpleITK python itk_check.py OUT --src SRC \
    --point 3495 9266 1837 --name x3495_y9266_z1837
```

Published: fields under `gs://allen_neuroglancer_ccf/warpfields/aind_894847/`,
states under `warpfields/states/aind_894847/`, viewed with the PR 1160 preview,
e.g. <https://neuroglancer--pr1160-25yzr4zh.web.app/#!https://storage.googleapis.com/allen_neuroglancer_ccf/warpfields/states/aind_894847/ccf_on_brain.json>.

## Results

**Conventions check** (`validate.py`, 4.6 M voxels of the pipeline's own
CCF-on-brain labels, `moved_ccf_anno_to_ls.nii.gz`, nearest-neighbour):

| chain | label agreement | brain-mask Dice | median Dice, 12 largest structures |
|---|---|---|---|
| brain → SPIM template (atlas-assets CCF-in-SPIM annotation) | 96.5% | 0.996 | 0.982 |
| brain → template → STPT 2020 (these template fields, CCF 2017) | 73.0% | 0.976 | 0.879 |

The first row reproduces the pipeline (the rest is ANTs `genericLabel` vs
nearest at boundaries), so the brain frame and the use of its ANTs files are
right. The second differs from the pipeline only in the template → CCF
registration; the 27% is how much the two template registrations disagree,
along the boundaries of the fine parcellation.

**Fields** (`make_fields.py`, 25 µm lattices, error bound 1 µm, `effort="fast"`):

| | forward (brain grid) | inverse (STPT placed in brain) |
|---|---|---|
| samples | 333 × 759 × 557 | 404 × 618 × 504 |
| size | 17 MB | 18 MB |
| decoded vs exact chain, median / p99 / max | 0.65 / 1.5 / 2.5 µm | 0.65 / 2.4 / 9.2 µm |
| min det J | 0.076 | 0.089 |

- placement L (affine fit over the brain): residual median 127 µm, p99 394 µm:
  the brain is far from affine to the atlas, so the fields carry real deformation;
- push(pull(d)) vs d: median 0.22 µm, p99 1.2 µm: the two directions (each a
  composition of two inverse-consistent ANTs pairs) agree.

Two more scenes put the SPIM LCA template in the views
(`make_fields.py --scenes spim stpt_spim`):

| scene | maps | size | decoded vs exact, median / p99 (fwd, inv) | min det J (fwd, inv) |
|---|---|---|---|---|
| `smartspim_894847_spim.zarr` | brain <-> SPIM template placed by L_spim (residual median 108 um) | 35 MB | 0.64 / 1.5, 0.61 / 1.9 um | 0.098, 0.073 |
| `stpt_spim_placed.zarr` | STPT placed by L <-> SPIM placed by L_spim | 22 MB | 0.58 / 1.2, 0.55 / 1.2 um | 0.25, 0.24 |

**Independent check with ITK** (`itk_check.py`): SimpleITK reads the ANTs
files itself (the `.mat`, `1Warp`, `transform-SPIM`) and the pipeline's own
level-3 image `prep_zarr_img.nii.gz` with its header, composes them as
`antsApplyTransforms -t A -t W -t transform-SPIM` would, and resamples the
brain onto a display-space slab around a point. At (x, y, z) = (3495, 9266,
1837) um in CCF space, over a 5 x 0.2 x 5 mm slab (5.3 M points at 10 um):

| vs ITK | median | p99 | max |
|---|---|---|---|
| `chain.push` (numpy) | 5e-10 um | 9e-10 um | 1e-9 um |
| decoded field drawn by Neuroglancer | 0.58 um | 2.0 um | 7.6 um (at the brain edge; 3.1 um within 300 um) |

The strong bending of the cortex there is in the registrations: det J of the
display -> brain map is 0.25-4.4 over the slab (0.63-2.75 within 300 um of the
point, no folds). Both the SyN warp (0.32-4.1) and the template fields
(0.48-2.6) contribute. ITK's `DisplacementFieldJacobianDeterminant` ignores the
direction matrix (these fields are permuted and flipped), so `itk_check.py`
computes det J in physical space itself. The script also writes the ITK
resample as an OME-Zarr and a state that overlays it (green) on the
Neuroglancer warp (magenta).

**In Neuroglancer** (the `warpfield-transform` branch, wasm rebuilt from this
crate): `make_states.py` writes, to `OUT/states/`,

- `ccf_on_brain.json`: the three channels in their own space, the CCF 2017
  annotation (volume and meshes), the STPT template and the SPIM template
  (hidden) warped onto them; the pipeline's static annotation alongside;
- `ccf_on_brain_affine.json`: the same, placed by the affines only;
- `brain_in_ccf.json`: the full-resolution channels warped into CCF space;
- `compare_{warped,static,affine}.json`: one coronal panel each, for the warped
  CCF, the pipeline's static annotation and the affine-only placement;
- `brain_vs_{template,spim_template}_in_{ccf,brain}_{coronal,horizontal}.json`:
  the brain (magenta) over the STPT or SPIM template (green), in either space;
- `meshes_on_brain.json`: the warped CCF meshes.

Every state also shows the field(s) of its display space coloured by det J
(white = volume kept, red = expansion, blue = compression, yellow = det J <=
min_det; the ANTs example's `field_shader.glsl`). The shader differences
neighbouring samples at a fixed spacing, so these layers read
`OUT/fields_s0/`, the fields' own OME metadata listing only s0 (Neuroglancer
would otherwise switch to coarser levels when zoomed out).

The warped CCF follows the tissue as closely as the static annotation does
(the affine placement misses the ventral surface and the hippocampus), and the
brain and the STPT template coincide in both spaces.

Serve `OUT` with CORS and pass its URL as `--fields-url`. A source `transform`
in Neuroglancer replaces the source's own default placement, so the template
matrices include their intrinsic -> mm RAS rotation (see `make_states.py`).
