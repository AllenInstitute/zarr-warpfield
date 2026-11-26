# Real fields: ANTs SPIM ↔ STPT mouse template registration

Two ANTs displacement fields that register the Allen adult mouse templates:

| | fixed grid (25 µm) | maps | raw | `.nii.gz` |
|---|---|---|---|---|
| `transform-SPIM.nii.gz` | STPT s1 grid, padded 32 voxels | SPIM → STPT space | 1.42 GB | 1.06 GB |
| `inverse-transform-SPIM.nii.gz` | SPIM s1 grid, padded 32 voxels | STPT → SPIM space | 2.76 GB | 1.54 GB |

Templates (OME-Zarr 0.6; voxel → RAS from their `mm RAS` coordinate transforms):

- `s3://allen-atlas-assets/templates/allen-adult-mouse-spim-lca-template/2024-05/template.ome.zarr`
- `s3://allen-atlas-assets/templates/allen-adult-mouse-stpt-template/2020/template.ome.zarr`

How the fields are applied (ANTs / ITK). Physical space is LPS, in mm, and
the stored vectors are LPS. A moving image M resampled into the fixed space is
W(x) = M(x + u(x)). u is linearly interpolated, with nearest-neighbour
extrapolation outside its grid. M is linearly interpolated, and is 0 outside.

`streaming.py` implements exactly this, chunk by chunk. It reproduces ITK's
`Resample` + `DisplacementFieldTransform` result to the fifth decimal of NCC
(0.75596 and 0.62834). Interpreting the vectors as RAS instead gives no overlap
at all.

```sh
uv run --with fsspec --with aiohttp python fetch_templates.py ATLAS_DIR          # s1 (25 µm) levels
uv run --with nibabel --with indexed_gzip python convert.py to-zarr transform-SPIM.nii.gz forward.zarr --error-bound 0.0025
uv run --with nibabel --with indexed_gzip python convert.py to-nifti forward.zarr forward.nii   # back to ANTs
uv run --with nibabel --with indexed_gzip python verify.py ATLAS_DIR transform-SPIM.nii.gz inverse-transform-SPIM.nii.gz OUT
uv run --with nibabel --with indexed_gzip --with matplotlib python figures.py ATLAS_DIR ... OUT OUT/figure.png
uv run python codec_timing.py OUT    # codec-only encode/decode timing -> OUT/codec_timing.json
uv run --with "zfpy==1.0.1.post0" python zfp_compare.py OUT   # vs zfp -> OUT/zfp_compare.json
```

Every step streams the data. The NIfTI is read in k-slabs, using
`indexed_gzip` to seek, zarr is read by region, and the templates through
memory maps. `convert.py` peaks at 1.2 GB (forward) and 1.9 GB (inverse), and
takes 11–12 s on 14 cores.

The zarr layout:
- the array is (k, j, i, 3) in ITK index order;
- the components are index-aligned, an exact signed permutation of LPS, so
  the conversion is lossless;
- the voxel → LPS affine is stored in `attributes.ants_displacement_field`;
- the codec is `warpfield` with chunks 64³×3 and `jacobian.min_det` 0.1.

For viewers, the fields are also written as an OME-Zarr 0.6 scene
(RFC-5 coordinate transformations):

```sh
uv run --with nibabel --with indexed_gzip python ome_field.py transform-SPIM.nii.gz OUT/forward
uv run --with nibabel --with indexed_gzip python ome_field.py inverse-transform-SPIM.nii.gz OUT/inverse
python ome_scene.py OUT/spim_stpt_ants.zarr --forward OUT/forward --inverse OUT/inverse \
    --input-space "STPT template 2020, mm RAS" --output-space "SPIM template LCA 2024-05, mm RAS"
```

```
spim_stpt_ants.zarr/              ome.scene: the two template spaces and a "bijection"
  coordinateTransformations/
    forward/                      field on the STPT grid: multiscale s0..s4, bounds/
    inverse/                      field on the SPIM grid
```

- **Each field** is an OME-Zarr multiscale image with axes (c, k, j, i).
  - `c` has `"type": "displacement"`. Its components are displacements in mm
    along the array axes of the coordinate system `intrinsic`.
  - An affine maps `intrinsic` to the templates' `mm RAS`.
  - `attributes.warpfield` holds what OME-Zarr has no place for: the error
    bound, each level's measured interpolation error, and the bounds array.
- **The bijection's forward transformation** maps an STPT point to SPIM
  space: affine to `intrinsic`, the forward field's `displacements`, then the
  affine back. Resampling SPIM through it puts SPIM in STPT space. The inverse
  transformation uses the inverse field.
- **Neuroglancer** takes a field group, or the scene, as a data source's
  `"transformSource"`. It finds the field of the opposite direction (used to
  move mesh vertices) through the scene's bijection.

## Results

| field | store | size | vs raw | vs `.nii.gz` | max error | NCC(warp, NIfTI warp) | mean / max image diff | Jacobian violations |
|---|---|---|---|---|---|---|---|---|
| forward | zstd-19, lossless | 871 MB | 1.6× | 1.2× | 0 | 1 | 0 | 0 |
| forward | warpfield ε = 1 µm | 4.8 MB | 293× | 218× | 1.000 µm | 0.999987 | 0.05% / 4.6% | 0 |
| forward | warpfield ε = 2.5 µm | 2.9 MB | 485× | 361× | 2.500 µm | 0.999928 | 0.13% / 12% | 0 |
| forward | warpfield ε = 6.25 µm | 1.9 MB | 731× | 544× | 6.250 µm | 0.999562 | 0.32% / 27% | 0 |
| inverse | zstd-19, lossless | 958 MB | 2.9× | 1.6× | 0 | 1 | 0 | 0 |
| inverse | warpfield ε = 1 µm | 5.9 MB | 467× | 261× | 1.000 µm | 0.999987 | 0.05% / 8% | 0 |
| inverse | warpfield ε = 2.5 µm | 3.7 MB | 739× | 413× | 2.500 µm | 0.999926 | 0.12% / 19% | 0 |
| inverse | warpfield ε = 6.25 µm | 2.5 MB | 1098× | 613× | 6.250 µm | 0.999553 | 0.30% / 50% | 0 |

- **Error** is the maximum absolute error over all samples and components.
- **Image differences** are fractions of the moving template's intensity range,
  over about 37 M brain voxels. The maximum occurs at the sharpest edges, where
  a 0.1-voxel shift moves a whole edge. The mean is about 0.1%.
- **Jacobian:** neither field has a cell with det J ≤ 0, and none appear after
  compression.

NCC to the fixed template is 0.75596 with the original forward field and
0.75572 at ε = 2.5 µm; for the inverse it is 0.62834 and 0.62822.

Forward-then-inverse round trip at 200 k STPT brain points:
- original fields: median 0.23 µm, p99 0.82 µm, max 2.5 µm;
- 2.5 µm stores: median 2.0 µm, p99 4.5 µm, max 9.3 µm.

The per-component bound allows up to √3·ε per vector for each of the two
fields, plus the original's own mismatch.

Reading a whole store, slab by slab, takes 0.9–1.6 s.
