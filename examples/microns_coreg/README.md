# MICrONS: in vivo stack and 2P fields warped onto the EM

This example leaves the EM in place and warps the functional imaging onto it in
Neuroglancer. Both the in vivo structural stack and the 2P imaging fields, which
are registered to that stack, are moved into EM space with a warpfield
displacement field. The field is built from the em_coregistration transform
(AllenInstitute/em_coregistration, branch `phase3`).

```sh
# the transform solutions and landmarks are in em_coregistration/data
uv run --with scipy python make_fields.py OUT \
    --forward phase3_staged_transform_solution.json \
    --inverse inverse_phase3_staged_transform_solution.json \
    --landmarks 17797_2Pfix_EMmoving_Phase3.csv          # ~25 min, streaming
uv run python make_states.py OUT --fields-url http://127.0.0.1:9078   # OUT served there
```

- `coreg_transform.py` evaluates em_coregistration `TransformList` JSON. It
  supports `PolynomialModel`, `ChunkedModel` and `SplineModel`, the last being
  affine plus |r| radial kernels, i.e. the 3-D biharmonic spline. It is
  vectorized and matches the repo's own code to within 4e-15 mm.
- `make_fields.py` writes `OUT/microns_em_2p.zarr`, an OME-Zarr 0.6 scene with
  forward and inverse fields at 5 µm spacing, in EM micrometres, axes
  (z, y, x). It also writes `OUT/placement.json`.
  - The placement **L** (stack µm → EM µm) is the inverse of an affine fit of
    the forward transform T over minnie65. Every moving layer is placed with L,
    and the field stores only the remainder, u(x) = L(T(x)) − x.
- `make_states.py` writes three Neuroglancer states:
  - `warped.json`: EM (minnie65 + 35), the blood-vessel segmentation, the
    segmentation, the stack (vessels red, cells green) and all 19 scans' 2P
    fields, warped;
  - `affine.json`: the same layers, placed by L only;
  - `field.json`: the field with det J.

  They need the Neuroglancer `warpfield-transform` branch, which also adds the
  channel-dimension support the 2-channel stack needs.

## Numbers (phase3 solution)

| | median | p95 | max |
|---|---|---|---|
| forward fit residual at the 2,934 used landmarks (µm) | 3.2 | 8.4 | 27 |
| inverse fit residual (µm) | 0.10 | 0.52 | 2.7 |
| forward → inverse round trip at the landmarks (µm) | 2.5 | 6.9 | 39 |

- **Field size:** 170 × 298 × 350 samples forward and 258 × 186 × 305 inverse;
  4 MB and 8 MB at ε = 0.1 µm.
- **Non-affine displacement** (T beyond L): median 45 µm and max 280 µm in
  minnie65; up to 370 µm in minnie35, which is extrapolated.
- **Decoded field vs exact T**, at random points:

  | region | p50 (µm) | p90 (µm) | p99 (µm) | max (µm) |
  |---|---|---|---|---|
  | minnie65 | 0.06 | 1.9 | 10 | 39 |
  | minnie35 | 0.05 | 0.09 | 0.12 | 23 |

  The tail comes from the steps described below.
- **End-to-end check.** At 40 random EM points, the value Neuroglancer shows for
  the warped stack matches the stack sampled in Python at T(x), with
  correlation 0.991 for cells and 0.994 for vessels. The difference is nearest
  vs trilinear sampling.

## Things to know about the transform

- **It is discontinuous.** Its two `ChunkedModel` stages are independent affines
  over slabs along the stack's x axis (4 and 20 slabs). T therefore jumps, by a median of 7 µm
  and up to 38 µm, on 22 surfaces roughly perpendicular to EM z (the slabs are along the stack's x, which L maps to EM z). A sampled field
  turns each jump into a ramp one grid cell (5 µm) wide. Away from those
  surfaces, 5 µm sampling reproduces T to about 0.003 µm (median).
- **It folds.** Those steps give min det J −12.7 forward and −15.4 inverse.
- **It extrapolates over minnie35.** The landmarks only cover minnie65.
  `attributes.warpfield.support` records the supported region.
- **The stack voxel convention is assumed.** The landmark files give stack
  coordinates in mm without saying whether they are voxel corners or centres.
  `--stack-offset` (default 0) shifts them; the difference is at most one µm,
  well inside the fit residual.
