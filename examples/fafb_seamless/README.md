# FAFB v14 → v14.1 (SEAMLeSS) as a warpfield store

A proof of concept: a 100-section, 32.8 µm cutout of the SEAMLeSS alignment
field that turned FAFB v14 into v14.1 (FlyWire), converted to a warpfield
OME-Zarr store and used to view v14 warped into v14.1 in Neuroglancer.

```
uv run python examples/fafb_seamless/make_field.py    # ~1 min: download + encode
uv run python examples/fafb_seamless/validate.py      # ~15 s: check vs the v14.1 render
uv run python examples/fafb_seamless/make_states.py   # Neuroglancer states

# the same at a 4 nm error bound (1 px at 4 nm), and a state comparing the two
uv run python examples/fafb_seamless/make_field.py --error-bound 4 --name fafb_v14_to_v14.1_eb4
uv run python examples/fafb_seamless/validate.py --name fafb_v14_to_v14.1_eb4
uv run python examples/fafb_seamless/make_states.py --name fafb_v14_to_v14.1_eb4 --compare fafb_v14_to_v14.1

# 8 nm (2 px at 4 nm): the 16 nm field only, and with a pyramid whose error
# bound doubles per level (8, 16, 32, 64 nm), as the sample spacing does
uv run python examples/fafb_seamless/make_field.py --error-bound 8 --levels 1 --name fafb_v14_to_v14.1_eb8_s0
uv run python examples/fafb_seamless/make_field.py --error-bound 8 --level-error-growth 2 --name fafb_v14_to_v14.1_eb8

# a 9x larger field of view (6144^2 samples, 98 um square), 8 nm doubling, 5 levels: ~4 min
uv run python examples/fafb_seamless/make_field.py --size 6144 --error-bound 8 --level-error-growth 2 --levels 5 --name fafb_v14_to_v14.1_eb8_wide
uv run python examples/fafb_seamless/validate.py --name fafb_v14_to_v14.1_eb8_wide --center 74067 42119 84000 32000 64000 51000 84000 51000
uv run python examples/fafb_seamless/make_states.py --name fafb_v14_to_v14.1_eb8_wide --zoom 24
```

The build works tile by tile: each work unit is one 1024² tile (one source
chunk) through 25 sections, and shards are 1024² × 1 section, so memory does
not grow with the field of view (the wide run peaks at 2.6 GB with 16 units in
flight).

Defaults: the region around v14 (74067, 42119, 4384) (4 × 4 × 40 nm), sections
4334–4433, 2048² samples at 16 nm (16 nm-grid x [17408, 19456), y [9216, 11264)).

## Source

- Field: `precomputed://https://td.princeton.edu/sseung-archive/fafbv14-vf/v0`,
  scale `16_16_40`, int16, 2 channels, chunks 1024 × 1024 × 1.
- v14 images: `gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig` (4 nm up)
  and `fafb_v14_clahe` (8 nm up).
- v14.1 render (ground truth): `gs://microns-seunglab/drosophila_v0/alignment/image_rechunked`.

Conventions (from SEAMLeSS `aligner.py` / `utilities/helpers.py`, and checked
against the render):

- v14.1(p) = v14(p + u(p)), p and u in one absolute frame: Neuroglancer's warp.
- int16 / 4 = 4 nm pixels, so the stored integer is the displacement in **nm**.
- **Channel 0 is the y displacement, channel 1 is x**, despite SEAMLeSS's
  docstrings, which say (x, y). Read as (x, y), warped v14 has NCC ≈ 0 with the
  render. Read as (y, x), NCC is 0.998.
- Samples are at pixel centres (fields and images), the same convention as Neuroglancer.
- Section z of v14.1 is section z of v14 (and of the field).

## Store

`out/fafb_seamless/fafb_v14_to_v14.1.zarr`: OME-Zarr 0.6, axes (c, z, y, x) in
nm, with c the RFC-5 displacement axis (dz = 0, dy, dx).

- **Chunks:** one section thick (3, 1, 256, 256); shards (3, 1, 1024, 1024).
- **Codec roles:** (component, **independent**, spatial, spatial). Each
  section is coded on its own, with no prediction across sections and no seam
  constraint at section faces.
- **Levels s1–s3:** 2 × 2 means in-plane only. Sections are never mixed.

Why single sections in z: the fields are computed per section, and in this
cutout they jump by up to **989 nm between neighbouring sections**. Within a
section they also fold over at tissue folds (det J down to −3.8). Prediction
across z would only cost bits, and interpolating across z would blend
unrelated sections.

## Results

| | ε = 0.5 nm | ε = 4 nm | ε = 8 nm, doubling per level |
|---|---|---|---|
| warped v14 vs v14.1 render, NCC (z 4334 / 4384 / 4433) | 0.998 / 0.999 / 0.998 | 0.996 / 0.996 / 0.995 | 0.988 / 0.989 / 0.991 |
| same against v14.1 sections z ± 1 | 0.30–0.50 | 0.30–0.50 | 0.30–0.50 |
| residual shift (phase correlation, 16 nm px) | < 0.004 | < 0.007 | < 0.04 |
| max decode error (s0) | 0.5 nm | 4.0 nm | 8.0 nm |
| s0 / whole store (4 levels) | 166 MB / 218 MB | 14.1 MB / 22.3 MB | 5.7 MB / 7.6 MB |
| s0 / whole store vs raw int16 | 10.1× / 7.7× | 119× / 75× | 294× / 221× |
| whole FAFB, straight extrapolation (×39,400), s0 / whole | 6.5 / 8.6 TB | 0.55 / 0.9 TB | 0.22 / 0.30 TB |

The extrapolation scales the cutout (100 × 2048² samples) to the field's full
16 nm bounding box (7062 × 36864 × 63488); the cutout is dense tissue with a
fold, and much of the box is not tissue, so it is likely an overestimate.

At ε = 4 nm the decoded field differs from the ε = 0.5 nm one by a median of
1.7 nm (p90 3.1, p99 4.0). The field is smooth enough that the in-plane
coarse levels now take a third of the store: s1–s3 add 8.3 MB to s0's 14.1 MB.

Compression of s0, against raw int16 (1678 MB for 100 sections; 5-section
sweep for the other bounds):

| | ratio | MB / section |
|---|---|---|
| gzip (the source, as served) | 4.8× | 3.5 |
| zstd-19, lossless | 6.1× | 2.7 |
| warpfield ε = 0.5 nm (stored) | 10.1× | 1.66 |
| warpfield ε = 1 nm (¼ px at 4 nm) | 17.7× | 0.95 |
| warpfield ε = 2 nm (½ px) | 46× | 0.36 |
| warpfield ε = 4 nm (1 px) | 117× | 0.14 |

## Wider field of view (8 nm, doubling per level)

`fafb_v14_to_v14.1_eb8_wide`: 16 nm x [15360, 21504), y [7168, 13312),
sections 4334–4433 (98 × 98 µm × 100 sections), levels s0–s4 with error bounds
8, 16, 32, 64, 128 nm.

| | |
|---|---|
| build time (download 3.5 GB + encode), peak memory | 244 s, 2.6 GB |
| s0 / whole store | 66.5 MB / 92.6 MB |
| vs raw int16 (15.1 GB) | 227× / 163× |
| level interpolation error p99, s1–s4 | 15, 37, 75, 143 nm |
| level interpolation error max, s1–s4 | 350, 459, 519, 516 nm |
| warped v14 vs v14.1, NCC, 4 places × 3 sections | 0.987–0.993 |
| residual shift (16 nm px) | < 0.05 |
| whole FAFB, straight extrapolation (×4,380), s0 / whole | 0.29 / 0.41 TB |

The top-left corner of this cutout is outside the tissue (the v14.1 render is
empty there). The field there is unconstrained: up to 7.2 µm of displacement
and det J down to −101, against ≤ 1.8 µm in the tissue. It and the folds set
the max interpolation errors. Neuroglancer chooses one field level per warped
scale for the whole volume, so it now selects by the p99 error when the store
gives one. On the max, the 16 nm level was chosen at every zoom, and a
zoomed-out view crashed the tab.
