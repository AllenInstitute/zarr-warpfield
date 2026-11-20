# zarr-warpfield

An experimental, error-bounded, seam-free lossy codec for **registration
displacement fields** in Zarr v3. It ships with a procedural synthetic-data
generator that covers 2D, stacked-2D, 3D, stacked-3D-over-time and 4D fields
with sharp discontinuities.

What it guarantees, per chunk and without ever reading a neighbouring chunk:

- **Absolute error bound.** Every sample is within `error_bound` (default 0.25
  voxel), including samples on tears, folds and misaligned sections.
- **No chunk seams.** Chunk-face samples are coded at `face_error_bound`
  (default ε/4). The error jump across a chunk boundary is ≤ 2·ε_face, and is
  statistically indistinguishable from jumps inside chunks.
- **Jacobian guarantee** (optional `jacobian` config). Cells inside a chunk keep
  `det J > min_det` if they had it, and never flip sign. The encoder promotes
  just the offending samples to tighter bounds.
- **Round trips.** Store only the forward field and invert the decoded field.
  The true forward map then round-trips within the inversion tolerance + ε.
- **Idempotent recompression.** Without `jacobian`, decode → encode → decode is
  a fixed point.

The normative spec draft is in the Allen `zarr-extensions` fork, at
`codecs/warpfield/` on the `warpfield` branch.

## How it works

The codec is a closed-loop multilevel interpolation predictor (SZ3/QoZ family)
with error-bounded residual quantization, plus three additions:

1. **Clamped anchor lattice + face classes.** Chunk corners are the coarsest
   anchors, nothing is extrapolated, and face samples are predicted only from
   their own face and coded at ε_face.
2. **ENO stencil choice.** A cubic stencil that would straddle a step is
   replaced by the one-sided quadratic. The choice is made from decoded values,
   so it needs no side information, and discontinuities cost bits in proportion
   to their area.
3. **Per-sample error classes** (interior / face / tight / verbatim), stored as
   sparse index lists. The Jacobian loop and float-rounding safety net both
   use them.

## Install and test

```sh
uv sync --extra fast               # builds the Rust backend too (needs cargo); plain `uv sync` = numpy only
uv run pytest                      # bounds, seams, Jacobian, round trips, zarr, Rust == numpy, wasm == numpy
```

## Implementations

All three produce the same quantization codes and bit-identical decoded values;
the tests check this on every preset, predictor and boundary mode, both dtypes,
and NaN / Inf / overflow inputs.

| | where | use |
|---|---|---|
| numpy reference | `src/zarr_warpfield` | readable reference, always available |
| Rust | `rust/warpfield` (crate), `rust/warpfield-py` (module `zarr_warpfield_rs`) | used automatically by `zarr_warpfield` when installed; `ZARR_WARPFIELD_BACKEND=python` forces numpy |
| WebAssembly + TypeScript | `rust/warpfield-wasm`, `rust/warpfield-wasm/js` (`@zarr-warpfield/wasm`) | browsers and node; no imports, no wasm-bindgen |

Single-core throughput on the full presets (Apple M-series; MB of float32
field per second; ε = 0.25 with the Jacobian pass):

| | encode | decode |
|---|---|---|
| numpy reference | 4–9 MB/s | 21–43 MB/s |
| Rust, zstd level 19 (reference sizes) | 10–18 MB/s | 290–480 MB/s |
| Rust, zstd level 3 (15–25% larger) | 30–205 MB/s | 275–525 MB/s |
| wasm in node | 27–88 MB/s | 190–340 MB/s |
| zfp fixed-accuracy, tol 0.25 | 330–610 MB/s | 250–990 MB/s |

The Rust backend releases the GIL and the zarr codec runs chunks on threads, so
decoding scales with cores (1.3–3.2 GB/s on 14 cores). `scripts/throughput.py`
measures all of these; see `rust/README.md` for building and the APIs.

## Use

```python
import zarr
from zarr_warpfield import WarpFieldCodec

codec = WarpFieldCodec(
    axis_roles=["spatial", "spatial", "spatial", "component"],   # [Z, Y, X, 3]
    error_bound=0.25,
    jacobian={"grid_spacing": [16, 16, 16], "min_det": 0.1},
)
a = zarr.create_array("field.zarr", shape=(2048, 2048, 2048, 3), chunks=(128, 128, 128, 3),
                      dtype="float32", serializer=codec, compressors=None, fill_value=0.0)
```

Axis roles are `spatial` (predicted, Jacobian), `smooth` (predicted, e.g. time
or section index), `independent` (not predicted across) and `component`. Use
`zarr_warpfield.encode/decode` for raw chunks without zarr.

## Synthetic data, benchmark, gallery

```sh
uv run python scripts/make_synthetic.py            # zarr v3 groups: raw / warpfield / labels
uv run --with zfpy==1.0.1.post0 python scripts/benchmark.py   # ratio/error/seam/Jacobian table
uv run --with zfpy==1.0.1.post0 python scripts/visualize.py   # out/gallery/index.html
```

zfpy is optional. It is only used for the comparison, and is kept out of the
dev environment because its 1.0.1.post0 version string breaks numcodecs'
import inside zarr.

`zarr_warpfield.synth.Preset.region()` evaluates any region of a field from
global coordinates. Arrays of any size can therefore be written chunk by
chunk (`make_synthetic.py --shape ...`).

## Status and limits

- Encoding at the reference zstd level 19 spends most of its time in zstd;
  lower levels are 5–10x faster for 15–25% larger chunks. A dedicated entropy
  coder for the quantization codes (rANS) should give both.
- Cells straddling two chunks are not Jacobian-checked. Their perturbation is
  bounded by the face bound, and the benchmark counts them (a handful at most).
- Partial edge chunks are padded with `fill_value` by zarr. The padding costs a
  few bytes and does not affect the bound.
- A Neuroglancer decoder (`src/datasource/zarr/codec/warpfield`) is the planned
  next step. The spec fixes float64 operation order so that it can be
  bit-exact.
