# warpfield in Rust

A Rust implementation of the zarr v3 `warpfield` codec, with Python,
WebAssembly/TypeScript and Java bindings. It is a line-for-line port of the numpy
reference in `src/warpfield`: same float64 operation order, same visiting
order, round-half-even quantization. So it emits the same quantization codes and
class lists, and decodes to bit-identical values. The byte format is the one in
the spec; only the zstd frames can differ between zstd libraries and levels.

```
rust/
├── warpfield/          core crate: encode / decode, no Python or JS dependencies
├── warpfield-py/       PyO3 module `warpfield_rs`, package `warpfield-rs` (maturin, abi3 >= 3.11)
├── warpfield-wasm/     wasm32 cdylib with a plain C ABI
│   └── js/             TypeScript wrapper, package `@warpfield/wasm`
└── warpfield-jni/      JNI cdylib `warpfield_jni`, the native half of ../java (org.warpfield.Warpfield)
```

## Core crate

```rust
use warpfield::{Config, EncodeOptions, JacobianConfig, Role, decode, encode};

let shape = [64, 64, 64, 3];                       // one chunk, C order
let mut cfg = Config::new(
    vec![Role::Spatial, Role::Spatial, Role::Spatial, Role::Component], 0.25);
cfg.jacobian = Some(JacobianConfig { grid_spacing: vec![16.0; 3], min_det: 0.1,
                                     tight_error_bound: None });
// or: let cfg = Config::from_json(r#"{"axis_roles": [...], "error_bound": 0.25}"#)?;
let enc = encode(&field, &shape, &cfg, &EncodeOptions::default())?;   // f32 or f64
let out: Vec<f32> = decode(&enc.bytes, &shape, &cfg)?;
```

Features:
- `zstd-native` (default): libzstd via the `zstd` crate.
- `zstd-pure`: pure-Rust `ruzstd`, for targets without a C toolchain such as
  wasm32. Its encoder only has a fast level, so chunks come out a little larger.
  It decodes any frame.
- `json` (default): `Config::from_json`.

`EncodeOptions::zstd_level` defaults to 19 like the reference. Level 3 encodes
5–10x faster for 15–25% larger chunks.

```sh
cargo test -p warpfield                                         # libzstd
cargo test -p warpfield --no-default-features --features zstd-pure,json
```

## Python

```sh
uv sync --extra fast        # from the repo root; or: pip install ./rust/warpfield-py
```

`warpfield.encode` / `decode` (and so the tiles helpers and the zarr codec) use
the Rust backend whenever `warpfield_rs` imports. Set `WARPFIELD_BACKEND=python` to
force numpy, or pass `backend="rust" | "python"` per call. The GIL is
released, and `WarpFieldCodec` codes chunks with `asyncio.to_thread`, so zarr
reads decode in parallel.

The low-level module:

```python
import json, warpfield_rs as rs
buf, classes, n_codes, n_tight, n_verbatim, rounds = rs.encode(arr, json.dumps(cfg), classes=True, zstd_level=19)
out = rs.decode(buf, arr.shape, "float32", json.dumps(cfg))
rs.decode_into(buf, out_array, json.dumps(cfg))    # into an existing C-contiguous array
```

`tests/test_rust.py` checks that Rust and numpy produce identical streams,
classes and decoded bits. It covers every preset, predictor, boundary mode,
ε, both dtypes, the Jacobian pass, and NaN/Inf/overflow input.

## WebAssembly / TypeScript

```sh
cd rust/warpfield-wasm/js
npm install
npm run build:wasm       # cargo build --target wasm32-unknown-unknown → warpfield.wasm (~390 KB)
npm run build            # tsc → dist/
npm test                 # node --test (node >= 22.18 runs the .ts directly)
```

```ts
import { WarpField } from "@warpfield/wasm";

const wf = await WarpField.instantiate(fetch(new URL("warpfield.wasm", import.meta.url)));
const config = zarrMetadata.codecs.find((c) => c.name === "warpfield").configuration;
const field = wf.decode(chunkBytes, [64, 64, 64, 3], "float32", config);   // Float32Array
const bytes = wf.encode(field, [64, 64, 64, 3], config);
```

The module has no imports. It exports `wf_alloc`/`wf_free`, `wf_decode`,
`wf_encode` and error/result accessors (see `warpfield-wasm/src/lib.rs`), so
it can also be loaded without the wrapper, for example from Neuroglancer's
worker-side codec registry. `tests/test_wasm.py` runs it under node. The test
decodes Python-encoded chunks and compares the bits, and checks that
wasm-encoded chunks carry the same streams as Python's.

## Throughput

Measured with `scripts/throughput.py` on the full presets, single core, Apple
M-series, MB of float32 field per second:

| preset | numpy enc / dec | Rust enc (zstd 19 / 3) | Rust dec | wasm enc / dec | zfp tol 0.25 enc / dec |
|---|---|---|---|---|---|
| 2d | 7 / 21 | 15 / 205 | 400–500 | 65 / 190 | 380 / 710 |
| 2d_stack | 8 / 42 | 17 / 131 | 475–525 | 88 / 340 | 610 / 930 |
| 3d | 9 / 43 | 18 / 83 | 415–425 | 69 / 306 | 600 / 990 |
| 3d_time_stack | 4 / 30 | 11 / 31 | 285–295 | 28 / 230 | 330 / 250 |
| 4d | 4 / 32 | 11 / 29 | 275–295 | 27 / 223 | 610 / 360 |

- **Decode** is 10–20x faster than numpy, and within about 2x of zfp
  (faster than zfp on the 4D presets).
- **Encode** is limited by zstd at level 19, and on 3d_time_stack and 4d by
  the extra Jacobian rounds, each of which re-runs the pass.
- **Multi-threaded** (14 threads), decode reaches 1.3–3.2 GB/s.
