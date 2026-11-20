"""The wasm build (via node) must be bit-identical to the numpy reference.

Needs node >= 22.18 (on PATH or in $NODE) and the built module
rust/warpfield-wasm/js/warpfield.wasm (`npm run build:wasm` in that directory).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from zarr_warpfield import Config, JacobianConfig, chunked, core, synth
from zarr_warpfield import format as fmt

JS = Path(__file__).resolve().parents[1] / "rust" / "warpfield-wasm" / "js"
WASM = JS / "warpfield.wasm"
NODE = os.environ.get("NODE") or shutil.which("node")

pytestmark = pytest.mark.skipif(
    not (NODE and WASM.exists()), reason="node or warpfield.wasm not available"
)


def _cases():
    for name in synth.PRESET_NAMES:
        p = synth.preset(name, "small")
        u, _ = p.full()
        sl = next(chunked.chunk_slices(u.shape, p.chunks))
        blk = np.ascontiguousarray(u[sl])
        h = (float(p.spacing),) * p.ks
        yield (
            f"{name}_eno_jac",
            blk,
            Config(p.axis_roles, 0.25, jacobian=JacobianConfig(h, 0.1)),
        )
        yield (
            f"{name}_cubic",
            blk.astype(np.float64),
            Config(p.axis_roles, 0.1, predictor="cubic"),
        )
    x = np.random.default_rng(1).normal(0, 3, (17, 9, 2)).astype(np.float32)
    x[3, 4, 0], x[5, 1, 1], x[0, 0, 1] = np.nan, np.inf, 3e38
    yield "nonfinite", x, Config(("spatial", "spatial", "component"), 0.25)


def test_wasm_matches_python(tmp_path):
    cases = list(_cases())
    for name, arr, cfg in cases:
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {"shape": list(arr.shape), "dtype": arr.dtype.name, "config": cfg.to_dict()}
            )
        )
        (tmp_path / f"{name}.raw").write_bytes(arr.tobytes())
        (tmp_path / f"{name}.encoded").write_bytes(core.encode(arr, cfg, backend="python"))
    r = subprocess.run(
        [NODE, str(JS / "test" / "crosscheck.ts"), str(tmp_path), str(WASM)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    for name, arr, cfg in cases:
        uint = np.uint32 if arr.dtype == np.float32 else np.uint64
        py_buf = (tmp_path / f"{name}.encoded").read_bytes()
        ref = core.decode(py_buf, arr.shape, arr.dtype, cfg, backend="python")
        got = np.frombuffer((tmp_path / f"{name}.wasm_decoded").read_bytes(), arr.dtype)
        np.testing.assert_array_equal(got.view(uint), ref.ravel().view(uint), err_msg=name)
        # wasm-encoded bytes (pure-Rust zstd) carry the same streams as Python's
        w_buf = (tmp_path / f"{name}.wasm_encoded").read_bytes()
        a, b = fmt.read_chunk(py_buf, arr.dtype), fmt.read_chunk(w_buf, arr.dtype)
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:], strict=True):
            np.testing.assert_array_equal(
                np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)
            )
        out = core.decode(w_buf, arr.shape, arr.dtype, cfg, backend="python")
        np.testing.assert_array_equal(out.view(uint), ref.view(uint), err_msg=name)
