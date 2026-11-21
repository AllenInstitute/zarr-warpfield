"""The Rust backend must be bit-identical to the numpy reference."""

import numpy as np
import pytest

from warpfield import Config, JacobianConfig, _backend, chunked, core, synth
from warpfield import format as fmt

pytestmark = pytest.mark.skipif(not _backend.available(), reason="zarr-warpfield-rs not built")


def _bits(a: np.ndarray) -> np.ndarray:
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def check_same(arr: np.ndarray, cfg: Config) -> None:
    pb, pst, pcls = core.encode(arr, cfg, stats=True, backend="python")
    rb, rst, rcls = core.encode(arr, cfg, stats=True, backend="rust")
    a, b = fmt.read_chunk(pb, arr.dtype), fmt.read_chunk(rb, arr.dtype)
    assert a[0] == b[0]
    for x, y in zip(a[1:], b[1:], strict=True):
        np.testing.assert_array_equal(
            _bits(x) if x.dtype.kind == "f" else x, _bits(y) if y.dtype.kind == "f" else y
        )
    np.testing.assert_array_equal(pcls, rcls)
    assert (pst.n_codes, pst.n_tight, pst.n_verbatim, pst.jacobian_rounds) == (
        rst.n_codes,
        rst.n_tight,
        rst.n_verbatim,
        rst.jacobian_rounds,
    )
    ref = core.decode(pb, arr.shape, arr.dtype, cfg, backend="python")
    for buf in (pb, rb):
        out = core.decode(buf, arr.shape, arr.dtype, cfg, backend="rust")
        np.testing.assert_array_equal(_bits(out), _bits(ref))
    np.testing.assert_array_equal(
        _bits(core.decode(rb, arr.shape, arr.dtype, cfg, backend="python")), _bits(ref)
    )


def _first_chunk(name):
    p = synth.preset(name, "small")
    u, _ = p.full()
    sl = next(chunked.chunk_slices(u.shape, p.chunks))
    return p, np.ascontiguousarray(u[sl])


@pytest.mark.parametrize("name", synth.PRESET_NAMES)
@pytest.mark.parametrize("predictor", ["linear", "cubic", "eno"])
@pytest.mark.parametrize("jac", [False, True])
def test_presets_identical(name, predictor, jac):
    p, blk = _first_chunk(name)
    j = JacobianConfig((float(p.spacing),) * p.ks, 0.1) if jac else None
    check_same(blk, Config(p.axis_roles, 0.25, predictor=predictor, jacobian=j))


@pytest.mark.parametrize("eps", [0.05, 0.5])
@pytest.mark.parametrize("boundary", ["clamped", "extrapolate"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_settings_identical(eps, boundary, dtype):
    p, blk = _first_chunk("2d")
    cfg = Config(p.axis_roles, eps, face_error_bound=eps / 2, boundary=boundary)
    check_same(blk.astype(dtype), cfg)


def test_whole_preset_chunked_identical():
    p = synth.preset("4d", "small")
    u, _ = p.full()
    cfg = Config(p.axis_roles, 0.25, jacobian=JacobianConfig((float(p.spacing),) * 3, 0.1))
    for sl in chunked.chunk_slices(u.shape, p.chunks):
        check_same(np.ascontiguousarray(u[sl]), cfg)


def test_edge_cases_identical():
    rng = np.random.default_rng(0)
    roles = ("spatial", "spatial", "component")
    x = rng.normal(0, 3, (17, 9, 2)).astype(np.float32)
    x[3, 4, 0] = np.nan
    x[5, 1, 1] = np.inf
    x[8, 8, 0] = -np.inf
    x[0, 0, 1] = 3e38
    x[16, 8, 0] = -1e30
    check_same(x, Config(roles, 0.25))
    check_same(x.astype(np.float64) * 1e300, Config(roles, 0.25))
    # size-one axes, independent axes, no component axis
    check_same(
        rng.normal(size=(1, 7, 1, 2)).astype(np.float32),
        Config(("independent", "spatial", "spatial", "component"), 0.1),
    )
    check_same(
        rng.normal(size=(3, 5, 6)).astype(np.float32),
        Config(("independent", "smooth", "spatial"), 0.1),
    )
    check_same(rng.normal(size=(1,)).astype(np.float64), Config(("spatial",), 0.1))


def test_rust_errors():
    cfg = Config(("spatial", "spatial", "component"), 0.25)
    buf = core.encode(np.zeros((8, 8, 2), np.float32), cfg, backend="rust")
    with pytest.raises(fmt.WarpFieldFormatError):
        core.decode(buf[:-3], (8, 8, 2), np.float32, cfg, backend="rust")
    with pytest.raises(fmt.WarpFieldFormatError):
        core.decode(buf, (8, 9, 2), np.float32, cfg, backend="rust")
    with pytest.raises(fmt.WarpFieldFormatError):
        core.decode(b"XXXX" + buf[4:], (8, 8, 2), np.float32, cfg, backend="rust")
    with pytest.raises(core.WarpFieldSpecError):
        core.encode(np.zeros((8, 8), np.float32), cfg, backend="rust")
