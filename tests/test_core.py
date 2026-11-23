import numpy as np
import pytest

from warpfield import Config, JacobianConfig, WarpFieldSpecError, decode, encode, huffman
from warpfield import format as fmt
from warpfield.core import TUNE_CANDIDATES, VERBATIM

S2 = ("spatial", "spatial", "component")


def step_field(shape=(70, 90), seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    u = np.stack([10 * np.sin(x / 30) + 5 * np.cos(y / 41), 8 * np.cos(y / 25 + x / 50)], -1)
    u[..., 1] += 6 * (y > 30 + 10 * np.sin(x / 15))  # a tear
    u[..., 0] += 3 * (x > 60)  # a straight step
    u += 0.02 * rng.standard_normal(u.shape)
    return u.astype(dtype)


def check_bound(u, cfg, *, dtype=None):
    buf, st, cls = encode(u, cfg, stats=True)
    d = decode(buf, u.shape, u.dtype, cfg)
    assert d.dtype == u.dtype
    bound = cfg.eps_table()[cls]
    err = np.abs(d.astype(np.float64) - u.astype(np.float64))
    assert (err <= bound).all(), float((err - bound).max())
    return buf, d, cls


@pytest.mark.parametrize("predictor", ["linear", "cubic", "eno"])
@pytest.mark.parametrize("eb", [0.05, 0.1, 0.25, 0.5])
def test_error_bound_with_steps(predictor, eb):
    u = step_field()
    check_bound(u, Config(S2, eb, predictor=predictor))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtypes(dtype):
    check_bound(step_field(dtype=dtype), Config(S2, 0.1))


def test_face_samples_use_face_bound():
    u = step_field()
    cfg = Config(S2, 0.25, face_error_bound=0.01)
    buf, d, cls = check_bound(u, cfg)
    err = np.abs(d - u)
    for face in (err[0], err[-1], err[:, 0], err[:, -1]):
        assert face.max() <= 0.01


def test_eno_is_smaller_than_cubic_on_steps():
    u = step_field((128, 128))
    n_eno = len(encode(u, Config(S2, 0.25, predictor="eno")))
    n_cub = len(encode(u, Config(S2, 0.25, predictor="cubic")))
    assert n_eno < n_cub


def test_non_finite_values_are_kept_exactly():
    u = step_field()
    u[5, 7, 0] = np.nan
    u[9, 9, 1] = np.inf
    cfg = Config(S2, 0.25)
    buf, st, cls = encode(u, cfg, stats=True)
    d = decode(buf, u.shape, u.dtype, cfg)
    assert np.isnan(d[5, 7, 0]) and d[9, 9, 1] == np.inf
    assert cls[5, 7, 0] == VERBATIM
    fin = np.isfinite(u)
    assert np.abs(d[fin] - u[fin]).max() <= 0.25


def test_huge_values_become_verbatim():
    u = step_field()
    u[3, 3, 0] = 1e30
    check_bound(u, Config(S2, 0.25))


@pytest.mark.parametrize("schedule", [(1.0, 1.0), (1.1, 1.5), (2.0, 4.0)])
def test_idempotent_recompression(schedule):
    # with the level schedule held fixed, decoded values are on the
    # quantization lattice and re-encode to themselves
    u = step_field()
    cfg = Config(S2, 0.25)
    d1 = decode(encode(u, cfg, schedule=schedule), u.shape, u.dtype, cfg)
    d2 = decode(encode(d1, cfg, schedule=schedule), u.shape, u.dtype, cfg)
    assert np.array_equal(d1, d2)


@pytest.mark.parametrize("schedule", [None, (1.0, 1.0), (1.25, 2.0), (2.0, 4.0)])
def test_schedule_keeps_bounds(schedule):
    u = step_field()
    cfg = Config(S2, 0.25)
    buf = encode(u, cfg, schedule=schedule)
    if schedule is not None:
        assert fmt.read_header(buf)[3] == schedule
    else:
        assert fmt.read_header(buf)[3] in TUNE_CANDIDATES
    check_bound(u, cfg)
    d = decode(buf, u.shape, u.dtype, cfg)
    assert np.abs(d.astype(np.float64) - u).max() <= 0.25


@pytest.mark.parametrize("bad", [(0.9, 2.0), (1.5, 0.5), (float("nan"), 2.0)])
def test_schedule_rejects_bad_values(bad):
    with pytest.raises(WarpFieldSpecError):
        encode(step_field(), Config(S2, 0.25), schedule=bad)


def test_huffman_roundtrip_and_errors():
    rng = np.random.default_rng(3)
    q = rng.geometric(0.4, 5000) * rng.choice([-1, 1], 5000) * (rng.random(5000) < 0.2)
    q = np.concatenate([q, [5000, -100000, 1 << 40]]).astype(np.int64)
    buf = huffman.encode(q)
    np.testing.assert_array_equal(huffman.decode(buf, q.size), q)
    np.testing.assert_array_equal(huffman.decode(huffman.encode(q[:0]), 0), q[:0])
    with pytest.raises(fmt.WarpFieldFormatError):
        huffman.decode(buf[:20], q.size)


def _version_1(shape, codes, tight, verb, vals) -> bytes:
    """A chunk in the version 1 layout (varint codes, no schedule)."""
    import struct

    import zstandard

    c = zstandard.ZstdCompressor(level=3)
    sections = [
        c.compress(fmt.varint_encode(fmt.zigzag(codes))),
        c.compress(fmt.pack_indices(tight)),
        c.compress(fmt.pack_indices(verb)),
        c.compress(vals.astype(vals.dtype.newbyteorder("<")).tobytes()),
    ]
    head = fmt.MAGIC + struct.pack("<BBH", 1, len(shape), 0)
    head += struct.pack(f"<{len(shape)}Q", *shape)
    head += struct.pack("<4I", *(len(x) for x in sections))
    head += struct.pack("<3Q", codes.size, len(tight), len(verb))
    return head + b"".join(sections)


def test_reads_version_1():
    u = step_field()
    cfg = Config(S2, 0.25)
    v2 = encode(u, cfg, schedule=(1.0, 1.0))
    v1 = _version_1(*fmt.read_chunk(v2, u.dtype))
    assert fmt.read_header(v1)[:2] == (1, 0)
    np.testing.assert_array_equal(
        decode(v1, u.shape, u.dtype, cfg), decode(v2, u.shape, u.dtype, cfg)
    )
    np.testing.assert_array_equal(
        decode(v1, u.shape, u.dtype, cfg, backend="python"), decode(v2, u.shape, u.dtype, cfg)
    )


def test_roles_and_layouts():
    rng = np.random.default_rng(1)
    # component axis first, an independent axis in the middle, a smooth axis
    u = np.cumsum(rng.standard_normal((2, 5, 3, 20, 24)), axis=-1).astype(np.float32)
    cfg = Config(("component", "independent", "smooth", "spatial", "spatial"), 0.1)
    check_bound(u, cfg)


def test_size_one_axes():
    u = step_field((1, 40))
    check_bound(u, Config(S2, 0.1))
    check_bound(step_field((2, 3)), Config(S2, 0.1))


def test_extrapolate_baseline_is_bounded():
    check_bound(step_field(), Config(S2, 0.25, face_error_bound=0.25, boundary="extrapolate"))


def test_jacobian_promotes_samples():
    # A one-cell compression ramp (det ~ 0.2) that the default bound can fold.
    from warpfield.jacobian import det_jacobian, violations

    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:64, 0:64].astype(np.float64)
    h = 1.0
    ux = -0.8 * np.clip(x - 31, 0, 1) + 0.05 * rng.standard_normal(x.shape)
    u = np.stack([0.05 * rng.standard_normal(x.shape), ux], -1).astype(np.float32)
    d0 = det_jacobian(u, 2, (0, 1), (h, h))
    plain = Config(S2, 0.25)
    dp = decode(encode(u, plain), u.shape, u.dtype, plain)
    assert violations(d0, det_jacobian(dp, 2, (0, 1), (h, h)), 0.1).any()
    cfg = Config(S2, 0.25, jacobian=JacobianConfig((h, h), min_det=0.1))
    buf, d, cls = check_bound(u, cfg)
    assert not violations(d0, det_jacobian(d, 2, (0, 1), (h, h)), 0.1).any()
    assert (cls >= 2).any()


def test_tight_ladder_promotes_through_tiers():
    from warpfield.jacobian import det_jacobian, violations

    # a ramp to det J ~ 0.13, just above min_det: the first tier is not
    # enough at its corners, so one tier needs verbatim samples
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:64, 0:64].astype(np.float64)
    ux = -0.87 * np.clip(x - 31, 0, 1) + 0.05 * rng.standard_normal(x.shape)
    u = np.stack([0.05 * rng.standard_normal(x.shape), ux], -1).astype(np.float32)
    d0 = det_jacobian(u, 2, (0, 1), (1.0, 1.0))
    one = Config(S2, 0.25, jacobian=JacobianConfig((1.0, 1.0), 0.1))
    lad = Config(
        S2,
        0.25,
        jacobian={"grid_spacing": [1, 1], "min_det": 0.1, "tight_error_bound": [0.03, 0.01]},
    )
    assert lad.to_dict()["jacobian"]["tight_error_bound"] == [0.03, 0.01]
    assert Config.from_dict(lad.to_dict()).jacobian == lad.jacobian
    assert list(lad.eps_table()) == [0.25, 0.0625, 0.03, 0.0, 0.01]
    # a one-entry list is the plain number
    assert (
        Config(
            S2, 0.25, jacobian={"grid_spacing": [1, 1], "tight_error_bound": [0.03]}
        ).jacobian.tight_error_bound
        == 0.03
    )
    b1 = encode(u, one)
    assert b1[4] == 2  # no ladder: still version 2
    buf, d, cls = check_bound(u, lad)
    assert not violations(d0, det_jacobian(d, 2, (0, 1), (1.0, 1.0)), 0.1).any()
    _, _, c1 = check_bound(u, one)
    # the ladder settles some samples at its second tier where one tier needs verbatim
    assert (cls == 4).any() and (c1 == 3).sum() >= (cls == 3).sum()


@pytest.mark.parametrize(
    "kw",
    [
        dict(axis_roles=("spatial", "bogus")),
        dict(axis_roles=("component", "component", "spatial")),
        dict(axis_roles=("independent", "component")),
        dict(axis_roles=S2, error_bound=0),
        dict(axis_roles=S2, face_error_bound=1.0),
        dict(axis_roles=S2, predictor="quintic"),
        dict(axis_roles=S2, jacobian={"grid_spacing": [1, 1, 1]}),
        dict(axis_roles=("spatial", "spatial"), jacobian={"grid_spacing": [1, 1]}),
        dict(axis_roles=S2, jacobian={"grid_spacing": [1, 1], "tight_error_bound": []}),
        dict(
            axis_roles=S2, jacobian={"grid_spacing": [1, 1], "tight_error_bound": [0.01, 0.02]}
        ),
        dict(axis_roles=S2, jacobian={"grid_spacing": [1, 1], "tight_error_bound": [0.1] * 5}),
    ],
)
def test_config_rejections(kw):
    with pytest.raises(WarpFieldSpecError):
        Config(**kw)


def test_rank_mismatch():
    with pytest.raises(WarpFieldSpecError):
        encode(np.zeros((4, 4), np.float32), Config(S2))


def test_rejects_integer_data():
    with pytest.raises(WarpFieldSpecError):
        encode(np.zeros((4, 4, 2), np.int32), Config(S2))


def test_bad_magic():
    with pytest.raises(fmt.WarpFieldFormatError):
        decode(b"XXXX" + bytes(100), (4, 4, 2), np.float32, Config(S2))


def test_varint_roundtrip():
    rng = np.random.default_rng(0)
    q = np.concatenate([rng.integers(-5, 5, 1000), rng.integers(-(2**40), 2**40, 50), [0]])
    z = fmt.zigzag(q)
    assert np.array_equal(fmt.unzigzag(fmt.varint_decode(fmt.varint_encode(z), z.size)), q)
    idx = np.sort(rng.choice(10**6, 300, replace=False))
    assert np.array_equal(fmt.unpack_indices(fmt.pack_indices(idx), idx.size), idx)
