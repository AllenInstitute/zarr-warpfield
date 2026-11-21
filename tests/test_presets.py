"""Array-level guarantees on every synthetic preset (small scale)."""

import numpy as np
import pytest

from warpfield import Config, JacobianConfig, chunked, jacobian, seams, synth

MIN_DET = 0.1


def configs(p):
    jc = JacobianConfig((p.spacing,) * p.ks, min_det=MIN_DET)
    return {
        "warpfield": Config(p.axis_roles, 0.25, jacobian=jc),
        "baseline": Config(p.axis_roles, 0.25, face_error_bound=0.25, boundary="extrapolate"),
    }


@pytest.fixture(scope="module", params=synth.PRESET_NAMES)
def case(request):
    p = synth.preset(request.param, "small")
    u, lab = p.full()
    out = {k: chunked.roundtrip(u, p.chunks, c) for k, c in configs(p).items()}
    return p, u, lab, out


def test_preset_has_discontinuities(case):
    p, u, lab, _ = case
    assert all((lab == k).any() for k in (1, 2, 4, 6))
    if p.misaligned:
        assert (lab == 5).any()


def test_every_sample_within_its_bound(case):
    p, u, lab, out = case
    for name, cfg in configs(p).items():
        r = out[name]
        err = np.abs(r.decoded.astype(np.float64) - u)
        assert (err <= cfg.eps_table()[r.classes]).all(), name
        assert err.max() <= 0.25
        # ...including right on the discontinuities
        assert err[lab > 0].max() <= 0.25


def test_no_seams(case):
    p, u, lab, out = case
    ca = u.ndim - 1
    axes = tuple(i for i, r in enumerate(p.axis_roles) if r in ("spatial", "smooth"))
    s_wf = seams.seam_stats(out["warpfield"].decoded - u, p.chunks, axes, ca)["all"]
    s_bl = seams.seam_stats(out["baseline"].decoded - u, p.chunks, axes, ca)["all"]
    # boundary-straddling error jumps are statistically like interior ones...
    assert s_wf["ratio"] <= 1.3
    # ...and are bounded by 2 * face_error_bound
    assert s_wf["boundary_max"] <= 2 * 0.25 / 4 + 1e-6
    # the naive chunked baseline does show seams
    assert s_bl["ratio"] >= 1.4


def test_jacobian_guarantee(case):
    p, u, lab, out = case
    ca = u.ndim - 1
    h = (p.spacing,) * p.ks
    d0 = jacobian.det_jacobian(u, ca, p.spatial_axes, h)
    d1 = jacobian.det_jacobian(out["warpfield"].decoded, ca, p.spatial_axes, h)
    bad = jacobian.violations(d0, d1, MIN_DET)
    # Inside chunks there are none; any left must be on cells straddling a chunk face.
    straddle = np.zeros(bad.shape, dtype=bool)
    for a in p.spatial_axes:
        i = np.arange(bad.shape[a])
        m = (i + 1) % p.chunks[a] == 0
        shp = [1] * bad.ndim
        shp[a] = -1
        straddle |= m.reshape(shp)
    assert not (bad & ~straddle).any()
    assert bad.sum() <= 1e-3 * bad.size
    # fold sign is preserved: decoded det is negative wherever the original is
    assert (d1[d0 < 0] < 0).mean() >= 0.99


def test_compression_ratio_reasonable(case):
    p, u, lab, out = case
    # small presets are feature-dense with tiny chunks; see the full-scale benchmark
    assert out["warpfield"].ratio(u) > 10
