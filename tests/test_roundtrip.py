"""Inverse-transform round trips through the decoded field."""

import numpy as np
import pytest

from zarr_warpfield import Config, chunked, inverse, synth


@pytest.fixture(scope="module")
def field2d():
    p = synth.preset("2d", "small")
    u, lab = p.full()
    cfg = Config(p.axis_roles, 0.25, jacobian={"grid_spacing": [p.spacing] * 2, "min_det": 0.1})
    dec = chunked.roundtrip(u, p.chunks, cfg).decoded
    return p, u.astype(np.float64), dec.astype(np.float64), lab


def test_forward_only_roundtrip_bound(field2d):
    """Invert the decoded forward field; the true forward round trip is off by <= eps.

    |phi(psi_hat(y)) - y| <= |phi_hat(psi_hat(y)) - y| + eps holds at every y,
    because multilinear interpolation keeps |U - U_hat| <= eps everywhere.
    """
    p, u, dec, lab = field2d
    h = np.array([p.spacing] * 2)
    v_hat, res = inverse.invert(dec, h)
    v_hat = v_hat.astype(np.float64)
    rt_hat = inverse.roundtrip(dec, v_hat, h)
    rt_true = inverse.roundtrip(u, v_hat, h)
    assert (rt_true <= rt_hat + 0.25 + 1e-6).all()
    # away from the discontinuities the inverse converges, so the round trip is ~eps
    smooth = lab == 0
    assert np.median(rt_hat[smooth]) < 1e-3
    assert np.percentile(rt_true[smooth], 99) <= 0.25 + 1e-3


def test_separate_inverse_bound(field2d):
    """Compress f and g = f^-1 separately.

    |g_hat(f_hat(x)) - x| <= eps_g + Lip(g) eps_f + r0.
    """
    p, u, dec, lab = field2d
    h = np.array([p.spacing] * 2)
    v, _ = inverse.invert(u, h)
    v = v.astype(np.float64)
    cfg = Config(p.axis_roles, 0.25)
    v_hat = chunked.roundtrip(v.astype(np.float32), p.chunks, cfg).decoded.astype(np.float64)
    r0 = inverse.roundtrip_fg(u, v, h)
    got = inverse.roundtrip_fg(dec, v_hat, h)
    lip = inverse.lipschitz_inf(v, h)
    assert (got <= 0.25 + lip * 0.25 + r0 + 1e-5).all()
