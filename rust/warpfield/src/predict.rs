//! Multilevel schedule and per-target stencils. Mirrors `zarr_warpfield.predict`
//! operation for operation, so results are bit-identical to the Python reference.

use crate::config::{Boundary, Predictor};

/// ENO switches from the cubic to a one-sided quadratic stencil when one side's
/// second divided difference exceeds the other's by this factor.
pub const ENO_RATIO: f64 = 3.0;

/// Smallest power of two >= max(N) - 1 (at least 1).
pub fn top_stride(pshape: &[usize]) -> usize {
    let m = pshape
        .iter()
        .map(|&n| n as i64 - 1)
        .max()
        .unwrap_or(1)
        .max(1) as usize;
    let mut s = 1usize;
    while s < m {
        s *= 2;
    }
    s
}

/// Indices known at stride `s` along an axis of extent `n`.
pub fn known(s: usize, n: usize, boundary: Boundary) -> Vec<usize> {
    let mut k: Vec<usize> = (0..n).step_by(s).collect();
    if boundary == Boundary::Clamped && *k.last().unwrap() != n - 1 {
        k.push(n - 1);
    }
    k
}

/// known(h) \ known(2h), ascending.
pub fn new_points(h: usize, n: usize, boundary: Boundary) -> Vec<usize> {
    let b = known(2 * h, n, boundary);
    known(h, n, boundary)
        .into_iter()
        .filter(|x| b.binary_search(x).is_err())
        .collect()
}

/// Stencil of one target along its axis: slots l2, l1, r1, r2.
#[derive(Clone, Debug)]
pub struct Tap {
    pub slots: [usize; 4],
    pub have: [bool; 4],
    pub pos: [f64; 4],
    pub w_lin: [f64; 4],
    pub w_ql: [f64; 4],
    pub w_qr: [f64; 4],
    pub w_cub: [f64; 4],
    pub w_ext: [f64; 4],
}

impl Tap {
    #[inline(always)]
    fn ok_lin(&self) -> bool {
        self.have[1] && self.have[2]
    }
    #[inline(always)]
    fn ok_ql(&self) -> bool {
        self.ok_lin() && self.have[0]
    }
    #[inline(always)]
    fn ok_qr(&self) -> bool {
        self.ok_lin() && self.have[3]
    }
    #[inline(always)]
    fn ok_cub(&self) -> bool {
        self.ok_ql() && self.have[3]
    }
}

fn lagrange(xs: &[f64], t: f64) -> Vec<f64> {
    (0..xs.len())
        .map(|i| {
            let mut w = 1.0f64;
            for (j, &xj) in xs.iter().enumerate() {
                if i != j {
                    w = w * (t - xj) / (xs[i] - xj);
                }
            }
            w
        })
        .collect()
}

fn weights(idx: &[usize], ok: bool, pos: &[f64; 4], t: f64) -> [f64; 4] {
    let mut w = [0.0f64; 4];
    if ok {
        let xs: Vec<f64> = idx.iter().map(|&i| pos[i]).collect();
        for (&i, wi) in idx.iter().zip(lagrange(&xs, t)) {
            w[i] = wi;
        }
    }
    w
}

/// Stencils for `targets` (none of which are in `anchors`, both ascending).
pub fn stencil(targets: &[usize], anchors: &[usize]) -> Vec<Tap> {
    let a = anchors;
    let na = a.len() as i64;
    let at = |i: i64| a[i.clamp(0, na - 1) as usize];
    targets
        .iter()
        .map(|&tg| {
            let p = a.partition_point(|&x| x < tg) as i64; // searchsorted, side="left"
            let have = [p >= 2, p >= 1, p < na, p + 1 < na];
            let l1 = at(p - 1);
            let l2 = if have[0] { at(p - 2) } else { l1 };
            let r1 = if have[2] { at(p) } else { l1 };
            let r2 = if have[3] { at(p + 1) } else { r1 };
            let slots = [l2, l1, r1, r2];
            let pos = slots.map(|s| s as f64);
            let t = tg as f64;
            let ok_lin = have[1] && have[2];
            let ok_ql = ok_lin && have[0];
            let ok_qr = ok_lin && have[3];
            let ok_cub = ok_ql && have[3];
            let mut w_ext = [0.0f64; 4];
            let no_r = !have[2];
            if no_r && have[0] {
                let e = lagrange(&[pos[0], pos[1]], t);
                w_ext[0] = e[0];
                w_ext[1] = e[1];
            } else if no_r {
                w_ext[1] = 1.0;
            }
            Tap {
                slots,
                have,
                pos,
                w_lin: weights(&[1, 2], ok_lin, &pos, t),
                w_ql: weights(&[0, 1, 2], ok_ql, &pos, t),
                w_qr: weights(&[1, 2, 3], ok_qr, &pos, t),
                w_cub: weights(&[0, 1, 2, 3], ok_cub, &pos, t),
                w_ext,
            }
        })
        .collect()
}

/// Fixed summation order l2, l1, r1, r2, all four terms (zero weights included,
/// so non-finite neighbours propagate exactly as in the reference).
#[inline(always)]
fn apply(w: &[f64; 4], v: &[f64; 4]) -> f64 {
    let mut acc = w[0] * v[0];
    acc += w[1] * v[1];
    acc += w[2] * v[2];
    acc + w[3] * v[3]
}

#[inline(always)]
fn dd2(xa: f64, xb: f64, xc: f64, fa: f64, fb: f64, fc: f64) -> f64 {
    ((fc - fb) / (xc - xb) - (fb - fa) / (xb - xa)) / (xc - xa)
}

/// Prediction of one target from its four slot values.
#[inline(always)]
pub fn predict(t: &Tap, v: &[f64; 4], predictor: Predictor, tau: f64) -> f64 {
    let base = |t: &Tap| {
        if t.ok_lin() {
            apply(&t.w_lin, v)
        } else {
            apply(&t.w_ext, v)
        }
    };
    match predictor {
        Predictor::Linear => base(t),
        Predictor::Cubic => {
            if t.ok_cub() {
                apply(&t.w_cub, v)
            } else if t.ok_ql() {
                apply(&t.w_ql, v)
            } else if t.ok_qr() {
                apply(&t.w_qr, v)
            } else {
                base(t)
            }
        }
        Predictor::Eno => {
            if t.ok_cub() {
                let x = &t.pos;
                let dl = dd2(x[0], x[1], x[2], v[0], v[1], v[2]).abs();
                let dr = dd2(x[1], x[2], x[3], v[1], v[2], v[3]).abs();
                let smooth = (dl <= ENO_RATIO * dr + tau) && (dr <= ENO_RATIO * dl + tau);
                if smooth {
                    apply(&t.w_cub, v)
                } else if dl <= dr {
                    apply(&t.w_ql, v)
                } else {
                    apply(&t.w_qr, v)
                }
            } else if t.ok_ql() {
                apply(&t.w_ql, v)
            } else if t.ok_qr() {
                apply(&t.w_qr, v)
            } else {
                base(t)
            }
        }
    }
}
