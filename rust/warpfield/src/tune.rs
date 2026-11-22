//! Per-chunk choice of the level schedule, after QoZ / HPEZ (Liu et al.,
//! SC 2022 and SIGMOD 2024): code a central sub-block of the chunk with each
//! candidate schedule and keep the cheapest.
//!
//! Deterministic by construction, so the Python reference picks the same
//! schedule: the cost is an integer, Σ over nonzero codes of
//! 1 + bit length of zigzag(q), plus 64 per sample that falls back to verbatim,
//! and ties go to the earlier candidate.

use crate::config::{Config, VERBATIM};
use crate::element::Element;
use crate::error::Result;
use crate::format::zigzag;
use crate::layout::{Layout, c_strides};
use crate::pass::Schedule;

/// Candidate (alpha, beta) pairs, in tie-break order. (1, 1) is uniform.
pub const CANDIDATES: [(f64, f64); 7] = [
    (1.0, 1.0),
    (1.1, 1.5),
    (1.1, 3.0),
    (1.25, 2.0),
    (1.5, 2.0),
    (1.5, 3.0),
    (2.0, 4.0),
];

/// Extent of the tuning sub-block along each predicted axis (2^6 + 1, so its
/// own lattice is complete). Chunks up to this size are tuned on themselves.
pub const SAMPLE: usize = 65;

/// Central sub-block of a canonical array (all batch entries).
fn sub_block(x: &[f64], cshape: &[usize], extent: usize) -> (Vec<f64>, Vec<usize>) {
    let mut start = vec![0usize; cshape.len()];
    let mut sub = cshape.to_vec();
    for ax in 1..cshape.len() {
        if cshape[ax] > extent {
            start[ax] = (cshape[ax] - extent) / 2;
            sub[ax] = extent;
        }
    }
    let strides = c_strides(cshape);
    let n: usize = sub.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; sub.len()];
    for _ in 0..n {
        let lin: usize = (0..sub.len())
            .map(|a| (start[a] + idx[a]) * strides[a])
            .sum();
        out.push(x[lin]);
        for a in (0..sub.len()).rev() {
            idx[a] += 1;
            if idx[a] < sub[a] {
                break;
            }
            idx[a] = 0;
        }
    }
    (out, sub)
}

pub fn cost(codes: &[i64], n_verbatim: usize) -> u64 {
    let mut c = 64 * n_verbatim as u64;
    for &q in codes {
        if q != 0 {
            c += 1 + (64 - zigzag(q).leading_zeros()) as u64;
        }
    }
    c
}

/// The cheapest candidate schedule for the canonical chunk `x` of layout `lay`,
/// each candidate run through the full encoder loop (Jacobian included) on
/// the central sub-block.
pub(crate) fn tune<T: Element>(cfg: &Config, lay: &Layout, x: &[f64]) -> Result<Schedule> {
    let (xs, sub) = sub_block(x, &lay.canonical_shape(), SAMPLE);
    let mut sl = lay.clone();
    sl.pred_shape = sub[1..].to_vec();
    let mut best = (u64::MAX, Schedule::UNIFORM);
    for (alpha, beta) in CANDIDATES {
        let sched = Schedule { alpha, beta };
        let (cls, codes, _) = crate::code_canonical::<T>(cfg, &sl, &xs, sched)?;
        let nv = cls.iter().filter(|&&c| c == VERBATIM).count();
        let c = cost(&codes, nv);
        if c < best.0 {
            best = (c, sched);
        }
    }
    Ok(best.1)
}
