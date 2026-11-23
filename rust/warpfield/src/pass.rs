//! The closed-loop multilevel pass shared by the encoder and the decoder.
//!
//! Samples are visited in exactly the order of the Python reference: blocks in
//! schedule order (anchors, then per level h and per predicted axis j), and
//! within a block in C order over (batch, list_0, list_1, ...). Quantization
//! codes are emitted / consumed in that order.

use crate::config::{Config, N_CLASSES, VERBATIM};
use crate::element::Element;
use crate::error::{Result, format};
use crate::layout::c_strides;
use crate::predict::{Tap, known, new_points, predict, stencil, top_stride};

pub(crate) const Q_LIMIT: f64 = 4503599627370496.0; // 2**52

/// Visit every sample of the block batch × lists[0] × lists[1] × ... in C order.
/// `offs[i][k]` is `lists[i][k] * stride(i)`. `f(linear index, per-axis list index)`.
#[inline(always)]
fn for_block(nb: usize, sb: usize, offs: &[Vec<usize>], mut f: impl FnMut(usize, &[usize])) {
    let d = offs.len();
    if offs.iter().any(|o| o.is_empty()) {
        return;
    }
    let mut idx = vec![0usize; d];
    let last = &offs[d - 1];
    for b in 0..nb {
        idx.fill(0);
        loop {
            let mut base = b * sb;
            for i in 0..d - 1 {
                base += offs[i][idx[i]];
            }
            for (k, &o) in last.iter().enumerate() {
                idx[d - 1] = k;
                f(base + o, &idx);
            }
            let mut a = d - 1;
            let mut done = true;
            while a > 0 {
                a -= 1;
                idx[a] += 1;
                if idx[a] < offs[a].len() {
                    done = false;
                    break;
                }
                idx[a] = 0;
            }
            if done {
                break;
            }
        }
    }
}

/// What the pass does with each sample once its prediction is known.
pub(crate) trait Quantizer {
    fn sample(&mut self, lin: usize, pred: f64, recon: &mut [f64]);
    /// Per-class bounds for the samples that follow (the level schedule).
    fn set_eps(&mut self, eps: [f64; N_CLASSES]);
    /// Called after each block (0: the anchors, then one per level and axis
    /// with targets, in pass order).
    fn end_block(&mut self, _step: usize, _recon: &mut [f64]) {}
}

/// Level schedule (QoZ / HPEZ): the level with interpolation step h = 2^L is
/// coded at `eps / min(alpha^L, beta)`, so coarse samples, which every finer
/// level predicts from, are coded more tightly. alpha = beta = 1 is uniform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    pub alpha: f64,
    pub beta: f64,
}

impl Schedule {
    pub const UNIFORM: Schedule = Schedule {
        alpha: 1.0,
        beta: 1.0,
    };

    /// min(alpha^L, beta) for step h = 2^L, alpha^L by repeated multiplication.
    pub fn factor(&self, h: usize) -> f64 {
        let mut f = 1.0f64;
        let mut s = 1usize;
        while s < h {
            f *= self.alpha;
            s *= 2;
        }
        f.min(self.beta)
    }

    pub fn eps(&self, base: &[f64; N_CLASSES], h: usize) -> [f64; N_CLASSES] {
        let f = self.factor(h);
        base.map(|e| e / f)
    }
}

/// Encodes every sample: its code goes to `qs[lin]` and `lin` to `order`, so
/// the code stream is `qs` over `order`, skipping verbatim samples. With a
/// Jacobian `check`, each block then tightens the corners of the cells it
/// completes and breaks (`incr::fix_block`).
pub(crate) struct Encoder<'a, T: Element> {
    pub x: &'a [f64],
    pub cls: &'a mut [u8],
    pub eps: [f64; N_CLASSES],
    pub qs: &'a mut [i64],
    pub pred: &'a mut [f64],
    pub order: Vec<u32>,
    pub check: Option<&'a crate::incr::Check<'a>>,
    pub promoted: &'a mut [Vec<usize>],
    pub stuck: &'a mut [Vec<usize>],
    pub _t: std::marker::PhantomData<T>,
}

impl<T: Element> Quantizer for Encoder<'_, T> {
    fn set_eps(&mut self, eps: [f64; N_CLASSES]) {
        self.eps = eps;
    }
    #[inline(always)]
    fn sample(&mut self, lin: usize, pred: f64, recon: &mut [f64]) {
        let (c, q, r) = crate::incr::quantize::<T>(self.x[lin], pred, self.cls[lin], &self.eps);
        self.cls[lin] = c;
        self.qs[lin] = q;
        self.pred[lin] = pred;
        recon[lin] = r;
        self.order.push(lin as u32);
    }
    fn end_block(&mut self, step: usize, recon: &mut [f64]) {
        if let Some(ch) = self.check {
            crate::incr::fix_block::<T>(
                ch,
                step,
                &self.eps,
                self.x,
                recon,
                self.cls,
                self.qs,
                self.pred,
                &mut self.promoted[step],
                &mut self.stuck[step],
                None,
            );
        }
    }
}

pub(crate) struct Decoder<'a, T: Element> {
    pub codes: &'a [i64],
    pub cursor: usize,
    pub cls: &'a [u8],
    pub eps: [f64; N_CLASSES],
    pub short: bool,
    pub _t: std::marker::PhantomData<T>,
}

impl<T: Element> Quantizer for Decoder<'_, T> {
    fn set_eps(&mut self, eps: [f64; N_CLASSES]) {
        self.eps = eps;
    }
    #[inline(always)]
    fn sample(&mut self, lin: usize, pred: f64, recon: &mut [f64]) {
        let c = self.cls[lin];
        if c != VERBATIM {
            let Some(&q) = self.codes.get(self.cursor) else {
                self.short = true;
                return;
            };
            self.cursor += 1;
            let w = 2.0 * self.eps[c as usize];
            recon[lin] = T::round(pred + (q as f64) * w);
        }
    }
}

/// Run the pass over the canonical array `recon` of shape `cshape`.
pub(crate) fn run<Q: Quantizer>(
    cfg: &Config,
    sched: Schedule,
    cshape: &[usize],
    recon: &mut [f64],
    q: &mut Q,
) {
    let base_eps = cfg.eps_table();
    let nb = cshape[0];
    let pshape = &cshape[1..];
    let d = pshape.len();
    let strides = c_strides(cshape);
    let sb = strides[0];
    let boundary = cfg.boundary;
    let offsets = |lists: &[Vec<usize>]| -> Vec<Vec<usize>> {
        lists
            .iter()
            .enumerate()
            .map(|(i, l)| l.iter().map(|&v| v * strides[i + 1]).collect())
            .collect()
    };

    // anchors: the coarsest lattice (the chunk corners), predicted as zero
    let s_top = top_stride(pshape);
    let lists: Vec<Vec<usize>> = pshape.iter().map(|&n| known(s_top, n, boundary)).collect();
    q.set_eps(sched.eps(&base_eps, s_top));
    for_block(nb, sb, &offsets(&lists), |lin, _| q.sample(lin, 0.0, recon));
    q.end_block(0, recon);
    let mut step = 0usize;

    let mut h = s_top / 2;
    while h >= 1 {
        let tau = cfg.error_bound / ((h * h) as f64);
        q.set_eps(sched.eps(&base_eps, h));
        for j in 0..d {
            let n = pshape[j];
            let tg = new_points(h, n, boundary);
            if tg.is_empty() {
                continue;
            }
            let taps: Vec<Tap> = stencil(&tg, &known(2 * h, n, boundary));
            let lists: Vec<Vec<usize>> = (0..d)
                .map(|i| match i.cmp(&j) {
                    std::cmp::Ordering::Less => known(h, pshape[i], boundary),
                    std::cmp::Ordering::Equal => tg.clone(),
                    std::cmp::Ordering::Greater => known(2 * h, pshape[i], boundary),
                })
                .collect();
            let sj = strides[j + 1];
            let tg_off: Vec<usize> = tg.iter().map(|&t| t * sj).collect();
            let slot_off: Vec<[usize; 4]> = taps.iter().map(|t| t.slots.map(|s| s * sj)).collect();
            let predictor = cfg.predictor;
            for_block(nb, sb, &offsets(&lists), |lin, idx| {
                let k = idx[j];
                let base = lin - tg_off[k];
                let so = &slot_off[k];
                let v = [
                    recon[base + so[0]],
                    recon[base + so[1]],
                    recon[base + so[2]],
                    recon[base + so[3]],
                ];
                let mut p = predict(&taps[k], &v, predictor, tau);
                if !p.is_finite() {
                    p = 0.0;
                }
                q.sample(lin, p, recon);
            });
            step += 1;
            q.end_block(step, recon);
        }
        h /= 2;
    }
}

pub(crate) fn check_decoded<T: Element>(d: &Decoder<'_, T>) -> Result<()> {
    if d.short {
        return format("too few quantization codes for this chunk shape");
    }
    if d.cursor != d.codes.len() {
        return format(format!(
            "{} unused quantization codes",
            d.codes.len() - d.cursor
        ));
    }
    Ok(())
}
