//! Incremental re-encoding for the Jacobian loop.
//!
//! When a round of the loop tightens some samples, only those samples, and the
//! samples predicted from them (transitively), can change. A sample's value
//! depends only on its class, its original value and the four stencil slots it
//! is predicted from, all coded earlier in the pass. So a round re-codes the
//! tightened samples in pass order and follows the dependency edges forward,
//! stopping wherever a reconstruction comes out bit-identical. The result is
//! exactly that of a full re-run of the pass with the new classes (which is
//! what the Python reference does): every sample not visited has the same
//! inputs, hence the same code, class and reconstruction.
//!
//! With a Jacobian constraint the pass also checks, after each block, the
//! cells that block completes, and tightens a broken cell's corners in that
//! block on the spot (`fix_block`): nothing has been predicted from them yet,
//! so fixing them cannot break anything downstream. A round re-derives those
//! tightenings: any block whose cells could check differently (a dependency
//! changed) undoes them and runs the fix again from the round's classes, as a
//! full pass would.

use crate::config::{Config, N_CLASSES, VERBATIM};
use crate::element::Element;
use crate::jacobian::{Geometry, violation};
use crate::layout::c_strides;
use crate::pass::{Q_LIMIT, Schedule};
use crate::predict::{Tap, known, new_points, stencil, top_stride};

/// One block of the pass after the anchors: level `h`, axis `j`.
struct Step {
    h: usize,
    j: usize,
    tg: Vec<usize>,
    taps: Vec<Tap>,
    /// Axis-j position -> index into `tg`, or -1.
    inv: Vec<i32>,
    /// Axis-j position -> indices into `tg` of the targets using it as a slot.
    feeds: Vec<Vec<u32>>,
    /// Per axis i != j: is the position on this block's lattice.
    lattice: Vec<Vec<bool>>,
    /// The block's per-axis index lists (as in `pass::run`).
    lists: Vec<Vec<usize>>,
}

/// The pass's block structure for one canonical shape.
pub(crate) struct Plan {
    pshape: Vec<usize>,
    strides: Vec<usize>, // canonical strides (batch axis first)
    psize: usize,
    s_top: usize,
    steps: Vec<Step>,
    /// Per predicted offset: 0 anchors, 1 + step index otherwise.
    step_of: Vec<u16>,
}

fn membership(list: &[usize], n: usize) -> Vec<bool> {
    let mut m = vec![false; n];
    for &v in list {
        m[v] = true;
    }
    m
}

impl Plan {
    pub fn new(cfg: &Config, cshape: &[usize]) -> Plan {
        let boundary = cfg.boundary;
        let pshape = cshape[1..].to_vec();
        let d = pshape.len();
        let strides = c_strides(cshape);
        let psize: usize = pshape.iter().product();
        let pstr = c_strides(&pshape);
        let s_top = top_stride(&pshape);
        let mut step_of = vec![u16::MAX; psize];
        let mut steps = Vec::new();
        let mut h = s_top / 2;
        while h >= 1 {
            for j in 0..d {
                let n = pshape[j];
                let tg = new_points(h, n, boundary);
                if tg.is_empty() {
                    continue;
                }
                let taps = stencil(&tg, &known(2 * h, n, boundary));
                let mut inv = vec![-1i32; n];
                let mut feeds = vec![Vec::new(); n];
                for (k, (&t, tap)) in tg.iter().zip(&taps).enumerate() {
                    inv[t] = k as i32;
                    for &s in &tap.slots {
                        if s < n && !feeds[s].contains(&(k as u32)) {
                            feeds[s].push(k as u32);
                        }
                    }
                }
                let lists: Vec<Vec<usize>> = (0..d)
                    .map(|i| match i.cmp(&j) {
                        std::cmp::Ordering::Less => known(h, pshape[i], boundary),
                        std::cmp::Ordering::Equal => tg.clone(),
                        std::cmp::Ordering::Greater => known(2 * h, pshape[i], boundary),
                    })
                    .collect();
                let lattice = (0..d).map(|i| membership(&lists[i], pshape[i])).collect();
                steps.push(Step {
                    h,
                    j,
                    tg,
                    taps,
                    inv,
                    feeds,
                    lattice,
                    lists,
                });
            }
            h /= 2;
        }
        // which block codes each predicted position
        let pstr = &pstr;
        let mark = |lists: &[Vec<usize>], id: u16, step_of: &mut [u16]| {
            let mut offs: Vec<usize> = vec![0];
            for (i, l) in lists.iter().enumerate() {
                offs = offs
                    .iter()
                    .flat_map(|&o| {
                        let si = pstr[i];
                        l.iter().map(move |&v| o + v * si)
                    })
                    .collect();
            }
            for o in offs {
                step_of[o] = id;
            }
        };
        let anchor_lists: Vec<Vec<usize>> =
            pshape.iter().map(|&n| known(s_top, n, boundary)).collect();
        mark(&anchor_lists, 0, &mut step_of);
        for (si, st) in steps.iter().enumerate() {
            mark(&st.lists, 1 + si as u16, &mut step_of);
        }
        debug_assert!(step_of.iter().all(|&s| s != u16::MAX));
        Plan {
            pshape,
            strides,
            psize,
            s_top,
            steps,
            step_of,
        }
    }

    #[inline]
    fn coord(&self, lin: usize, axis: usize) -> usize {
        (lin / self.strides[axis + 1]) % self.pshape[axis]
    }

    /// Number of blocks: the anchors plus one per level and axis with targets.
    pub fn n_blocks(&self) -> usize {
        self.steps.len() + 1
    }
}

/// Per chunk shape: the pass's blocks, and which block completes each
/// Jacobian cell. Built once per shape and thread (tuning codes the same
/// sub-block shape seven times per chunk, and every chunk of an array shares
/// its shape).
pub(crate) struct Tables {
    pub plan: Plan,
    /// Block that codes each cell's last dependency.
    cell_step: Vec<u16>,
    cells_by_step: Vec<Vec<usize>>,
    /// (independent row, predicted offset of corner 0) per cell.
    origin: Vec<(u32, u32)>,
}

type TablesKey = (Vec<usize>, u8, Vec<usize>, usize);

thread_local! {
    static TABLES: std::cell::RefCell<Vec<(TablesKey, std::rc::Rc<Tables>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The tables for this chunk shape and geometry (cached per thread).
pub(crate) fn tables(
    cfg: &Config,
    cshape: &[usize],
    g: &Geometry,
    spatial_in_pred: &[usize],
) -> std::rc::Rc<Tables> {
    let key: TablesKey = (
        cshape.to_vec(),
        cfg.boundary as u8,
        spatial_in_pred.to_vec(),
        g.n_cells(),
    );
    if let Some(t) = TABLES.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, t)| t.clone())
    }) {
        return t;
    }
    let plan = Plan::new(cfg, cshape);
    let k = g.n_spatial();
    let mut cell_step = vec![0u16; g.n_cells()];
    let mut cells_by_step = vec![Vec::new(); plan.n_blocks()];
    let mut origin = Vec::with_capacity(g.n_cells());
    for (ci, cs) in cell_step.iter_mut().enumerate() {
        let (i, offs) = g.dep_offsets(ci);
        origin.push((i as u32, offs[0] as u32));
        *cs = offs[..=k].iter().map(|&o| plan.step_of[o]).max().unwrap();
        cells_by_step[*cs as usize].push(ci);
    }
    let t = std::rc::Rc::new(Tables {
        plan,
        cell_step,
        cells_by_step,
        origin,
    });
    TABLES.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= 8 {
            c.remove(0);
        }
        c.push((key, t.clone()));
    });
    t
}

/// The Jacobian check during the pass, for one chunk.
pub(crate) struct Check<'a> {
    pub g: &'a Geometry,
    pub d0: &'a [f64],
    pub min_det: f64,
    t: &'a Tables,
    /// Spatial steps of a cell in predicted offsets.
    step: Vec<usize>,
    /// Per class: the loosest tight tier with a smaller bound, else verbatim.
    nxt: [u8; N_CLASSES],
}

impl<'a> Check<'a> {
    pub fn new(
        g: &'a Geometry,
        d0: &'a [f64],
        min_det: f64,
        t: &'a Tables,
        eps: &[f64; N_CLASSES],
        ladder: &[u8],
    ) -> Check<'a> {
        let mut nxt = [VERBATIM; N_CLASSES];
        for (c, n) in nxt.iter_mut().enumerate() {
            for &tier in ladder.iter().rev() {
                if eps[tier as usize] < eps[c] {
                    *n = tier;
                }
            }
        }
        Check {
            g,
            d0,
            min_det,
            t,
            step: g.step(),
            nxt,
        }
    }

    /// det J of the listed cells.
    fn dets(&self, u: &[f64], cells: &[usize]) -> Vec<f64> {
        cells
            .iter()
            .map(|&c| {
                let (i, off) = self.t.origin[c];
                self.g.cell_det(u, &self.step, i as usize, off as usize)
            })
            .collect()
    }

    /// Mark the blocks whose cell checks depend on sample `lin`.
    fn mark_dirty(&self, lin: usize, dirty: &mut [bool], scratch: &mut Vec<usize>) {
        scratch.clear();
        self.g.dep_cells_of_sample(lin, scratch);
        for &c in scratch.iter() {
            dirty[self.t.cell_step[c] as usize] = true;
        }
    }
}

/// After block `step` is coded: check the cells it completes; while some are
/// broken, move their dependency samples in this block one tier tighter (then
/// verbatim) and re-quantize them. Same result as the Python reference's
/// `_run_pass` with `check`, which re-checks every completed cell on each
/// iteration: only cells next to a just-tightened sample can change, so only
/// those are re-checked here. Appends to `stuck` the cells still broken at
/// the end; nothing later in the pass changes their inputs, so these are
/// exactly the block's cells that are broken after the pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fix_block<T: Element>(
    ch: &Check<'_>,
    step: usize,
    eps: &[f64; N_CLASSES],
    x: &[f64],
    recon: &mut [f64],
    cls: &mut [u8],
    qs: &mut [i64],
    pred: &[f64],
    promoted: &mut Vec<usize>,
    stuck: &mut Vec<usize>,
    mut changed: Option<&mut Vec<usize>>,
) {
    let k = ch.g.n_spatial();
    let all: &[usize] = &ch.t.cells_by_step[step];
    if all.is_empty() {
        return;
    }
    let mut check: Vec<usize> = Vec::new();
    let mut first = true;
    let mut ever_bad: Vec<usize> = Vec::new();
    let mut hits: Vec<usize> = Vec::new();
    let mut scratch: Vec<usize> = Vec::with_capacity(4);
    loop {
        let list: &[usize] = if first { all } else { &check };
        first = false;
        let dets = ch.dets(recon, list);
        let bad: Vec<usize> = list
            .iter()
            .zip(&dets)
            .filter(|&(&c, &v)| violation(ch.d0[c], v, ch.min_det))
            .map(|(&c, _)| c)
            .collect();
        ever_bad.extend_from_slice(&bad);
        hits.clear();
        for &c in &bad {
            let (i, offs) = ch.g.dep_offsets(c);
            for &o in &offs[..=k] {
                if ch.t.plan.step_of[o] as usize != step {
                    continue;
                }
                for comp in 0..k {
                    let lin = ch.g.sample(i, comp, o);
                    if cls[lin] != VERBATIM {
                        hits.push(lin);
                    }
                }
            }
        }
        hits.sort_unstable();
        hits.dedup();
        if hits.is_empty() {
            break;
        }
        check.clear();
        for &h in &hits {
            let (c, q, r) = quantize::<T>(x[h], pred[h], ch.nxt[cls[h] as usize], eps);
            cls[h] = c;
            qs[h] = q;
            if r.to_bits() != recon[h].to_bits() {
                recon[h] = r;
                if let Some(chg) = changed.as_deref_mut() {
                    chg.push(h);
                }
            }
            // the cells of this block that depend on it may check (or hit)
            // differently now, even if its value came out the same
            scratch.clear();
            ch.g.dep_cells_of_sample(h, &mut scratch);
            check.extend(
                scratch
                    .iter()
                    .filter(|&&c| ch.t.cell_step[c] as usize == step),
            );
            promoted.push(h);
        }
        // cells broken before whose hits were all exact stay as they were;
        // re-check the touched ones
        check.sort_unstable();
        check.dedup();
    }
    // the block's broken cells now: only cells ever found broken can be
    ever_bad.sort_unstable();
    ever_bad.dedup();
    let dets = ch.dets(recon, &ever_bad);
    stuck.extend(
        ever_bad
            .iter()
            .zip(&dets)
            .filter(|&(&c, &v)| violation(ch.d0[c], v, ch.min_det))
            .map(|(&c, _)| c),
    );
}

/// Quantize one sample of class `c` (see `pass::Encoder`): (class, code, recon).
#[inline(always)]
pub(crate) fn quantize<T: Element>(
    x: f64,
    pred: f64,
    c: u8,
    eps: &[f64; N_CLASSES],
) -> (u8, i64, f64) {
    if c != VERBATIM {
        let e = eps[c as usize];
        let w = 2.0 * e;
        let q = ((x - pred) / w).round_ties_even();
        let r = T::round(pred + q * w);
        if x.is_finite() && (x - r).abs() <= e && q.abs() < Q_LIMIT {
            return (c, q as i64, r);
        }
    }
    (VERBATIM, 0, x)
}

/// State of one canonical chunk between rounds of the Jacobian loop.
pub(crate) struct State<'a> {
    pub x: &'a [f64],
    pub recon: Vec<f64>,
    pub cls: Vec<u8>,
    pub qs: Vec<i64>,
    /// Each sample's prediction in the last pass that coded it.
    pub pred: Vec<f64>,
    /// Per block: samples the in-pass check tightened (may repeat).
    pub promoted: Vec<Vec<usize>>,
    /// Per block: cells it completes that are still broken after its check.
    pub stuck: Vec<Vec<usize>>,
}

impl State<'_> {
    /// Every cell broken after the pass (the blocks' `stuck` lists).
    pub fn broken(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.stuck.iter().flatten().copied().collect();
        v.sort_unstable();
        v
    }
}

/// Re-code `seeds` (samples whose class changed; `cur` holds every sample's
/// class before quantization) and everything that depends on them. Returns
/// the samples whose reconstruction changed, or None as soon as more than
/// `limit` have: the state is then partly updated, and the caller re-runs the
/// full pass instead (cheaper than this bookkeeping for large changes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rerun<T: Element>(
    cfg: &Config,
    sched: Schedule,
    plan: &Plan,
    st: &mut State<'_>,
    cur: &[u8],
    seeds: &[usize],
    limit: usize,
    check: Option<&Check<'_>>,
) -> Option<Vec<usize>> {
    let base_eps = cfg.eps_table();
    let nsteps = plan.steps.len() + 1;
    let mut dirty = vec![false; nsteps];
    let mut scratch = Vec::with_capacity(4);
    let mut by_step: Vec<Vec<usize>> = vec![Vec::new(); nsteps];
    for &s in seeds {
        by_step[plan.step_of[s % plan.psize] as usize].push(s);
    }
    let mut changed: Vec<usize> = Vec::new();
    let mut queued = vec![u32::MAX; st.recon.len()];

    let recode = |st: &mut State<'_>,
                  lin: usize,
                  pred: f64,
                  eps: &[f64; N_CLASSES],
                  changed: &mut Vec<usize>| {
        let (c, q, r) = quantize::<T>(st.x[lin], pred, cur[lin], eps);
        st.cls[lin] = c;
        st.qs[lin] = q;
        st.pred[lin] = pred;
        if r.to_bits() != st.recon[lin].to_bits() {
            st.recon[lin] = r;
            changed.push(lin);
        }
    };

    // After a block's candidates are re-coded: if any of the cells it completes
    // could check differently, undo last round's in-pass tightenings there and
    // run the check again from this round's classes.
    let in_pass = |s: usize,
                   cand: &[usize],
                   queued: &[u32],
                   eps: &[f64; N_CLASSES],
                   st: &mut State<'_>,
                   changed: &mut Vec<usize>,
                   dirty: &mut [bool],
                   scratch: &mut Vec<usize>| {
        let Some(ch) = check else { return };
        for &lin in cand {
            ch.mark_dirty(lin, dirty, scratch);
        }
        if !dirty[s] {
            return;
        }
        let before = changed.len();
        let mut old = std::mem::take(&mut st.promoted[s]);
        old.sort_unstable();
        old.dedup();
        for lin in old {
            if queued[lin] == s as u32 {
                continue; // re-coded from this round's class already
            }
            let (c, q, r) = quantize::<T>(st.x[lin], st.pred[lin], cur[lin], eps);
            st.cls[lin] = c;
            st.qs[lin] = q;
            if r.to_bits() != st.recon[lin].to_bits() {
                st.recon[lin] = r;
                changed.push(lin);
            }
        }
        let mut promoted = Vec::new();
        let mut stuck = Vec::new();
        fix_block::<T>(
            ch,
            s,
            eps,
            st.x,
            &mut st.recon,
            &mut st.cls,
            &mut st.qs,
            &st.pred,
            &mut promoted,
            &mut stuck,
            Some(changed),
        );
        st.promoted[s] = promoted;
        st.stuck[s] = stuck;
        for &lin in &changed[before..] {
            ch.mark_dirty(lin, dirty, scratch);
        }
    };

    // anchors: predicted as zero, depend on nothing
    let eps = sched.eps(&base_eps, plan.s_top);
    for &lin in &by_step[0] {
        queued[lin] = 0;
        recode(st, lin, 0.0, &eps, &mut changed);
    }
    in_pass(
        0,
        &by_step[0],
        &queued,
        &eps,
        st,
        &mut changed,
        &mut dirty,
        &mut scratch,
    );

    let mut cand: Vec<usize> = Vec::new();
    for (si, step) in plan.steps.iter().enumerate() {
        let tag = (si + 1) as u32;
        cand.clear();
        for &lin in &by_step[si + 1] {
            if queued[lin] != tag {
                queued[lin] = tag;
                cand.push(lin);
            }
        }
        let j = step.j;
        let sj = plan.strides[j + 1];
        let d = plan.pshape.len();
        // targets of this block whose slots changed
        for &lin in &changed {
            let cj = plan.coord(lin, j);
            let f = &step.feeds[cj];
            if f.is_empty() {
                continue;
            }
            if !(0..d).all(|i| i == j || step.lattice[i][plan.coord(lin, i)]) {
                continue;
            }
            let row = lin - cj * sj;
            for &k in f {
                let t = row + step.tg[k as usize] * sj;
                if queued[t] != tag {
                    queued[t] = tag;
                    cand.push(t);
                }
            }
        }
        let eps = sched.eps(&base_eps, step.h);
        let tau = cfg.error_bound / ((step.h * step.h) as f64);
        for &t in &cand {
            let cj = plan.coord(t, j);
            let k = step.inv[cj] as usize;
            let tap = &step.taps[k];
            let row = t - cj * sj;
            let v = [
                st.recon[row + tap.slots[0] * sj],
                st.recon[row + tap.slots[1] * sj],
                st.recon[row + tap.slots[2] * sj],
                st.recon[row + tap.slots[3] * sj],
            ];
            let mut p = crate::predict::predict(tap, &v, cfg.predictor, tau);
            if !p.is_finite() {
                p = 0.0;
            }
            recode(st, t, p, &eps, &mut changed);
        }
        in_pass(
            si + 1,
            &cand,
            &queued,
            &eps,
            st,
            &mut changed,
            &mut dirty,
            &mut scratch,
        );
        if changed.len() > limit {
            return None;
        }
    }
    Some(changed)
}
