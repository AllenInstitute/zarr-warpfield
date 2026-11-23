//! # warpfield
//!
//! Error-bounded, seam-free lossy compression of registration displacement
//! fields: the reference implementation of the zarr v3 `warpfield` codec, in
//! Rust. Output is bit-identical to the Python reference (`zarr_warpfield`):
//! same quantization codes, same class lists, same decoded values.
//!
//! ```
//! use warpfield::{Config, Role, encode, decode, EncodeOptions};
//! let shape = [33, 40, 2];
//! let data: Vec<f32> = (0..33 * 40 * 2).map(|i| (i as f32 * 0.01).sin() * 5.0).collect();
//! let cfg = Config::new(vec![Role::Spatial, Role::Spatial, Role::Component], 0.25);
//! let enc = encode(&data, &shape, &cfg, &EncodeOptions::default()).unwrap();
//! let out: Vec<f32> = decode(&enc.bytes, &shape, &cfg).unwrap();
//! assert!(data.iter().zip(&out).all(|(a, b)| (a - b).abs() <= 0.25));
//! ```
//!
//! Guarantees, per sample: |x - x̂| ≤ `error_bound` inside a chunk, ≤
//! `face_error_bound` (default `error_bound / 4`) on chunk faces, so the error
//! jump across a chunk boundary is ≤ 2·face bound; with `jacobian` set, no cell
//! inside a chunk newly falls below `min_det` or changes the sign of det J.

mod config;
mod element;
mod error;
mod format;
mod frames;
mod huffman;
mod incr;
mod jacobian;
mod layout;
mod pass;
mod predict;
mod tune;

pub use config::{Boundary, Config, JacobianConfig, Predictor, Role};
pub use config::{FACE, INTERIOR, MAX_TIERS, N_CLASSES, TIGHT, VERBATIM, is_tight, tight_class};
pub use element::Element;
pub use error::{Error, Result};
pub use layout::Layout;
pub use pass::Schedule;

use std::marker::PhantomData;

/// Rounds of Jacobian tightening before flagged samples go straight to verbatim.
pub const MAX_TIGHT_ROUNDS: usize = 3;

/// Encoder effort: how hard to look for a small encoding. Decoding is the
/// same either way, and so are all error and Jacobian guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Effort {
    /// Tune the level schedule on the central 33^d of the chunk; choose the
    /// code coder at zstd level 3 and compress at level 12. About 2-7x
    /// faster than `Max`, for stores 1-17% larger (most on folded fields).
    #[default]
    Fast,
    /// Tune on the central 65^d (the whole chunk up to 64^d); compress both
    /// coders at zstd level 19 and keep the smaller.
    Max,
}

impl Effort {
    pub fn parse(s: &str) -> Result<Effort> {
        match s {
            "fast" => Ok(Effort::Fast),
            "max" => Ok(Effort::Max),
            _ => Err(Error::Spec(format!(
                "effort must be \"fast\" or \"max\", not {s:?}"
            ))),
        }
    }
    /// zstd level of the sections.
    pub fn zstd_level(self) -> i32 {
        match self {
            Effort::Fast => 12,
            Effort::Max => 19,
        }
    }
    /// Level at which both code coders are tried (None: at `zstd_level`).
    pub fn pick_level(self) -> Option<i32> {
        match self {
            Effort::Fast => Some(3),
            Effort::Max => None,
        }
    }
    /// Extent of the central tuning sub-block per predicted axis.
    pub fn tune_extent(self) -> usize {
        match self {
            Effort::Fast => 33,
            Effort::Max => tune::SAMPLE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// zstd compression level for the sections (does not affect decoded values).
    pub zstd_level: i32,
    /// Also return the per-sample error class map.
    pub classes: bool,
    /// Level schedule; None picks one per chunk (see `tune`).
    pub schedule: Option<Schedule>,
    /// Per-sample minimum class (C order of the input chunk): a tight tier
    /// class (`tight_class`) or VERBATIM forces those samples to that class
    /// from the start where it tightens them; 0 and 1 leave them to the encoder.
    /// Used by writers that check the Jacobian of cells straddling chunk faces
    /// (one-sample halo).
    pub min_classes: Option<Vec<u8>>,
    /// How the code section's coder (varints or Huffman) is chosen. None:
    /// both are compressed at `zstd_level` and the smaller kept. Some(p):
    /// both at level p, then only the smaller one again at `zstd_level`.
    pub pick_level: Option<i32>,
    /// Extent per predicted axis of the central sub-block the schedule is
    /// tuned on.
    pub tune_extent: usize,
}

impl EncodeOptions {
    /// The settings of an effort level.
    pub fn with_effort(effort: Effort) -> Self {
        EncodeOptions {
            zstd_level: effort.zstd_level(),
            classes: false,
            schedule: None,
            min_classes: None,
            pick_level: effort.pick_level(),
            tune_extent: effort.tune_extent(),
        }
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions::with_effort(Effort::default())
    }
}

#[derive(Clone, Debug)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// Per-sample class (0 interior, 1 face, 2 tight, 3 verbatim, 3 + t tight
    /// tier t > 0), C order of the input chunk; only if requested.
    pub classes: Option<Vec<u8>>,
    pub n_codes: usize,
    pub n_tight: usize,
    pub n_verbatim: usize,
    pub jacobian_rounds: usize,
}

fn check(cfg: &Config, shape: &[usize], len: usize) -> Result<Layout> {
    cfg.validate()?;
    let lay = Layout::new(&cfg.axis_roles, shape)?;
    if lay.size() != len {
        return Err(Error::Spec(format!(
            "buffer holds {len} values but shape {shape:?} needs {}",
            lay.size()
        )));
    }
    if let Some(j) = &cfg.jacobian {
        let ns = lay.spatial_in_pred().len();
        if lay.n_comp != ns {
            return Err(Error::Spec(format!(
                "jacobian needs one component per spatial axis ({ns}), but the component \
                 axis has extent {}",
                lay.n_comp
            )));
        }
        debug_assert_eq!(j.grid_spacing.len(), ns);
    }
    Ok(lay)
}

/// The closed loop on one canonical array: code it with `sched`, then, with a
/// Jacobian constraint, tighten or store verbatim the samples of violating
/// cells until none are left. Returns (classes, codes, rounds).
pub(crate) fn code_canonical<T: Element>(
    cfg: &Config,
    lay: &Layout,
    x: &[f64],
    sched: Schedule,
    min_classes: Option<&[u8]>,
) -> Result<(Vec<u8>, Vec<i64>, usize)> {
    let cshape = lay.canonical_shape();
    let n = x.len();
    let base = layout::face_classes(&cshape);
    let eps = cfg.eps_table();
    let n_tiers = cfg.tight_bounds().len();
    let geo = cfg
        .jacobian
        .as_ref()
        .map(|j| jacobian::Geometry::new(lay, &j.grid_spacing));
    let d0 = geo.as_ref().map(|g| g.det(x));
    // forced class per sample, 0 = none (the face class applies)
    let mut force = vec![0u8; n];
    if let Some(m) = min_classes {
        for i in 0..n {
            let c = m[i];
            if c >= TIGHT && (c == VERBATIM || (is_tight(c) && (tier_of(c) < n_tiers))) {
                // only where it tightens (a face bound may already be below a tier)
                if eps[c as usize] < eps[base[i] as usize] {
                    force[i] = c;
                }
            } else if c > FACE {
                return Err(Error::Spec(
                    "min_classes names a tight tier the configuration lacks".into(),
                ));
            }
        }
    }
    let max_rounds = MAX_TIGHT_ROUNDS + n_tiers - 1;
    let mut cur: Vec<u8> = (0..n)
        .map(|i| if force[i] != 0 { force[i] } else { base[i] })
        .collect();

    // the pass's block structure and, with a Jacobian constraint, the check
    // run after each block
    let tables = geo
        .as_ref()
        .map(|g| incr::tables(cfg, &cshape, g, &lay.spatial_in_pred()));
    let ladder: Vec<u8> = (0..n_tiers).map(tight_class).collect();
    let check = match (&geo, &d0, &cfg.jacobian, &tables) {
        (Some(g), Some(d0), Some(jc), Some(t)) => {
            Some(incr::Check::new(g, d0, jc.min_det, t, &eps, &ladder))
        }
        _ => None,
    };
    let n_blocks = tables.as_ref().map_or(0, |t| t.plan.n_blocks());

    // round 0: the full pass
    let mut st = incr::State {
        x,
        recon: vec![0.0f64; n],
        cls: cur.clone(),
        qs: vec![0i64; n],
        pred: vec![0.0f64; n],
        promoted: vec![Vec::new(); n_blocks],
        stuck: vec![Vec::new(); n_blocks],
    };
    let order = {
        let mut enc = pass::Encoder::<T> {
            x,
            cls: &mut st.cls,
            eps,
            qs: &mut st.qs,
            pred: &mut st.pred,
            order: Vec::with_capacity(n),
            check: check.as_ref(),
            promoted: &mut st.promoted,
            stuck: &mut st.stuck,
            _t: PhantomData,
        };
        pass::run(cfg, sched, &cshape, &mut st.recon, &mut enc);
        enc.order
    };
    let codes_of = |st: &incr::State<'_>| -> Vec<i64> {
        order
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| st.cls[i] != VERBATIM)
            .map(|i| st.qs[i])
            .collect()
    };
    let (Some(g), Some(_)) = (&geo, &check) else {
        let codes = codes_of(&st);
        return Ok((st.cls, codes, 0));
    };
    // every cell was checked as its block completed it, and nothing changes a
    // cell's inputs after that: the broken cells are the blocks' stuck lists
    let mut bad: Vec<usize> = st.broken();

    // Later rounds re-code only the tightened samples and what depends on them
    // (`incr`), and re-check only the cells those touch; when that grows past
    // a fraction of the chunk, a full pass and full check are cheaper. Either
    // way the result is that of a full re-run (what the Python reference does).
    let plan = &tables
        .as_ref()
        .expect("tables exist with a Jacobian constraint")
        .plan;
    let mut seen = vec![false; n];
    let limit = n / 16; // measured: n/64..n/4 all within 20% on MICrONS
    let mut rounds = 0usize;
    while !bad.is_empty() {
        rounds += 1;
        let hit = g.cells_to_sample_list(&bad, &mut seen);
        let mut seeds = Vec::new();
        for i in hit {
            if force[i] == VERBATIM {
                continue;
            }
            // the loosest tier tighter than the sample's class, else verbatim
            let e = eps[cur[i] as usize];
            force[i] = if rounds <= max_rounds {
                (0..n_tiers)
                    .map(tight_class)
                    .find(|&c| eps[c as usize] < e)
                    .unwrap_or(VERBATIM)
            } else {
                VERBATIM
            };
            if force[i] != cur[i] {
                cur[i] = force[i];
                seeds.push(i);
            }
        }
        if seeds.is_empty() {
            // verbatim cells reproduce the original determinant exactly
            return Err(Error::Spec("jacobian promotion made no progress".into()));
        }
        let changed = if seeds.len() > limit {
            None
        } else {
            incr::rerun::<T>(
                cfg,
                sched,
                plan,
                &mut st,
                &cur,
                &seeds,
                limit,
                check.as_ref(),
            )
        };
        if changed.is_none() {
            // a full pass from this round's classes
            st.cls.copy_from_slice(&cur);
            st.recon.fill(0.0);
            for p in st.promoted.iter_mut().chain(st.stuck.iter_mut()) {
                p.clear();
            }
            let mut enc = pass::Encoder::<T> {
                x,
                cls: &mut st.cls,
                eps,
                qs: &mut st.qs,
                pred: &mut st.pred,
                order: Vec::new(),
                check: check.as_ref(),
                promoted: &mut st.promoted,
                stuck: &mut st.stuck,
                _t: PhantomData,
            };
            pass::run(cfg, sched, &cshape, &mut st.recon, &mut enc);
        }
        bad = st.broken();
    }
    let codes = codes_of(&st);
    Ok((st.cls, codes, rounds))
}

/// Encode one chunk (C order, `shape` matching `cfg.axis_roles`).
pub fn encode<T: Element>(
    data: &[T],
    shape: &[usize],
    cfg: &Config,
    opts: &EncodeOptions,
) -> Result<Encoded> {
    let lay = check(cfg, shape, data.len())?;
    let cshape = lay.canonical_shape();
    let n = lay.size();
    let x: Vec<f64> = lay.to_canonical(data, T::to_f64);
    let sched = match opts.schedule {
        Some(s) => {
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // rejects NaN too
            if !(s.alpha >= 1.0 && s.beta >= 1.0 && s.alpha.is_finite() && s.beta.is_finite()) {
                return Err(Error::Spec(
                    "schedule alpha and beta must be finite and >= 1".into(),
                ));
            }
            s
        }
        None => tune::tune::<T>(cfg, &lay, &x, opts.tune_extent)?,
    };
    let min_c: Option<Vec<u8>> = match &opts.min_classes {
        Some(m) if m.len() != n => {
            return Err(Error::Spec(format!(
                "min_classes has {} entries, the chunk has {n}",
                m.len()
            )));
        }
        Some(m) => Some(lay.to_canonical(m, |v| v)),
        None => None,
    };
    let (cls, codes, rounds) = code_canonical::<T>(cfg, &lay, &x, sched, min_c.as_deref())?;
    let verbatim: Vec<usize> = (0..n).filter(|&i| cls[i] == VERBATIM).collect();
    let tight: Vec<usize> = (0..n).filter(|&i| is_tight(cls[i])).collect();
    let tiers: Vec<u8> = tight.iter().map(|&i| tier_of(cls[i]) as u8).collect();
    let mut values = Vec::with_capacity(verbatim.len() * T::SIZE);
    for &i in &verbatim {
        T::from_f64(x[i]).write_le(&mut values);
    }
    let bytes = format::write(
        &cshape,
        sched,
        &codes,
        &tight,
        &tiers,
        &verbatim,
        &values,
        opts.zstd_level,
        opts.pick_level,
    )?;
    let classes = opts.classes.then(|| {
        let mut c = vec![0u8; n];
        lay.from_canonical(&cls, &mut c, |v| v);
        c
    });
    Ok(Encoded {
        bytes,
        classes,
        n_codes: codes.len(),
        n_tight: tight.len(),
        n_verbatim: verbatim.len(),
        jacobian_rounds: rounds,
    })
}

/// Decode one chunk into `out` (C order, `shape` matching `cfg.axis_roles`).
pub fn decode_into<T: Element>(
    buf: &[u8],
    shape: &[usize],
    cfg: &Config,
    out: &mut [T],
) -> Result<()> {
    let lay = check(cfg, shape, out.len())?;
    let cshape = lay.canonical_shape();
    let chunk = format::read(buf, T::SIZE)?;
    if chunk.shape != cshape {
        return Err(Error::Format(format!(
            "chunk was encoded with canonical shape {:?}, expected {:?}",
            chunk.shape, cshape
        )));
    }
    let mut cls = lay.face_classes();
    let n_tiers = cfg.tight_bounds().len();
    for (&i, &t) in chunk.tight.iter().zip(&chunk.tiers) {
        if t as usize >= n_tiers {
            return Err(Error::Format(
                "chunk uses a tight tier the configuration lacks".into(),
            ));
        }
        cls[i] = tight_class(t as usize);
    }
    let mut recon = vec![0.0f64; lay.size()];
    for (k, &i) in chunk.verbatim.iter().enumerate() {
        cls[i] = VERBATIM;
        recon[i] = T::read_le(&chunk.values[k * T::SIZE..(k + 1) * T::SIZE]).to_f64();
    }
    let mut dec = pass::Decoder::<T> {
        codes: &chunk.codes,
        cursor: 0,
        cls: &cls,
        eps: cfg.eps_table(),
        short: false,
        _t: PhantomData,
    };
    pass::run(cfg, chunk.schedule, &cshape, &mut recon, &mut dec);
    pass::check_decoded(&dec)?;
    lay.from_canonical(&recon, out, T::from_f64);
    Ok(())
}

/// Decode one chunk.
pub fn decode<T: Element + Default>(buf: &[u8], shape: &[usize], cfg: &Config) -> Result<Vec<T>> {
    let mut out = vec![T::default(); shape.iter().product()];
    decode_into(buf, shape, cfg, &mut out)?;
    Ok(out)
}

fn tier_of(c: u8) -> usize {
    if c == TIGHT {
        0
    } else {
        (c - VERBATIM) as usize
    }
}

/// Per-class absolute error bound of a sample class (see `Encoded::classes`).
pub fn class_error_bound(cfg: &Config, class: u8) -> f64 {
    cfg.eps_table()[class as usize]
}
