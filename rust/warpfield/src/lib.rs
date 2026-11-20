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
mod jacobian;
mod layout;
mod pass;
mod predict;

pub use config::{Boundary, Config, JacobianConfig, Predictor, Role};
pub use config::{FACE, INTERIOR, TIGHT, VERBATIM};
pub use element::Element;
pub use error::{Error, Result};
pub use layout::Layout;

use std::marker::PhantomData;

/// Rounds of Jacobian tightening before flagged samples go straight to verbatim.
pub const MAX_TIGHT_ROUNDS: usize = 3;
/// zstd level used by the Python reference.
pub const DEFAULT_ZSTD_LEVEL: i32 = 19;

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// zstd compression level for the four sections (does not affect decoded values).
    pub zstd_level: i32,
    /// Also return the per-sample error class map.
    pub classes: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            zstd_level: DEFAULT_ZSTD_LEVEL,
            classes: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// Per-sample class (0 interior, 1 face, 2 tight, 3 verbatim), C order of the
    /// input chunk; only if requested.
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
    let base = lay.face_classes();
    let eps = cfg.eps_table();
    let eps_tight = cfg.eps_tight();
    let geo = cfg
        .jacobian
        .as_ref()
        .map(|j| jacobian::Geometry::new(&lay, &j.grid_spacing));
    let d0 = geo.as_ref().map(|g| g.det(&x));
    let mut forced_tight = vec![false; n];
    let mut forced_verb = vec![false; n];
    let mut recon = vec![0.0f64; n];
    let mut rounds = 0usize;
    let (cls, codes) = loop {
        let mut cls = base.clone();
        for i in 0..n {
            if forced_verb[i] {
                cls[i] = VERBATIM;
            } else if forced_tight[i] {
                cls[i] = TIGHT;
            }
        }
        recon.fill(0.0);
        let mut enc = pass::Encoder::<T> {
            x: &x,
            cls: &mut cls,
            eps,
            codes: Vec::new(),
            _t: PhantomData,
        };
        pass::run(cfg, &cshape, &mut recon, &mut enc);
        let codes = enc.codes;
        let (Some(g), Some(d0), Some(jc)) = (&geo, &d0, &cfg.jacobian) else {
            break (cls, codes);
        };
        let bad = jacobian::violations(d0, &g.det(&recon), jc.min_det);
        if !bad.iter().any(|&b| b) {
            break (cls, codes);
        }
        rounds += 1;
        let mut hit = vec![false; n];
        g.cells_to_samples(&bad, &mut hit);
        let mut progress = false;
        for i in 0..n {
            if !hit[i] || forced_verb[i] {
                continue;
            }
            progress = true;
            let can_tighten =
                rounds <= MAX_TIGHT_ROUNDS && !forced_tight[i] && eps[base[i] as usize] > eps_tight;
            if can_tighten {
                forced_tight[i] = true;
            } else {
                forced_verb[i] = true;
            }
        }
        if !progress {
            // verbatim cells reproduce the original determinant exactly
            return Err(Error::Spec("jacobian promotion made no progress".into()));
        }
    };
    let verbatim: Vec<usize> = (0..n).filter(|&i| cls[i] == VERBATIM).collect();
    let tight: Vec<usize> = (0..n).filter(|&i| cls[i] == TIGHT).collect();
    let mut values = Vec::with_capacity(verbatim.len() * T::SIZE);
    for &i in &verbatim {
        T::from_f64(x[i]).write_le(&mut values);
    }
    let bytes = format::write(&cshape, &codes, &tight, &verbatim, &values, opts.zstd_level)?;
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
    for &i in &chunk.tight {
        cls[i] = TIGHT;
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
    pass::run(cfg, &cshape, &mut recon, &mut dec);
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

/// Per-class absolute error bound of a sample class (0..=3).
pub fn class_error_bound(cfg: &Config, class: u8) -> f64 {
    cfg.eps_table()[class as usize]
}
