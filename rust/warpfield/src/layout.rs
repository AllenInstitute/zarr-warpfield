//! Canonical layout: (B, *P) with B = independent × component flattened, then the
//! predicted axes in their original order.

use crate::config::{FACE, Role};
use crate::error::{Result, spec};

#[derive(Clone, Debug)]
pub struct Layout {
    pub perm: Vec<usize>,
    pub shape: Vec<usize>,
    pub n_indep: usize,
    pub n_comp: usize,
    pub pred_shape: Vec<usize>,
    pub pred_roles: Vec<Role>,
}

pub(crate) fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

impl Layout {
    pub fn new(roles: &[Role], shape: &[usize]) -> Result<Layout> {
        if roles.len() != shape.len() {
            return spec(format!(
                "chunk rank {} does not match {} axis_roles",
                shape.len(),
                roles.len()
            ));
        }
        if shape.contains(&0) {
            return spec("chunk extents must be at least 1");
        }
        let pick = |f: &dyn Fn(Role) -> bool| -> Vec<usize> {
            (0..roles.len()).filter(|&i| f(roles[i])).collect()
        };
        let ind = pick(&|r| r == Role::Independent);
        let comp = pick(&|r| r == Role::Component);
        let pred = pick(&|r| r.predicted());
        let mut perm = ind.clone();
        perm.extend(&comp);
        perm.extend(&pred);
        Ok(Layout {
            perm,
            shape: shape.to_vec(),
            n_indep: ind.iter().map(|&i| shape[i]).product(),
            n_comp: comp.first().map(|&i| shape[i]).unwrap_or(1),
            pred_shape: pred.iter().map(|&i| shape[i]).collect(),
            pred_roles: pred.iter().map(|&i| roles[i]).collect(),
        })
    }

    pub fn canonical_shape(&self) -> Vec<usize> {
        let mut s = vec![self.n_indep * self.n_comp];
        s.extend(&self.pred_shape);
        s
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn spatial_in_pred(&self) -> Vec<usize> {
        (0..self.pred_roles.len())
            .filter(|&i| self.pred_roles[i] == Role::Spatial)
            .collect()
    }

    /// For each canonical linear index (in order), the linear index of the same
    /// sample in the original C-order array, visited by `f(canonical, original)`.
    fn walk(&self, mut f: impl FnMut(usize, usize)) {
        let src_strides = c_strides(&self.shape);
        let t_shape: Vec<usize> = self.perm.iter().map(|&a| self.shape[a]).collect();
        let t_strides: Vec<usize> = self.perm.iter().map(|&a| src_strides[a]).collect();
        let n = self.size();
        if n == 0 {
            return;
        }
        let r = t_shape.len();
        let inner = t_shape[r - 1];
        let istride = t_strides[r - 1];
        let mut idx = vec![0usize; r];
        let mut off = 0usize; // source offset of idx with idx[r-1] = 0
        let mut dst = 0usize;
        loop {
            let mut o = off;
            for _ in 0..inner {
                f(dst, o);
                dst += 1;
                o += istride;
            }
            // advance the outer odometer (axes 0..r-1)
            let mut a = r - 1;
            loop {
                if a == 0 {
                    return;
                }
                a -= 1;
                idx[a] += 1;
                off += t_strides[a];
                if idx[a] < t_shape[a] {
                    break;
                }
                off -= t_strides[a] * t_shape[a];
                idx[a] = 0;
            }
        }
    }

    pub fn to_canonical<T: Copy, U>(&self, src: &[T], conv: impl Fn(T) -> U) -> Vec<U> {
        let mut out = Vec::with_capacity(self.size());
        self.walk(|_, o| out.push(conv(src[o])));
        out
    }

    pub fn from_canonical<T: Copy, U>(&self, src: &[T], dst: &mut [U], conv: impl Fn(T) -> U) {
        self.walk(|c, o| dst[o] = conv(src[c]));
    }

    /// Class map with chunk-face samples (index 0 or N-1 on any predicted axis
    /// of extent >= 2) set to FACE.
    pub fn face_classes(&self) -> Vec<u8> {
        face_classes(&self.canonical_shape())
    }
}

/// FACE on the first and last index of every predicted axis (extent >= 2) of a
/// canonical array (batch axis first), INTERIOR elsewhere.
pub(crate) fn face_classes(cs: &[usize]) -> Vec<u8> {
    let n: usize = cs.iter().product();
    let mut cls = vec![0u8; n];
    let strides = c_strides(cs);
    for ax in 1..cs.len() {
        let nj = cs[ax];
        if nj < 2 {
            continue;
        }
        let st = strides[ax];
        let outer: usize = cs[..ax].iter().product();
        let block = st * nj;
        for o in 0..outer {
            for end in [0, nj - 1] {
                let base = o * block + end * st;
                cls[base..base + st].fill(FACE);
            }
        }
    }
    cls
}
