//! det(I + du/dx) on forward-difference cells, on the canonical layout
//! (n_indep, n_comp, *P). Same operation order as `zarr_warpfield.jacobian`.

use crate::layout::{Layout, c_strides};

pub(crate) struct Geometry {
    n_indep: usize,
    k: usize,
    psize: usize,
    pstrides: Vec<usize>,
    spatial: Vec<usize>,
    cell_shape: Vec<usize>,
    spacing: Vec<f64>,
}

impl Geometry {
    pub fn new(lay: &Layout, spacing: &[f64]) -> Geometry {
        let spatial = lay.spatial_in_pred();
        let mut cell_shape = lay.pred_shape.clone();
        for &a in &spatial {
            cell_shape[a] -= 1;
        }
        Geometry {
            n_indep: lay.n_indep,
            k: spatial.len(),
            psize: lay.pred_shape.iter().product(),
            pstrides: c_strides(&lay.pred_shape),
            spatial,
            cell_shape,
            spacing: spacing.to_vec(),
        }
    }

    pub fn n_cells(&self) -> usize {
        self.n_indep * self.cell_shape.iter().product::<usize>()
    }

    /// Visit cells in C order over (indep, *cell_shape): f(cell index, pred offset of corner 0).
    fn for_cells(&self, mut f: impl FnMut(usize, usize, usize)) {
        let ncell: usize = self.cell_shape.iter().product();
        if ncell == 0 {
            return;
        }
        let d = self.cell_shape.len();
        let mut idx = vec![0usize; d];
        let mut ci = 0usize;
        for i in 0..self.n_indep {
            idx.fill(0);
            let mut off = 0usize;
            loop {
                f(ci, i, off);
                ci += 1;
                let mut a = d;
                let mut carry = true;
                while carry && a > 0 {
                    a -= 1;
                    idx[a] += 1;
                    off += self.pstrides[a];
                    if idx[a] < self.cell_shape[a] {
                        carry = false;
                    } else {
                        off -= self.pstrides[a] * self.cell_shape[a];
                        idx[a] = 0;
                    }
                }
                if carry {
                    break;
                }
            }
        }
    }

    /// Determinant per cell.
    #[allow(clippy::needless_range_loop)]
    pub fn det(&self, u: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n_cells()];
        let k = self.k;
        let ps = self.psize;
        let step: Vec<usize> = self.spatial.iter().map(|&a| self.pstrides[a]).collect();
        self.for_cells(|ci, i, off| {
            let mut jm = [[0.0f64; 3]; 3];
            for a in 0..k {
                let base = (i * k + a) * ps + off;
                let u0 = u[base];
                for b in 0..k {
                    let d = (u[base + step[b]] - u0) / self.spacing[b];
                    jm[a][b] = d + if a == b { 1.0 } else { 0.0 };
                }
            }
            out[ci] = match k {
                1 => jm[0][0],
                2 => jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0],
                _ => {
                    jm[0][0] * (jm[1][1] * jm[2][2] - jm[1][2] * jm[2][1])
                        - jm[0][1] * (jm[1][0] * jm[2][2] - jm[1][2] * jm[2][0])
                        + jm[0][2] * (jm[1][0] * jm[2][1] - jm[1][1] * jm[2][0])
                }
            };
        });
        out
    }

    /// Mark every component of every corner sample of each flagged cell, on the
    /// canonical array (n_indep * k, *P).
    pub fn cells_to_samples(&self, bad: &[bool], out: &mut [bool]) {
        let k = self.k;
        let ps = self.psize;
        let step: Vec<usize> = self.spatial.iter().map(|&a| self.pstrides[a]).collect();
        self.for_cells(|ci, i, off| {
            if !bad[ci] {
                return;
            }
            for corner in 0..(1usize << k) {
                let mut o = off;
                for (b, s) in step.iter().enumerate() {
                    if (corner >> b) & 1 == 1 {
                        o += s;
                    }
                }
                for c in 0..k {
                    out[(i * k + c) * ps + o] = true;
                }
            }
        });
    }
}

pub(crate) fn violations(d0: &[f64], d1: &[f64], min_det: f64) -> Vec<bool> {
    d0.iter()
        .zip(d1)
        .map(|(&a, &b)| {
            (a > min_det && b <= min_det) || (a > 0.0 && b <= 0.0) || (a < 0.0 && b >= 0.0)
        })
        .collect()
}
