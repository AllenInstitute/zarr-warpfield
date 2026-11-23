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

    pub fn step(&self) -> Vec<usize> {
        self.spatial.iter().map(|&a| self.pstrides[a]).collect()
    }

    /// det J of the cell with corner 0 at predicted offset `off` of independent row `i`.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    pub fn cell_det(&self, u: &[f64], step: &[usize], i: usize, off: usize) -> f64 {
        let k = self.k;
        let mut jm = [[0.0f64; 3]; 3];
        for a in 0..k {
            let base = (i * k + a) * self.psize + off;
            let u0 = u[base];
            for b in 0..k {
                let d = (u[base + step[b]] - u0) / self.spacing[b];
                jm[a][b] = d + if a == b { 1.0 } else { 0.0 };
            }
        }
        match k {
            1 => jm[0][0],
            2 => jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0],
            _ => {
                jm[0][0] * (jm[1][1] * jm[2][2] - jm[1][2] * jm[2][1])
                    - jm[0][1] * (jm[1][0] * jm[2][2] - jm[1][2] * jm[2][0])
                    + jm[0][2] * (jm[1][0] * jm[2][1] - jm[1][1] * jm[2][0])
            }
        }
    }

    /// Determinant per cell.
    pub fn det(&self, u: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n_cells()];
        let step = self.step();
        self.for_cells(|ci, i, off| out[ci] = self.cell_det(u, &step, i, off));
        out
    }

    /// (independent row, predicted offset of corner 0) of cell `ci`.
    pub fn cell_origin(&self, ci: usize) -> (usize, usize) {
        let ncell: usize = self.cell_shape.iter().product();
        let (i, mut r) = (ci / ncell, ci % ncell);
        let mut off = 0;
        for a in (0..self.cell_shape.len()).rev() {
            off += (r % self.cell_shape[a]) * self.pstrides[a];
            r /= self.cell_shape[a];
        }
        (i, off)
    }

    /// Predicted offsets of the samples cell `ci`'s determinant depends on:
    /// its origin x, then x + e_b per spatial axis b.
    pub fn dep_offsets(&self, ci: usize) -> (usize, [usize; 4]) {
        let (i, off) = self.cell_origin(ci);
        let mut o = [off; 4];
        for (b, &a) in self.spatial.iter().enumerate() {
            o[b + 1] = off + self.pstrides[a];
        }
        (i, o)
    }

    /// Canonical sample `lin` of component `c` at predicted offset `off`, row `i`.
    #[inline]
    pub fn sample(&self, i: usize, c: usize, off: usize) -> usize {
        (i * self.k + c) * self.psize + off
    }

    pub fn n_spatial(&self) -> usize {
        self.k
    }

    /// Push the cells whose determinant depends on canonical sample `lin`
    /// (those with origin at its position, or at its position minus e_b).
    pub fn dep_cells_of_sample(&self, lin: usize, out: &mut Vec<usize>) {
        let i = lin / (self.k * self.psize);
        let p = lin % self.psize;
        let d = self.cell_shape.len();
        let ncell: usize = self.cell_shape.iter().product();
        let mut pc = [0usize; 8];
        let mut r = p;
        for (c, &s) in pc.iter_mut().zip(&self.pstrides) {
            *c = r / s;
            r %= s;
        }
        'cell: for back in 0..=self.k {
            // back 0: origin here; back b + 1: origin one step back along spatial b
            let mut ci = 0usize;
            for (a, &pa) in pc.iter().enumerate().take(d) {
                let mut c = pa;
                if back > 0 && self.spatial[back - 1] == a {
                    if c == 0 {
                        continue 'cell;
                    }
                    c -= 1;
                }
                if c >= self.cell_shape[a] {
                    continue 'cell;
                }
                ci = ci * self.cell_shape[a] + c;
            }
            out.push(i * ncell + ci);
        }
    }

    /// Every component of every corner sample of the listed cells, once each
    /// (`seen` is scratch, all false on entry and on return).
    pub fn cells_to_sample_list(&self, cells: &[usize], seen: &mut [bool]) -> Vec<usize> {
        let mut v = Vec::new();
        let k = self.k;
        let ps = self.psize;
        let step = self.step();
        for &ci in cells {
            let (i, off) = self.cell_origin(ci);
            for corner in 0..(1usize << k) {
                let mut o = off;
                for (b, s) in step.iter().enumerate() {
                    if (corner >> b) & 1 == 1 {
                        o += s;
                    }
                }
                for c in 0..k {
                    let lin = (i * k + c) * ps + o;
                    if !seen[lin] {
                        seen[lin] = true;
                        v.push(lin);
                    }
                }
            }
        }
        for &lin in &v {
            seen[lin] = false;
        }
        v.sort_unstable();
        v
    }
}

/// Whether a cell with original determinant `a` and decoded `b` violates.
#[inline]
pub(crate) fn violation(a: f64, b: f64, min_det: f64) -> bool {
    (a > min_det && b <= min_det) || (a > 0.0 && b <= 0.0) || (a < 0.0 && b >= 0.0)
}
