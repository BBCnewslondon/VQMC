use crate::constants::EPS;
use crate::vec3::Vec3;
use std::str::FromStr;

pub trait Wavefunction: Send + Sync {
    fn n_electrons(&self) -> usize;

    /// Natural log of the (real) trial wavefunction, ln Ψ_T(R).
    fn log_psi(&self, positions: &[Vec3]) -> f64;

    /// Gradient of ln Ψ_T with respect to electron i.
    fn grad_log_psi(&self, i: usize, positions: &[Vec3]) -> Vec3;

    /// Laplacian of ln Ψ_T with respect to electron i.
    fn laplacian_log_psi(&self, i: usize, positions: &[Vec3]) -> f64;
}

fn safe_unit(v: Vec3) -> Vec3 {
    let r = v.norm().max(EPS);
    v / r
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orbital {
    OneS,
    TwoS,
    TwoPX,
    TwoPY,
    TwoPZ,
}

impl FromStr for Orbital {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let key = s.trim().to_ascii_lowercase().replace('_', "");
        match key.as_str() {
            "1s" => Ok(Self::OneS),
            "2s" => Ok(Self::TwoS),
            "2px" | "2p x" => Ok(Self::TwoPX),
            "2py" | "2p y" => Ok(Self::TwoPY),
            "2pz" | "2p z" => Ok(Self::TwoPZ),
            _ => Err(format!(
                "Unknown orbital '{s}'. Use one of: 1s,2s,2px,2py,2pz"
            )),
        }
    }
}

fn orbital_value_grad_lap(orb: Orbital, alpha: f64, rvec: Vec3) -> (f64, Vec3, f64) {
    let r = rvec.norm().max(EPS);
    let r_hat = safe_unit(rvec);

    match orb {
        Orbital::OneS => {
            // φ(r) = exp(-α r)
            let phi = (-alpha * r).exp();
            let dphi_dr = -alpha * phi;
            let grad = r_hat * dphi_dr;
            // ∇^2 φ = (α^2 - 2α/r) φ
            let lap = (alpha * alpha - 2.0 * alpha / r) * phi;
            (phi, grad, lap)
        }
        Orbital::TwoS => {
            // Minimal 2s-like STO: φ(r) = (1 - α r/2) exp(-α r/2)
            let g = (-0.5 * alpha * r).exp();
            let f = 1.0 - 0.5 * alpha * r;
            let phi = f * g;

            // dφ/dr = -α (1 - α r/4) exp(-α r/2)
            let dphi_dr = -alpha * (1.0 - 0.25 * alpha * r) * g;
            let grad = r_hat * dphi_dr;

            // ∇^2 φ = (1/r^2) d/dr (r^2 dφ/dr)
            // = -α exp(-α r/2) * (2/r - 5α/4 + α^2 r/8)
            let lap = -alpha * g * (2.0 / r - 1.25 * alpha + (alpha * alpha) * r / 8.0);
            (phi, grad, lap)
        }
        Orbital::TwoPX | Orbital::TwoPY | Orbital::TwoPZ => {
            // Minimal 2p-like STO: φ(r) = (coord) exp(-α r/2)
            let g = (-0.5 * alpha * r).exp();
            let (coord, axis) = match orb {
                Orbital::TwoPX => (rvec.x, Vec3::new(1.0, 0.0, 0.0)),
                Orbital::TwoPY => (rvec.y, Vec3::new(0.0, 1.0, 0.0)),
                Orbital::TwoPZ => (rvec.z, Vec3::new(0.0, 0.0, 1.0)),
                _ => unreachable!(),
            };

            let phi = coord * g;

            // ∇(coord * g) = axis*g + coord*∇g, with ∇g = g' r_hat and g' = -α/2 g
            let grad = axis * g + r_hat * (-0.5 * alpha * coord * g);

            // ∇^2(coord * g) = coord ∇^2 g + 2 ∇coord · ∇g
            // For g(r)=exp(-α r/2): ∇^2 g = (α^2/4 - α/r) g, and ∇coord·∇g = axis·(g' r_hat)
            // => ∇^2 φ = coord*(α^2/4 - 2α/r) g
            let lap = (alpha * alpha / 4.0 - 2.0 * alpha / r) * phi;

            (phi, grad, lap)
        }
    }
}

fn lu_decompose(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<usize>, i32)> {
    let n = a.len();
    if n == 0 {
        return None;
    }
    for row in a {
        if row.len() != n {
            return None;
        }
    }

    let mut lu = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign = 1i32;

    for k in 0..n {
        let mut p = k;
        let mut max = lu[k][k].abs();
        for i in (k + 1)..n {
            let v = lu[i][k].abs();
            if v > max {
                max = v;
                p = i;
            }
        }
        if max < EPS {
            return None;
        }
        if p != k {
            lu.swap(k, p);
            piv.swap(k, p);
            sign = -sign;
        }

        let pivot = lu[k][k];
        for i in (k + 1)..n {
            lu[i][k] /= pivot;
            let lik = lu[i][k];
            for j in (k + 1)..n {
                lu[i][j] -= lik * lu[k][j];
            }
        }
    }

    Some((lu, piv, sign))
}

fn lu_solve(lu: &[Vec<f64>], piv: &[usize], b: &[f64]) -> Vec<f64> {
    let n = lu.len();
    let mut x = vec![0.0; n];

    // Apply permutation: x = P b
    for i in 0..n {
        x[i] = b[piv[i]];
    }

    // Forward substitution (L has unit diagonal)
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu[i][j] * x[j];
        }
    }

    // Back substitution (U)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }

    x
}

fn inverse_and_logabsdet(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, f64)> {
    let n = a.len();
    let (lu, piv, _sign) = lu_decompose(a)?;
    let mut logabsdet = 0.0;
    for i in 0..n {
        logabsdet += lu[i][i].abs().max(EPS).ln();
    }

    let mut inv = vec![vec![0.0; n]; n];
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        let x = lu_solve(&lu, &piv, &e);
        for row in 0..n {
            inv[row][col] = x[row];
        }
    }

    Some((inv, logabsdet))
}

fn default_orbitals_for_count(n: usize) -> Vec<Orbital> {
    match n {
        0 => vec![],
        1 => vec![Orbital::OneS],
        2 => vec![Orbital::OneS, Orbital::TwoS],
        3 => vec![Orbital::OneS, Orbital::TwoS, Orbital::TwoPX],
        4 => vec![Orbital::OneS, Orbital::TwoS, Orbital::TwoPX, Orbital::TwoPY],
        5 => vec![
            Orbital::OneS,
            Orbital::TwoS,
            Orbital::TwoPX,
            Orbital::TwoPY,
            Orbital::TwoPZ,
        ],
        _ => panic!("Unsupported determinant size {n}; extend orbital basis if needed"),
    }
}

/// Antisymmetric Slater–Jastrow trial function.
///
/// Ψ = |D_↑| |D_↓| exp( Σ_{i<j} u(r_ij) ), with u(s)= s / (2(1+β s))
///
/// - Spin assignment is by electron index: 0..n_up are ↑, n_up..N are ↓.
/// - Determinants use a minimal orbital basis: 1s, 2s, then 2p_x/2p_y/2p_z as needed.
#[derive(Clone, Debug)]
pub struct SlaterJastrowAtom {
    pub n_electrons: usize,
    pub n_up: usize,
    pub alpha: f64,
    pub beta: f64,

    /// Optional explicit orbital list for the spin-up determinant (length must equal n_up).
    pub up_orbitals: Option<Vec<Orbital>>,
    /// Optional explicit orbital list for the spin-down determinant (length must equal n_down).
    pub down_orbitals: Option<Vec<Orbital>>,
}

impl SlaterJastrowAtom {
    fn u(&self, s: f64) -> f64 {
        0.5 * s / (1.0 + self.beta * s)
    }

    fn u_prime(&self, s: f64) -> f64 {
        let denom = 1.0 + self.beta * s;
        0.5 / (denom * denom)
    }

    fn u_second(&self, s: f64) -> f64 {
        let denom = 1.0 + self.beta * s;
        -self.beta / (denom * denom * denom)
    }

    fn sector_indices(&self, up: bool) -> Vec<usize> {
        if up {
            (0..self.n_up).collect()
        } else {
            (self.n_up..self.n_electrons).collect()
        }
    }

    fn sector_orbitals(&self, up: bool, n_spin: usize) -> Vec<Orbital> {
        let custom = if up {
            self.up_orbitals.as_ref()
        } else {
            self.down_orbitals.as_ref()
        };

        if let Some(list) = custom {
            assert_eq!(
                list.len(),
                n_spin,
                "Custom orbital list length must equal spin-sector electron count"
            );
            list.clone()
        } else {
            default_orbitals_for_count(n_spin)
        }
    }

    fn det_logabs_and_inv(
        &self,
        positions: &[Vec3],
        indices: &[usize],
        orbitals: &[Orbital],
    ) -> (f64, Vec<Vec<f64>>) {
        let n = indices.len();
        if n == 0 {
            return (0.0, vec![]);
        }
        let mut mat = vec![vec![0.0; n]; n];
        for (row, &ei) in indices.iter().enumerate() {
            for (col, &orb) in orbitals.iter().enumerate() {
                let (phi, _g, _l) = orbital_value_grad_lap(orb, self.alpha, positions[ei]);
                mat[row][col] = phi;
            }
        }

        let (inv, logabsdet) = inverse_and_logabsdet(&mat)
            .unwrap_or_else(|| (vec![vec![0.0; n]; n], f64::NEG_INFINITY));
        (logabsdet, inv)
    }
}

impl Wavefunction for SlaterJastrowAtom {
    fn n_electrons(&self) -> usize {
        self.n_electrons
    }

    fn log_psi(&self, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        assert!(self.n_up <= self.n_electrons);

        let up_idx = self.sector_indices(true);
        let dn_idx = self.sector_indices(false);
        let up_orb = self.sector_orbitals(true, up_idx.len());
        let dn_orb = self.sector_orbitals(false, dn_idx.len());

        let (logdet_up, _inv_up) = self.det_logabs_and_inv(positions, &up_idx, &up_orb);
        let (logdet_dn, _inv_dn) = self.det_logabs_and_inv(positions, &dn_idx, &dn_orb);

        let mut sum_u = 0.0;
        for i in 0..self.n_electrons {
            for j in (i + 1)..self.n_electrons {
                let rij = (positions[i] - positions[j]).norm().max(EPS);
                sum_u += self.u(rij);
            }
        }

        logdet_up + logdet_dn + sum_u
    }

    fn grad_log_psi(&self, i: usize, positions: &[Vec3]) -> Vec3 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        assert!(i < self.n_electrons);
        assert!(self.n_up <= self.n_electrons);

        // Jastrow gradient
        let mut grad_j = Vec3::new(0.0, 0.0, 0.0);
        for j in 0..self.n_electrons {
            if j == i {
                continue;
            }
            let rij_vec = positions[i] - positions[j];
            let rij = rij_vec.norm().max(EPS);
            let rij_hat = rij_vec / rij;
            grad_j += rij_hat * self.u_prime(rij);
        }

        // Slater sector gradient
        let (indices, orbitals, row_in_sector) = if i < self.n_up {
            let idx = self.sector_indices(true);
            let row = idx.iter().position(|&e| e == i).unwrap();
            (idx, self.sector_orbitals(true, self.n_up), row)
        } else {
            let idx = self.sector_indices(false);
            let row = idx.iter().position(|&e| e == i).unwrap();
            (
                idx,
                self.sector_orbitals(false, self.n_electrons - self.n_up),
                row,
            )
        };

        if indices.is_empty() {
            return grad_j;
        }

        let (_logdet, inv) = self.det_logabs_and_inv(positions, &indices, &orbitals);
        let n = indices.len();

        let mut grad_d = Vec3::new(0.0, 0.0, 0.0);
        for a in 0..n {
            let (_phi, gphi, _lphi) = orbital_value_grad_lap(orbitals[a], self.alpha, positions[i]);
            // inv[row][col] corresponds to (A^{-1})_{row, col}
            // A_{row=electron_in_sector, col=orbital}
            // grad ln det for this electron row: sum_a (A^{-1})_{a,row} ∇ A_{row,a}
            grad_d += gphi * inv[a][row_in_sector];
        }

        grad_d + grad_j
    }

    fn laplacian_log_psi(&self, i: usize, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        assert!(i < self.n_electrons);
        assert!(self.n_up <= self.n_electrons);

        // Jastrow laplacian
        let mut lap_j = 0.0;
        for j in 0..self.n_electrons {
            if j == i {
                continue;
            }
            let rij = (positions[i] - positions[j]).norm().max(EPS);
            let up = self.u_prime(rij);
            let upp = self.u_second(rij);
            lap_j += upp + 2.0 * up / rij;
        }

        // Slater sector laplacian
        let (indices, orbitals, row_in_sector) = if i < self.n_up {
            let idx = self.sector_indices(true);
            let row = idx.iter().position(|&e| e == i).unwrap();
            (idx, self.sector_orbitals(true, self.n_up), row)
        } else {
            let idx = self.sector_indices(false);
            let row = idx.iter().position(|&e| e == i).unwrap();
            (
                idx,
                self.sector_orbitals(false, self.n_electrons - self.n_up),
                row,
            )
        };

        if indices.is_empty() {
            return lap_j;
        }

        let (_logdet, inv) = self.det_logabs_and_inv(positions, &indices, &orbitals);
        let n = indices.len();

        // term1 = Σ_a inv[a,row] ∇^2 φ_a(r_i)
        let mut term1 = 0.0;
        // term2 = Σ_{a,c} inv[a,row] inv[c,row] (∇φ_a·∇φ_c)
        let mut term2 = 0.0;

        let mut grads: Vec<Vec3> = Vec::with_capacity(n);
        for a in 0..n {
            let (_phi, gphi, lphi) = orbital_value_grad_lap(orbitals[a], self.alpha, positions[i]);
            grads.push(gphi);
            term1 += inv[a][row_in_sector] * lphi;
        }

        for a in 0..n {
            for c in 0..n {
                term2 += inv[a][row_in_sector] * inv[c][row_in_sector] * grads[a].dot(grads[c]);
            }
        }

        let lap_d = term1 - term2;
        lap_d + lap_j
    }
}

/// Ψ = exp(-α Σ_i r_i)
#[derive(Clone, Copy, Debug)]
pub struct HydrogenicAtom {
    pub n_electrons: usize,
    pub alpha: f64,
}

impl Wavefunction for HydrogenicAtom {
    fn n_electrons(&self) -> usize {
        self.n_electrons
    }

    fn log_psi(&self, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        let sum_r: f64 = positions.iter().map(|r| r.norm()).sum();
        -self.alpha * sum_r
    }

    fn grad_log_psi(&self, i: usize, positions: &[Vec3]) -> Vec3 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        let ri_hat = safe_unit(positions[i]);
        ri_hat * (-self.alpha)
    }

    fn laplacian_log_psi(&self, i: usize, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        let r = positions[i].norm().max(EPS);
        -2.0 * self.alpha / r
    }
}

/// Ψ = exp(-α Σ_i r_i) * exp( Σ_{i<j} u(r_ij) ), with u(s)= s / (2(1+β s))
#[derive(Clone, Copy, Debug)]
pub struct PadeJastrowAtom {
    pub n_electrons: usize,
    pub alpha: f64,
    pub beta: f64,
}

impl PadeJastrowAtom {
    fn u(&self, s: f64) -> f64 {
        0.5 * s / (1.0 + self.beta * s)
    }

    fn u_prime(&self, s: f64) -> f64 {
        // u'(s) = 1 / (2 (1+β s)^2)
        let denom = 1.0 + self.beta * s;
        0.5 / (denom * denom)
    }

    fn u_second(&self, s: f64) -> f64 {
        // u''(s) = -β / (1+β s)^3
        let denom = 1.0 + self.beta * s;
        -self.beta / (denom * denom * denom)
    }
}

impl Wavefunction for PadeJastrowAtom {
    fn n_electrons(&self) -> usize {
        self.n_electrons
    }

    fn log_psi(&self, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);

        let sum_r: f64 = positions.iter().map(|r| r.norm()).sum();
        let mut sum_u = 0.0;
        for i in 0..self.n_electrons {
            for j in (i + 1)..self.n_electrons {
                let rij = (positions[i] - positions[j]).norm().max(EPS);
                sum_u += self.u(rij);
            }
        }
        -self.alpha * sum_r + sum_u
    }

    fn grad_log_psi(&self, i: usize, positions: &[Vec3]) -> Vec3 {
        debug_assert_eq!(positions.len(), self.n_electrons);

        let mut grad = safe_unit(positions[i]) * (-self.alpha);
        for j in 0..self.n_electrons {
            if j == i {
                continue;
            }
            let rij_vec = positions[i] - positions[j];
            let rij = rij_vec.norm().max(EPS);
            let rij_hat = rij_vec / rij;
            grad += rij_hat * self.u_prime(rij);
        }
        grad
    }

    fn laplacian_log_psi(&self, i: usize, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);

        let r = positions[i].norm().max(EPS);
        let mut lap = -2.0 * self.alpha / r;

        for j in 0..self.n_electrons {
            if j == i {
                continue;
            }
            let rij = (positions[i] - positions[j]).norm().max(EPS);
            let up = self.u_prime(rij);
            let upp = self.u_second(rij);
            // For a scalar f(s) with s = |ri-rj|: ∇_ri^2 f = f''(s) + 2 f'(s)/s
            lap += upp + 2.0 * up / rij;
        }
        lap
    }
}

/// Backwards-compatible helpers for Helium.
pub type HydrogenicHe = HydrogenicAtom;
pub type PadeJastrowHe = PadeJastrowAtom;
