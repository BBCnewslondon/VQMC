use crate::constants::EPS;
use crate::vec3::Vec3;

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

/// Ψ = exp(-α (r1 + r2)) for Helium-like 2-electron atoms.
#[derive(Clone, Copy, Debug)]
pub struct HydrogenicHe {
    pub alpha: f64,
}

impl Wavefunction for HydrogenicHe {
    fn n_electrons(&self) -> usize {
        2
    }

    fn log_psi(&self, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), 2);
        let r1 = positions[0].norm();
        let r2 = positions[1].norm();
        -self.alpha * (r1 + r2)
    }

    fn grad_log_psi(&self, i: usize, positions: &[Vec3]) -> Vec3 {
        debug_assert_eq!(positions.len(), 2);
        let ri_hat = safe_unit(positions[i]);
        ri_hat * (-self.alpha)
    }

    fn laplacian_log_psi(&self, i: usize, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), 2);
        let r = positions[i].norm().max(EPS);
        -2.0 * self.alpha / r
    }
}

/// Ψ = exp(-α(r1+r2)) * exp(u(r12)), with u(s)= s / (2(1+β s))
#[derive(Clone, Copy, Debug)]
pub struct PadeJastrowHe {
    pub alpha: f64,
    pub beta: f64,
}

impl PadeJastrowHe {
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

    fn r12_hat(positions: &[Vec3]) -> (Vec3, f64) {
        let r12 = positions[0] - positions[1];
        let s = r12.norm().max(EPS);
        (r12 / s, s)
    }
}

impl Wavefunction for PadeJastrowHe {
    fn n_electrons(&self) -> usize {
        2
    }

    fn log_psi(&self, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), 2);
        let r1 = positions[0].norm();
        let r2 = positions[1].norm();
        let r12 = (positions[0] - positions[1]).norm();
        let u = 0.5 * r12 / (1.0 + self.beta * r12);
        -self.alpha * (r1 + r2) + u
    }

    fn grad_log_psi(&self, i: usize, positions: &[Vec3]) -> Vec3 {
        debug_assert_eq!(positions.len(), 2);

        let ri_hat = safe_unit(positions[i]);
        let (r12_hat, s) = Self::r12_hat(positions);
        let up = self.u_prime(s);

        match i {
            0 => ri_hat * (-self.alpha) + r12_hat * up,
            1 => ri_hat * (-self.alpha) - r12_hat * up,
            _ => panic!("electron index out of bounds"),
        }
    }

    fn laplacian_log_psi(&self, i: usize, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), 2);

        let r = positions[i].norm().max(EPS);
        let (_r12_hat, s) = Self::r12_hat(positions);

        let lap_orb = -2.0 * self.alpha / r;
        let up = self.u_prime(s);
        let upp = self.u_second(s);
        // For a scalar f(s) with s = |r1-r2|: ∇_r1^2 f = f''(s) + 2 f'(s)/s
        let lap_u = upp + 2.0 * up / s;

        lap_orb + lap_u
    }
}
