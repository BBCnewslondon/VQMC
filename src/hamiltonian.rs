use crate::constants::EPS;
use crate::vec3::Vec3;
use crate::wavefunction::Wavefunction;

/// Non-relativistic Coulomb Hamiltonian in atomic units.
///
/// H = -1/2 Σ_i ∇_i^2 - Z Σ_i 1/r_i + Σ_{i<j} 1/r_ij
#[derive(Clone, Copy, Debug)]
pub struct CoulombHamiltonian {
    pub z: f64,
    pub n_electrons: usize,
}

impl CoulombHamiltonian {
    pub fn potential(&self, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);

        let mut v = 0.0;
        for i in 0..self.n_electrons {
            let ri = positions[i].norm().max(EPS);
            v += -self.z / ri;
        }
        for i in 0..self.n_electrons {
            for j in (i + 1)..self.n_electrons {
                let rij = (positions[i] - positions[j]).norm().max(EPS);
                v += 1.0 / rij;
            }
        }
        v
    }

    pub fn local_energy(&self, wf: &dyn Wavefunction, positions: &[Vec3]) -> f64 {
        debug_assert_eq!(positions.len(), self.n_electrons);
        debug_assert_eq!(wf.n_electrons(), self.n_electrons);

        let mut kinetic = 0.0;
        for i in 0..self.n_electrons {
            let grad = wf.grad_log_psi(i, positions);
            let lap = wf.laplacian_log_psi(i, positions);
            kinetic += -0.5 * (lap + grad.dot(grad));
        }

        kinetic + self.potential(positions)
    }
}
