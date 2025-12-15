use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use crate::vec3::Vec3;
use crate::wavefunction::Wavefunction;

#[derive(Clone, Debug)]
pub struct Walker {
    pub positions: Vec<Vec3>,
    pub log_psi: f64,
}

pub struct MetropolisSampler {
    rng: StdRng,
    normal: Normal<f64>,
    pub step_size: f64,
}

impl MetropolisSampler {
    pub fn new(step_size: f64, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            rng,
            normal: Normal::new(0.0, 1.0).expect("normal dist"),
            step_size,
        }
    }

    pub fn init_walker(&mut self, wf: &dyn Wavefunction) -> Walker {
        let mut positions = Vec::with_capacity(wf.n_electrons());
        for _ in 0..wf.n_electrons() {
            // Small Gaussian cloud around origin (bohr)
            let dx = self.normal.sample(&mut self.rng) * 0.5;
            let dy = self.normal.sample(&mut self.rng) * 0.5;
            let dz = self.normal.sample(&mut self.rng) * 0.5;
            positions.push(Vec3::new(dx, dy, dz));
        }
        let log_psi = wf.log_psi(&positions);
        Walker { positions, log_psi }
    }

    /// Propose a single-electron move; return whether it was accepted.
    pub fn step(&mut self, wf: &dyn Wavefunction, walker: &mut Walker) -> bool {
        let n = walker.positions.len();
        let i = self.rng.gen_range(0..n);

        let displacement = Vec3::new(
            self.normal.sample(&mut self.rng) * self.step_size,
            self.normal.sample(&mut self.rng) * self.step_size,
            self.normal.sample(&mut self.rng) * self.step_size,
        );

        let old_pos = walker.positions[i];
        walker.positions[i] = old_pos + displacement;
        let new_log_psi = wf.log_psi(&walker.positions);

        let delta = new_log_psi - walker.log_psi;
        // Acceptance for |Ψ|^2 sampling: min(1, exp(2Δ))
        let accept_prob = (2.0 * delta).exp().min(1.0);
        let u: f64 = self.rng.gen();

        if u < accept_prob {
            walker.log_psi = new_log_psi;
            true
        } else {
            walker.positions[i] = old_pos;
            false
        }
    }
}
