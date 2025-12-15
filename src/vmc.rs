use crate::hamiltonian::CoulombHamiltonian;
use crate::metropolis::{MetropolisSampler, Walker};
use crate::wavefunction::Wavefunction;

#[derive(Clone, Debug)]
pub struct VmcConfig {
    pub steps: usize,
    pub burn_in: usize,
    pub step_size: f64,
    pub seed: Option<u64>,
    pub sample_every: usize,
}

#[derive(Clone, Debug)]
pub struct VmcResult {
    pub mean_energy_ha: f64,
    pub stderr_energy_ha: f64,
    pub acceptance_rate: f64,
    pub samples: usize,
}

#[derive(Clone, Debug, Default)]
struct RunningStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance_unbiased(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    fn stderr(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.variance_unbiased() / self.n as f64).sqrt()
        }
    }
}

pub fn run_vmc(h: &CoulombHamiltonian, wf: &dyn Wavefunction, cfg: &VmcConfig) -> VmcResult {
    assert_eq!(wf.n_electrons(), h.n_electrons);
    assert!(cfg.sample_every >= 1);

    let mut sampler = MetropolisSampler::new(cfg.step_size, cfg.seed);
    let mut walker: Walker = sampler.init_walker(wf);

    let mut accepted = 0usize;
    let mut attempted = 0usize;

    // Burn-in
    for _ in 0..cfg.burn_in {
        attempted += 1;
        if sampler.step(wf, &mut walker) {
            accepted += 1;
        }
    }

    let mut stats = RunningStats::default();

    // Sampling
    let mut sample_counter = 0usize;
    for step in 0..cfg.steps {
        attempted += 1;
        if sampler.step(wf, &mut walker) {
            accepted += 1;
        }

        if step % cfg.sample_every == 0 {
            let e = h.local_energy(wf, &walker.positions);
            stats.push(e);
            sample_counter += 1;
        }
    }

    VmcResult {
        mean_energy_ha: stats.mean(),
        stderr_energy_ha: stats.stderr(),
        acceptance_rate: if attempted == 0 {
            0.0
        } else {
            accepted as f64 / attempted as f64
        },
        samples: sample_counter,
    }
}
