use clap::Parser;

use vmc_atom::constants::HARTREE_TO_EV;
use vmc_atom::hamiltonian::CoulombHamiltonian;
use vmc_atom::vmc::{run_vmc, VmcConfig};
use vmc_atom::wavefunction::{HydrogenicHe, PadeJastrowHe, Wavefunction};

#[derive(Parser, Debug)]
#[command(name = "vmc-atom")]
#[command(about = "Variational Quantum Monte Carlo (VMC) for small atoms", long_about = None)]
struct Cli {
    /// Nuclear charge Z (Helium=2)
    #[arg(long, default_value_t = 2.0)]
    z: f64,

    /// Number of Monte Carlo steps (post burn-in) used for averaging
    #[arg(long, default_value_t = 200_000)]
    steps: usize,

    /// Burn-in steps (discarded)
    #[arg(long, default_value_t = 20_000)]
    burn_in: usize,

    /// Proposal step size (Gaussian sigma, in bohr)
    #[arg(long, default_value_t = 0.7)]
    step_size: f64,

    /// Random seed (optional)
    #[arg(long)]
    seed: Option<u64>,

    /// Trial wavefunction: "hydrogenic" or "pade"
    #[arg(long, default_value = "pade")]
    wf: String,

    /// Variational parameter alpha (orbital exponent)
    #[arg(long, default_value_t = 1.6875)]
    alpha: f64,

    /// Variational parameter beta (Padé-Jastrow only)
    #[arg(long, default_value_t = 0.3)]
    beta: f64,
}

fn main() {
    let cli = Cli::parse();

    let h = CoulombHamiltonian {
        z: cli.z,
        n_electrons: 2,
    };

    let config = VmcConfig {
        steps: cli.steps,
        burn_in: cli.burn_in,
        step_size: cli.step_size,
        seed: cli.seed,
        sample_every: 1,
    };

    let wavefunction: Box<dyn Wavefunction> = match cli.wf.as_str() {
        "hydrogenic" | "hyd" => Box::new(HydrogenicHe { alpha: cli.alpha }),
        "pade" | "jastrow" => Box::new(PadeJastrowHe {
            alpha: cli.alpha,
            beta: cli.beta,
        }),
        other => {
            eprintln!("Unknown --wf={other}. Use 'hydrogenic' or 'pade'.");
            std::process::exit(2);
        }
    };

    let result = run_vmc(&h, wavefunction.as_ref(), &config);

    println!("Accepted: {:.2}%", 100.0 * result.acceptance_rate);
    println!(
        "E = {:.6} ± {:.6} Ha",
        result.mean_energy_ha, result.stderr_energy_ha
    );
    println!(
        "E = {:.3} ± {:.3} eV",
        result.mean_energy_ha * HARTREE_TO_EV,
        result.stderr_energy_ha * HARTREE_TO_EV
    );
}
