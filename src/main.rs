use clap::Parser;

use vmc_atom::constants::HARTREE_TO_EV;
use vmc_atom::hamiltonian::CoulombHamiltonian;
use vmc_atom::vmc::{run_vmc, VmcConfig};
use vmc_atom::wavefunction::{
    HydrogenicAtom, Orbital, PadeJastrowAtom, SlaterJastrowAtom, Wavefunction,
};

#[derive(Parser, Debug)]
#[command(name = "vmc-atom")]
#[command(about = "Variational Quantum Monte Carlo (VMC) for small atoms", long_about = None)]
struct Cli {
    /// Nuclear charge Z (Helium=2)
    #[arg(long, default_value_t = 2.0)]
    z: f64,

    /// Number of electrons (Helium=2, Lithium=3)
    #[arg(long, default_value_t = 2)]
    electrons: usize,

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

    /// Trial wavefunction: "hydrogenic", "pade", or "slater"
    #[arg(long, default_value = "pade")]
    wf: String,

    /// Number of spin-up electrons for `--wf slater` (default: closed-shell-ish)
    #[arg(long)]
    n_up: Option<usize>,

    /// Comma-separated orbital list for the spin-up Slater determinant (e.g. "1s,2s" or "1s,2s,2px")
    #[arg(long)]
    up_orbitals: Option<String>,

    /// Comma-separated orbital list for the spin-down Slater determinant (e.g. "1s")
    #[arg(long)]
    down_orbitals: Option<String>,

    /// Variational parameter alpha (orbital exponent)
    #[arg(long, default_value_t = 1.6875)]
    alpha: f64,

    /// Variational parameter beta (Padé-Jastrow only)
    #[arg(long, default_value_t = 0.3)]
    beta: f64,

    /// Write a CSV trace of sampled energies (for plotting)
    #[arg(long)]
    trace_csv: Option<std::path::PathBuf>,

    /// Emit one trace row every N samples
    #[arg(long, default_value_t = 1)]
    trace_every: usize,
}

fn parse_orbital_list(s: &str) -> Result<Vec<Orbital>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(part.parse::<Orbital>()?);
    }
    if out.is_empty() {
        Err("Orbital list was empty".to_string())
    } else {
        Ok(out)
    }
}

fn main() {
    let cli = Cli::parse();

    let h = CoulombHamiltonian {
        z: cli.z,
        n_electrons: cli.electrons,
    };

    let config = VmcConfig {
        steps: cli.steps,
        burn_in: cli.burn_in,
        step_size: cli.step_size,
        seed: cli.seed,
        sample_every: 1,
        trace_csv: cli.trace_csv.clone(),
        trace_every: cli.trace_every,
    };

    let default_n_up = match cli.electrons {
        1 => 1,
        2 => 1,
        3 => 2,
        n => (n + 1) / 2,
    };

    let wavefunction: Box<dyn Wavefunction> = match cli.wf.as_str() {
        "hydrogenic" | "hyd" => Box::new(HydrogenicAtom {
            n_electrons: cli.electrons,
            alpha: cli.alpha,
        }),
        "pade" | "jastrow" => Box::new(PadeJastrowAtom {
            n_electrons: cli.electrons,
            alpha: cli.alpha,
            beta: cli.beta,
        }),
        "slater" | "slater-jastrow" => {
            let n_up = cli.n_up.unwrap_or(default_n_up);
            if n_up > cli.electrons {
                eprintln!("Invalid --n-up={n_up}: must be <= --electrons");
                std::process::exit(2);
            }

            let up_orbitals = match cli.up_orbitals.as_deref() {
                Some(s) => match parse_orbital_list(s) {
                    Ok(list) => Some(list),
                    Err(e) => {
                        eprintln!("Invalid --up-orbitals: {e}");
                        std::process::exit(2);
                    }
                },
                None => None,
            };

            let down_orbitals = match cli.down_orbitals.as_deref() {
                Some(s) => match parse_orbital_list(s) {
                    Ok(list) => Some(list),
                    Err(e) => {
                        eprintln!("Invalid --down-orbitals: {e}");
                        std::process::exit(2);
                    }
                },
                None => None,
            };

            Box::new(SlaterJastrowAtom {
                n_electrons: cli.electrons,
                n_up,
                alpha: cli.alpha,
                beta: cli.beta,
                up_orbitals,
                down_orbitals,
            })
        }
        other => {
            eprintln!("Unknown --wf={other}. Use 'hydrogenic', 'pade', or 'slater'.");
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
