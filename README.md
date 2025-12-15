# vmc-atom

Variational Monte Carlo (VMC) in Rust for small atoms in **atomic units**.

- Samples electron configurations with the Metropolis algorithm using $|\Psi_T|^2$.
- Estimates the ground-state energy via the variational principle.
- Trial wavefunctions are pluggable via a `Wavefunction` trait.

## Physics (atomic units)

For a Helium-like two-electron atom (nuclear charge $Z$), the Hamiltonian is

$$\hat H = -\tfrac{1}{2}(\nabla_1^2 + \nabla_2^2) - Z(\tfrac{1}{r_1}+\tfrac{1}{r_2}) + \tfrac{1}{r_{12}}.$$

The code computes the **local energy**

$$E_L(R) = \frac{\hat H\Psi_T(R)}{\Psi_T(R)} = -\tfrac{1}{2}\sum_i \left(\nabla_i^2\ln\Psi_T + |\nabla_i\ln\Psi_T|^2\right) + V(R).$$

Energy is printed in Hartree and eV using $1\ \mathrm{Ha} = 27.211386245988\ \mathrm{eV}$.

## Trial wavefunctions

- `hydrogenic`: $\Psi_T = \exp(-\alpha(r_1+r_2))$
- `pade`: $\Psi_T = \exp(-\alpha(r_1+r_2))\,\exp(u(r_{12}))$, with $u(s)=\tfrac{s}{2(1+\beta s)}$

## Run

```bash
cargo run --release -- --wf pade --alpha 1.6875 --beta 0.30 --steps 200000 --burn-in 20000 --step-size 0.7
```

Try the simpler trial function:

```bash
cargo run --release -- --wf hydrogenic --alpha 1.7
```

Notes:
- If acceptance is extremely low/high, adjust `--step-size`.
- Monte Carlo samples are correlated; reported standard error is a naive estimate (no reblocking).
