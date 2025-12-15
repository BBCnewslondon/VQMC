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

- `hydrogenic`: $\Psi_T = \exp(-\alpha\sum_i r_i)$
- `pade`: $\Psi_T = \exp(-\alpha\sum_i r_i)\,\exp(\sum_{i<j} u(r_{ij}))$, with $u(s)=\tfrac{s}{2(1+\beta s)}$
- `slater`: Slater–Jastrow $\Psi_T = |D_\uparrow||D_\downarrow|\exp(\sum_{i<j} u(r_{ij}))$ (antisymmetric within each spin sector)

For `--wf slater`, the default orbital fill order is a minimal basis: 1s, 2s, then 2p$_x$/2p$_y$/2p$_z$ (as needed for larger spin determinants).

## Run

```bash
cargo run --release -- --wf pade --alpha 1.6875 --beta 0.30 --steps 200000 --burn-in 20000 --step-size 0.7
```

Lithium (very simple 3-electron demo):

```bash
cargo run --release -- --z 3 --electrons 3 --wf pade --alpha 2.7 --beta 0.30 --steps 200000 --burn-in 20000 --step-size 0.7
```

Lithium with antisymmetry (Slater–Jastrow, default spin split is 2 up / 1 down):

```bash
cargo run --release -- --z 3 --electrons 3 --wf slater --alpha 2.7 --beta 0.30 --steps 200000 --burn-in 20000 --step-size 0.7
```

Explicit Slater orbital configuration (example: Li 1s$^2$2s$^1$ mapped as $D_\uparrow$=[1s,2s], $D_\downarrow$=[1s]):

```bash
cargo run --release -- --z 3 --electrons 3 --wf slater --n-up 2 --up-orbitals "1s,2s" --down-orbitals "1s" --alpha 2.7 --beta 0.30
```

Try the simpler trial function:

```bash
cargo run --release -- --wf hydrogenic --alpha 1.7
```

## Visualize convergence

Write a CSV trace during a run:

```bash
cargo run --release -- --wf slater --z 2 --electrons 2 --n-up 1 --alpha 1.6875 --beta 0.30 \
	--steps 50000 --burn-in 5000 --step-size 0.7 --seed 1 \
	--trace-csv trace.csv --trace-every 1
```

Plot it (PNG):

```bash
python3 scripts/plot_trace.py trace.csv --out trace.png
```

Notes:
- If acceptance is extremely low/high, adjust `--step-size`.
- Monte Carlo samples are correlated; reported standard error is a naive estimate (no reblocking).
- For multi-electron atoms, physically accurate trial states include antisymmetry (Slater determinants) and spin; use `--wf slater` for a minimal antisymmetric baseline.
