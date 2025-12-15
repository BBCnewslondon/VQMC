#!/usr/bin/env python3

import argparse
import csv
import sys


def read_trace(path: str):
    samples = []
    steps = []
    energies = []
    means = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample", "step", "energy_ha", "mean_ha"}
        if set(reader.fieldnames or []) != required:
            # Allow extra columns, but require at least these
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV must contain columns {sorted(required)}, got {reader.fieldnames}"
                )

        for row in reader:
            samples.append(int(row["sample"]))
            steps.append(int(row["step"]))
            energies.append(float(row["energy_ha"]))
            means.append(float(row["mean_ha"]))

    return samples, steps, energies, means


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot VMC energy trace CSV")
    ap.add_argument("csv", help="Input CSV produced by vmc-atom --trace-csv")
    ap.add_argument(
        "--out",
        default="trace.png",
        help="Output image path (default: trace.png)",
    )
    ap.add_argument(
        "--title",
        default="VMC energy trace (Hartree)",
        help="Plot title",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window instead of only saving",
    )

    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        # In some devcontainers, `python3` points to a user-managed Python that doesn't include
        # Ubuntu's `/usr/lib/python3/dist-packages` on sys.path.
        sys.path.append("/usr/lib/python3/dist-packages")
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(
                "matplotlib is required for plotting. Install it e.g. `sudo apt-get install python3-matplotlib` \
(or `pip install matplotlib`).\n\nError: "
                + str(e),
                file=sys.stderr,
            )
            return 2

    samples, steps, energies, means = read_trace(args.csv)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(samples, energies, linewidth=0.8, alpha=0.35, label="E_L samples")
    ax.plot(samples, means, linewidth=2.0, label="running mean")

    ax.set_title(args.title)
    ax.set_xlabel("sample")
    ax.set_ylabel("energy (Ha)")
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
