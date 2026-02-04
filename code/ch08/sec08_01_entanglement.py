from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def trim_png_whitespace(path: Path, *, threshold: int, pad_px: int) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Trimming PNG whitespace requires Pillow. Install pillow.") from exc

    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)
    if arr.size == 0:
        return

    rgb = arr[..., :3]
    alpha = arr[..., 3]
    non_bg = (alpha > 0) & np.any(rgb < threshold, axis=-1)
    coords = np.argwhere(non_bg)
    if coords.size == 0:
        return

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0) + 1
    y0, x0 = (int(v) for v in top_left)
    y1, x1 = (int(v) for v in bottom_right)

    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(img.width, x1 + pad_px)
    y1 = min(img.height, y1 + pad_px)

    trimmed = img.crop((x0, y0, x1, y1))
    trimmed.convert("RGB").save(path)


def save_figure(
    fig: plt.Figure,
    out_pdf: Path,
    *,
    tight: bool = False,
    pad_inches: float = 0.05,
    trim_png: bool = False,
    trim_threshold: int = 250,
    trim_pad_px: int = 12,
) -> None:
    out_png = out_pdf.with_suffix(".png")
    bbox_inches = "tight" if tight else None
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_visible(False)
        suptitle.set_text("")
    for ax in fig.get_axes():
        ax.set_title("")
        ax.title.set_visible(False)
    fig.savefig(
        out_pdf,
        facecolor="white",
        transparent=False,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    fig.savefig(
        out_png,
        dpi=300,
        facecolor="white",
        transparent=False,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    if trim_png:
        trim_png_whitespace(out_png, threshold=trim_threshold, pad_px=trim_pad_px)


def draw_circuit(circuit: QuantumCircuit, out_pdf: Path) -> None:
    fig = circuit.draw("mpl", style={"backgroundcolor": "white"})
    save_figure(
        fig,
        out_pdf,
        tight=True,
        pad_inches=0.08,
        trim_png=True,
        trim_threshold=250,
        trim_pad_px=12,
    )
    plt.close(fig)


def run_counts(
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed_simulator: int,
    seed_transpiler: int,
) -> dict[str, int]:
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator, seed_transpiler=seed_transpiler)
    job = simulator.run(compiled, shots=shots, seed_simulator=seed_simulator)
    return job.result().get_counts(compiled)


def plot_histogram(
    estimated_probability: dict[str, float],
    *,
    out_pdf: Path,
) -> None:
    bitstrings = ["00", "01", "10", "11"]
    values = [float(estimated_probability.get(k, 0.0)) for k in bitstrings]

    fig, ax = plt.subplots(figsize=(3.4, 2.4), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(bitstrings, values, color="#1f77b4")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("bitstring", fontsize=8)
    ax.set_ylabel("estimated probability", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True)
    plt.close(fig)


def build_bell_circuit(*, measure_basis: str, add_phase_flip: bool = False) -> QuantumCircuit:
    if measure_basis not in {"Z", "X"}:
        raise ValueError("measure_basis must be 'Z' or 'X'.")

    circuit = QuantumCircuit(2, 2)
    circuit.h(1)
    circuit.cx(1, 0)
    if add_phase_flip:
        circuit.z(1)
    if measure_basis == "X":
        circuit.h(0)
        circuit.h(1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed-simulator", type=int, default=7)
    parser.add_argument("--seed-transpiler", type=int, default=7)
    args = parser.parse_args()

    if args.shots <= 0:
        raise SystemExit("--shots must be positive.")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    circuits = {
        "phi_plus_z": build_bell_circuit(measure_basis="Z", add_phase_flip=False),
        "phi_plus_x": build_bell_circuit(measure_basis="X", add_phase_flip=False),
        "phi_minus_z": build_bell_circuit(measure_basis="Z", add_phase_flip=True),
        "phi_minus_x": build_bell_circuit(measure_basis="X", add_phase_flip=True),
    }

    figure_paths = {
        "phi_plus_circuit_z": "ch08_circuit_bell_z.pdf",
        "phi_plus_circuit_x": "ch08_circuit_bell_x.pdf",
        "phi_plus_hist_z": "ch08_hist_bell_z.pdf",
        "phi_plus_hist_x": "ch08_hist_bell_x.pdf",
        "phi_minus_circuit_z": "ch08_circuit_bell_minus_z.pdf",
        "phi_minus_circuit_x": "ch08_circuit_bell_minus_x.pdf",
        "phi_minus_hist_z": "ch08_hist_bell_minus_z.pdf",
        "phi_minus_hist_x": "ch08_hist_bell_minus_x.pdf",
    }

    draw_circuit(circuits["phi_plus_z"], args.outdir / figure_paths["phi_plus_circuit_z"])
    draw_circuit(circuits["phi_plus_x"], args.outdir / figure_paths["phi_plus_circuit_x"])
    draw_circuit(circuits["phi_minus_z"], args.outdir / figure_paths["phi_minus_circuit_z"])
    draw_circuit(circuits["phi_minus_x"], args.outdir / figure_paths["phi_minus_circuit_x"])

    results: dict[str, dict[str, object]] = {}
    seed_offsets = {
        "phi_plus_z": 0,
        "phi_plus_x": 1,
        "phi_minus_z": 2,
        "phi_minus_x": 3,
    }
    for key in ["phi_plus_z", "phi_plus_x", "phi_minus_z", "phi_minus_x"]:
        seed_offset = seed_offsets[key]
        seed_simulator = int(args.seed_simulator) + seed_offset
        seed_transpiler = int(args.seed_transpiler) + seed_offset
        circuit = circuits[key]
        counts_raw = run_counts(
            circuit,
            shots=args.shots,
            seed_simulator=seed_simulator,
            seed_transpiler=seed_transpiler,
        )
        counts = {k: int(counts_raw.get(k, 0)) for k in ["00", "01", "10", "11"]}
        estimated = {k: counts[k] / args.shots for k in counts}
        results[key] = {
            "seed_simulator": seed_simulator,
            "seed_transpiler": seed_transpiler,
            "counts": counts,
            "estimated_probability": estimated,
        }

    plot_histogram(
        results["phi_plus_z"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["phi_plus_hist_z"],
    )
    plot_histogram(
        results["phi_plus_x"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["phi_plus_hist_x"],
    )
    plot_histogram(
        results["phi_minus_z"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["phi_minus_hist_z"],
    )
    plot_histogram(
        results["phi_minus_x"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["phi_minus_hist_x"],
    )

    record = {
        "shots": args.shots,
        "seed_simulator": args.seed_simulator,
        "seed_transpiler": args.seed_transpiler,
        "seed_offsets": seed_offsets,
        "figures": figure_paths,
        "results": results,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }

    json_path = args.datadir / "ch08_entanglement_counts.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(
        f"shots={args.shots:5d} seed_simulator={args.seed_simulator} seed_transpiler={args.seed_transpiler}"
    )
    for key in ["phi_plus_z", "phi_plus_x", "phi_minus_z", "phi_minus_x"]:
        entry = results[key]
        counts = entry["counts"]
        est = entry["estimated_probability"]
        print(
            f"{key:>11s} seed_simulator={entry['seed_simulator']} seed_transpiler={entry['seed_transpiler']} "
            f"counts={counts} estimated={est}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
