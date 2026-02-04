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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def circuit_probabilities(circuit: QuantumCircuit) -> dict[str, float]:
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(circuit_no_meas)
    probs = state.probabilities()
    return {"0": float(probs[0]), "1": float(probs[1])}


def sample_counts(
    probability: dict[str, float],
    *,
    shots: int,
    seed: int,
) -> dict[str, int]:
    p0 = float(probability.get("0", 0.0))
    p1 = float(probability.get("1", 0.0))
    total = p0 + p1
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Probability must have a positive finite total.")
    p0 /= total
    p1 /= total

    rng = np.random.default_rng(seed)
    samples = rng.choice(["0", "1"], size=shots, p=[p0, p1])
    count_0 = int(np.sum(samples == "0"))
    count_1 = shots - count_0
    return {"0": count_0, "1": count_1}


def plot_estimated_probability(
    estimated: dict[str, float],
    *,
    out_pdf: Path,
) -> None:
    p0 = float(estimated["0"])
    p1 = float(estimated["1"])
    fig, ax = plt.subplots(figsize=(3.2, 2.4), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(["0", "1"], [p0, p1], color="#1f77b4")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("estimated probability")
    ax.set_ylabel("measurement outcome")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True)
    plt.close(fig)


def build_circuits() -> dict[str, QuantumCircuit]:
    circuits: dict[str, QuantumCircuit] = {}

    circuit_h_z = QuantumCircuit(1, 1)
    circuit_h_z.h(0)
    circuit_h_z.measure(0, 0)
    circuits["H_measZ"] = circuit_h_z

    circuit_h_x = QuantumCircuit(1, 1)
    circuit_h_x.h(0)
    circuit_h_x.h(0)
    circuit_h_x.measure(0, 0)
    circuits["H_measX"] = circuit_h_x

    circuit_hz_x = QuantumCircuit(1, 1)
    circuit_hz_x.h(0)
    circuit_hz_x.z(0)
    circuit_hz_x.h(0)
    circuit_hz_x.measure(0, 0)
    circuits["HZ_measX"] = circuit_hz_x

    return circuits


measurement_basis = {
    "H_measZ": "Z",
    "H_measX": "X",
    "HZ_measX": "X",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.shots <= 0:
        raise SystemExit("--shots must be positive.")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    circuits = build_circuits()
    figure_paths = {
        "H_measZ": "ch05_circuit_h_measure_z.pdf",
        "H_measX": "ch05_circuit_h_measure_x.pdf",
        "HZ_measX": "ch05_circuit_hz_measure_x.pdf",
        "H_measZ_hist": "ch05_hist_h_meas_z.pdf",
        "H_measX_hist": "ch05_hist_h_meas_x.pdf",
        "HZ_measX_hist": "ch05_hist_hz_meas_x.pdf",
    }

    for key, circuit in circuits.items():
        draw_circuit(circuit, args.outdir / figure_paths[key])

    results: dict[str, dict[str, object]] = {}
    for key in ["H_measZ", "H_measX", "HZ_measX"]:
        circuit = circuits[key]
        probability = circuit_probabilities(circuit)
        counts = sample_counts(probability, shots=args.shots, seed=args.seed)
        count_0 = int(counts.get("0", 0))
        count_1 = int(counts.get("1", 0))
        estimated = {"0": count_0 / args.shots, "1": count_1 / args.shots}
        results[key] = {
            "measurement_basis": measurement_basis[key],
            "probability": probability,
            "counts": {"0": count_0, "1": count_1},
            "estimated_probability": estimated,
        }

    plot_estimated_probability(
        results["H_measZ"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["H_measZ_hist"],
    )
    plot_estimated_probability(
        results["H_measX"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["H_measX_hist"],
    )
    plot_estimated_probability(
        results["HZ_measX"]["estimated_probability"],  # type: ignore[arg-type]
        out_pdf=args.outdir / figure_paths["HZ_measX_hist"],
    )

    record = {
        "shots": args.shots,
        "seed": args.seed,
        "figures": figure_paths,
        "results": results,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch05_interference_counts.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(f"shots={args.shots:5d} seed={args.seed}")
    for key in ["H_measZ", "H_measX", "HZ_measX"]:
        data = results[key]
        basis = data["measurement_basis"]
        counts = data["counts"]
        p1_hat = data["estimated_probability"]["1"]  # type: ignore[index]
        print(f"{key:>8s} basis={basis} counts={counts} p1_hat={p1_hat:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
