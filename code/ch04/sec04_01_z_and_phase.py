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


def build_circuits() -> dict[str, QuantumCircuit]:
    circuits: dict[str, QuantumCircuit] = {}

    circuit_h = QuantumCircuit(1, 1)
    circuit_h.h(0)
    circuit_h.measure(0, 0)
    circuits["H"] = circuit_h

    circuit_hz = QuantumCircuit(1, 1)
    circuit_hz.h(0)
    circuit_hz.z(0)
    circuit_hz.measure(0, 0)
    circuits["HZ"] = circuit_hz

    circuit_hh = QuantumCircuit(1, 1)
    circuit_hh.h(0)
    circuit_hh.h(0)
    circuit_hh.measure(0, 0)
    circuits["HH"] = circuit_hh

    circuit_hzh = QuantumCircuit(1, 1)
    circuit_hzh.h(0)
    circuit_hzh.z(0)
    circuit_hzh.h(0)
    circuit_hzh.measure(0, 0)
    circuits["HZH"] = circuit_hzh

    return circuits


measurement_basis = {
    "H": "Z",
    "HZ": "Z",
    "HH": "X",
    "HZH": "X",
}


def plot_phase_scan(*, out_pdf: Path, num_points: int = 361) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, num_points)
    p1 = 0.5 * (1.0 - np.cos(phi))

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(phi, p1, color="#1f77b4", linewidth=2.0)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("relative phase phi")
    ax.set_ylabel("probability of 1")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_pdf, tight=True, pad_inches=0.08, trim_png=True)
    plt.close(fig)


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
        "H": "ch04_circuit_h.pdf",
        "HZ": "ch04_circuit_hz.pdf",
        "HH": "ch04_circuit_hh_measure_x.pdf",
        "HZH": "ch04_circuit_hzh_measure_x.pdf",
        "PHASE_SCAN": "ch04_phase_scan.pdf",
    }

    for key, circuit in circuits.items():
        draw_circuit(circuit, args.outdir / figure_paths[key])

    plot_phase_scan(out_pdf=args.outdir / figure_paths["PHASE_SCAN"])

    results: dict[str, dict[str, object]] = {}
    for key, circuit in circuits.items():
        probability = circuit_probabilities(circuit)
        counts = sample_counts(probability, shots=args.shots, seed=args.seed)
        count_0 = int(counts.get("0", 0))
        count_1 = int(counts.get("1", 0))
        p1_hat = count_1 / args.shots
        results[key] = {
            "measurement_basis": measurement_basis[key],
            "probability": probability,
            "counts": {"0": count_0, "1": count_1},
            "estimated_p1": p1_hat,
        }

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
    json_path = args.datadir / "ch04_phase_counts.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(f"shots={args.shots:5d} seed={args.seed}")
    for key in ["H", "HZ", "HH", "HZH"]:
        data = results[key]
        basis = data["measurement_basis"]
        counts = data["counts"]
        p1_hat = data["estimated_p1"]
        print(f"{key:>3s} basis={basis} counts={counts} p1_hat={p1_hat:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
