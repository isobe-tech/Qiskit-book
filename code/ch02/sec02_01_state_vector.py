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
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


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


def build_circuit(alpha: complex, beta: complex) -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.initialize([alpha, beta], 0)
    circuit.measure(0, 0)
    return circuit


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
    result = job.result()
    return result.get_counts(compiled)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--p1", type=float, default=0.25)
    parser.add_argument("--seed-simulator", type=int, default=7)
    parser.add_argument("--seed-transpiler", type=int, default=7)
    args = parser.parse_args()

    if not (0.0 <= args.p1 <= 1.0):
        raise SystemExit("--p1 must be between 0 and 1.")
    if args.shots <= 0:
        raise SystemExit("--shots must be positive.")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    alpha = complex(np.sqrt(1.0 - args.p1))
    beta = complex(np.sqrt(args.p1))
    state = Statevector([alpha, beta])
    probs = np.abs(state.data) ** 2
    prob_0 = float(probs[0])
    prob_1 = float(probs[1])

    circuit = build_circuit(alpha, beta)

    circuit_draw_path = args.outdir / "ch02_circuit_prepare_measure.pdf"
    circuit_fig = circuit.draw("mpl", style={"backgroundcolor": "white"})
    save_figure(
        circuit_fig,
        circuit_draw_path,
        tight=True,
        pad_inches=0.08,
        trim_png=True,
        trim_threshold=250,
        trim_pad_px=12,
    )
    plt.close(circuit_fig)

    counts = run_counts(
        circuit,
        shots=args.shots,
        seed_simulator=args.seed_simulator,
        seed_transpiler=args.seed_transpiler,
    )
    count_0 = int(counts.get("0", 0))
    count_1 = int(counts.get("1", 0))
    p1_hat = count_1 / args.shots

    record = {
        "alpha": [float(alpha.real), float(alpha.imag)],
        "beta": [float(beta.real), float(beta.imag)],
        "probability": {"0": prob_0, "1": prob_1},
        "shots": args.shots,
        "counts": {"0": count_0, "1": count_1},
        "estimated_probability": {"0": count_0 / args.shots, "1": p1_hat},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seed_simulator": args.seed_simulator,
        "seed_transpiler": args.seed_transpiler,
    }
    json_path = args.datadir / "ch02_statevector_counts.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(f"alpha={alpha:.6f} beta={beta:.6f} probability_of_1={prob_1:.4f}")
    print(f"shots={args.shots:5d} counts={counts} p1_hat={p1_hat:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
