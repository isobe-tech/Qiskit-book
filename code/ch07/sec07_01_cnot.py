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


def build_circuit(input_bits: str) -> QuantumCircuit:
    if len(input_bits) != 2 or any(bit not in "01" for bit in input_bits):
        raise ValueError("input_bits must be a 2-bit string.")

    circuit = QuantumCircuit(2, 2)
    if input_bits[0] == "1":
        circuit.x(1)
    if input_bits[1] == "1":
        circuit.x(0)
    circuit.cx(1, 0)
    circuit.measure([0, 1], [0, 1])
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
    return job.result().get_counts(compiled)


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

    circuit_cnot = QuantumCircuit(2, 2)
    circuit_cnot.cx(1, 0)
    circuit_cnot.measure([0, 1], [0, 1])
    draw_circuit(circuit_cnot, args.outdir / "ch07_circuit_cnot.pdf")

    truth_table = {
        "00": "00",
        "01": "01",
        "10": "11",
        "11": "10",
    }

    results: dict[str, dict[str, object]] = {}
    for input_bits, expected_bits in truth_table.items():
        circuit = build_circuit(input_bits)
        counts_raw = run_counts(
            circuit,
            shots=args.shots,
            seed_simulator=args.seed_simulator,
            seed_transpiler=args.seed_transpiler,
        )
        counts = {k: int(v) for k, v in counts_raw.items()}
        expected_count = int(counts.get(expected_bits, 0))
        expected_prob = expected_count / args.shots
        results[input_bits] = {
            "expected": expected_bits,
            "counts": counts,
            "expected_count": expected_count,
            "expected_probability": expected_prob,
        }

    record = {
        "shots": args.shots,
        "seed_simulator": args.seed_simulator,
        "seed_transpiler": args.seed_transpiler,
        "results": results,
        "figures": {
            "cnot_circuit": "ch07_circuit_cnot.pdf",
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }

    json_path = args.datadir / "ch07_cnot_counts.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(
        f"shots={args.shots:5d} seed_simulator={args.seed_simulator} seed_transpiler={args.seed_transpiler}"
    )
    for input_bits in ["00", "01", "10", "11"]:
        res = results[input_bits]
        print(
            f"input={input_bits} expected={res['expected']} count={res['expected_count']} p={res['expected_probability']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
