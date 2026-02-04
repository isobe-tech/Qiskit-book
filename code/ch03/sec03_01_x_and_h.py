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


def build_circuits() -> dict[str, QuantumCircuit]:
    circuits: dict[str, QuantumCircuit] = {}

    circuit_h = QuantumCircuit(1, 1)
    circuit_h.h(0)
    circuit_h.measure(0, 0)
    circuits["H"] = circuit_h

    circuit_hh = QuantumCircuit(1, 1)
    circuit_hh.h(0)
    circuit_hh.h(0)
    circuit_hh.measure(0, 0)
    circuits["HH"] = circuit_hh

    circuit_x = QuantumCircuit(1, 1)
    circuit_x.x(0)
    circuit_x.measure(0, 0)
    circuits["X"] = circuit_x

    circuit_hxh = QuantumCircuit(1, 1)
    circuit_hxh.h(0)
    circuit_hxh.x(0)
    circuit_hxh.h(0)
    circuit_hxh.measure(0, 0)
    circuits["HXH"] = circuit_hxh

    circuit_xhh = QuantumCircuit(1, 1)
    circuit_xhh.x(0)
    circuit_xhh.h(0)
    circuit_xhh.h(0)
    circuit_xhh.measure(0, 0)
    circuits["XHH"] = circuit_xhh

    return circuits


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

    circuits = build_circuits()

    figure_paths = {
        "H": "ch03_circuit_h.pdf",
        "HH": "ch03_circuit_hh.pdf",
        "X": "ch03_circuit_x.pdf",
        "HXH": "ch03_circuit_hxh_measure_x.pdf",
        "XHH": "ch03_circuit_xhh_measure_x.pdf",
    }

    for key, circuit in circuits.items():
        draw_circuit(circuit, args.outdir / figure_paths[key])

    expected_p1 = {
        "H": 0.5,
        "HH": 0.0,
        "X": 1.0,
        "HXH": 0.0,
        "XHH": 1.0,
    }

    measurement_basis = {
        "H": "Z",
        "HH": "Z",
        "X": "Z",
        "HXH": "X",
        "XHH": "X",
    }

    results: dict[str, dict[str, object]] = {}
    for key, circuit in circuits.items():
        counts = run_counts(
            circuit,
            shots=args.shots,
            seed_simulator=args.seed_simulator,
            seed_transpiler=args.seed_transpiler,
        )
        count_0 = int(counts.get("0", 0))
        count_1 = int(counts.get("1", 0))
        p1_hat = count_1 / args.shots
        results[key] = {
            "measurement_basis": measurement_basis[key],
            "expected_p1": expected_p1[key],
            "counts": {"0": count_0, "1": count_1},
            "estimated_p1": p1_hat,
        }

    record = {
        "shots": args.shots,
        "seed_simulator": args.seed_simulator,
        "seed_transpiler": args.seed_transpiler,
        "figures": figure_paths,
        "results": results,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch03_x_h_counts.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(f"shots={args.shots:5d} seed_simulator={args.seed_simulator} seed_transpiler={args.seed_transpiler}")
    for key in ["H", "HH", "X", "HXH", "XHH"]:
        data = results[key]
        counts = data["counts"]
        p1_hat = data["estimated_p1"]
        basis = data["measurement_basis"]
        print(f"{key:>3s} basis={basis} counts={counts} p1_hat={p1_hat:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
