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
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def trim_png_whitespace(path: Path, *, threshold: int, pad_px: int) -> None:
    from PIL import Image

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


def all_bitstrings(*, n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2**n)]


def plot_probability_bar(
    *,
    out_pdf: Path,
    probabilities: dict[str, float],
) -> None:
    keys = sorted(probabilities.keys())
    y = np.array([probabilities[k] for k in keys], dtype=float)
    x = np.arange(len(keys))
    colors = ["#4C78A8" if v > 1e-12 else "#BDBDBD" for v in y]

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(5.4, 2.8), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, keys)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def oracle_gate_constant(*, n: int, value: int) -> Gate:
    qc = QuantumCircuit(n + 1, name="Uf")
    if value not in (0, 1):
        raise ValueError("value must be 0 or 1.")
    if value == 1:
        qc.x(n)
    return qc.to_gate(label=r"$U_f$")


def oracle_gate_xi(*, n: int, i: int) -> Gate:
    if i < 0 or i >= n:
        raise ValueError("i out of range.")
    qc = QuantumCircuit(n + 1, name="Uf")
    qc.cx(i, n)
    return qc.to_gate(label=r"$U_f$")


def oracle_gate_parity(*, n: int) -> Gate:
    qc = QuantumCircuit(n + 1, name="Uf")
    for i in range(n):
        qc.cx(i, n)
    return qc.to_gate(label=r"$U_f$")


def build_dj_unitary(*, n: int, oracle: Gate) -> QuantumCircuit:
    qc = QuantumCircuit(n + 1)
    qc.h(range(n))
    qc.x(n)
    qc.h(n)
    qc.append(oracle, list(range(n + 1)))
    qc.h(range(n))
    return qc


def build_dj_measurement_circuit(*, n: int, oracle: Gate) -> QuantumCircuit:
    unitary = build_dj_unitary(n=n, oracle=oracle)
    qc = QuantumCircuit(n + 1, n)
    qc.compose(unitary, inplace=True)
    qc.measure(range(n), range(n))
    return qc


def input_probabilities_from_statevector(*, n: int, unitary: QuantumCircuit) -> dict[str, float]:
    state = Statevector.from_instruction(unitary)
    probs = state.probabilities_dict(qargs=list(range(n)))
    keys = all_bitstrings(n=n)
    return {k: float(probs.get(k, 0.0)) for k in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()

    n = int(args.n)
    if n < 1:
        raise SystemExit("--n must be >= 1")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    # Circuit figure uses an oracle gate label to keep the diagram readable.
    oracle_for_diagram = oracle_gate_parity(n=n)
    circuit_diagram = build_dj_measurement_circuit(n=n, oracle=oracle_for_diagram)
    draw_circuit(circuit_diagram, args.outdir / "ch14_circuit_dj.pdf")

    # Example probabilities: constant vs balanced.
    oracle_const0 = oracle_gate_constant(n=n, value=0)
    oracle_balanced = oracle_gate_parity(n=n)

    unitary_const0 = build_dj_unitary(n=n, oracle=oracle_const0)
    unitary_balanced = build_dj_unitary(n=n, oracle=oracle_balanced)

    p_const0 = input_probabilities_from_statevector(n=n, unitary=unitary_const0)
    p_balanced = input_probabilities_from_statevector(n=n, unitary=unitary_balanced)

    plot_probability_bar(out_pdf=args.outdir / "ch14_probs_constant.pdf", probabilities=p_const0)
    plot_probability_bar(out_pdf=args.outdir / "ch14_probs_balanced.pdf", probabilities=p_balanced)

    record = {
        "n": n,
        "examples": {
            "constant_0": {"probs": p_const0},
            "balanced_example": {"type": "parity", "probs": p_balanced},
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch14_deutsch_jozsa.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
