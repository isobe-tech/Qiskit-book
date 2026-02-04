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

    tick_labelsize = 10
    rotation = 0
    if len(keys) > 8:
        tick_labelsize = 8
        rotation = 90

    with plt.rc_context(
        {"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": tick_labelsize, "ytick.labelsize": 10}
    ):
        fig, ax = plt.subplots(figsize=(9.0, 3.6), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, keys, rotation=rotation)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def oracle_gate_bv(*, a: str, b: int = 0) -> Gate:
    if any(ch not in "01" for ch in a) or len(a) == 0:
        raise ValueError("a must be a non-empty bitstring of 0/1.")
    if b not in (0, 1):
        raise ValueError("b must be 0 or 1.")

    n = len(a)
    qc = QuantumCircuit(n + 1, name="Ua")

    # y <- y ⊕ b
    if b == 1:
        qc.x(n)

    # y <- y ⊕ (a · x mod 2)
    # Qiskit bitstring convention is little-endian; reverse the string so that
    # the rightmost character maps to qubit 0.
    for i, bit in enumerate(reversed(a)):
        if bit == "1":
            qc.cx(i, n)

    return qc.to_gate(label=r"$U_a$")


def apply_oracle_bv(qc: QuantumCircuit, *, a: str, b: int = 0, ancilla: int) -> None:
    if any(ch not in "01" for ch in a) or len(a) == 0:
        raise ValueError("a must be a non-empty bitstring of 0/1.")
    if b not in (0, 1):
        raise ValueError("b must be 0 or 1.")

    # y <- y ⊕ b
    if b == 1:
        qc.x(ancilla)

    # y <- y ⊕ (a · x mod 2)
    # Qiskit bitstring convention is little-endian; reverse the string so that
    # the rightmost character maps to qubit 0.
    for i, bit in enumerate(reversed(a)):
        if bit == "1":
            qc.cx(i, ancilla)


def build_bv_unitary(*, a: str, b: int = 0) -> QuantumCircuit:
    n = len(a)
    qc = QuantumCircuit(n + 1)
    qc.h(range(n))
    qc.x(n)
    qc.h(n)
    apply_oracle_bv(qc, a=a, b=b, ancilla=n)
    qc.h(range(n))
    return qc


def build_bv_measurement_circuit(*, a: str, b: int = 0) -> QuantumCircuit:
    n = len(a)
    unitary = build_bv_unitary(a=a, b=b)
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
    parser.add_argument("--a", type=str, default="1011")
    parser.add_argument("--b", type=int, default=0)
    args = parser.parse_args()

    a = str(args.a)
    b = int(args.b)
    n = len(a)

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    circuit = build_bv_measurement_circuit(a=a, b=b)
    draw_circuit(circuit, args.outdir / "ch15_circuit_bv.pdf")

    unitary = build_bv_unitary(a=a, b=b)
    probs = input_probabilities_from_statevector(n=n, unitary=unitary)
    plot_probability_bar(out_pdf=args.outdir / "ch15_probs_bv.pdf", probabilities=probs)

    record = {
        "a": a,
        "b": b,
        "n": n,
        "probs": probs,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch15_bernstein_vazirani.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
