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


def probabilities_full(*, n: int, circuit: QuantumCircuit) -> dict[str, float]:
    probs = Statevector.from_instruction(circuit).probabilities_dict()
    keys = all_bitstrings(n=n)
    return {k: float(probs.get(k, 0.0)) for k in keys}


def plot_probability_bar(*, out_pdf: Path, probabilities: dict[str, float], highlight_key: str | None = None) -> None:
    keys = sorted(probabilities.keys())
    y = np.array([probabilities[k] for k in keys], dtype=float)
    x = np.arange(len(keys))
    colors = []
    for k, v in zip(keys, y):
        if highlight_key is not None and k == highlight_key:
            colors.append("#F58518")
        else:
            colors.append("#4C78A8" if v > 1e-12 else "#BDBDBD")

    tick_labelsize = 10
    rotation = 0
    if len(keys) > 8:
        tick_labelsize = 8
        rotation = 90

    with plt.rc_context(
        {"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": tick_labelsize, "ytick.labelsize": 10}
    ):
        fig, ax = plt.subplots(figsize=(9.2, 3.6), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, keys, rotation=rotation)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def plot_success_probability_vs_k(*, out_pdf: Path, ks: list[int], probs: list[float], k_opt: int) -> None:
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.6, 3.2), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(ks, probs, marker="o", linewidth=1.6, color="#4C78A8")
        ax.axvline(k_opt, color="#F58518", linestyle="--", linewidth=1.4)
        if k_opt in ks:
            ax.plot([k_opt], [probs[ks.index(k_opt)]], marker="o", color="#F58518")
        ax.set_xlabel("iterations k")
        ax.set_ylabel("success probability")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(min(ks), max(ks))
        ax.grid(True, which="both", axis="y", color="#DDDDDD", linewidth=0.8)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def oracle_gate_mark_state(*, n: int, w: str) -> Gate:
    if len(w) != n or any(ch not in "01" for ch in w):
        raise ValueError("w must be a bitstring of length n.")

    qc = QuantumCircuit(n, name="O")
    # Map |w> to |11...1>.
    for i, bit in enumerate(reversed(w)):
        if bit == "0":
            qc.x(i)

    # Phase flip on |11...1>.
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    # Unmap.
    for i, bit in enumerate(reversed(w)):
        if bit == "0":
            qc.x(i)

    return qc.to_gate(label=r"$O$")


def diffusion_gate(*, n: int) -> Gate:
    qc = QuantumCircuit(n, name="D")
    qc.h(range(n))
    qc.x(range(n))

    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))
    return qc.to_gate(label=r"$D$")


def build_grover_unitary(*, n: int, w: str, k: int) -> QuantumCircuit:
    if k < 0:
        raise ValueError("k must be >= 0.")

    oracle = oracle_gate_mark_state(n=n, w=w)
    diff = diffusion_gate(n=n)

    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(k):
        qc.append(oracle, list(range(n)))
        qc.append(diff, list(range(n)))
    return qc


def build_grover_measurement_circuit(*, n: int, w: str, k: int) -> QuantumCircuit:
    unitary = build_grover_unitary(n=n, w=w, k=k)
    qc = QuantumCircuit(n, n)
    qc.compose(unitary, inplace=True)
    qc.measure(range(n), range(n))
    return qc


def recommended_k(*, n: int) -> int:
    n_items = 2**n
    theta = np.arcsin(1.0 / np.sqrt(n_items))
    k = int(np.round((np.pi / (4.0 * theta)) - 0.5))
    return max(0, k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--w", type=str, default="1011")
    parser.add_argument("--max_k", type=int, default=8)
    args = parser.parse_args()

    n = int(args.n)
    if n < 2:
        raise SystemExit("--n must be >= 2")

    w = str(args.w)
    if len(w) != n or any(ch not in "01" for ch in w):
        raise SystemExit("--w must be a 0/1 bitstring of length --n")

    max_k = int(args.max_k)
    if max_k < 0:
        raise SystemExit("--max_k must be >= 0")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    k_opt = recommended_k(n=n)
    k_over = min(max_k, k_opt + 2)

    # Circuit figure: show one Grover iteration (k=1) as the repeating unit.
    circuit = build_grover_measurement_circuit(n=n, w=w, k=1)
    draw_circuit(circuit, args.outdir / "ch16_circuit_grover.pdf")

    # Probability distributions at selected k.
    probs_k0 = probabilities_full(n=n, circuit=build_grover_unitary(n=n, w=w, k=0))
    probs_kopt = probabilities_full(n=n, circuit=build_grover_unitary(n=n, w=w, k=k_opt))
    probs_kover = probabilities_full(n=n, circuit=build_grover_unitary(n=n, w=w, k=k_over))

    plot_probability_bar(out_pdf=args.outdir / "ch16_probs_k0.pdf", probabilities=probs_k0, highlight_key=w)
    plot_probability_bar(out_pdf=args.outdir / "ch16_probs_kopt.pdf", probabilities=probs_kopt, highlight_key=w)
    plot_probability_bar(out_pdf=args.outdir / "ch16_probs_kover.pdf", probabilities=probs_kover, highlight_key=w)

    # Success probability curve.
    ks = list(range(max_k + 1))
    success = []
    for k in ks:
        probs = probabilities_full(n=n, circuit=build_grover_unitary(n=n, w=w, k=k))
        success.append(float(probs.get(w, 0.0)))
    plot_success_probability_vs_k(out_pdf=args.outdir / "ch16_success_vs_k.pdf", ks=ks, probs=success, k_opt=k_opt)

    record = {
        "n": n,
        "N": 2**n,
        "w": w,
        "k_opt": int(k_opt),
        "k_over": int(k_over),
        "max_k": int(max_k),
        "success_probability": {str(k): float(p) for k, p in zip(ks, success)},
        "probs_selected": {
            "k0": probs_k0,
            "k_opt": probs_kopt,
            "k_over": probs_kover,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch16_grover.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
