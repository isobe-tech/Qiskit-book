from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.synthesis.qft import synth_qft_full


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


def draw_circuit(circuit: QuantumCircuit, out_pdf: Path, *, fold: int | None = None) -> None:
    draw_kwargs: dict[str, object] = {"style": {"backgroundcolor": "white"}}
    if fold is not None:
        draw_kwargs["fold"] = int(fold)
    fig = circuit.draw("mpl", **draw_kwargs)
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


def plot_probability_bar(*, out_pdf: Path, probabilities: dict[str, float], highlight_keys: set[str] | None = None) -> None:
    keys = sorted(probabilities.keys())
    y = np.array([probabilities[k] for k in keys], dtype=float)
    x = np.arange(len(keys))

    highlight_keys = highlight_keys or set()
    colors = []
    for k, v in zip(keys, y):
        if k in highlight_keys:
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
        fig, ax = plt.subplots(figsize=(9.2, 3.2), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, keys, rotation=rotation)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", axis="y", color="#DDDDDD", linewidth=0.8)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def build_even_superposition(*, n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="prep")
    # Keep the least significant qubit in |0> (even numbers) and put the rest into |+>.
    for q in range(1, n):
        qc.h(q)
    return qc


def build_measurement_circuit(*, unitary: QuantumCircuit) -> QuantumCircuit:
    n = unitary.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(unitary, inplace=True)
    qc.measure(range(n), range(n))
    return qc


def qft_circuit(*, n: int, approximation_degree: int) -> QuantumCircuit:
    return synth_qft_full(
        num_qubits=n,
        do_swaps=True,
        approximation_degree=approximation_degree,
        insert_barriers=False,
        inverse=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--approx", type=int, default=2)
    parser.add_argument("--depth_n", type=int, default=8)
    parser.add_argument("--max_approx", type=int, default=5)
    args = parser.parse_args()

    n = int(args.n)
    if n < 2:
        raise SystemExit("--n must be >= 2")

    approx = int(args.approx)
    if approx < 0:
        raise SystemExit("--approx must be >= 0")

    depth_n = int(args.depth_n)
    if depth_n < 2:
        raise SystemExit("--depth-n must be >= 2")

    max_approx = int(args.max_approx)
    if max_approx < 0:
        raise SystemExit("--max-approx must be >= 0")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    prep = build_even_superposition(n=n)
    qft_exact = qft_circuit(n=n, approximation_degree=0)
    qft_approx = qft_circuit(n=n, approximation_degree=approx)

    unitary_prep = QuantumCircuit(n)
    unitary_prep.compose(prep, inplace=True)

    unitary_exact = QuantumCircuit(n)
    unitary_exact.compose(prep, inplace=True)
    unitary_exact.compose(qft_exact, inplace=True)

    unitary_approx = QuantumCircuit(n)
    unitary_approx.compose(prep, inplace=True)
    unitary_approx.compose(qft_approx, inplace=True)

    probs_prep = probabilities_full(n=n, circuit=unitary_prep)
    probs_exact = probabilities_full(n=n, circuit=unitary_exact)
    probs_approx = probabilities_full(n=n, circuit=unitary_approx)

    highlight = {format(0, f"0{n}b"), format(2 ** (n - 1), f"0{n}b")}

    plot_probability_bar(out_pdf=args.outdir / "ch17_probs_even.pdf", probabilities=probs_prep)
    plot_probability_bar(out_pdf=args.outdir / "ch17_probs_qft.pdf", probabilities=probs_exact, highlight_keys=highlight)
    plot_probability_bar(out_pdf=args.outdir / "ch17_probs_qft_approx.pdf", probabilities=probs_approx, highlight_keys=highlight)

    draw_circuit(build_measurement_circuit(unitary=unitary_prep), args.outdir / "ch17_circuit_even.pdf", fold=22)
    draw_circuit(build_measurement_circuit(unitary=unitary_exact), args.outdir / "ch17_circuit_qft.pdf", fold=15)
    # For the approx circuit, some fold values create an almost-empty second row (measurement only).
    # Pick a fold that keeps both rows populated so the figure stays readable after LaTeX scaling.
    draw_circuit(build_measurement_circuit(unitary=unitary_approx), args.outdir / "ch17_circuit_qft_approx.pdf", fold=12)

    # Depth vs approximation: show the effect of omitting small controlled rotations.
    depths = []
    approx_degrees = list(range(0, max_approx + 1))
    for d in approx_degrees:
        depths.append(int(qft_circuit(n=depth_n, approximation_degree=d).decompose().depth()))

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.0, 3.0), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(approx_degrees, depths, marker="o", linewidth=1.6, color="#4C78A8")
        ax.set_xlabel("approximation degree")
        ax.set_ylabel("circuit depth")
        ax.grid(True, which="both", axis="y", color="#DDDDDD", linewidth=0.8)
        fig.tight_layout()
    save_figure(fig, args.outdir / "ch17_depth_vs_approx.pdf", tight=True, pad_inches=0.03, trim_png=True)
    plt.close(fig)

    out_json = {
        "meta": {
            "python": sys.version,
            "platform": platform.platform(),
            "qiskit": qiskit.__version__,
        },
        "params": {
            "n": n,
            "approximation_degree": approx,
            "depth_n": depth_n,
            "max_approx": max_approx,
        },
        "probabilities": {
            "prep_even": probs_prep,
            "after_qft_exact": probs_exact,
            "after_qft_approx": probs_approx,
        },
        "metrics": {
            "tvd_exact_vs_approx": total_variation_distance(probs_exact, probs_approx),
        },
        "depth_vs_approximation_degree": {"degrees": approx_degrees, "depths": depths},
        "circuits": {
            "qft_exact_depth": int(qft_exact.decompose().depth()),
            "qft_approx_depth": int(qft_approx.decompose().depth()),
        },
    }

    (args.datadir / "ch17_qft_intro.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2) + "\n")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print("TVD(exact, approx):", out_json["metrics"]["tvd_exact_vs_approx"])
    print("depth(QFT exact):", out_json["circuits"]["qft_exact_depth"])
    print("depth(QFT approx):", out_json["circuits"]["qft_approx_depth"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
