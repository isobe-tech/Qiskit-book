from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import QuantumCircuit
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


def draw_circuit(circuit: QuantumCircuit, out_pdf: Path, *, fold: int | None = None) -> None:
    draw_kwargs: dict[str, object] = {"style": {"backgroundcolor": "white"}}
    if fold is not None:
        draw_kwargs["fold"] = int(fold)
    fig = circuit.draw("mpl", **draw_kwargs)
    save_figure(fig, out_pdf, tight=True, pad_inches=0.08, trim_png=True, trim_threshold=250, trim_pad_px=12)
    plt.close(fig)


def plot_hybrid_loop(out_pdf: Path) -> None:
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(7.4, 3.2), facecolor="white")
        ax.set_facecolor("white")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")

        def box(x: float, y: float, w: float, h: float, text: str) -> None:
            rect = plt.Rectangle((x, y), w, h, facecolor="#F2F2F2", edgecolor="black", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")

        box(0.6, 2.4, 2.5, 1.0, "parameters")
        box(3.5, 2.4, 2.7, 1.0, "quantum circuit")
        box(6.8, 2.4, 2.5, 1.0, "measure shots")
        box(6.8, 0.6, 2.5, 1.0, "estimate cost")
        box(3.5, 0.6, 2.7, 1.0, "classical update")

        def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.2})

        arrow(3.1, 2.9, 3.5, 2.9)
        arrow(6.2, 2.9, 6.8, 2.9)
        arrow(8.05, 2.4, 8.05, 1.6)
        arrow(6.8, 1.1, 6.2, 1.1)
        arrow(3.5, 1.1, 2.0, 2.4)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def circuit_single_qubit(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc


def true_expectation_z(theta: float) -> float:
    # For |psi(theta)> = Ry(theta)|0>, <Z> = cos(theta).
    return float(math.cos(theta))


def sample_expectation_z(*, theta: float, shots: int, rng: np.random.Generator) -> float:
    # Measure Z. Outcome is +1 on |0>, -1 on |1>.
    # For Ry(theta)|0>, p(1) = sin^2(theta/2).
    if shots <= 0:
        raise ValueError("shots must be positive")
    p1 = float(math.sin(theta / 2) ** 2)
    n1 = int(rng.binomial(shots, p1))
    return float((shots - 2 * n1) / shots)


@dataclass(frozen=True)
class Trace:
    thetas: list[float]
    measured: list[float]
    true: list[float]


def optimize_1d(
    *,
    shots: int,
    rng: np.random.Generator,
    iters: int = 40,
    theta0: float = 0.25,
    step0: float = 1.2,
    cooling: float = 0.92,
) -> Trace:
    theta = float(theta0)
    step = float(step0)
    thetas: list[float] = []
    measured: list[float] = []
    true: list[float] = []

    def measure_cost(t: float) -> float:
        return sample_expectation_z(theta=t, shots=shots, rng=rng)

    for _ in range(iters):
        c0 = measure_cost(theta)
        c_plus = measure_cost(theta + step)
        c_minus = measure_cost(theta - step)

        if c_plus < c0 and c_plus <= c_minus:
            theta = theta + step
            c_best = c_plus
        elif c_minus < c0 and c_minus < c_plus:
            theta = theta - step
            c_best = c_minus
        else:
            c_best = c0

        theta = float((theta + math.pi) % (2 * math.pi) - math.pi)
        thetas.append(theta)
        measured.append(float(c_best))
        true.append(true_expectation_z(theta))
        step *= cooling

    return Trace(thetas=thetas, measured=measured, true=true)


def plot_optimization_traces(*, out_pdf: Path, traces: dict[int, Trace]) -> None:
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.6, 3.2), facecolor="white")
        ax.set_facecolor("white")

        xs = None
        for shots, tr in traces.items():
            xs = np.arange(1, len(tr.measured) + 1)
            ax.plot(xs, tr.measured, marker="o", markersize=3.0, linewidth=1.2, label=f"measured ({shots} shots)")
            ax.plot(xs, tr.true, linewidth=1.8, linestyle="--", label=f"true ({shots} shots path)")

        ax.set_xlabel("iteration")
        ax.set_ylabel("cost  <Z>")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, which="both", axis="y", color="#DDDDDD", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def plot_shot_cost_scaling(*, out_pdf: Path, theta: float, rng: np.random.Generator) -> dict[str, list[float]]:
    shot_list = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    repeats = 400

    true_val = true_expectation_z(theta)
    rms_errors = []
    stds = []
    for shots in shot_list:
        samples = np.array([sample_expectation_z(theta=theta, shots=shots, rng=rng) for _ in range(repeats)], dtype=float)
        rms_errors.append(float(np.sqrt(np.mean((samples - true_val) ** 2))))
        stds.append(float(np.std(samples, ddof=1)))

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.6, 3.2), facecolor="white")
        ax.set_facecolor("white")

        xs = np.array(shot_list, dtype=float)
        ax.plot(xs, stds, marker="o", linewidth=1.6, color="#4C78A8", label="std of estimate")
        ax.plot(xs, rms_errors, marker="o", linewidth=1.6, color="#F58518", label="RMS error")

        # Reference slope: 1/sqrt(N)
        ref = (stds[0] * np.sqrt(xs[0])) / np.sqrt(xs)
        ax.plot(xs, ref, linestyle="--", linewidth=1.2, color="#666666", label=r"~ $1/\sqrt{N}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots N")
        ax.set_ylabel("error size")
        ax.grid(True, which="both", color="#DDDDDD", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)
    return {"shots": shot_list, "std": stds, "rms_error": rms_errors, "theta": [float(theta)], "true": [float(true_val)]}


def qaoa_circuit_edge(*, gamma: float, beta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    # Cost unitary: exp(-i gamma Z0 Z1). A global phase is irrelevant for measurement probabilities.
    qc.rzz(2 * gamma, 0, 1)
    # Mixer: exp(-i beta X) = Rx(2 beta)
    qc.rx(2 * beta, 0)
    qc.rx(2 * beta, 1)
    return qc


def maxcut_value_from_bitstring(bits: str) -> int:
    # One edge between the two bits.
    return 1 if bits in {"01", "10"} else 0


def qaoa_expected_cut(*, gamma: float, beta: float) -> float:
    qc = qaoa_circuit_edge(gamma=gamma, beta=beta)
    probs = Statevector.from_instruction(qc).probabilities_dict()
    exp_val = 0.0
    for bits, p in probs.items():
        exp_val += float(p) * float(maxcut_value_from_bitstring(bits))
    return float(exp_val)


def plot_qaoa_probs(*, out_pdf: Path, gamma: float, beta: float) -> dict[str, float]:
    qc = qaoa_circuit_edge(gamma=gamma, beta=beta)
    probs = Statevector.from_instruction(qc).probabilities_dict()
    keys = ["00", "01", "10", "11"]
    y = [float(probs.get(k, 0.0)) for k in keys]
    colors = ["#BDBDBD", "#F58518", "#F58518", "#BDBDBD"]

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(4.8, 2.8), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(np.arange(len(keys)), y, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(np.arange(len(keys)), keys)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", axis="y", color="#DDDDDD", linewidth=0.8)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)
    return {k: float(probs.get(k, 0.0)) for k in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--shots_low", type=int, default=200)
    parser.add_argument("--shots_high", type=int, default=2000)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    rng = np.random.default_rng(int(args.seed))

    plot_hybrid_loop(args.outdir / "ch18_hybrid_loop.pdf")
    draw_circuit(circuit_single_qubit(theta=0.7), args.outdir / "ch18_circuit_param.pdf")

    traces = {
        int(args.shots_low): optimize_1d(shots=int(args.shots_low), rng=rng, iters=int(args.iters), theta0=0.25),
        int(args.shots_high): optimize_1d(shots=int(args.shots_high), rng=rng, iters=int(args.iters), theta0=0.25),
    }
    plot_optimization_traces(out_pdf=args.outdir / "ch18_optim_trace.pdf", traces=traces)

    shot_cost_data = plot_shot_cost_scaling(out_pdf=args.outdir / "ch18_shot_cost.pdf", theta=math.pi / 3, rng=rng)

    # QAOA demo: grid-search best parameters for one-edge MaxCut.
    gammas = np.linspace(0.0, math.pi, 181)
    betas = np.linspace(0.0, math.pi / 2, 181)
    best = (-1.0, 0.0, 0.0)
    for g in gammas:
        for b in betas:
            val = qaoa_expected_cut(gamma=float(g), beta=float(b))
            if val > best[0]:
                best = (val, float(g), float(b))

    best_val, best_gamma, best_beta = best
    qaoa_probs = plot_qaoa_probs(out_pdf=args.outdir / "ch18_qaoa_probs.pdf", gamma=best_gamma, beta=best_beta)
    qc_qaoa = QuantumCircuit(2, 2)
    qc_qaoa.compose(qaoa_circuit_edge(gamma=best_gamma, beta=best_beta), inplace=True)
    qc_qaoa.measure([0, 1], [0, 1])
    draw_circuit(qc_qaoa, args.outdir / "ch18_circuit_qaoa.pdf", fold=18)

    out_json = {
        "meta": {"python": sys.version, "platform": platform.platform(), "qiskit": qiskit.__version__},
        "params": {
            "seed": int(args.seed),
            "iters": int(args.iters),
            "shots_low": int(args.shots_low),
            "shots_high": int(args.shots_high),
        },
        "vqe_like_demo": {
            "traces": {
                str(shots): {"theta": tr.thetas, "measured_cost": tr.measured, "true_cost": tr.true}
                for shots, tr in traces.items()
            }
        },
        "shot_cost_demo": shot_cost_data,
        "qaoa_demo": {"best_expected_cut": float(best_val), "best_gamma": float(best_gamma), "best_beta": float(best_beta), "probs": qaoa_probs},
    }
    (args.datadir / "ch18_variational_intro.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2) + "\n")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print("QAOA best expected cut:", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
