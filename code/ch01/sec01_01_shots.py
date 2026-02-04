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
from qiskit.visualization import plot_histogram


def build_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def run_experiment(
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed_simulator: int,
    seed_transpiler: int,
    memory: bool = False,
) -> dict[str, int]:
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator, seed_transpiler=seed_transpiler)
    job = simulator.run(compiled, shots=shots, seed_simulator=seed_simulator, memory=memory)
    result = job.result()
    return result.get_counts(compiled)


def run_experiment_memory(
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed_simulator: int,
    seed_transpiler: int,
) -> list[str]:
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator, seed_transpiler=seed_transpiler)
    job = simulator.run(compiled, shots=shots, seed_simulator=seed_simulator, memory=True)
    result = job.result()
    return list(result.get_memory(compiled))


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--seed-simulator", type=int, default=7)
    parser.add_argument("--seed-transpiler", type=int, default=7)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    circuit = build_circuit()

    circuit_draw_path = args.outdir / "ch01_circuit_h_measure.pdf"
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

    shot_list = [100, 1000, 10000]
    records: list[dict[str, object]] = []

    for shots in shot_list:
        counts = run_experiment(
            circuit,
            shots=shots,
            seed_simulator=args.seed_simulator,
            seed_transpiler=args.seed_transpiler,
        )

        count_0 = int(counts.get("0", 0))
        count_1 = int(counts.get("1", 0))
        p_hat = count_1 / shots
        deviation = abs(p_hat - 0.5)

        records.append(
            {
                "shots": shots,
                "counts": {"0": count_0, "1": count_1},
                "p_hat": p_hat,
                "deviation_from_half": deviation,
            }
        )

        with plt.rc_context(
            {
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            }
        ):
            p0 = count_0 / shots
            p1 = count_1 / shots
            fig, ax = plt.subplots(figsize=(3.2, 2.6), facecolor="white")
            ax.set_facecolor("white")
            ax.barh(["0", "1"], [p0, p1], color="#1f77b4")
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("estimated probability", fontsize=8)
            ax.set_ylabel("measurement outcome", fontsize=8)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout()
        fig_path = args.outdir / f"ch01_hist_shots_{shots}.pdf"
        save_figure(fig, fig_path, tight=True, pad_inches=0.03)
        plt.close(fig)

    running_shots = 200
    memory = run_experiment_memory(
        circuit,
        shots=running_shots,
        seed_simulator=args.seed_simulator,
        seed_transpiler=args.seed_transpiler,
    )
    bits = np.array([1 if m.strip() == "1" else 0 for m in memory], dtype=float)
    n = np.arange(1, running_shots + 1, dtype=float)
    running_p_hat = np.cumsum(bits) / n

    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(n, running_p_hat, linewidth=1.2, label=r"$\hat{p}$ up to shot $n$")
    ax.plot(n, 0.5 + 0.5 / np.sqrt(n), linestyle="--", linewidth=1.0, label=r"$0.5 \pm \frac{1}{2\sqrt{n}}$")
    ax.plot(n, 0.5 - 0.5 / np.sqrt(n), linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="black", linewidth=0.8)
    ax.set_xlim(1, running_shots)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("shot index n")
    ax.set_ylabel("running estimate of p")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, args.outdir / "ch01_running_estimate.pdf")
    plt.close(fig)

    run_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seed_simulator": args.seed_simulator,
        "seed_transpiler": args.seed_transpiler,
        "shot_list": shot_list,
        "running_shots": running_shots,
    }
    run_info_path = args.datadir / "ch01_run_info.json"
    run_info_path.write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    json_path = args.datadir / "ch01_shots_counts.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    shots = np.array([r["shots"] for r in records], dtype=float)
    deviation = np.array([r["deviation_from_half"] for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(shots, deviation, marker="o", label="empirical deviation")
    ax.plot(shots, 0.5 / np.sqrt(shots), linestyle="--", label=r"$\frac{1}{2\sqrt{N}}$")
    ax.set_xscale("log")
    ax.set_xlabel("shots")
    ax.set_ylabel("absolute deviation from 0.5")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, args.outdir / "ch01_deviation_scaling.pdf")
    plt.close(fig)

    print("Saved figures to:", args.outdir)
    print("Saved data to:", args.datadir)
    print("Run info:", run_info_path)
    for r in records:
        shots = r["shots"]
        counts = r["counts"]
        p_hat = r["p_hat"]
        deviation = r["deviation_from_half"]
        print(f"shots={shots:5d} counts={counts} p_hat={p_hat:.4f} deviation={deviation:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
