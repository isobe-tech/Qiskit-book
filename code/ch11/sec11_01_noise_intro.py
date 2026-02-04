from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Operator


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


def kraus_phase_flip(*, p: float) -> Kraus:
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be within [0, 1].")

    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Kraus([np.sqrt(1 - p) * I, np.sqrt(p) * Z])


def kraus_amplitude_damping(*, gamma: float) -> Kraus:
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be within [0, 1].")

    e0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    e1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return Kraus([e0, e1])


def kraus_depolarizing_1q(*, p: float) -> Kraus:
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be within [0, 1].")
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    a = np.sqrt(1 - p)
    b = np.sqrt(p / 3) if p > 0 else 0.0
    return Kraus([a * I, b * X, b * Y, b * Z])


def kraus_depolarizing_2q(*, p: float) -> Kraus:
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be within [0, 1].")
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = [I, X, Y, Z]
    non_identity = []
    for a, b in itertools.product(paulis, paulis):
        if a is I and b is I:
            continue
        non_identity.append(np.kron(a, b))

    a0 = np.sqrt(1 - p)
    a1 = np.sqrt(p / 15) if p > 0 else 0.0
    kraus_ops = [a0 * np.kron(I, I), *[a1 * op for op in non_identity]]
    return Kraus(kraus_ops)


def compose_kraus(left: Kraus, right: Kraus) -> Kraus:
    ops = [b @ a for a in left.data for b in right.data]
    return Kraus(ops)


def probability_dict(dm: DensityMatrix) -> dict[str, float]:
    probs = dm.probabilities_dict()
    return {str(k): float(v) for k, v in probs.items()}


def apply_readout_error_probs(
    probs: dict[str, float],
    *,
    p01: float,
    p10: float,
    num_qubits: int,
) -> dict[str, float]:
    if p01 < 0.0 or p01 > 1.0 or p10 < 0.0 or p10 > 1.0:
        raise ValueError("Readout error probabilities must be within [0, 1].")

    keys = ["".join(bits) for bits in itertools.product("01", repeat=num_qubits)]
    out = {k: 0.0 for k in keys}

    for true_bits in keys:
        p_true = probs.get(true_bits, 0.0)
        if p_true == 0.0:
            continue
        for meas_bits in keys:
            p_meas_given_true = 1.0
            for bt, bm in zip(true_bits, meas_bits):
                if bt == "0":
                    p_meas_given_true *= (1 - p01) if bm == "0" else p01
                else:
                    p_meas_given_true *= p10 if bm == "0" else (1 - p10)
            out[meas_bits] += p_true * p_meas_given_true

    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def sample_counts(probs: dict[str, float], *, shots: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    keys = sorted(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=float)
    p = p / p.sum()
    samples = rng.choice(keys, size=shots, p=p)
    counts: dict[str, int] = {k: 0 for k in keys}
    for s in samples:
        counts[str(s)] += 1
    return counts


def simulate_density_matrix_with_noise(
    circuit: QuantumCircuit,
    *,
    kraus_thermal_1q: Kraus,
    kraus_depol_1q: Kraus,
    kraus_depol_2q: Kraus,
) -> DensityMatrix:
    num_qubits = circuit.num_qubits
    dm = DensityMatrix.from_label("0" * num_qubits)

    for ci in circuit.data:
        if ci.operation.name == "measure":
            continue
        qargs = [circuit.find_bit(q).index for q in ci.qubits]
        dm = dm.evolve(Operator(ci.operation), qargs=qargs)

        for q in qargs:
            dm = dm.evolve(kraus_thermal_1q, qargs=[q])

        if len(qargs) == 1:
            dm = dm.evolve(kraus_depol_1q, qargs=qargs)
        elif len(qargs) == 2:
            dm = dm.evolve(kraus_depol_2q, qargs=qargs)
        else:
            raise ValueError(f"Unsupported gate width: {len(qargs)}")

    return dm


def plot_histogram(
    counts: dict[str, int],
    *,
    out_pdf: Path,
    order: list[str],
) -> None:
    values = [counts.get(k, 0) for k in order]
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(4.2, 2.6), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(order, values, color="#4C78A8", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("bitstring")
        ax.set_ylabel("count")
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        fig.tight_layout()
    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def plot_depth_success(
    depths: list[int],
    *,
    p0_ideal: list[float],
    p0_noisy: list[float],
    out_pdf: Path,
) -> None:
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(5.2, 3.0), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(depths, p0_ideal, marker="o", linewidth=1.6, label="ideal")
        ax.plot(depths, p0_noisy, marker="o", linewidth=1.6, label="noisy")
        ax.set_xlabel("number of identity gates")
        ax.set_ylabel("P(measure 0)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.legend(loc="lower left")
        fig.tight_layout()
    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def draw_noise_categories(out_pdf: Path) -> None:
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        # Keep generous vertical margins so labels/arrows never touch box frames
        # even after LaTeX scaling.
        fig, ax = plt.subplots(figsize=(7.2, 10.8), facecolor="white")
        ax.set_facecolor("white")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 18.0)
        ax.axis("off")

        def box(x: float, y: float, w: float, h: float, text: str) -> None:
            rect = plt.Rectangle((x, y), w, h, facecolor="#F2F2F2", edgecolor="black", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")

        # Add ample vertical spacing so the labels/arrows never touch the boxes.
        # Ensure all boxes stay within the visible axis range (avoid clipping).
        # (The PDF often scales this figure down, so give extra air.)
        h = 1.20
        gap = 6.20
        y_mid = 8.20
        y_gate = y_mid + h + gap
        y_readout = y_mid - (h + gap)

        box(0.3, y_mid, 2.6, h, "during circuit")
        box(3.2, y_gate, 3.0, h, "gate error")
        box(3.2, y_mid, 3.0, h, "relaxation")
        box(3.2, y_readout, 3.0, h, "readout error")
        box(6.6, y_mid, 3.1, h, "measured counts")

        y_flow = y_mid + h / 2
        ax.annotate("", xy=(3.2, y_flow), xytext=(2.9, y_flow), arrowprops={"arrowstyle": "->", "lw": 1.2})
        ax.annotate("", xy=(6.6, y_flow), xytext=(6.2, y_flow), arrowprops={"arrowstyle": "->", "lw": 1.2})

        x_center = 4.7
        arrow_common = {"arrowstyle": "-|>", "lw": 1.0, "shrinkA": 72, "shrinkB": 72}

        # Gate/relaxation: drift during circuit
        ax.annotate("", xy=(x_center, y_mid + h), xytext=(x_center, y_gate), arrowprops=arrow_common)
        # Readout/relaxation: misread at end
        ax.annotate("", xy=(x_center, y_mid), xytext=(x_center, y_readout + h), arrowprops=arrow_common)

        x_label = x_center + 2.40
        ax.text(
            x_label,
            (y_gate + (y_mid + h)) / 2,
            "state drifts",
            ha="left",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.60},
        )
        ax.text(
            x_label,
            ((y_readout + h) + y_mid) / 2,
            "0/1 misread",
            ha="left",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.60},
        )
        fig.tight_layout()
    # This diagram is mostly whitespace; trim it but keep a large safety margin so
    # box frames never get clipped by aggressive trimming.
    save_figure(fig, out_pdf, tight=True, pad_inches=0.10, trim_png=True, trim_threshold=250, trim_pad_px=72)
    plt.close(fig)


def build_bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(1)
    circuit.cx(1, 0)
    circuit.measure([0, 1], [0, 1])
    return circuit


def build_depth_circuit(*, num_identity: int) -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    for _ in range(num_identity):
        circuit.id(0)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--p-depol-1q", type=float, default=0.010)
    parser.add_argument("--p-depol-2q", type=float, default=0.060)
    parser.add_argument("--gamma-amp", type=float, default=0.008)
    parser.add_argument("--p-phase", type=float, default=0.010)
    parser.add_argument("--readout-p01", type=float, default=0.030)
    parser.add_argument("--readout-p10", type=float, default=0.050)

    parser.add_argument("--depths", type=int, nargs="+", default=[0, 4, 8, 12, 16, 24, 32, 48, 64])
    parser.add_argument("--depth-figure", type=int, default=8)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    kraus_depol_1q = kraus_depolarizing_1q(p=args.p_depol_1q)
    kraus_depol_2q = kraus_depolarizing_2q(p=args.p_depol_2q)
    kraus_amp = kraus_amplitude_damping(gamma=args.gamma_amp)
    kraus_phase = kraus_phase_flip(p=args.p_phase)
    kraus_thermal_1q = compose_kraus(kraus_amp, kraus_phase)

    bell = build_bell_circuit()
    depth_circuit_for_fig = build_depth_circuit(num_identity=args.depth_figure)

    draw_circuit(bell, args.outdir / "ch11_circuit_bell.pdf")
    draw_circuit(depth_circuit_for_fig, args.outdir / "ch11_circuit_depth.pdf")
    draw_noise_categories(args.outdir / "ch11_noise_categories.pdf")

    bell_ideal_probs_true = {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}
    bell_ideal_counts = sample_counts(bell_ideal_probs_true, shots=args.shots, seed=args.seed)

    bell_dm_noisy = simulate_density_matrix_with_noise(
        bell,
        kraus_thermal_1q=kraus_thermal_1q,
        kraus_depol_1q=kraus_depol_1q,
        kraus_depol_2q=kraus_depol_2q,
    )
    bell_noisy_probs_true = probability_dict(bell_dm_noisy)
    bell_noisy_probs_measured = apply_readout_error_probs(
        bell_noisy_probs_true, p01=args.readout_p01, p10=args.readout_p10, num_qubits=2
    )
    bell_noisy_counts = sample_counts(bell_noisy_probs_measured, shots=args.shots, seed=args.seed + 1)

    order_2q = ["00", "01", "10", "11"]
    plot_histogram(bell_ideal_counts, out_pdf=args.outdir / "ch11_hist_bell_ideal.pdf", order=order_2q)
    plot_histogram(bell_noisy_counts, out_pdf=args.outdir / "ch11_hist_bell_noisy.pdf", order=order_2q)

    depths = list(dict.fromkeys(args.depths))
    if any(d < 0 for d in depths):
        raise SystemExit("--depths must be non-negative.")

    p0_ideal: list[float] = []
    p0_noisy: list[float] = []

    for d in depths:
        circuit = build_depth_circuit(num_identity=d)

        p0_ideal.append(1.0)

        dm_noisy = simulate_density_matrix_with_noise(
            circuit,
            kraus_thermal_1q=kraus_thermal_1q,
            kraus_depol_1q=kraus_depol_1q,
            kraus_depol_2q=kraus_depol_2q,
        )
        noisy_true = probability_dict(dm_noisy)
        noisy_meas = apply_readout_error_probs(noisy_true, p01=args.readout_p01, p10=args.readout_p10, num_qubits=1)
        p0_noisy.append(float(noisy_meas["0"]))

    plot_depth_success(depths, p0_ideal=p0_ideal, p0_noisy=p0_noisy, out_pdf=args.outdir / "ch11_depth_success.pdf")

    record = {
        "shots": int(args.shots),
        "seed": int(args.seed),
        "noise_params": {
            "p_depol_1q": float(args.p_depol_1q),
            "p_depol_2q": float(args.p_depol_2q),
            "gamma_amp": float(args.gamma_amp),
            "p_phase": float(args.p_phase),
            "readout_p01": float(args.readout_p01),
            "readout_p10": float(args.readout_p10),
        },
        "bell": {
            "ideal_probs_true": bell_ideal_probs_true,
            "ideal_counts": bell_ideal_counts,
            "noisy_probs_true": bell_noisy_probs_true,
            "noisy_probs_measured": bell_noisy_probs_measured,
            "noisy_counts": bell_noisy_counts,
        },
        "depth": {"depths": depths, "p0_ideal_measured": p0_ideal, "p0_noisy_measured": p0_noisy},
        "figures": {
            "circuit_bell": "ch11_circuit_bell",
            "circuit_depth": "ch11_circuit_depth",
            "noise_categories": "ch11_noise_categories",
            "hist_bell_ideal": "ch11_hist_bell_ideal",
            "hist_bell_noisy": "ch11_hist_bell_noisy",
            "depth_success": "ch11_depth_success",
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch11_noise_intro.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print("bell counts (ideal):", {k: bell_ideal_counts.get(k, 0) for k in order_2q})
    print("bell counts (noisy):", {k: bell_noisy_counts.get(k, 0) for k in order_2q})
    print("depth success (ideal):", [round(x, 4) for x in p0_ideal])
    print("depth success (noisy):", [round(x, 4) for x in p0_noisy])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
