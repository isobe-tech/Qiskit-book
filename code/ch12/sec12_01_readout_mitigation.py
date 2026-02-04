from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
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


def apply_readout_error_probs(probs: dict[str, float], *, p01: float, p10: float) -> dict[str, float]:
    if p01 < 0.0 or p01 > 1.0 or p10 < 0.0 or p10 > 1.0:
        raise ValueError("Readout error probabilities must be within [0, 1].")

    p0 = float(probs.get("0", 0.0))
    p1 = float(probs.get("1", 0.0))
    q0 = (1 - p01) * p0 + p10 * p1
    q1 = p01 * p0 + (1 - p10) * p1
    total = q0 + q1
    if total == 0:
        return {"0": 0.0, "1": 0.0}
    return {"0": q0 / total, "1": q1 / total}


def sample_counts(probs: dict[str, float], *, shots: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    keys = ["0", "1"]
    p = np.array([probs.get(k, 0.0) for k in keys], dtype=float)
    p = p / p.sum()
    samples = rng.choice(keys, size=shots, p=p)
    return {"0": int(np.sum(samples == "0")), "1": int(np.sum(samples == "1"))}


def counts_to_prob(counts: dict[str, int]) -> np.ndarray:
    shots = int(counts.get("0", 0) + counts.get("1", 0))
    if shots <= 0:
        raise ValueError("shots must be positive.")
    return np.array([counts.get("0", 0) / shots, counts.get("1", 0) / shots], dtype=float)


def estimate_confusion_matrix(counts0: dict[str, int], counts1: dict[str, int]) -> np.ndarray:
    # M[i, j] = P(measured=i | prepared=j)
    q0 = counts_to_prob(counts0)
    q1 = counts_to_prob(counts1)
    return np.column_stack([q0, q1])


def correct_probabilities(
    q: np.ndarray,
    *,
    Mhat: np.ndarray,
    clip: bool,
) -> np.ndarray:
    phat = np.linalg.inv(Mhat) @ q
    if clip:
        phat = np.clip(phat, 0.0, 1.0)
        total = float(phat.sum())
        if total > 0:
            phat = phat / total
    return phat


def multinomial_resample(counts: dict[str, int], *, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    shots = int(counts.get("0", 0) + counts.get("1", 0))
    p = counts_to_prob(counts)
    draw = rng.multinomial(shots, p)
    return {"0": int(draw[0]), "1": int(draw[1])}


def bootstrap_corrected_probabilities(
    *,
    target_counts: dict[str, int],
    calib_counts0: dict[str, int],
    calib_counts1: dict[str, int],
    num_bootstrap: int,
    seed: int,
    clip: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.zeros((num_bootstrap, 2), dtype=float)
    for i in range(num_bootstrap):
        s0 = int(rng.integers(0, 2**31 - 1))
        s1 = int(rng.integers(0, 2**31 - 1))
        s2 = int(rng.integers(0, 2**31 - 1))

        boot_target = multinomial_resample(target_counts, seed=s0)
        boot_calib0 = multinomial_resample(calib_counts0, seed=s1)
        boot_calib1 = multinomial_resample(calib_counts1, seed=s2)

        q = counts_to_prob(boot_target)
        Mhat = estimate_confusion_matrix(boot_calib0, boot_calib1)
        samples[i] = correct_probabilities(q, Mhat=Mhat, clip=clip)
    return samples


def plot_probability_comparison(
    *,
    out_pdf: Path,
    title: str,
    p_true: np.ndarray,
    q_meas: np.ndarray,
    p_corr: np.ndarray,
    corr_ci: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    labels = ["0", "1"]
    x = np.arange(len(labels))
    width = 0.26

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(5.4, 2.8), facecolor="white")
        ax.set_facecolor("white")

        ax.bar(x - width, p_true, width=width, label="true", color="#4C78A8", edgecolor="black", linewidth=0.4)
        ax.bar(x, q_meas, width=width, label="measured", color="#F58518", edgecolor="black", linewidth=0.4)

        if corr_ci is None:
            ax.bar(x + width, p_corr, width=width, label="corrected", color="#54A24B", edgecolor="black", linewidth=0.4)
        else:
            lo, hi = corr_ci
            yerr = np.vstack([p_corr - lo, hi - p_corr])
            ax.bar(
                x + width,
                p_corr,
                width=width,
                label="corrected",
                color="#54A24B",
                edgecolor="black",
                linewidth=0.4,
                yerr=yerr,
                capsize=3,
                ecolor="black",
                error_kw={"linewidth": 0.8},
            )

        ax.set_xticks(x, labels)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="upper center", ncol=3, frameon=False)
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom")
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def statevector_probabilities_1q(circuit: QuantumCircuit) -> dict[str, float]:
    # Circuit must not contain measurements.
    state = Statevector.from_instruction(circuit)
    probs = state.probabilities_dict()
    return {"0": float(probs.get("0", 0.0)), "1": float(probs.get("1", 0.0))}


def build_target_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def build_target_unitary() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def build_calibration_circuit(*, prepared: int) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    if prepared == 1:
        qc.x(0)
    qc.measure(0, 0)
    return qc


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    actual_p01: float
    actual_p10: float
    calib_p01: float
    calib_p10: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--calib-shots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-bootstrap", type=int, default=800)
    parser.add_argument("--no-clip", action="store_true")

    parser.add_argument("--actual-p01", type=float, default=0.030)
    parser.add_argument("--actual-p10", type=float, default=0.050)
    parser.add_argument("--calib-p01", type=float, default=0.030)
    parser.add_argument("--calib-p10", type=float, default=0.050)

    parser.add_argument("--drift-actual-p01", type=float, default=0.060)
    parser.add_argument("--drift-actual-p10", type=float, default=0.020)
    parser.add_argument("--drift-calib-p01", type=float, default=0.030)
    parser.add_argument("--drift-calib-p10", type=float, default=0.050)
    args = parser.parse_args()
    clip = not bool(args.no_clip)

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    target = build_target_circuit()
    calib0 = build_calibration_circuit(prepared=0)
    calib1 = build_calibration_circuit(prepared=1)
    draw_circuit(target, args.outdir / "ch12_circuit_target.pdf")
    draw_circuit(calib0, args.outdir / "ch12_circuit_calib_0.pdf")
    draw_circuit(calib1, args.outdir / "ch12_circuit_calib_1.pdf")

    p_true_dict = statevector_probabilities_1q(build_target_unitary())
    p_true = np.array([p_true_dict["0"], p_true_dict["1"]], dtype=float)

    scenarios = [
        Scenario(
            name="matched",
            title="calibration matches",
            actual_p01=float(args.actual_p01),
            actual_p10=float(args.actual_p10),
            calib_p01=float(args.calib_p01),
            calib_p10=float(args.calib_p10),
        ),
        Scenario(
            name="drift",
            title="calibration drifts",
            actual_p01=float(args.drift_actual_p01),
            actual_p10=float(args.drift_actual_p10),
            calib_p01=float(args.drift_calib_p01),
            calib_p10=float(args.drift_calib_p10),
        ),
    ]

    results: dict[str, object] = {}

    for idx, sc in enumerate(scenarios):
        calib_counts0 = sample_counts(
            apply_readout_error_probs({"0": 1.0, "1": 0.0}, p01=sc.calib_p01, p10=sc.calib_p10),
            shots=int(args.calib_shots),
            seed=int(args.seed + 10 + 3 * idx),
        )
        calib_counts1 = sample_counts(
            apply_readout_error_probs({"0": 0.0, "1": 1.0}, p01=sc.calib_p01, p10=sc.calib_p10),
            shots=int(args.calib_shots),
            seed=int(args.seed + 11 + 3 * idx),
        )
        Mhat = estimate_confusion_matrix(calib_counts0, calib_counts1)

        q_meas_dict = apply_readout_error_probs(p_true_dict, p01=sc.actual_p01, p10=sc.actual_p10)
        target_counts = sample_counts(q_meas_dict, shots=int(args.shots), seed=int(args.seed + 12 + 3 * idx))
        q = counts_to_prob(target_counts)

        p_corr = correct_probabilities(q, Mhat=Mhat, clip=clip)

        samples = bootstrap_corrected_probabilities(
            target_counts=target_counts,
            calib_counts0=calib_counts0,
            calib_counts1=calib_counts1,
            num_bootstrap=int(args.num_bootstrap),
            seed=int(args.seed + 99 + 5 * idx),
            clip=clip,
        )
        lo = np.quantile(samples, 0.16, axis=0)
        hi = np.quantile(samples, 0.84, axis=0)

        plot_probability_comparison(
            out_pdf=args.outdir / f"ch12_probs_{sc.name}.pdf",
            title=sc.title,
            p_true=p_true,
            q_meas=q,
            p_corr=p_corr,
            corr_ci=(lo, hi),
        )

        results[sc.name] = {
            "actual": {"p01": sc.actual_p01, "p10": sc.actual_p10},
            "calibration": {"p01": sc.calib_p01, "p10": sc.calib_p10},
            "calibration_counts0": calib_counts0,
            "calibration_counts1": calib_counts1,
            "Mhat": Mhat.tolist(),
            "true_probs": {"0": float(p_true[0]), "1": float(p_true[1])},
            "measured_counts": target_counts,
            "measured_probs": {"0": float(q[0]), "1": float(q[1])},
            "corrected_probs": {"0": float(p_corr[0]), "1": float(p_corr[1])},
            "corrected_ci_16_84": {"0": [float(lo[0]), float(hi[0])], "1": [float(lo[1]), float(hi[1])]},
        }

    record = {
        "shots": int(args.shots),
        "calib_shots": int(args.calib_shots),
        "seed": int(args.seed),
        "num_bootstrap": int(args.num_bootstrap),
        "clip": bool(clip),
        "figures": {
            "circuit_target": "ch12_circuit_target",
            "circuit_calib_0": "ch12_circuit_calib_0",
            "circuit_calib_1": "ch12_circuit_calib_1",
            "probs_matched": "ch12_probs_matched",
            "probs_drift": "ch12_probs_drift",
        },
        "results": results,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch12_readout_mitigation.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
