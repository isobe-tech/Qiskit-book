from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

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


def statevector_probabilities(circuit: QuantumCircuit) -> dict[str, float]:
    state = Statevector.from_instruction(circuit)
    probs = state.probabilities_dict()
    # Keep keys as Qiskit bitstrings.
    return {k: float(v) for k, v in probs.items()}


def all_bitstrings(*, num_qubits: int) -> list[str]:
    return [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]


def sample_counts(
    probs: dict[str, float],
    *,
    shots: int,
    seed: int,
    num_qubits: int,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    keys = all_bitstrings(num_qubits=num_qubits)
    p = np.array([probs.get(k, 0.0) for k in keys], dtype=float)
    p = p / p.sum()
    samples = rng.choice(keys, size=shots, p=p)
    return {k: int(np.sum(samples == k)) for k in keys}


def plot_probabilities(
    *,
    out_pdf: Path,
    probabilities: dict[str, float],
) -> None:
    keys = sorted(probabilities.keys())
    y = np.array([probabilities[k] for k in keys], dtype=float)
    x = np.arange(len(keys))

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(5.4, 2.8), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x, y, color="#4C78A8", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, keys)
        ax.set_ylabel("probability")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def plot_counts(
    *,
    out_pdf: Path,
    counts: dict[str, int],
) -> None:
    keys = sorted(counts.keys())
    y = np.array([counts[k] for k in keys], dtype=float)
    x = np.arange(len(keys))

    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(5.4, 2.8), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x, y, color="#4C78A8", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, keys)
        ax.set_ylabel("counts")
        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def add_oracle_mark_11(qc: QuantumCircuit) -> None:
    qc.cz(0, 1)


def add_diffusion_2q(qc: QuantumCircuit) -> None:
    qc.h([0, 1])
    qc.x([0, 1])
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.x([0, 1])
    qc.h([0, 1])


def build_unitary_uniform() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    return qc


def build_unitary_oracle() -> QuantumCircuit:
    qc = build_unitary_uniform()
    add_oracle_mark_11(qc)
    return qc


def build_unitary_grover_once() -> QuantumCircuit:
    qc = build_unitary_uniform()
    add_oracle_mark_11(qc)
    add_diffusion_2q(qc)
    return qc


def build_measurement_circuit(unitary: QuantumCircuit) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.compose(unitary, inplace=True)
    qc.measure([0, 1], [0, 1])
    return qc


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    unitary_builder: Callable[[], QuantumCircuit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    stages = [
        Stage(name="uniform", title="uniform superposition", unitary_builder=build_unitary_uniform),
        Stage(name="oracle", title="after oracle", unitary_builder=build_unitary_oracle),
        Stage(name="grover", title="after oracle + diffusion", unitary_builder=build_unitary_grover_once),
    ]

    record: dict[str, object] = {
        "shots": int(args.shots),
        "seed": int(args.seed),
        "stages": {},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }

    for idx, st in enumerate(stages):
        unitary = st.unitary_builder()
        circuit = build_measurement_circuit(unitary)

        draw_circuit(circuit, args.outdir / f"ch13_circuit_{st.name}.pdf")

        probs = statevector_probabilities(unitary)
        keys = all_bitstrings(num_qubits=2)
        probs_full = {k: float(probs.get(k, 0.0)) for k in keys}
        plot_probabilities(out_pdf=args.outdir / f"ch13_probs_{st.name}.pdf", probabilities=probs_full)

        counts = sample_counts(
            probs,
            shots=int(args.shots),
            seed=int(args.seed + 10 * idx),
            num_qubits=2,
        )
        plot_counts(out_pdf=args.outdir / f"ch13_counts_{st.name}.pdf", counts=counts)

        record["stages"][st.name] = {
            "unitary_depth": int(unitary.depth()),
            "unitary_size": int(unitary.size()),
            "unitary_count_ops": {k: int(v) for k, v in unitary.count_ops().items()},
            "probs": probs_full,
            "counts": counts,
        }

    json_path = args.datadir / "ch13_algorithm_skeleton.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
