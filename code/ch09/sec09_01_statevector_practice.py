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
from qiskit.quantum_info import Statevector


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


def plot_statevector_component(
    state: Statevector,
    *,
    component: str,
    out_pdf: Path,
) -> None:
    component = component.lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be 'real' or 'imag'.")

    vec = np.asarray(state.data, dtype=np.complex128)
    values = vec.real if component == "real" else vec.imag

    num_qubits = state.num_qubits
    basis_labels = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]

    fig, ax = plt.subplots(figsize=(3.4, 2.6), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(basis_labels, values, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("computational basis state")
    ax.set_ylabel(f"{component} amplitude")
    ax.grid(True, axis="y", alpha=0.25)

    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if np.isfinite(max_abs) and max_abs > 0.0:
        ax.set_ylim(-1.2 * max_abs, 1.2 * max_abs)

    fig.tight_layout()
    save_figure(fig, out_pdf, tight=True, pad_inches=0.03, trim_png=True)
    plt.close(fig)


def complex_to_json_list(values: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in values.tolist()]


def build_bell_state(*, phase_minus: bool) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(1)
    circuit.cx(1, 0)
    if phase_minus:
        circuit.z(1)
    return circuit


def probabilities_dict(state: Statevector) -> dict[str, float]:
    probs = state.probabilities_dict()
    return {k: float(v) for k, v in probs.items()}


def inner_product(a: Statevector, b: Statevector) -> complex:
    data_a = np.asarray(a.data, dtype=np.complex128)
    data_b = np.asarray(b.data, dtype=np.complex128)
    return complex(np.vdot(data_a, data_b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--phase", type=float, default=float(np.pi / 3.0))
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    circuits = {
        "phi_plus": build_bell_state(phase_minus=False),
        "phi_minus": build_bell_state(phase_minus=True),
    }

    figure_paths = {
        "phi_plus_circuit": "ch09_circuit_phi_plus.pdf",
        "phi_minus_circuit": "ch09_circuit_phi_minus.pdf",
        "phi_plus_real": "ch09_statevector_phi_plus_real.pdf",
        "phi_plus_imag": "ch09_statevector_phi_plus_imag.pdf",
        "phi_minus_real": "ch09_statevector_phi_minus_real.pdf",
        "phi_minus_imag": "ch09_statevector_phi_minus_imag.pdf",
    }

    draw_circuit(circuits["phi_plus"], args.outdir / figure_paths["phi_plus_circuit"])
    draw_circuit(circuits["phi_minus"], args.outdir / figure_paths["phi_minus_circuit"])

    states = {key: Statevector.from_instruction(circuit) for key, circuit in circuits.items()}
    phi_plus = states["phi_plus"]
    phi_minus = states["phi_minus"]

    global_phase = complex(np.cos(args.phase) + 1j * np.sin(args.phase))
    phi_plus_phased = Statevector(global_phase * phi_plus.data)

    overlap = inner_product(phi_plus, phi_minus)
    fidelity = float(abs(overlap) ** 2)

    plot_statevector_component(phi_plus, component="real", out_pdf=args.outdir / figure_paths["phi_plus_real"])
    plot_statevector_component(phi_plus, component="imag", out_pdf=args.outdir / figure_paths["phi_plus_imag"])
    plot_statevector_component(phi_minus, component="real", out_pdf=args.outdir / figure_paths["phi_minus_real"])
    plot_statevector_component(phi_minus, component="imag", out_pdf=args.outdir / figure_paths["phi_minus_imag"])

    record = {
        "figures": figure_paths,
        "states": {
            "phi_plus": {
                "statevector": complex_to_json_list(np.asarray(phi_plus.data)),
                "probabilities": probabilities_dict(phi_plus),
            },
            "phi_minus": {
                "statevector": complex_to_json_list(np.asarray(phi_minus.data)),
                "probabilities": probabilities_dict(phi_minus),
            },
            "phi_plus_global_phase": {
                "phase": float(args.phase),
                "global_phase": [float(global_phase.real), float(global_phase.imag)],
                "equiv_to_phi_plus": bool(phi_plus.equiv(phi_plus_phased)),
            },
        },
        "comparisons": {
            "phi_plus_equiv_phi_minus": bool(phi_plus.equiv(phi_minus)),
            "inner_product": [float(overlap.real), float(overlap.imag)],
            "fidelity": fidelity,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch09_statevector_practice.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(f"phi_plus_equiv_phi_minus={record['comparisons']['phi_plus_equiv_phi_minus']}")
    print(f"fidelity(phi_plus, phi_minus)={fidelity:.6f}")
    print(f"phi_plus_equiv_global_phase={record['states']['phi_plus_global_phase']['equiv_to_phi_plus']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
