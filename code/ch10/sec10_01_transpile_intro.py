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
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

try:
    import qiskit_aer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    qiskit_aer = None

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def build_logical_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4, 4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return circuit


def build_coupling_map() -> tuple[CouplingMap, list[list[int]]]:
    edges = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
    return CouplingMap(edges), edges


def draw_coupling_map(out_pdf: Path, *, edges: list[list[int]]) -> None:
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(4.6, 1.8), facecolor="white")
        ax.set_facecolor("white")

        positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}
        undirected = {tuple(sorted(edge)) for edge in edges}

        for a, b in sorted(undirected):
            xa, ya = positions[a]
            xb, yb = positions[b]
            ax.plot([xa, xb], [ya, yb], color="black", linewidth=1.6, zorder=1)

        for q, (x, y) in positions.items():
            ax.scatter([x], [y], s=380, facecolor="white", edgecolor="black", linewidth=1.6, zorder=2)
            ax.text(x, y, str(q), ha="center", va="center", fontsize=12, zorder=3)

        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()

    save_figure(fig, out_pdf, tight=True, pad_inches=0.02, trim_png=True, trim_threshold=250, trim_pad_px=10)
    plt.close(fig)


def get_layout_physical_to_virtual(circuit: QuantumCircuit) -> dict[int, int] | None:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return None
    initial = getattr(layout, "initial_layout", None)
    if initial is None:
        return None
    physical_bits = initial.get_physical_bits()
    return {int(k): int(v._index) for k, v in physical_bits.items()}


def summarize_metrics(circuit: QuantumCircuit) -> dict[str, int]:
    ops = circuit.count_ops()
    cx = int(ops.get("cx", 0))
    swap = int(ops.get("swap", 0))
    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "cx": cx,
        "swap": swap,
        "two_qubit_gates": cx + swap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--seed-transpiler", type=int, default=7)
    parser.add_argument("--opt-levels", type=int, nargs="+", default=[0, 3])
    args = parser.parse_args()

    opt_levels = list(dict.fromkeys(args.opt_levels))
    if any(level < 0 or level > 3 for level in opt_levels):
        raise SystemExit("--opt-levels must be within 0..3.")

    ensure_dir(args.outdir)
    ensure_dir(args.datadir)

    logical = build_logical_circuit()
    coupling, edges = build_coupling_map()
    basis_gates = ["cx", "rz", "sx", "x"]

    draw_circuit(logical, args.outdir / "ch10_circuit_logical.pdf")
    draw_coupling_map(args.outdir / "ch10_coupling_map_linear.pdf", edges=edges)

    compiled: dict[str, QuantumCircuit] = {}
    for level in opt_levels:
        compiled_key = f"ol{level}"
        compiled[compiled_key] = transpile(
            logical,
            coupling_map=coupling,
            optimization_level=level,
            seed_transpiler=args.seed_transpiler,
        )
        draw_circuit(compiled[compiled_key], args.outdir / f"ch10_circuit_transpiled_{compiled_key}.pdf")

    compiled_native = transpile(
        logical,
        basis_gates=basis_gates,
        coupling_map=coupling,
        optimization_level=max(opt_levels),
        seed_transpiler=args.seed_transpiler,
    )
    draw_circuit(compiled_native, args.outdir / "ch10_circuit_transpiled_native.pdf")

    record = {
        "seed_transpiler": args.seed_transpiler,
        "coupling_edges": edges,
        "basis_gates": basis_gates,
        "figures": {
            "logical": "ch10_circuit_logical",
            "coupling_map": "ch10_coupling_map_linear",
            "transpiled": {key: f"ch10_circuit_transpiled_{key}" for key in compiled},
            "native": "ch10_circuit_transpiled_native",
        },
        "metrics": {
            "logical": summarize_metrics(logical),
            **{key: summarize_metrics(circuit) for key, circuit in compiled.items()},
            "native": summarize_metrics(compiled_native),
        },
        "layout": {key: get_layout_physical_to_virtual(circuit) for key, circuit in compiled.items()},
        "layout_native": get_layout_physical_to_virtual(compiled_native),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": getattr(qiskit_aer, "__version__", None),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    json_path = args.datadir / "ch10_transpile_intro.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figures to: {args.outdir}")
    print(f"Saved data to: {args.datadir}")
    print(f"seed_transpiler={args.seed_transpiler} opt_levels={opt_levels}")
    print("metrics:")
    print("  logical:", record["metrics"]["logical"])
    for key in compiled:
        print(f"  {key:>4}:", record["metrics"][key])
    print("  native :", record["metrics"]["native"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
