#!/usr/bin/env python3
"""
Publication-style CKA heatmap between expert Similarity (rows) and Sigma (cols).

This is meant to pair visually with `plot_joint_tsne_perp2_min_style.py`:
- Nature-ish rcParams
- Paper-friendly CKA heatmap styling (magma-like, as commonly used in CKA figures)

Example:
  python scripts/ablation/plot_cka_heatmap_min_style.py \
    --data-dir logs/expert_similarity/flame-moe-38m/20251226_104551 \
    --layers 2 5 9
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLOR_SIM = "#E64B35"
COLOR_SIGMA = "#4DBBD5"

# Typography (paper): make numeric information most prominent.
AXIS_LABEL_FONTSIZE = (
    8  # axis titles: "Expert Co-occurrence" / "Expert Output Similarity"
)
TICK_LABEL_FONTSIZE = 10  # tick numbers: 2 / 5 / 9
CELL_VALUE_FONTSIZE = 12  # cell annotations: 0.68 / 0.06 / ...
CBAR_TICK_FONTSIZE = 8


def setup_nature_style() -> None:
    """Match the look of `plot_joint_tsne_perp2_min_style.py` for side-by-side figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def diagonal_min(matrix: np.ndarray) -> np.ndarray:
    """Set diagonal to row-wise minimum of off-diagonal elements (diag=min)."""
    mat = matrix.copy()
    mask = ~np.eye(mat.shape[0], dtype=bool)
    row_mins = [mat[i, mask[i]].min() for i in range(mat.shape[0])]
    np.fill_diagonal(mat, row_mins)
    return mat


def preprocess_matrix(mat: np.ndarray, diag_strategy: str) -> np.ndarray:
    mat = mat.astype(np.float64, copy=True)
    if diag_strategy == "keep":
        return mat
    np.fill_diagonal(mat, 0.0)
    if diag_strategy == "zero":
        return mat
    if diag_strategy == "min":
        return diagonal_min(mat)
    raise ValueError(f"Unknown --diag-strategy: {diag_strategy!r}")


def centered(mat: np.ndarray) -> np.ndarray:
    """Double-center a square matrix: H @ mat @ H."""
    n = mat.shape[0]
    H = np.eye(n) - np.ones((n, n), dtype=mat.dtype) / n
    return H @ mat @ H


def cka_from_gram(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA computed on Gram-like matrices (centered kernel alignment)."""
    if K.shape != L.shape:
        raise ValueError(f"Shape mismatch: {K.shape} vs {L.shape}")
    Kc = centered(K)
    Lc = centered(L)
    denom = np.linalg.norm(Kc, ord="fro") * np.linalg.norm(Lc, ord="fro")
    if denom <= 0:
        return float("nan")
    return float((Kc * Lc).sum() / denom)


def load_layer(data_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_dir / f"results_layer{layer}.npz")
    return (
        data["similarity"].astype(np.float64),
        data["sigma"].astype(np.float64),
    )


def default_output_path(data_dir: Path, layers: list[int]) -> Path:
    suffix = "_".join(str(x) for x in layers)
    return data_dir / f"cka_heatmap_sim_vs_sigma_layers_{suffix}.svg"


def plot_cka_heatmap(
    M: np.ndarray,
    layers: list[int],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    setup_nature_style()

    # CKA is typically interpreted in [0, 1]. We clip for visualization so the
    # colorbar is consistent and the background stays near-white like the demo.
    M_plot = np.clip(M, 0.0, 1.0)
    vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    # Match the screenshot's red tone by taking the *positive half* of RdBu_r:
    # - 0.0 -> pure white (original midpoint)
    # - 1.0 -> saturated red (original positive end)
    base = plt.get_cmap("RdBu_r")
    cmap = ListedColormap(base(np.linspace(0.5, 1.0, 256)), name="RdBu_r_pos")
    im = ax.imshow(M_plot, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    # Match naming used in joint t-SNE figures.
    ax.set_xlabel(
        "Expert Co-occurrence",
        labelpad=4,
        color=COLOR_SIGMA,
        fontweight="bold",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Expert Output Similarity",
        labelpad=4,
        color=COLOR_SIM,
        fontweight="bold",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], rotation=0)
    ax.set_yticklabels([str(l) for l in layers], rotation=0)
    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        labelsize=TICK_LABEL_FONTSIZE,
        colors="#222222",
    )

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M_plot[i, j]
            if not np.isfinite(val):
                txt = "nan"
            else:
                txt = f"{val:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=CELL_VALUE_FONTSIZE,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(0.0, 1.01, 0.25))
    cbar.ax.tick_params(length=0, labelsize=CBAR_TICK_FONTSIZE)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("#AAAAAA")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot a paper-style CKA heatmap.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="logs/expert_similarity/flame-moe-38m/20251226_104551",
        help="Directory containing results_layer*.npz files",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 5, 9],
        help="Layer indices to compare (default: 2 5 9)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: <data-dir>/cka_heatmap_sim_vs_sigma_layers_<layers>.png)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    layers = list(args.layers)

    sim_by_layer: dict[int, np.ndarray] = {}
    sigma_by_layer: dict[int, np.ndarray] = {}
    diag_strategy = "min"  # match joint t-SNE diag=min preprocessing
    for layer in layers:
        sim, sigma = load_layer(data_dir, layer)
        sim_by_layer[layer] = preprocess_matrix(sim, diag_strategy)
        sigma_by_layer[layer] = preprocess_matrix(sigma, diag_strategy)

    # Sanity: all matrices must be the same shape to compute cross-layer CKA.
    shapes = {sim_by_layer[l].shape for l in layers} | {
        sigma_by_layer[l].shape for l in layers
    }
    if len(shapes) != 1:
        raise ValueError(
            f"All layers must have same matrix shape; got: {sorted(shapes)}"
        )

    M = np.zeros((len(layers), len(layers)), dtype=np.float64)
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            M[i, j] = cka_from_gram(sim_by_layer[li], sigma_by_layer[lj])

    output_path = (
        Path(args.output) if args.output else default_output_path(data_dir, layers)
    )
    plot_cka_heatmap(
        M,
        layers=layers,
        output_path=output_path,
    )

    print("Saved:", output_path)
    print("CKA matrix rows=sim layers, cols=sigma layers")
    with np.printoptions(precision=4, suppress=True):
        print(M)


if __name__ == "__main__":
    main()
