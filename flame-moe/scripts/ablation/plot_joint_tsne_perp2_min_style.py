"""
Fixed-parameter joint t-SNE plot (perp=2, diag=min) with adjusted styling.
- Wider and flatter layout
- Lighter, less prominent correspondence lines
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Fixed parameters (perp=2, diag=min, tuned)
PERPLEXITY = 2
MAX_ITER = 1500
EARLY_EXAGGERATION = 6.0
LEARNING_RATE = 20
RANDOM_STATE = 42

# Style tweaks
FIGSIZE = (8.0, 3.2)  # compact layout
LINE_COLOR = "#9E9E9E"  # light gray, but visible
LINE_ALPHA = 0.5
LINE_WIDTH = 0.8

# Colors
COLOR_SIM = "#E64B35"
COLOR_SIGMA = "#4DBBD5"


def setup_nature_style() -> None:
    """Configure matplotlib for Nature-style publication figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
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
    """Set diagonal to row-wise minimum of off-diagonal elements."""
    mat = matrix.copy()
    mask = ~np.eye(mat.shape[0], dtype=bool)
    row_mins = [mat[i, mask[i]].min() for i in range(mat.shape[0])]
    np.fill_diagonal(mat, row_mins)
    return mat


def compute_joint_tsne_no_preprocess(
    sim_mat: np.ndarray,
    sigma_mat: np.ndarray,
    perplexity: int = PERPLEXITY,
    max_iter: int = MAX_ITER,
    early_exaggeration: float = EARLY_EXAGGERATION,
    learning_rate: int | float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
):
    """Compute joint t-SNE without diagonal preprocessing (assumes already done)."""
    E = sim_mat.shape[0]

    scaler = StandardScaler()
    sim_norm = scaler.fit_transform(sim_mat)
    sigma_norm = scaler.fit_transform(sigma_mat)

    combined_data = np.vstack([sim_norm, sigma_norm])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
        init="pca",
        metric="cosine",
    )

    emb_combined = tsne.fit_transform(combined_data)
    emb_sim = emb_combined[:E]
    emb_sigma = emb_combined[E:]

    distances = np.linalg.norm(emb_sim - emb_sigma, axis=1)
    avg_displacement = np.mean(distances)

    return emb_sim, emb_sigma, avg_displacement


def plot_single_layer(
    ax,
    emb_sim,
    emb_sigma,
    avg_displacement,
    layer_idx,
):
    E = emb_sim.shape[0]

    for i in range(E):
        ax.plot(
            [emb_sim[i, 0], emb_sigma[i, 0]],
            [emb_sim[i, 1], emb_sigma[i, 1]],
            color=LINE_COLOR,
            alpha=LINE_ALPHA,
            linewidth=LINE_WIDTH,
            zorder=3,
        )

    ax.scatter(
        emb_sim[:, 0],
        emb_sim[:, 1],
        c=COLOR_SIM,
        label="Expert Output Similarity",
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.8,
        zorder=2,
        marker="o",
    )

    ax.scatter(
        emb_sigma[:, 0],
        emb_sigma[:, 1],
        c=COLOR_SIGMA,
        label="Expert Co-occurrence",
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.8,
        zorder=2,
        marker="^",
    )

    ax.set_title(f"Layer {layer_idx}", fontweight="bold", pad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("#AAAAAA")


def plot_joint_tsne_multi_layer(data_dict, output_path=None):
    setup_nature_style()

    layers = sorted(data_dict.keys())
    n_layers = len(layers)

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(1, n_layers, figure=fig, wspace=0.01)  # minimal spacing

    embeddings = []
    for layer in layers:
        sim_mat, sigma_mat, _ = data_dict[layer]
        emb_sim, emb_sigma, avg_disp = compute_joint_tsne_no_preprocess(
            sim_mat,
            sigma_mat,
        )
        embeddings.append((layer, emb_sim, emb_sigma, avg_disp))

    all_points = np.vstack(
        [np.vstack([emb_sim, emb_sigma]) for _, emb_sim, emb_sigma, _ in embeddings]
    )
    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * span
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    xlim = (x_center - span / 2 - pad, x_center + span / 2 + pad)
    ylim = (y_center - span / 2 - pad, y_center + span / 2 + pad)

    for idx, layer in enumerate(layers):
        _, emb_sim, emb_sigma, avg_disp = embeddings[idx]
        ax = fig.add_subplot(gs[0, idx])
        plot_single_layer(
            ax,
            emb_sim,
            emb_sigma,
            avg_disp,
            layer_idx=layer,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")  # keep subplot square

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_SIM,
            markersize=8,
            label="Expert Output Similarity",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COLOR_SIGMA,
            markersize=8,
            label="Expert Co-occurrence",
        ),
        plt.Line2D(
            [0],
            [0],
            color=LINE_COLOR,
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            label="Expert Correspondence",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        borderpad=0.3,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.2, wspace=0.01)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Figure saved to: {output_path}")

    return fig


def load_layer_data(data_dir: Path, layer_idx: int):
    npz_path = data_dir / f"results_layer{layer_idx}.npz"
    data = np.load(npz_path)
    correlation = {
        "pearson": float(data["sigma_pearson_r"]),
        "spearman": float(data["sigma_spearman_r"]),
    }
    return data["similarity"], data["sigma"], correlation


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate joint t-SNE visualization for expert similarity analysis"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="logs/expert_similarity/flame-moe-38m/20251226_104551",
        help="Directory containing results_layer*.npz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: <data-dir>/joint_tsne_perp2_min_style.png)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 5, 9],
        help="Layer indices to plot (default: 2 5 9)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    layers = args.layers

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "joint_tsne_perp2_min_style.svg"

    data_dict = {}
    for layer in layers:
        sim_mat, sigma_mat, correlation = load_layer_data(data_dir, layer)
        sim_c = sim_mat.copy()
        sigma_c = sigma_mat.copy()
        np.fill_diagonal(sim_c, 0)
        np.fill_diagonal(sigma_c, 0)
        sim_p = diagonal_min(sim_c)
        sigma_p = diagonal_min(sigma_c)
        data_dict[layer] = (sim_p, sigma_p, correlation)

    plot_joint_tsne_multi_layer(data_dict, output_path=output_path)


if __name__ == "__main__":
    main()
