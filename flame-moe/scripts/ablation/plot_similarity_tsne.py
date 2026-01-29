"""
t-SNE visualization for expert output similarity matrix only.
Uses the same styling as joint t-SNE plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Fixed parameters (same as joint t-SNE)
PERPLEXITY = 2
MAX_ITER = 1500
EARLY_EXAGGERATION = 6.0
LEARNING_RATE = 20
RANDOM_STATE = 42

# Style tweaks
FIGSIZE = (8.0, 3.2)  # compact layout

# Color for similarity (same red as joint plot)
COLOR_SIM = "#E64B35"


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


def compute_tsne(
    sim_mat: np.ndarray,
    perplexity: int = PERPLEXITY,
    max_iter: int = MAX_ITER,
    early_exaggeration: float = EARLY_EXAGGERATION,
    learning_rate: int | float = LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    type="moe",
):
    """Compute t-SNE embedding for similarity matrix."""
    scaler = StandardScaler()
    sim_norm = scaler.fit_transform(sim_mat)

    if type == "moe":
        tsne = TSNE(
            n_components=2,
            perplexity=2,
            random_state=random_state,
            # init="pca",
            # metric="cosine",
        )
    elif type == "mope":
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=2,
            # init="pca",
            # metric="cosine",
        )

    embedding = tsne.fit_transform(sim_norm)
    return embedding


def plot_single_layer(ax, embedding, layer_idx, panel_label):
    """Plot t-SNE embedding for a single layer."""
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=COLOR_SIM,
        label="Expert Output Similarity",
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.8,
        zorder=2,
        marker="o",
    )

    # ax.text(
    #     -0.12,
    #     1.05,
    #     f"({panel_label})",
    #     transform=ax.transAxes,
    #     fontsize=11,
    #     fontweight="bold",
    #     va="bottom",
    #     ha="left",
    # )

    ax.set_title(f"Layer {layer_idx}", fontweight="bold", pad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_aspect("equal")  # keep subplot square

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("#AAAAAA")


def plot_similarity_tsne_multi_layer(data_dict, type, output_path=None):
    """Plot t-SNE embeddings for multiple layers."""
    setup_nature_style()

    layers = sorted(data_dict.keys())
    n_layers = len(layers)

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(1, n_layers, figure=fig, wspace=0.01)  # minimal spacing
    panel_labels = ["a", "b", "c", "d", "e", "f"][:n_layers]

    embeddings = []
    for layer in layers:
        sim_mat = data_dict[layer]
        embedding = compute_tsne(sim_mat, type=type)
        embeddings.append((layer, embedding))

    # Compute unified axis limits
    all_points = np.vstack([emb for _, emb in embeddings])
    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.05 * span
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    xlim = (x_center - span / 2 - pad, x_center + span / 2 + pad)
    ylim = (y_center - span / 2 - pad, y_center + span / 2 + pad)

    for idx, (layer, panel_label) in enumerate(zip(layers, panel_labels)):
        _, embedding = embeddings[idx]
        ax = fig.add_subplot(gs[0, idx])
        plot_single_layer(ax, embedding, layer_idx=layer, panel_label=panel_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Add legend
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
    ]
    # fig.legend(
    #     handles=handles,
    #     loc="lower center",
    #     ncol=1,
    #     bbox_to_anchor=(0.5, 0.06),
    #     frameon=True,
    #     framealpha=0.9,
    #     edgecolor="#CCCCCC",
    #     borderpad=0.3,
    #     handletextpad=0.6,
    #     columnspacing=1.2,
    # )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.2, wspace=0.01)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Figure saved to: {output_path}")

    return fig


def load_layer_data(data_dir: Path, layer_idx: int):
    """Load similarity matrix from results file."""
    npz_path = data_dir / f"results_layer{layer_idx}.npz"
    data = np.load(npz_path)
    return data["similarity"]


def main(data_dir: Path, type: str) -> None:
    layers = [2, 5, 9]

    output_path = data_dir / "similarity_tsne.svg"

    data_dict = {}
    for layer in layers:
        sim_mat = load_layer_data(data_dir, layer)
        # Apply same preprocessing as joint plot
        sim_c = sim_mat.copy()
        np.fill_diagonal(sim_c, 0)
        sim_p = diagonal_min(sim_c)
        data_dict[layer] = sim_p

    plot_similarity_tsne_multi_layer(
        data_dict,
        type=type,
        output_path=output_path,
    )


if __name__ == "__main__":
    main(
        data_dir=Path("logs/expert_similarity/flame-moe-98m-11-25/20260118_191952"),
        type="moe",
    )
    main(
        data_dir=Path("logs/expert_similarity/flame-mope-98m/20260119_140549"),
        type="mope",
    )
