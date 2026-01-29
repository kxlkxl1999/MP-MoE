#!/usr/bin/env python3
"""
Expert Embedding PCA Visualization: Layer 9 only, MoE vs MoPE comparison.

Usage:
    python scripts/ablation/plot_expert_pca_layer9.py --data-dir logs/expert_embedding_tsne
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Publication-quality settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_embeddings(data_dir: str, model_type: str, layer: int) -> np.ndarray:
    """Load expert embeddings from npz file."""
    file_path = os.path.join(data_dir, model_type, f"all_embeddings_layer{layer}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    data = np.load(file_path)
    embeddings = data["expert_outputs"]  # [E, T, H]
    print(f"Loaded {model_type} layer {layer}: {embeddings.shape}")
    return embeddings


def apply_pca(embeddings: np.ndarray, expert_ids: tuple) -> np.ndarray:
    """Apply PCA to selected expert embeddings."""
    selected = embeddings[list(expert_ids)]
    _, _, H = selected.shape
    flat = selected.reshape(-1, H)

    scaler = StandardScaler()
    flat_norm = scaler.fit_transform(flat)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(flat_norm)

    return reduced


def main():
    parser = argparse.ArgumentParser(description="Plot PCA visualization for Layer 9")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="logs/expert_embedding_tsne",
        help="Directory containing moe/ and mope/ subdirectories",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs=3,
        default=[15, 29, 46],
        help="Expert IDs to visualize",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=9,
        help="Layer number to visualize",
    )
    args = parser.parse_args()

    expert_ids = tuple(args.experts)
    layer = args.layer

    # Load embeddings
    print("Loading embeddings...")
    moe_emb = load_embeddings(args.data_dir, "moe", layer)
    mope_emb = load_embeddings(args.data_dir, "mope", layer)

    # Apply PCA
    print("Applying PCA...")
    moe_pca = apply_pca(moe_emb, expert_ids)
    mope_pca = apply_pca(mope_emb, expert_ids)

    # Colors for 3 experts
    colors = ["#E64B35", "#F4A460", "#808080"]  # Red, Orange, Gray

    # Create figure: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    T = moe_emb.shape[1]  # tokens per expert

    # Compute limits based on actual data range (not centered at origin)
    all_data = np.vstack([moe_pca, mope_pca])
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()

    # Add small margin (5%)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)

    for ax, (data, title) in zip(
        axes, [(moe_pca, "Classic MoE"), (mope_pca, "MP-MoE")]
    ):
        for i, eid in enumerate(expert_ids):
            start, end = i * T, (i + 1) * T
            ax.scatter(
                data[start:end, 0],
                data[start:end, 1],
                c=colors[i],
                label=f"Expert {eid}",
                alpha=0.6,
                s=8,
                edgecolors="none",
            )

        ax.set_title(title)
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.legend(loc="upper right", framealpha=0.9, markerscale=3)

        # Use same limits for both plots
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Add light grid
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(args.data_dir, "pca_layer9.svg")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
