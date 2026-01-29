#!/usr/bin/env python3
"""
Expert Similarity Analysis: Compare expert output similarity with co-occurrence patterns.

This script:
1. Loads a trained FLAME-MoE model
2. Runs inference to collect MoE layer inputs
3. Passes each token through ALL experts (bypassing routing)
4. Computes cosine similarity between expert outputs
5. Compares with co-occurrence matrix (PMI normalized)

Usage:
    torchrun --nproc_per_node=2 scripts/ablation/expert_similarity_analysis.py \
        --load /path/to/checkpoint \
        --target-layer 5 \
        --stats-dir /path/to/stats \
        --num-tokens 1024
"""

import glob
import os
import re
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy import stats as scipy_stats

# Add Megatron-LM to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Megatron-LM"))

from megatron.core import mpu
from megatron.core.transformer.moe.moe_layer import MoELayer, MoPELayer
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def get_model_provider():
    """Returns the model provider function."""
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_block_spec,
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )

    def model_provider(pre_process=True, post_process=True):
        args = get_args()
        config = core_transformer_config_from_args(args)

        # Match training config: use transformer_engine (default)
        # TENorm from transformer_engine supports RMSNorm
        # FusedLayerNorm from apex does NOT support RMSNorm
        use_te = args.transformer_impl == "transformer_engine"

        if args.num_experts:
            # Use get_gpt_decoder_block_spec for MoE models (handles moe_layer_freq)
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config, use_transformer_engine=use_te
            )
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts,
                    args.moe_grouped_gemm,
                    args.qk_layernorm,
                    args.multi_latent_attention,
                    args.moe_use_legacy_grouped_gemm,
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts,
                    args.moe_grouped_gemm,
                    args.qk_layernorm,
                    args.multi_latent_attention,
                    args.moe_use_legacy_grouped_gemm,
                )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )
        return model

    return model_provider


def setup_model_and_data():
    """Initialize megatron, build model, load checkpoint."""
    args = get_args()

    # Clear MOE_MASK_EXPERTS to ensure no masking during analysis
    if "MOE_MASK_EXPERTS" in os.environ:
        del os.environ["MOE_MASK_EXPERTS"]
        print_rank_0("Cleared MOE_MASK_EXPERTS environment variable")

    # Build model
    model_provider = get_model_provider()
    model = model_provider()
    model = model.cuda()

    # Load checkpoint
    if args.load:
        load_checkpoint([model], None, None, strict=False)
    else:
        raise ValueError("--load is required to specify checkpoint path")

    model.eval()
    return model


def find_moe_layer(model, target_layer: int) -> Tuple[MoELayer, int]:
    """
    Find the MoE layer corresponding to the target layer number.

    Args:
        model: The GPT model
        target_layer: 1-indexed layer number (as in MCore convention)

    Returns:
        (moe_layer, layer_idx): The MoELayer module and its 0-indexed position
    """
    # model.decoder.layers is a ModuleList
    # layer_number is 1-indexed in MCore
    layer_idx = target_layer - 1  # Convert to 0-indexed

    if layer_idx < 0 or layer_idx >= len(model.decoder.layers):
        raise ValueError(
            f"Target layer {target_layer} out of range. "
            f"Model has {len(model.decoder.layers)} layers (1-indexed: 1 to {len(model.decoder.layers)})"
        )

    layer = model.decoder.layers[layer_idx]
    mlp = layer.mlp

    # Handle MoEOrMoPE wrapper - unwrap to get the actual MoELayer
    if hasattr(mlp, "impl"):
        # MoEOrMoPE wraps MoELayer or MoPELayer in self.impl
        actual_moe = mlp.impl
        print_rank_0(f"Unwrapped {type(mlp).__name__} -> {type(actual_moe).__name__}")
        mlp = actual_moe

    if not isinstance(mlp, (MoELayer, MoPELayer)):
        raise ValueError(
            f"Layer {target_layer} (index {layer_idx}) is not an MoE/MoPE layer. "
            f"Got {type(mlp).__name__} instead."
        )

    # Verify layer number
    if hasattr(mlp, "layer_number"):
        assert mlp.layer_number == target_layer, (
            f"Layer number mismatch: expected {target_layer}, got {mlp.layer_number}"
        )

    return mlp, layer_idx


def build_validation_data_iterator(num_tokens: int):
    """
    Build validation data iterator using Megatron's data loading infrastructure.

    Args:
        num_tokens: Number of tokens to collect

    Returns:
        data_iterator: Iterator over validation batches
    """
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.legacy.data.data_samplers import build_pretraining_data_loader
    from megatron.training.utils import get_blend_and_blend_per_split

    args = get_args()
    tokenizer = get_tokenizer()

    # Get data blend configuration
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    # Build dataset config
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=1,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )

    def is_dataset_built_on_rank():
        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

    # Convert num_tokens to num_samples (sequences)
    # Each sample is one sequence of seq_length tokens
    num_samples = (num_tokens + args.seq_length - 1) // args.seq_length

    # Ensure we have at least one full batch (micro_batch_size samples)
    # Otherwise the dataloader may drop the incomplete batch
    micro_batch_size = args.micro_batch_size
    num_samples = max(num_samples, micro_batch_size)

    # Format: [train_samples, valid_samples, test_samples]
    train_val_test_num_samples = [0, num_samples, 0]

    print_rank_0(f"Building validation dataset: {num_samples} samples for {num_tokens} tokens...")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    if valid_ds is None:
        raise RuntimeError("Failed to build validation dataset")

    # Build dataloader
    valid_dataloader = build_pretraining_data_loader(valid_ds, 0)

    return iter(valid_dataloader)


def get_sample_hidden_states(model, num_tokens: int) -> torch.Tensor:
    """
    Get hidden states from validation data by running model forward pass.

    Uses actual validation data from the training dataset.
    ALL EP ranks must participate in forward pass (MoE all-to-all requires it).

    Returns:
        hidden_states: [num_tokens, hidden_size]
    """
    args = get_args()
    ep_rank = mpu.get_expert_model_parallel_rank()

    # ALL EP ranks must participate in forward pass due to MoE all-to-all communication
    print_rank_0("Loading validation data (all EP ranks participate in forward pass)...")
    try:
        data_iterator = build_validation_data_iterator(num_tokens)
        all_hidden_states = _collect_hidden_states_from_data(
            model, data_iterator, num_tokens
        )
    except Exception as e:
        print_rank_0(f"Warning: Failed to build validation iterator: {e}")
        print_rank_0("Falling back to random tokens...")
        all_hidden_states = _collect_hidden_states_random(model, num_tokens)

    print_rank_0(f"Collected hidden_states: {all_hidden_states.shape}")
    return all_hidden_states


def _collect_hidden_states_from_data(model, data_iterator, num_tokens: int) -> torch.Tensor:
    """Helper: collect hidden states from data iterator."""
    args = get_args()
    device = torch.cuda.current_device()

    collected_hidden_states = []
    total_tokens = 0

    def hook_fn(module, hook_args):
        hs = hook_args[0].detach().clone()
        if hs.dim() == 3:
            hs = hs.reshape(-1, hs.shape[-1])
        collected_hidden_states.append(hs)

    target_layer = int(os.environ.get("TARGET_LAYER", "5"))
    moe_layer, _ = find_moe_layer(model, target_layer)
    handle = moe_layer.register_forward_pre_hook(hook_fn)

    seq_length = args.seq_length
    num_batches = max((num_tokens + seq_length - 1) // seq_length, 1)

    with torch.no_grad():
        for batch_idx in range(num_batches + 10):  # extra iterations in case of short batches
            try:
                batch = next(data_iterator)
                tokens = batch["tokens"].cuda()
                position_ids = (
                    batch["position_ids"].cuda() if "position_ids" in batch else None
                )
                attention_mask = (
                    batch["attention_mask"].cuda()
                    if "attention_mask" in batch
                    else None
                )

                if position_ids is None:
                    position_ids = (
                        torch.arange(tokens.shape[1], device=device)
                        .unsqueeze(0)
                        .expand(tokens.shape[0], -1)
                    )

                _ = model(tokens, position_ids, attention_mask)
                total_tokens += tokens.numel()

                if total_tokens >= num_tokens:
                    break

            except StopIteration:
                print_rank_0(f"Validation iterator exhausted after {batch_idx} batches")
                break
            except Exception as e:
                print_rank_0(f"Forward pass error at batch {batch_idx}: {e}")
                break

    handle.remove()

    if not collected_hidden_states:
        raise RuntimeError("Failed to collect hidden states. Check model forward pass.")

    all_hidden_states = torch.cat(collected_hidden_states, dim=0)
    if all_hidden_states.shape[0] > num_tokens:
        all_hidden_states = all_hidden_states[:num_tokens]

    print_rank_0(f"Collected {all_hidden_states.shape[0]} tokens from validation data")
    return all_hidden_states.float()


def _collect_hidden_states_random(model, num_tokens: int) -> torch.Tensor:
    """Helper: collect hidden states from random tokens."""
    args = get_args()
    device = torch.cuda.current_device()

    vocab_size = args.padded_vocab_size
    seq_length = min(num_tokens, args.seq_length)
    batch_size = max((num_tokens + seq_length - 1) // seq_length, 1)

    torch.manual_seed(args.seed)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    position_ids = (
        torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
    )

    collected_hidden_states = []

    def hook_fn(module, hook_args):
        hs = hook_args[0].detach().clone()
        if hs.dim() == 3:
            hs = hs.reshape(-1, hs.shape[-1])
        collected_hidden_states.append(hs)

    target_layer = int(os.environ.get("TARGET_LAYER", "5"))
    moe_layer, _ = find_moe_layer(model, target_layer)
    handle = moe_layer.register_forward_pre_hook(hook_fn)

    with torch.no_grad():
        try:
            _ = model(tokens, position_ids, attention_mask=None)
        except Exception as e:
            print_rank_0(f"Forward pass error: {e}")

    handle.remove()

    if not collected_hidden_states:
        raise RuntimeError("Failed to collect hidden states")

    all_hidden_states = torch.cat(collected_hidden_states, dim=0)
    if all_hidden_states.shape[0] > num_tokens:
        all_hidden_states = all_hidden_states[:num_tokens]

    print_rank_0(f"Collected {all_hidden_states.shape[0]} tokens (random fallback)")
    return all_hidden_states.float()


def compute_expert_outputs(
    moe_layer: MoELayer, hidden_states: torch.Tensor
) -> torch.Tensor:
    """
    Compute output of each local expert for all tokens.

    Args:
        moe_layer: The MoE layer
        hidden_states: [T, H] input tensor

    Returns:
        expert_outputs: [num_local_experts, T, H]
    """
    args = get_args()
    experts = moe_layer.experts
    local_experts = experts.local_experts  # ModuleList of MLP modules
    num_local_experts = len(local_experts)

    print_rank_0(f"Computing outputs for {num_local_experts} local experts...")

    # Convert to model dtype (bf16) to match expert weights
    if args.bf16:
        hidden_states = hidden_states.bfloat16()
    elif args.fp16:
        hidden_states = hidden_states.half()

    expert_outputs = []
    with torch.no_grad():
        for i, expert_mlp in enumerate(local_experts):
            # Each expert_mlp is a standard MLP: input [T, H] -> output [T, H]
            output, _ = expert_mlp(hidden_states)
            # Convert back to float32 for similarity computation
            expert_outputs.append(output.float())

    # Stack: [num_local_experts, T, H]
    expert_outputs = torch.stack(expert_outputs, dim=0)
    print_rank_0(f"Expert outputs shape: {expert_outputs.shape}")

    return expert_outputs


def gather_all_expert_outputs(
    local_outputs: torch.Tensor, moe_layer: MoELayer
) -> torch.Tensor:
    """
    Gather expert outputs from all EP ranks.

    Args:
        local_outputs: [num_local_experts, T, H] from this rank
        moe_layer: The MoE layer (to get EP info)

    Returns:
        all_outputs: [num_global_experts, T, H]
    """
    ep_size = mpu.get_expert_model_parallel_world_size()

    if ep_size == 1:
        return local_outputs

    # All-gather across EP group
    ep_group = mpu.get_expert_model_parallel_group()
    ep_rank = mpu.get_expert_model_parallel_rank()

    # Prepare output buffer
    num_local = local_outputs.shape[0]
    num_global = num_local * ep_size
    T, H = local_outputs.shape[1], local_outputs.shape[2]

    # Gather list
    gathered = [torch.zeros_like(local_outputs) for _ in range(ep_size)]
    dist.all_gather(gathered, local_outputs, group=ep_group)

    # Concatenate in correct order
    all_outputs = torch.cat(gathered, dim=0)  # [num_global_experts, T, H]

    print_rank_0(f"Gathered all expert outputs: {all_outputs.shape}")
    return all_outputs


def compute_similarity_matrix(expert_outputs: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity between expert output vectors.

    Args:
        expert_outputs: [E, T, H] expert outputs

    Returns:
        similarity: [E, E] cosine similarity matrix
    """
    E, T, H = expert_outputs.shape

    # Center each expert's output (remove mean across tokens)
    # This reduces bias from input distribution
    expert_outputs = expert_outputs - expert_outputs.mean(dim=1, keepdim=True)

    # Flatten to [E, T*H]
    flat_outputs = expert_outputs.reshape(E, -1).float()

    # Normalize for cosine similarity
    flat_outputs = F.normalize(flat_outputs, p=2, dim=1)

    # Compute similarity matrix
    similarity = torch.mm(flat_outputs, flat_outputs.t())

    return similarity.cpu().numpy()


def load_cooccurrence_matrix(
    stats_dir: str, layer: int, model_type: str = "moe"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load co-occurrence matrix from saved stats.

    Args:
        stats_dir: Directory containing stats files
        layer: Layer number
        model_type: Model type ("moe" or "mope")

    Returns:
        (cooccurrence_matrix, expert_counts)
    """
    # Find the latest stats file for this layer
    pattern = os.path.join(stats_dir, f"layer{layer}_seg*_{model_type}.npz")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No stats files found for layer {layer} in {stats_dir}"
        )

    def extract_seg(path: str) -> int:
        match = re.search(r"_seg(\d+)_", path)
        return int(match.group(1)) if match else -1

    latest_file = max(files, key=extract_seg)
    print_rank_0(f"Loading co-occurrence from: {latest_file}")

    data = np.load(latest_file)
    C = data["cooccurrence_matrix"].astype(np.float64)
    u = data["expert_counts"].astype(np.float64)

    return C, u


def compute_pmi(C: np.ndarray, u: np.ndarray, topk: int = 6) -> np.ndarray:
    """
    Compute Pointwise Mutual Information from co-occurrence counts.

    PMI(i,j) = log(P(i,j) / (P(i) * P(j)))

    For co-occurrence:
    - C[i,j] = number of tokens that activated BOTH expert i AND expert j
    - u[i] = number of tokens that activated expert i
    - N = total number of tokens = u.sum() / topk

    Probability space (all in terms of tokens):
    - P(i) = u[i] / N = probability a random token activates expert i
    - P(i,j) = C[i,j] / N = probability a random token activates both i and j
    - Note: P(i) ≈ topk / num_experts under uniform routing

    Args:
        C: [E, E] co-occurrence matrix
        u: [E] expert usage counts
        topk: number of experts per token (for normalization)

    Returns:
        PMI: [E, E] PMI matrix
    """
    # Total number of tokens
    N = u.sum() / topk

    # All probabilities in token space
    P_ij = C / (N + 1e-10)  # P(token activates both i and j)
    P_i = u / (N + 1e-10)   # P(token activates i)

    # PMI = log(P(i,j) / (P(i) * P(j)))
    # Under independence: P(i,j) = P(i) * P(j)
    # PMI > 0 means i,j co-occur more than expected
    expected = np.outer(P_i, P_i)
    PMI = np.log((P_ij + 1e-10) / (expected + 1e-10))

    # Handle infinities
    PMI = np.clip(PMI, -20, 20)

    return PMI


def compute_sigma(C: np.ndarray, u: np.ndarray, topk: int = 6) -> np.ndarray:
    """
    Compute MoPE-style covariance matrix (Sigma) from co-occurrence counts.

    Sigma(i,j) = C[i,j]/T - (mu[i] * mu[j]) / T^2
               = P(i,j) - P(i) * P(j)

    This is the linear version of PMI (covariance vs log-ratio).

    Relationship with PMI:
    - PMI = log(P(i,j) / (P(i)*P(j))) = log(1 + Sigma / (P(i)*P(j)))
    - Both have same sign: Sigma > 0 ⟺ PMI > 0

    Args:
        C: [E, E] co-occurrence matrix
        u: [E] expert usage counts (mu in MoPE notation)
        topk: number of experts per token (for normalization)

    Returns:
        Sigma: [E, E] covariance matrix
    """
    # Total number of tokens
    T = u.sum() / topk

    # Sigma = C/T - (mu @ mu.T) / T^2
    # Which equals P(i,j) - P(i) * P(j)
    P_ij = C / (T + 1e-10)
    P_i = u / (T + 1e-10)
    expected = np.outer(P_i, P_i)

    Sigma = P_ij - expected

    return Sigma


def analyze_correlation(similarity: np.ndarray, pmi: np.ndarray) -> Dict[str, float]:
    """
    Analyze correlation between similarity and PMI matrices.
    Only uses upper triangular entries (excluding diagonal).

    Returns:
        dict with pearson_r, spearman_r, and p-values
    """
    E = similarity.shape[0]
    triu_idx = np.triu_indices(E, k=1)

    sim_flat = similarity[triu_idx]
    pmi_flat = pmi[triu_idx]

    # Pearson correlation
    pearson_r, pearson_p = scipy_stats.pearsonr(sim_flat, pmi_flat)

    # Spearman correlation
    spearman_r, spearman_p = scipy_stats.spearmanr(sim_flat, pmi_flat)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n_pairs": len(sim_flat),
    }


def plot_results(
    similarity: np.ndarray,
    pmi: np.ndarray,
    corr_pmi: Dict[str, float],
    output_dir: str,
    layer: int,
    sigma: np.ndarray = None,
    corr_sigma: Dict[str, float] = None,
):
    """Save heatmaps and scatter plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print_rank_0("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    E = similarity.shape[0]
    triu_idx = np.triu_indices(E, k=1)
    sim_flat = similarity[triu_idx]
    pmi_flat = pmi[triu_idx]

    # Figure 1: Similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"Expert Output Similarity (Layer {layer})")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Expert ID")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"similarity_layer{layer}.png"), dpi=150)
    plt.close()

    # Figure 2: PMI heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.percentile(np.abs(pmi[triu_idx]), 95)
    im = ax.imshow(pmi, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Expert Co-occurrence PMI (Layer {layer})")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Expert ID")
    plt.colorbar(im, ax=ax, label="PMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pmi_layer{layer}.png"), dpi=150)
    plt.close()

    # Figure 3: PMI scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pmi_flat, sim_flat, alpha=0.3, s=10)
    ax.set_xlabel("Co-occurrence PMI")
    ax.set_ylabel("Output Cosine Similarity")
    ax.set_title(
        f"Layer {layer}: Similarity vs PMI\n"
        f"Pearson r={corr_pmi['pearson_r']:.3f}, "
        f"Spearman ρ={corr_pmi['spearman_r']:.3f}"
    )
    z = np.polyfit(pmi_flat, sim_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pmi_flat.min(), pmi_flat.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Linear fit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_pmi_layer{layer}.png"), dpi=150)
    plt.close()

    # Figure 4: Sigma scatter plot (if provided)
    if sigma is not None and corr_sigma is not None:
        sigma_flat = sigma[triu_idx]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(sigma_flat, sim_flat, alpha=0.3, s=10, color="green")
        ax.set_xlabel("Co-occurrence Sigma (Covariance)")
        ax.set_ylabel("Output Cosine Similarity")
        ax.set_title(
            f"Layer {layer}: Similarity vs Sigma\n"
            f"Pearson r={corr_sigma['pearson_r']:.3f}, "
            f"Spearman ρ={corr_sigma['spearman_r']:.3f}"
        )
        z = np.polyfit(sigma_flat, sim_flat, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sigma_flat.min(), sigma_flat.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Linear fit")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_sigma_layer{layer}.png"), dpi=150)
        plt.close()

        # Figure 5: Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(pmi_flat, sim_flat, alpha=0.3, s=10)
        axes[0].set_xlabel("PMI")
        axes[0].set_ylabel("Similarity")
        axes[0].set_title(f"PMI (r={corr_pmi['pearson_r']:.3f})")

        axes[1].scatter(sigma_flat, sim_flat, alpha=0.3, s=10, color="green")
        axes[1].set_xlabel("Sigma")
        axes[1].set_ylabel("Similarity")
        axes[1].set_title(f"Sigma (r={corr_sigma['pearson_r']:.3f})")

        plt.suptitle(f"Layer {layer}: PMI vs Sigma Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_layer{layer}.png"), dpi=150)
        plt.close()

    print_rank_0(f"Plots saved to: {output_dir}")


def main():
    # Parse arguments using Megatron's argument parser
    # We add our custom arguments via extra_args_provider

    def extra_args_provider(parser):
        group = parser.add_argument_group("Expert Similarity Analysis")
        group.add_argument(
            "--target-layer",
            type=int,
            default=5,
            help="Target MoE layer number (1-indexed)",
        )
        group.add_argument(
            "--stats-dir",
            type=str,
            required=True,
            help="Directory containing co-occurrence stats (*.npz files)",
        )
        group.add_argument(
            "--num-tokens",
            type=int,
            default=1024,
            help="Number of tokens to analyze",
        )
        group.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Output directory for plots and results",
        )
        group.add_argument(
            "--topk",
            type=int,
            default=6,
            help="Number of experts per token (for PMI normalization)",
        )
        group.add_argument(
            "--model-type",
            type=str,
            default="moe",
            choices=["moe", "mope"],
            help="Model type: 'moe' or 'mope' (affects stats file naming)",
        )
        return parser

    # Initialize Megatron
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )

    args = get_args()

    # Set target layer in environment for hook
    os.environ["TARGET_LAYER"] = str(args.target_layer)

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            PROJECT_ROOT, "logs", "expert_similarity", timestamp
        )

    if mpu.get_data_parallel_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Validate setup
    print_rank_0("=" * 60)
    print_rank_0("Expert Similarity Analysis")
    print_rank_0("=" * 60)
    print_rank_0(f"Target layer: {args.target_layer}")
    print_rank_0(f"Num tokens: {args.num_tokens}")
    print_rank_0(f"Stats dir: {args.stats_dir}")
    print_rank_0(f"Output dir: {args.output_dir}")
    print_rank_0(f"EP size: {mpu.get_expert_model_parallel_world_size()}")
    print_rank_0(f"EP rank: {mpu.get_expert_model_parallel_rank()}")
    print_rank_0(f"Router topk (for PMI): {args.topk}")

    # Validate topk consistency with model config
    if hasattr(args, 'moe_router_topk') and args.moe_router_topk != args.topk:
        print_rank_0(f"WARNING: --topk ({args.topk}) differs from --moe-router-topk ({args.moe_router_topk})")
        print_rank_0("  PMI normalization may not match co-occurrence matrix!")
        print_rank_0(f"  Consider using --topk {args.moe_router_topk}")

    # Setup model
    print_rank_0("\n[1/6] Loading model...")
    model = setup_model_and_data()

    # Find target MoE layer
    print_rank_0("\n[2/6] Finding target MoE layer...")
    moe_layer, layer_idx = find_moe_layer(model, args.target_layer)
    num_local_experts = len(moe_layer.experts.local_experts)
    num_global_experts = moe_layer.num_moe_experts
    print_rank_0(f"Found MoE layer at index {layer_idx}")
    print_rank_0(
        f"Local experts: {num_local_experts}, Global experts: {num_global_experts}"
    )

    # Collect hidden states
    print_rank_0("\n[3/6] Collecting hidden states...")
    hidden_states = get_sample_hidden_states(model, args.num_tokens)
    print_rank_0(f"Hidden states: {hidden_states.shape}")

    # Compute expert outputs
    print_rank_0("\n[4/6] Computing expert outputs...")
    local_expert_outputs = compute_expert_outputs(moe_layer, hidden_states)

    # Gather all expert outputs across EP ranks
    all_expert_outputs = gather_all_expert_outputs(local_expert_outputs, moe_layer)

    # Only rank 0 does analysis and plotting
    if mpu.get_data_parallel_rank() == 0 and mpu.get_expert_model_parallel_rank() == 0:
        # Compute similarity matrix
        print_rank_0("\n[5/6] Computing similarity matrix...")
        similarity = compute_similarity_matrix(all_expert_outputs)
        print_rank_0(f"Similarity matrix shape: {similarity.shape}")

        # Load and normalize co-occurrence matrix
        print_rank_0("\n[6/6] Analyzing correlation with co-occurrence...")
        C, u = load_cooccurrence_matrix(args.stats_dir, args.target_layer, args.model_type)
        pmi = compute_pmi(C, u, args.topk)
        sigma = compute_sigma(C, u, args.topk)
        print_rank_0(f"Co-occurrence matrix shape: {C.shape}")

        # Sanity check
        assert similarity.shape == pmi.shape, (
            f"Shape mismatch: similarity {similarity.shape} vs PMI {pmi.shape}"
        )

        # Analyze correlation with both PMI and Sigma
        corr_pmi = analyze_correlation(similarity, pmi)
        corr_sigma = analyze_correlation(similarity, sigma)

        # Print results
        print_rank_0("\n" + "=" * 60)
        print_rank_0("RESULTS")
        print_rank_0("=" * 60)
        print_rank_0(f"Number of expert pairs analyzed: {corr_pmi['n_pairs']}")

        print_rank_0("\n--- PMI (log-ratio) ---")
        print_rank_0(
            f"Pearson correlation:  r = {corr_pmi['pearson_r']:.4f} (p = {corr_pmi['pearson_p']:.2e})"
        )
        print_rank_0(
            f"Spearman correlation: ρ = {corr_pmi['spearman_r']:.4f} (p = {corr_pmi['spearman_p']:.2e})"
        )

        print_rank_0("\n--- Sigma (linear covariance, MoPE-style) ---")
        print_rank_0(
            f"Pearson correlation:  r = {corr_sigma['pearson_r']:.4f} (p = {corr_sigma['pearson_p']:.2e})"
        )
        print_rank_0(
            f"Spearman correlation: ρ = {corr_sigma['spearman_r']:.4f} (p = {corr_sigma['spearman_p']:.2e})"
        )

        print_rank_0("\n--- Comparison ---")
        print_rank_0(f"PMI Pearson r:   {corr_pmi['pearson_r']:.4f}")
        print_rank_0(f"Sigma Pearson r: {corr_sigma['pearson_r']:.4f}")
        diff = corr_sigma['pearson_r'] - corr_pmi['pearson_r']
        if abs(diff) < 0.01:
            print_rank_0("→ PMI and Sigma show similar correlation with similarity")
        elif diff > 0:
            print_rank_0(f"→ Sigma shows stronger correlation (+{diff:.4f})")
        else:
            print_rank_0(f"→ PMI shows stronger correlation ({diff:.4f})")

        if corr_pmi["pearson_r"] > 0.3:
            print_rank_0("\n✓ Positive correlation detected!")
            print_rank_0(
                "  High co-occurrence → High similarity → Experts are REDUNDANT"
            )
        elif corr_pmi["pearson_r"] < -0.3:
            print_rank_0("\n✗ Negative correlation detected!")
            print_rank_0(
                "  High co-occurrence → Low similarity → Experts are COMPLEMENTARY"
            )
        else:
            print_rank_0("\n○ Weak correlation")
            print_rank_0("  No clear relationship between co-occurrence and similarity")

        # Save results
        results_file = os.path.join(
            args.output_dir, f"results_layer{args.target_layer}.npz"
        )
        np.savez_compressed(
            results_file,
            similarity=similarity,
            pmi=pmi,
            sigma=sigma,
            cooccurrence=C,
            expert_counts=u,
            # PMI correlations
            pmi_pearson_r=corr_pmi["pearson_r"],
            pmi_pearson_p=corr_pmi["pearson_p"],
            pmi_spearman_r=corr_pmi["spearman_r"],
            pmi_spearman_p=corr_pmi["spearman_p"],
            # Sigma correlations
            sigma_pearson_r=corr_sigma["pearson_r"],
            sigma_pearson_p=corr_sigma["pearson_p"],
            sigma_spearman_r=corr_sigma["spearman_r"],
            sigma_spearman_p=corr_sigma["spearman_p"],
            # Legacy names for compatibility
            pearson_r=corr_pmi["pearson_r"],
            pearson_p=corr_pmi["pearson_p"],
            spearman_r=corr_pmi["spearman_r"],
            spearman_p=corr_pmi["spearman_p"],
            target_layer=args.target_layer,
            num_tokens=args.num_tokens,
        )
        print_rank_0(f"\nResults saved to: {results_file}")

        # Plot
        plot_results(
            similarity, pmi, corr_pmi, args.output_dir, args.target_layer,
            sigma=sigma, corr_sigma=corr_sigma
        )

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    print_rank_0("\nDone!")


if __name__ == "__main__":
    main()
