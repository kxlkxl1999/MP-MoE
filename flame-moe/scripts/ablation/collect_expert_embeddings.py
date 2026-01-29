#!/usr/bin/env python3
"""
Expert Embedding Collection: Collect expert output embeddings for t-SNE visualization.

This script:
1. Loads a trained MoE/MoPE model
2. Runs inference to collect MoE layer inputs
3. Passes each token through ALL experts (bypassing routing)
4. Saves raw expert output embeddings for later t-SNE visualization

Usage:
    torchrun --nproc_per_node=2 scripts/ablation/collect_expert_embeddings.py \
        --load /path/to/checkpoint \
        --target-layer 5 \
        --num-tokens 1024 \
        --output-dir logs/expert_embedding_tsne/moe
"""

import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

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

        use_te = args.transformer_impl == "transformer_engine"

        if args.num_experts:
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
        actual_moe = mlp.impl
        print_rank_0(f"Unwrapped {type(mlp).__name__} -> {type(actual_moe).__name__}")
        mlp = actual_moe

    if not isinstance(mlp, (MoELayer, MoPELayer)):
        raise ValueError(
            f"Layer {target_layer} (index {layer_idx}) is not an MoE/MoPE layer. "
            f"Got {type(mlp).__name__} instead."
        )

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

    blend, blend_per_split = get_blend_and_blend_per_split(args)

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

    num_samples = (num_tokens + args.seq_length - 1) // args.seq_length
    micro_batch_size = args.micro_batch_size
    num_samples = max(num_samples, micro_batch_size)

    train_val_test_num_samples = [0, num_samples, 0]

    print_rank_0(f"Building validation dataset: {num_samples} samples for {num_tokens} tokens...")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    if valid_ds is None:
        raise RuntimeError("Failed to build validation dataset")

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
        for batch_idx in range(num_batches + 10):
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
            output, _ = expert_mlp(hidden_states)
            expert_outputs.append(output.float())

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

    # Gather list
    gathered = [torch.zeros_like(local_outputs) for _ in range(ep_size)]
    dist.all_gather(gathered, local_outputs, group=ep_group)

    # Concatenate in correct order
    all_outputs = torch.cat(gathered, dim=0)  # [num_global_experts, T, H]

    print_rank_0(f"Gathered all expert outputs: {all_outputs.shape}")
    return all_outputs


def main():
    # Parse arguments using Megatron's argument parser
    def extra_args_provider(parser):
        group = parser.add_argument_group("Expert Embedding Collection")
        group.add_argument(
            "--target-layer",
            type=int,
            default=5,
            help="Target MoE layer number (1-indexed)",
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
            help="Output directory for embeddings",
        )
        group.add_argument(
            "--model-type",
            type=str,
            default="moe",
            choices=["moe", "mope"],
            help="Model type: 'moe' or 'mope' (affects output naming)",
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
        args.output_dir = os.path.join(
            PROJECT_ROOT, "logs", "expert_embedding_tsne", args.model_type
        )

    if mpu.get_data_parallel_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Validate setup
    print_rank_0("=" * 60)
    print_rank_0("Expert Embedding Collection")
    print_rank_0("=" * 60)
    print_rank_0(f"Target layer: {args.target_layer}")
    print_rank_0(f"Num tokens: {args.num_tokens}")
    print_rank_0(f"Output dir: {args.output_dir}")
    print_rank_0(f"Model type: {args.model_type}")
    print_rank_0(f"EP size: {mpu.get_expert_model_parallel_world_size()}")
    print_rank_0(f"EP rank: {mpu.get_expert_model_parallel_rank()}")

    # Setup model
    print_rank_0("\n[1/4] Loading model...")
    model = setup_model_and_data()

    # Find target MoE layer
    print_rank_0("\n[2/4] Finding target MoE layer...")
    moe_layer, layer_idx = find_moe_layer(model, args.target_layer)
    num_local_experts = len(moe_layer.experts.local_experts)
    num_global_experts = moe_layer.num_moe_experts
    print_rank_0(f"Found MoE layer at index {layer_idx}")
    print_rank_0(
        f"Local experts: {num_local_experts}, Global experts: {num_global_experts}"
    )

    # Collect hidden states
    print_rank_0("\n[3/4] Collecting hidden states...")
    hidden_states = get_sample_hidden_states(model, args.num_tokens)
    print_rank_0(f"Hidden states: {hidden_states.shape}")

    # Compute expert outputs
    print_rank_0("\n[4/4] Computing expert outputs...")
    local_expert_outputs = compute_expert_outputs(moe_layer, hidden_states)

    # Gather all expert outputs across EP ranks
    all_expert_outputs = gather_all_expert_outputs(local_expert_outputs, moe_layer)

    # Only rank 0 saves results
    if mpu.get_data_parallel_rank() == 0 and mpu.get_expert_model_parallel_rank() == 0:
        # Save embeddings
        output_file = os.path.join(
            args.output_dir, f"all_embeddings_layer{args.target_layer}.npz"
        )
        np.savez_compressed(
            output_file,
            expert_outputs=all_expert_outputs.cpu().numpy(),  # [E, T, H]
            target_layer=args.target_layer,
            num_tokens=args.num_tokens,
            num_experts=num_global_experts,
            model_type=args.model_type,
        )
        print_rank_0(f"\nEmbeddings saved to: {output_file}")
        print_rank_0(f"Shape: {all_expert_outputs.shape} (experts, tokens, hidden_size)")

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    print_rank_0("\nDone!")


if __name__ == "__main__":
    main()
