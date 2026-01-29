# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import glob
import json
import os
import re
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.moe.OT_pruning import otep_batched
from megatron.core.transformer.transformer_config import TransformerConfig

_COOCCURRENCE_MATRICES_CACHE = None  # {layer_num: np.ndarray}


def _parse_moe_mask_experts():
    """Parse MOE_MASK_EXPERTS JSON like '{"2":[3,17],"5":[10]}'."""
    env_value = os.environ.get("MOE_MASK_EXPERTS", "").strip()
    if not env_value:
        print("[MOE_MASK] env MOE_MASK_EXPERTS is empty, no masking")
        return {}
    raw = json.loads(env_value)
    result = {int(k): set(int(x) for x in v) for k, v in raw.items()}
    print(f"[MOE_MASK] Parsed MOE_MASK_EXPERTS: {result}")
    return result


# ==================== Dynamic Cooccurrence Mask ====================
# Environment variables:
#   MOE_DYNAMIC_MASK_MODE: "cooccurrence" | "random" | "" (disabled)
#   MOE_COOCCURRENCE_STATS_DIR: path to stats directory with layer*_seg*_moe.npz
#   MOE_DYNAMIC_MASK_LAYERS: space-separated layer numbers, e.g. "2 5 9"


def _load_cooccurrence_matrices():
    """Load cooccurrence matrices from npz files (called once at startup)."""
    global _COOCCURRENCE_MATRICES_CACHE
    if _COOCCURRENCE_MATRICES_CACHE is not None:
        return _COOCCURRENCE_MATRICES_CACHE

    stats_dir = os.environ.get("MOE_COOCCURRENCE_STATS_DIR", "").strip()
    layers_str = os.environ.get("MOE_DYNAMIC_MASK_LAYERS", "").strip()

    if not stats_dir or not layers_str:
        _COOCCURRENCE_MATRICES_CACHE = {}
        return _COOCCURRENCE_MATRICES_CACHE

    layers = [int(x) for x in layers_str.split()]
    result = {}

    for layer in layers:
        pattern = os.path.join(stats_dir, f"layer{layer}_seg*_moe.npz")
        files = glob.glob(pattern)
        if not files:
            print(
                f"[DYNAMIC_MASK] Warning: no npz files for layer {layer} in {stats_dir}"
            )
            continue

        # Find latest segment
        def seg(path):
            match = re.search(r"_seg(\d+)_", path)
            return int(match.group(1)) if match else -1

        latest = max(files, key=seg)
        data = np.load(latest)
        C = data["cooccurrence_matrix"].astype(np.float64)

        # Normalize: C_norm[i,j] = C[i,j] / C[i,i]
        diag = np.maximum(np.diag(C), 1.0)
        C_norm = C / diag[:, None]
        np.fill_diagonal(C_norm, 0.0)

        result[layer] = C_norm
        print(
            f"[DYNAMIC_MASK] Loaded cooccurrence matrix for layer {layer}: {C_norm.shape} from {os.path.basename(latest)}"
        )

    _COOCCURRENCE_MATRICES_CACHE = result
    return result


def _parse_dynamic_mask_mode():
    """Parse MOE_DYNAMIC_MASK_MODE environment variable."""
    mode = os.environ.get("MOE_DYNAMIC_MASK_MODE", "").strip().lower()
    if mode not in ("", "cooccurrence", "random"):
        print(f"[DYNAMIC_MASK] Warning: unknown mode '{mode}', disabling dynamic mask")
        return ""
    if mode:
        print(f"[DYNAMIC_MASK] Dynamic mask mode: {mode}")
    return mode


def _parse_dynamic_mask_seed():
    """Parse optional MOE_DYNAMIC_MASK_SEED for deterministic random masking."""
    raw = os.environ.get("MOE_DYNAMIC_MASK_SEED", "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except ValueError:
        print(
            f"[DYNAMIC_MASK] Warning: invalid MOE_DYNAMIC_MASK_SEED='{raw}', ignoring"
        )
        return None
    print(f"[DYNAMIC_MASK] Dynamic mask seed: {seed}")
    return seed


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.config.num_moe_experts, self.config.hidden_size),
                dtype=torch.float32,
            )
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == "cpu":
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == "fp32":
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == "fp64":
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(
            input.to(router_dtype), self.weight.to(router_dtype)
        )
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        E = self.config.num_moe_experts
        self.moe_sigma = torch.eye(E, dtype=torch.float32)  # placeholder
        self.moe_mu = torch.zeros(E, self.config.hidden_size, dtype=torch.float32)

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "expert_bias",
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None
        self._moe_mask_experts = _parse_moe_mask_experts()

        # Dynamic cooccurrence mask (disabled by default)
        self._dynamic_mask_mode = _parse_dynamic_mask_mode()
        self._cooccurrence_matrices = _load_cooccurrence_matrices()
        self._dynamic_mask_seed = _parse_dynamic_mask_seed()
        self._dynamic_mask_rng = {}

        # OTEP warm-up: use standard topk during warm-up period
        self._otep_warmup_fraction = getattr(config, "moe_otep_warmup_fraction", 0.01)
        # Use register_buffer so _otep_batch_count is part of model state (but not saved to checkpoint)
        self.register_buffer(
            "_otep_batch_count", torch.tensor(0, dtype=torch.long), persistent=False
        )
        self._otep_warmup_batches = None  # will be set on first forward

    def _is_in_otep_warmup(self) -> bool:
        """Check if we are still in OTEP warm-up period.

        Only applies during training. Inference always uses OTEP.
        """
        if not self.training:
            return False
        if self._otep_warmup_fraction <= 0:
            return False
        # Lazy init: compute warmup batches from train_iters on first call
        if self._otep_warmup_batches is None:
            try:
                from megatron.core.num_microbatches_calculator import (
                    get_num_microbatches,
                )
                from megatron.training.global_vars import get_args

                args = get_args()
                train_iters = getattr(args, "train_iters", None) or 0
                num_mbs = get_num_microbatches()
                # Convert train_iters (optimizer steps) to microbatch count
                self._otep_warmup_batches = int(
                    train_iters * self._otep_warmup_fraction * num_mbs
                )
            except Exception:
                self._otep_warmup_batches = 0
        return self._otep_batch_count.item() <= self._otep_warmup_batches

    def _apply_expert_mask(self, logits: torch.Tensor) -> torch.Tensor:
        if self.layer_number is None or not self._moe_mask_experts:
            return logits
        mask = self._moe_mask_experts.get(self.layer_number)
        if not mask:
            return logits
        num_experts = logits.shape[-1]
        masked = [idx for idx in mask if 0 <= idx < num_experts]
        # Debug: print once per layer
        if not getattr(self, "_mask_debug_printed", False):
            print(
                f"[MOE_MASK] Layer {self.layer_number}: masking {len(masked)} experts: {sorted(masked)}"
            )
            self._mask_debug_printed = True
        neg_inf = torch.finfo(logits.dtype).min
        logits[:, masked] = neg_inf
        return logits

    def _apply_dynamic_cooccurrence_mask(
        self,
        scores: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> tuple:
        """Apply dynamic cooccurrence-based mask to routing results (vectorized).

        For each token, among its selected top-k experts:
        - cooccurrence mode: mask the expert in the highest cooccurrence pair
        - random mode: mask a random expert

        Args:
            scores: [num_tokens, num_experts] routing weights
            routing_map: [num_tokens, num_experts] bool, True for selected experts

        Returns:
            (scores, routing_map) with one expert masked per token (k -> k-1)
        """
        if not self._dynamic_mask_mode:
            return scores, routing_map

        if self.layer_number is None:
            return scores, routing_map

        C_norm = self._cooccurrence_matrices.get(self.layer_number)
        if C_norm is None:
            return scores, routing_map

        device = scores.device
        num_tokens, num_experts = scores.shape
        topk = self.topk
        selected_counts = routing_map.sum(dim=1)

        if topk <= 1:
            return scores, routing_map

        # Convert cooccurrence matrix to tensor (cached)
        if not hasattr(self, "_C_norm_tensor") or self._C_norm_tensor is None:
            self._C_norm_tensor = {}
        if self.layer_number not in self._C_norm_tensor:
            self._C_norm_tensor[self.layer_number] = torch.from_numpy(C_norm).to(
                device=device, dtype=scores.dtype
            )
        C_tensor = self._C_norm_tensor[self.layer_number]

        if not getattr(self, "_dynamic_mask_underfull_warned", False):
            underfull = selected_counts < topk
            if underfull.any():
                num_underfull = int(underfull.sum().item())
                print(
                    f"[DYNAMIC_MASK] Layer {self.layer_number}: {num_underfull}/{num_tokens} "
                    f"tokens have <{topk} selected experts; skipping dynamic mask for those tokens"
                )
                self._dynamic_mask_underfull_warned = True

        # Get top-k indices for each token [T, k]
        # Use scores to get indices (nonzero positions in routing_map)
        masked_scores = scores.masked_fill(~routing_map, float("-inf"))
        topk_scores, top_indices = masked_scores.topk(topk, dim=1)  # [T, k]
        valid = torch.isfinite(topk_scores)
        mask_tokens = selected_counts >= 2

        if self._dynamic_mask_mode == "cooccurrence":
            # Vectorized cooccurrence mask:
            # 1. Build [T, k, k] cooccurrence submatrices
            # 2. Find max off-diagonal element per token
            # 3. Mask the j expert from max pair

            # Gather cooccurrence values: C[top_indices[t,i], top_indices[t,j]]
            # Use advanced indexing: C_tensor[top_indices] -> [T, k, E]
            # Then gather again for second dimension
            idx_i = top_indices.unsqueeze(2).expand(-1, -1, topk)  # [T, k, k]
            idx_j = top_indices.unsqueeze(1).expand(-1, topk, -1)  # [T, k, k]

            # Gather: sub_C[t, i, j] = C_tensor[top_indices[t,i], top_indices[t,j]]
            sub_C = C_tensor[idx_i, idx_j]  # [T, k, k]

            # Mask diagonal (set to -inf)
            diag_mask = torch.eye(topk, device=device, dtype=torch.bool).unsqueeze(0)
            sub_C = sub_C.masked_fill(diag_mask, float("-inf"))

            # Mask invalid (non-selected) positions
            valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
            sub_C = sub_C.masked_fill(~valid_pairs, float("-inf"))

            # Find argmax in flattened [T, k*k] -> get j index
            flat_max_idx = sub_C.view(num_tokens, -1).argmax(dim=1)  # [T]
            j_local = flat_max_idx % topk  # [T], local index within top-k

            # Get the actual expert index to mask
            experts_to_mask = top_indices.gather(1, j_local.unsqueeze(1)).squeeze(
                1
            )  # [T]

        elif self._dynamic_mask_mode == "random":
            # Vectorized random mask:
            # Generate random index [0, topk) for each token
            rng = None
            if self._dynamic_mask_seed is not None:
                rng = self._dynamic_mask_rng.get(device)
                if rng is None:
                    rng = torch.Generator(device=device)
                    rng.manual_seed(self._dynamic_mask_seed)
                    self._dynamic_mask_rng[device] = rng

            if rng is None:
                rand_uniform = torch.rand(num_tokens, device=device)
            else:
                rand_uniform = torch.rand(num_tokens, device=device, generator=rng)
            rand_idx = (
                (rand_uniform * selected_counts.clamp(min=1).to(rand_uniform.dtype))
                .floor()
                .long()
            )

            # Get the actual expert index to mask
            experts_to_mask = top_indices.gather(1, rand_idx.unsqueeze(1)).squeeze(
                1
            )  # [T]

        else:
            return scores, routing_map

        # Build mask_to_remove using scatter
        mask_to_remove = torch.zeros(
            num_tokens, num_experts, dtype=torch.bool, device=device
        )
        mask_to_remove.scatter_(1, experts_to_mask.unsqueeze(1), True)
        mask_to_remove &= mask_tokens.unsqueeze(1)

        # Apply mask
        new_routing_map = routing_map & ~mask_to_remove
        new_scores = scores.clone()
        new_scores[mask_to_remove] = 0.0

        # Renormalize scores
        score_sums = new_scores.sum(dim=1, keepdim=True)
        new_scores = new_scores / (score_sums + 1e-9)

        # Debug: print once per layer
        if not getattr(self, "_dynamic_mask_debug_printed", False):
            masked_count = mask_to_remove.sum().item()
            print(
                f"[DYNAMIC_MASK] Layer {self.layer_number}: mode={self._dynamic_mask_mode}, "
                f"masked {masked_count}/{num_tokens} tokens"
            )
            self._dynamic_mask_debug_printed = True

        return new_scores, new_routing_map

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
            return logits

        assert self.config.moe_aux_loss_coeff == 0, (
            "Sinkhorn routing does not support aux loss."
        )
        if self.training:
            with torch.no_grad():  # Used to wrap code segments that need to disable gradient computation
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.
        Define an auxiliary loss switch_load_balancing_loss_func that measures the imbalance of expert usage,
        then merge this loss into model training in apply_load_balancing_loss
        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        # Consider expert capacity (capacity limit), i.e., each expert receives at most capacity_factor * (num_tokens / num_experts) tokens, to avoid overloading individual experts
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def seq_aux_loss_load_balancing(
        self, logits: torch.Tensor, bsz: int, seq_length: int
    ):
        """Apply loss-based load balancing to the logits tensor."""

        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=routing_map,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        if moe_aux_loss_coeff == 0:
            return activation
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        elif parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            sequence_partition_group = (
                parallel_state.get_tensor_and_context_parallel_group()
            )

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group,
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss",
                z_loss / moe_z_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        logits = self._apply_expert_mask(logits)

        if self.routing_type == "sinkhorn":  # default is aux_loss
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map = self.seq_aux_loss_load_balancing(
                logits, bsz, seq_length
            )
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        # elif self.routing_type == "otep":
        #     # ---------- 1) First use standard top-k Softmax with capacity ----------
        #     scores, routing_map, _ = topk_softmax_with_capacity(
        #         logits,
        #         self.topk,
        #         capacity_factor=self.config.moe_expert_capacity_factor,
        #         pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        #         drop_policy=self.config.moe_token_drop_policy,
        #         use_pre_softmax=self.config.moe_router_pre_softmax,
        #         num_groups=self.config.moe_router_num_groups,
        #         group_topk=self.config.moe_router_group_topk,
        #         scaling_factor=self.config.moe_router_topk_scaling_factor,
        #         deterministic_mode=self.config.deterministic_mode,
        #         score_function=self.score_function,
        #         expert_bias=self.expert_bias,
        #     )  # scores:[T,E] routing_map:[T,E]
        #
        #     # # ---------- 2) Call OTEP per-token to get new routing ----------
        #     # if self.moe_sigma is None:
        #     #     raise RuntimeError("OTEP routing selected but `self.moe_sigma` is None. "
        #     #                        "Please inject sigma from MoELayer statistics.")
        #     # sigma_np = self.moe_sigma.cpu().numpy()  # (E,E) only copy once
        #     #
        #     # T, E = scores.shape
        #     # device = scores.device
        #     # new_routing = torch.zeros_like(routing_map)  # Bool 0
        #     #
        #     # for t in range(T):  # per token
        #     #     mu_np = scores[t].detach().cpu().numpy()  # (E,)
        #     #     sel_vec = ot_based_ensemble_pruning(mu_np, sigma_np, k=self.topk)  # ndarray 0/1
        #     #     mask_t = torch.as_tensor(sel_vec, dtype=torch.bool, device=device)  # (E,)
        #     #     new_routing[t] = mask_t
        #     #
        #     # routing_map = new_routing  # Update routing
        #     # batched version
        #     routing_map = otep_batched(scores, self.moe_sigma.to(scores.device, dtype=scores.dtype), self.topk)
        #     scores = scores * routing_map

        elif self.routing_type == "otep":
            # OTEP warm-up: use standard topk during warm-up period
            # Only count during first forward pass (not recompute in backward)
            # In Megatron checkpoint: first forward has grad disabled, recompute has grad enabled
            if self.training and not torch.is_grad_enabled():
                self._otep_batch_count += 1
            in_warmup = self._is_in_otep_warmup()

            if in_warmup:
                # During warm-up, use standard aux_loss routing (same as aux_loss branch)
                scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
                    logits,
                    self.topk,
                    capacity_factor=self.config.moe_expert_capacity_factor,
                    pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                    drop_policy=self.config.moe_token_drop_policy,
                    use_pre_softmax=self.config.moe_router_pre_softmax,
                    num_groups=self.config.moe_router_num_groups,
                    group_topk=self.config.moe_router_group_topk,
                    scaling_factor=self.config.moe_router_topk_scaling_factor,
                    deterministic_mode=self.config.deterministic_mode,
                    score_function=self.score_function,
                    expert_bias=self.expert_bias,
                )

                # Compute probs_full for consistency with post-warmup phase
                probs_full = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)

                if self.training and self.config.moe_aux_loss_coeff:
                    probs_for_aux = torch.softmax(logits, dim=-1, dtype=torch.float32)
                    aux_loss_func = partial(
                        switch_load_balancing_loss_func,
                        probs=probs_for_aux,
                        tokens_per_expert=tokens_per_expert,
                        topk=self.topk,
                    )
                    scores = self.apply_load_balancing_loss(
                        activation=scores, load_balancing_loss_func=aux_loss_func
                    )
            else:
                # After warm-up, use OTEP routing
                # 0) Use "full logits" to get probabilities, feed to OTEP to generate K-candidate set for each token
                probs_full = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )  # [T,E]
                otep_mask = otep_batched(
                    probs_full,
                    self.moe_sigma.to(probs_full.device, dtype=probs_full.dtype),
                    self.topk,  # Exactly K True values per row
                )  # [T,E] bool

                # 1) Hard mask experts not in OTEP candidate set:
                #    - Select top-k (this step can only select from OTEP's K candidates after masking)
                #    - Normalize to get routing weights
                #    - Apply capacity constraints (capacity_factor)
                #    - Drop excess tokens according to drop_policy (don't "reassign" to k+1)
                #    - Optional pad_to_capacity
                masked_logits = logits.masked_fill(~otep_mask, float("-inf"))

                scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
                    masked_logits,
                    self.topk,
                    capacity_factor=self.config.moe_expert_capacity_factor,
                    pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                    drop_policy=self.config.moe_token_drop_policy,
                    use_pre_softmax=self.config.moe_router_pre_softmax,
                    num_groups=self.config.moe_router_num_groups,
                    group_topk=self.config.moe_router_group_topk,
                    scaling_factor=self.config.moe_router_topk_scaling_factor,
                    deterministic_mode=self.config.deterministic_mode,
                    score_function=self.score_function,
                    expert_bias=self.expert_bias,
                )
                # notes:
                # - scores: [T,E], at most K non-zero per row (after capacity/drop, excess set to 0)
                # - routing_map: [T,E] bool, final assignment after capacity/drop (excess is False)
                # - tokens_per_expert: [E], number of tokens each expert finally receives (excluding pad)

                # 2) Add MoE aux-loss (consistent with "aux_loss" branch)
                if self.training and self.config.moe_aux_loss_coeff:
                    # Use softmax of "unmasked original logits" as probs (consistent with official implementation)
                    probs_for_aux = torch.softmax(logits, dim=-1, dtype=torch.float32)
                    aux_loss_func = partial(
                        switch_load_balancing_loss_func,  # If you want sample-level seq-aux-loss, switch to sequence_load_balancing_loss_func
                        probs=probs_for_aux,
                        tokens_per_expert=tokens_per_expert,
                        topk=self.topk,
                    )
                    scores = self.apply_load_balancing_loss(
                        activation=scores,  # Attach load balancing loss to the routing weights used for weighting
                        load_balancing_loss_func=aux_loss_func,
                    )

        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        # Apply dynamic cooccurrence mask (if enabled)
        # This masks one expert per token based on cooccurrence or random selection
        scores, routing_map = self._apply_dynamic_cooccurrence_mask(scores, routing_map)

        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        if self.routing_type == "otep":
            return scores, routing_map, probs_full
        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.routing_type == "otep":
            scores, routing_map, probs_full = self.routing(logits)
            return scores, routing_map, probs_full

        scores, routing_map = self.routing(logits)
        return scores, routing_map
