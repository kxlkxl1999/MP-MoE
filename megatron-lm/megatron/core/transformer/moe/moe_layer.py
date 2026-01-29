# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import FunctionType
from typing import Optional, Union
from megatron.core import parallel_state as mpu
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import (
    MoEAlltoAllSEQTokenDispatcher,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

TokenDispatcherType = Union[
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEAlltoAllSEQTokenDispatcher,
    MoEFlexTokenDispatcher,
]

ModuleLike = Union[nn.Module, FunctionType]


def _resolve_figures_root() -> Path:
    """Locate FLAME-MoE repo root so Figures live beside Megatron-LM."""
    module_path = Path(__file__).resolve()
    for ancestor in module_path.parents:
        if ancestor.name == "Megatron-LM":
            return ancestor.parent / "Figures"
    # Fallback: stay relative to current module if Megatron-LM isn't found
    return module_path.parent / "Figures"


_FIGURES_ROOT = _resolve_figures_root()
PLOT_DIR = _FIGURES_ROOT / datetime.now().strftime("plot_%Y%m%d_%H%M%S_MOE")
PLOT_STATS_DIR = PLOT_DIR / "stats"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_STATS_DIR.mkdir(parents=True, exist_ok=True)


def _is_log_rank_dp0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True  # Allow output in single process / uninitialized state

    try:
        # Only check data parallel dimension, ensure each DP group writes only once
        return mpu.get_data_parallel_rank() == 0
    except Exception:
        # Fallback: degrade to global rank0
        return dist.get_rank() == 0


def _print_rank0(message: str):
    if not dist.is_available() or not dist.is_initialized():
        print(message, flush=True)
        return
    if dist.get_rank() == 0:
        print(message, flush=True)


def _dump_moe_stats_to_disk(
    *,
    layer_num: int,
    segment_idx: int,
    model_type: str,
    expert_counts,
    cooccurrence_matrix,
    extra_payload: Optional[dict] = None,
) -> str:
    """Persist aggregated routing statistics for offline plotting."""

    base_name = f"layer{layer_num}_seg{segment_idx}_{model_type}"
    filename = PLOT_STATS_DIR / f"{base_name}.npz"

    payload = {
        "expert_counts": np.array(expert_counts, copy=True),
        "cooccurrence_matrix": np.array(cooccurrence_matrix, copy=True),
        "timestamp": np.array(datetime.now().isoformat()),
    }
    if extra_payload:
        payload.update(extra_payload)

    np.savez_compressed(str(filename), **payload)
    return str(filename)


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Optional[Union[ModuleSpec, type]] = None
    shared_experts: Optional[Union[ModuleSpec, type]] = None


@dataclass
class _RoutingDumpState:
    cnts: int
    rank: int
    dump_dir: Path


class BaseMoELayer(MegatronModule, ABC):
    TRACKED_LAYERS = (2, 5, 9)
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.submodules: MoESubmodules = submodules or MoESubmodules()
        self.expert_parallel_size = (
            parallel_state.get_expert_model_parallel_world_size()
        )
        assert self.expert_parallel_size > 0, (
            "Expected non-negative expert parallel size"
        )

        num_moe_experts = self.config.num_moe_experts
        if num_moe_experts is None:
            raise RuntimeError("num_moe_experts must be set in config")
        self.num_moe_experts: int = int(num_moe_experts)

        assert self.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts: int = self.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = (
            self.config.moe_shared_expert_intermediate_size is not None
        )
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.num_moe_experts, self.local_expert_indices))
        self.moe_layer_recompute = config.moe_layer_recompute
        self.router: Optional[TopKRouter] = None
        self.experts: Optional[ModuleLike] = None
        self.shared_experts: Optional[ModuleLike] = None
        self.token_dispatcher: Optional[TokenDispatcherType] = None
        self.layer_number = layer_number
        self._dump_state: Optional[_RoutingDumpState] = None

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        if self.router is not None:
            self.router.set_layer_number(layer_number)

    def _build_router(self):
        """Initialize router."""
        return TopKRouter(config=self.config)

    def _build_token_dispatcher(self):
        """Initialize token dispatcher."""
        # Four choices only affect data communication process, not the return results
        # Default is allgather; official also recommends using alltoall when Expert Parallelism (EP) is enabled.
        dispatcher_type = self.config.moe_token_dispatcher_type
        args = (self.num_local_experts, self.local_expert_indices)
        if dispatcher_type == "allgather":
            return MoEAllGatherTokenDispatcher(*args, config=self.config)
        if dispatcher_type == "alltoall":
            return MoEAlltoAllTokenDispatcher(*args, config=self.config)
        if dispatcher_type == "alltoall_seq":
            return MoEAlltoAllSEQTokenDispatcher(*args, config=self.config)
        if dispatcher_type == "flex":
            return MoEFlexTokenDispatcher(*args, config=self.config)
        raise ValueError(f"Unsupported token dispatcher type: {dispatcher_type}")

    def _build_experts(self) -> ModuleLike:
        """Initialize experts."""
        if self.submodules is None or self.submodules.experts is None:
            raise RuntimeError("MoE experts specification is missing.")
        return build_module(
            self.submodules.experts, self.num_local_experts, self.config
        )

    def _build_shared_experts(self) -> Optional[ModuleLike]:
        """Initialize shared experts."""
        if not self.use_shared_expert or self.submodules is None:
            return None
        if self.submodules.shared_experts is None:
            return None
        shared = build_module(self.submodules.shared_experts, config=self.config)
        if self.shared_expert_overlap and self.token_dispatcher is not None:
            self.token_dispatcher.set_shared_experts(shared)
        return shared

    def _ensure_dump_state(self) -> _RoutingDumpState:
        """Ensure dump state is initialized."""
        if self._dump_state is not None:
            return self._dump_state
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed backend must be initialized for dump.")
        rank = torch.distributed.get_rank()
        dump_dir = Path(os.environ["EACT_SAVE"], str(self.layer_number))
        dump_dir.mkdir(parents=True, exist_ok=True)
        self._dump_state = _RoutingDumpState(cnts=0, rank=rank, dump_dir=dump_dir)
        return self._dump_state

    def _track_stats_this_layer(self) -> bool:
        """Check if stats should be tracked for this layer."""
        return getattr(self, "layer_number", None) in self.TRACKED_LAYERS

    def _init_moe_stats_buffers_if_needed(self):
        """Initialize MoE stats buffers if needed.
        Every tracked layer creates these buffers once."""
        if not self._track_stats_this_layer() or hasattr(self, "expert_usage_counter"):
            return
        E_global = int(self.num_moe_experts)
        self.expert_usage_counter = np.zeros(E_global, dtype=np.int64)
        self.expert_cooccurrence_counter = np.zeros(
            (E_global, E_global), dtype=np.int64
        )
        # Stats counting is tracked in optimizer-step units (not per-forward).
        # We still observe routing per forward/microbatch and accumulate into the
        # numpy buffers above, but we only advance the "step counter" once we've
        # completed `get_num_microbatches()` training forwards.
        self._stats_microbatches_in_step = 0
        self._stats_step_count = 0
        self._stats_segment_idx = 0

    def _get_stats_window_size(self):
        """Get the stats window size in optimizer steps.

        `--moe-plot-every` is defined in *optimizer-step* units (global steps).
        We convert training microbatch-forwards into step counts separately.
        """
        return max(1, int(getattr(self.config, "moe_plot_every", 200)))

    def _should_count_stats_this_forward(self) -> bool:
        """Decide whether to count stats in this forward pass.

        Defaults is closed. Appending --moe-layer-recompute to open it to lower memory usage."""
        return (not self.moe_layer_recompute) or (
            self.moe_layer_recompute and (not torch.is_grad_enabled())
        )

    def _maybe_track_routing_stats(self, routing_map: torch.Tensor):
        """Track routing statistics on the local rank and convert GPU tensors to CPU NumPy arrays."""
        self._init_moe_stats_buffers_if_needed()
        with torch.no_grad():
            token_per_expert = routing_map.sum(dim=0)
            route_float = routing_map.float()
            co_mat = torch.matmul(route_float.t(), route_float)

        # Skip stats collection and return directly (e.g., not in tracked layers)
        if not (
            self._track_stats_this_layer() and hasattr(self, "expert_usage_counter")
        ):
            return token_per_expert, co_mat

        # Only count *training* microbatches towards optimizer-step windows.
        # This avoids:
        # - eval/inference forwards skewing the expected frequency,
        # - --moe-layer-recompute causing double-counting (the recompute forward runs again).
        should_advance_counters = (
            self.training and self._should_count_stats_this_forward()
        )

        if not should_advance_counters:
            return token_per_expert, co_mat

        # NOTE: Stats logic must execute before return
        self.expert_usage_counter += token_per_expert.cpu().numpy().astype(np.int64)
        self.expert_cooccurrence_counter += co_mat.cpu().numpy().astype(np.int64)

        self._stats_microbatches_in_step += 1
        try:
            num_mbs = int(get_num_microbatches())
        except Exception:
            num_mbs = 1
        if self._stats_microbatches_in_step >= max(1, num_mbs):
            self._stats_microbatches_in_step = 0
            self._stats_step_count += 1

        return token_per_expert, co_mat

    def _reduce_moe_stats_for_plot(self, device):
        """Collect all rank stats, and return summed numpy arrays

        Unconditionally enters collective; returns (u_sum_np, C_sum_np), does not directly overwrite self.*"""
        E_global = int(self.num_moe_experts)
        if not hasattr(self, "expert_usage_counter"):
            u_local = torch.zeros(E_global, dtype=torch.int32, device=device)
            C_local = torch.zeros(E_global, E_global, dtype=torch.int32, device=device)
        else:
            u_local = torch.as_tensor(
                self.expert_usage_counter, device=device, dtype=torch.int32
            )
            C_local = torch.as_tensor(
                self.expert_cooccurrence_counter, device=device, dtype=torch.int32
            )
        u_sum = u_local.clone()
        C_sum = C_local.clone()
        if mpu.get_expert_model_parallel_world_size() > 1:  # EP
            ep_group = mpu.get_expert_model_parallel_group()
            torch.distributed.all_reduce(
                u_sum, op=torch.distributed.ReduceOp.SUM, group=ep_group
            )
            torch.distributed.all_reduce(
                C_sum, op=torch.distributed.ReduceOp.SUM, group=ep_group
            )
        if mpu.get_tensor_model_parallel_world_size() > 1:  # TP
            tp_group = mpu.get_tensor_model_parallel_group()
            torch.distributed.all_reduce(
                u_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
            torch.distributed.all_reduce(
                C_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
        return u_sum.int().cpu().numpy(), C_sum.int().cpu().numpy()

    def _handle_moe_stats_plotting(
        self,
        model_type: str,
        device: torch.device,
        extra_message: str = "",
        extra_rank0_dump_fn=None,
    ):
        """Save stats for plotting.

        Args:
            model_type: str, "moe" or "mope"
            device: torch.device, device to perform all_reduce
            extra_message: str, extra message to print on rank 0
            extra_rank0_dump_fn: Optional function returning extra payload dict for dump
        """
        if not (
            self._track_stats_this_layer() and hasattr(self, "expert_usage_counter")
        ):
            return
        if not self.training:
            return
        do_plot = self._stats_step_count >= self._get_stats_window_size()
        if not do_plot:
            return
        u_sum_np, C_sum_np = self._reduce_moe_stats_for_plot(device=device)
        if _is_log_rank_dp0():
            window = self._get_stats_window_size()
            tag = model_type.upper()
            message = (
                f"[{tag}] trigger plot at layer {self.layer_number}: "
                f"steps={self._stats_step_count}, window={window}"
            )
            if extra_message:
                message = f"{message}, {extra_message}"
            _print_rank0(
                message
            )  # _is_log_rank_dp0 has guarantee that only rank 0 in DP group will print

            self._stats_segment_idx += 1
            extra_payload = None
            if extra_rank0_dump_fn is not None:
                extra_payload = extra_rank0_dump_fn(self._stats_segment_idx)
            if self.layer_number is None:
                raise RuntimeError("MoE layer_number must be set to dump stats")
            stats_path = _dump_moe_stats_to_disk(
                layer_num=int(self.layer_number),
                segment_idx=self._stats_segment_idx,
                model_type=model_type,
                expert_counts=u_sum_np,
                cooccurrence_matrix=C_sum_np,
                extra_payload=extra_payload,
            )
            _print_rank0(f"[{tag}] saved stats to {stats_path}")
        if hasattr(self, "expert_usage_counter"):
            self.expert_usage_counter.fill(0)
            self.expert_cooccurrence_counter.fill(0)
        self._stats_microbatches_in_step = 0
        self._stats_step_count = 0


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ):
        super(MoELayer, self).__init__(
            config=config, submodules=submodules, layer_number=layer_number
        )
        self._init_moe_stats_buffers_if_needed()
        self.router = self._build_router()
        self.experts = self._build_experts()
        self.token_dispatcher = self._build_token_dispatcher()
        self.shared_experts = self._build_shared_experts()

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # Input hidden_states: shape = [num_tokens, hidden_size] or [batch_size * seq_len, hidden_size];
        # Returns output: same shape as input, token representations after expert processing
        # process MoE
        def custom_forward(hidden_states):
            if (
                self.router is None
                or self.token_dispatcher is None
                or self.experts is None
            ):
                raise RuntimeError("MoE layer modules are not initialized.")

            # routing
            # probs: shape = [num_tokens, num_experts] probability for each expert
            # routing_map: shape = [num_tokens, num_experts] 0-1 matrix indicating which experts each token is assigned to
            probs, routing_map = self.router(hidden_states)
            self._maybe_track_routing_stats(routing_map)

            # capture the activated expert ids
            if not self.training and self.config.test_mode:
                dump_state = self._ensure_dump_state()
                values, indices = torch.topk(probs, k=self.config.moe_router_topk)
                torch.save(
                    (values, indices),
                    dump_state.dump_dir / f"{dump_state.cnts}-{dump_state.rank}.pt",
                )
                dump_state.cnts += 1

            # dispatch tokens to experts
            # dispatched_input: shape = [num_experts, capacity, hidden_size];
            # tokens_per_expert: list of length num_experts, indicating the number of tokens each expert receives.
            (dispatched_input, tokens_per_expert) = (
                self.token_dispatcher.token_permutation(
                    hidden_states, probs, routing_map
                )
            )

            # experts forward
            # expert_output: shape = [num_experts, capacity, hidden_size]
            # mlp_bias: bias from expert output (used for residual/fusion)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)

            # token unpermutation
            # output restored to [num_tokens, hidden_size] or [batch_size * seq_len, hidden_size]
            output, mlp_bias = self.token_dispatcher.token_unpermutation(
                expert_output, mlp_bias
            )

            # shared expert forward
            if self.use_shared_expert and not self.shared_expert_overlap:
                if self.shared_experts is None:
                    raise RuntimeError("Shared expert module is not initialized.")
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)

            self._handle_moe_stats_plotting(
                model_type="moe", device=hidden_states.device
            )
            return output, mlp_bias

        if self.moe_layer_recompute:
            # Equivalent to PyTorch's torch.utils.checkpoint: forward pass doesn't save intermediate activations,
            # backward pass re-runs custom_forward to retrieve activations, significantly reducing activation memory
            # The checkpointed custom_forward will be executed again during backward. Therefore, if there is code
            # with "side effects" inside (such as logging counts, writing routing results to disk, accumulating aux loss),
            # care must be taken to avoid duplication. Megatron issues have reported that under --moe-layer-recompute,
            # the load balancing loss can be accumulated twice
            result = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
            if result is None:
                raise RuntimeError("MoE checkpoint returned no outputs.")
            output, mlp_bias = result
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


class MoPELayer(BaseMoELayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ):
        super(MoPELayer, self).__init__(
            config=config, submodules=submodules, layer_number=layer_number
        )
        # print_rank_0(f"[MoPE] __init__ => building MoPELayer(layer={self.layer_number})")
        self.moe_layer_recompute = config.moe_layer_recompute

        # Statistics products: mean matrix/covariance matrix, mu in [E_global, H]; sigma in [E_global, E_global]
        E_global = (
            self.num_local_experts
            * parallel_state.get_expert_model_parallel_world_size()
        )
        # self.normalize_sigma = False  # Whether to normalize covariance matrix to correlation matrix
        # Initialize to all zeros (CPU, FP32)
        self.moe_mu = torch.zeros(
            E_global, self.config.hidden_size, dtype=torch.float32, device="cpu"
        )
        self.moe_sigma = torch.eye(E_global, dtype=torch.float32, device="cpu")
        self.moe_sigma_inv = torch.eye(E_global, dtype=torch.float32, device="cpu")
        self.moe_sigma_std = torch.eye(E_global, dtype=torch.float32, device="cpu")

        # Internal counting and trigger flags
        self._moe_batch_count: int = 0
        self._moe_capture_next: bool = True
        self._cov_eps: float = 1e-12

        # Use co-occurrence matrix to construct sigma; when enabled, no longer perform dense forward to compute sigma
        self.moe_sigma_from_cooccurrence: bool = True

        # Global (for this layer) online counting: expert usage and co-occurrence (stored in CPU/FP32 or Long)
        self._coocc_counts = torch.zeros(
            E_global, E_global, dtype=torch.long, device="cpu"
        )  # C in [E,E]
        self._usage_counts = torch.zeros(
            E_global, dtype=torch.long, device="cpu"
        )  # u in [E]
        self._coocc_total_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu"
        )  # Cumulative token count

        # If you want to update to router by window, you can reuse the existing plotting window parameters
        self._sigma_update_every_forwards = int(
            getattr(self.config, "moe_sigma_update_every", 1) or 1
        )
        self._sigma_forward_counter = 0
        # Numerical stability term (can also inherit your existing _cov_eps)
        self._sigma_eps = 1e-12

        self._init_moe_stats_buffers_if_needed()
        self.router = self._build_router()
        self.token_dispatcher = self._build_token_dispatcher()
        self.experts = self._build_experts()
        self.shared_experts = self._build_shared_experts()

    def _aggregate_counts_for_sigma(self, device: torch.device):
        """For sigma, move CPU counts to device for all_reduce, then move back to CPU. Compatible with EP/TP aggregation."""
        C = self._coocc_counts.to(device, non_blocking=True)
        u = self._usage_counts.to(device, non_blocking=True)
        N = torch.tensor(
            [int(self._coocc_total_tokens.item())], dtype=torch.long, device=device
        )

        # EP aggregation
        if mpu.get_expert_model_parallel_world_size() > 1:
            ep_group = mpu.get_expert_model_parallel_group()
            torch.distributed.all_reduce(
                C, op=torch.distributed.ReduceOp.SUM, group=ep_group
            )
            torch.distributed.all_reduce(
                u, op=torch.distributed.ReduceOp.SUM, group=ep_group
            )
            torch.distributed.all_reduce(
                N, op=torch.distributed.ReduceOp.SUM, group=ep_group
            )

        # TP aggregation
        if mpu.get_tensor_model_parallel_world_size() > 1:
            tp_group = mpu.get_tensor_model_parallel_group()
            torch.distributed.all_reduce(
                C, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
            torch.distributed.all_reduce(
                u, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )
            torch.distributed.all_reduce(
                N, op=torch.distributed.ReduceOp.SUM, group=tp_group
            )

        return C.cpu(), u.cpu(), int(N.item())

    def _update_sigma_from_cooccurrence(self, device: torch.device, dtype: torch.dtype):
        """Use cumulative co-occurrence counts C, usage counts u, to compute Sigma = C/N - (u u^T)/N^2, and push to router."""
        C_cpu, u_cpu, N = self._aggregate_counts_for_sigma(device=device)
        if N <= 0:
            return  # No tokens collected yet

        # Second moment & baseline (independence assumption)
        C = C_cpu.to(torch.float32)  # [E,E]
        u = u_cpu.to(torch.float32)  # [E]
        Nf = float(N)

        second_moment = C / Nf  # C/N
        baseline = torch.outer(u, u) / (Nf * Nf)  # (u u^T) / N^2 = p p^T

        sigma = second_moment - baseline  # Covariance-style "excess co-occurrence"

        # Numerical and symmetry protection
        sigma = 0.5 * (sigma + sigma.t())  # Symmetrize
        sigma.diagonal().clamp_(min=self._sigma_eps)  # Floor diagonal to avoid 0/negative

        # sigma_for_inv = sigma.clone()
        # sigma_for_inv.diagonal().clamp_(min=self._sigma_eps)  # Prevent 0
        # sigma_inv = torch.linalg.pinv(sigma_for_inv)  # Direct pseudo-inverse is sufficient
        # self.moe_sigma_inv = sigma_inv.to("cpu")

        # Store CPU main copy and sync to router (GPU)
        self.moe_sigma = sigma.to("cpu")
        if self.router is None:
            raise RuntimeError("Router is not initialized.")
        self.router.moe_sigma = sigma.to(device=device, dtype=dtype, non_blocking=True)

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # if not hasattr(self, "_dbg2"):
        #     print(f"[MoPE] forward(layer={self.layer_number})")
        #     self._dbg2 = True

        self._moe_batch_count += 1

        def custom_forward(hidden_states: torch.Tensor):
            if (
                self.router is None
                or self.token_dispatcher is None
                or self.experts is None
            ):
                raise RuntimeError("MoPE layer modules are not initialized.")
            # routing
            probs, routing_map, _ = self.router(hidden_states)

            # plot stats
            token_per_expert, co_mat = self._maybe_track_routing_stats(routing_map)

            # update sigma from cooccurrence
            # Only advance sigma statistics during training, and avoid double-counting
            # under --moe-layer-recompute (checkpointed forward is replayed in backward).
            should_advance_sigma = (
                self.training and self._should_count_stats_this_forward()
            )
            if (
                should_advance_sigma
                and token_per_expert is not None
                and co_mat is not None
            ):
                with torch.no_grad():
                    # cooccur sigma
                    T = routing_map.size(0)
                    token_per_expert_long = token_per_expert.to(torch.long)
                    co_mat_long = co_mat.to(torch.long)
                    # Accumulate to "layer-wide sigma driver counters" (CPU stores main copy)
                    self._usage_counts += token_per_expert_long.cpu()
                    self._coocc_counts += co_mat_long.cpu()
                    self._coocc_total_tokens += torch.tensor(T, dtype=torch.long)
                    # Control update frequency (update sigma every N forwards; default N=1)
                    self._sigma_forward_counter += 1
                    if self._sigma_forward_counter >= self._sigma_update_every_forwards:
                        self._update_sigma_from_cooccurrence(
                            device=hidden_states.device, dtype=torch.float32
                        )
                        self._sigma_forward_counter = 0

            # Inference test export
            if not self.training and self.config.test_mode:
                dump_state = self._ensure_dump_state()
                # Based on routing_map rather than top-k(probs)
                selected_indices = routing_map.nonzero(as_tuple=False)  # [N_sel, 2]
                selected_values = probs[selected_indices[:, 0], selected_indices[:, 1]]
                torch.save(
                    (selected_values.cpu(), selected_indices.cpu()),
                    dump_state.dump_dir / f"{dump_state.cnts}-{dump_state.rank}.pt",
                )
                dump_state.cnts += 1

            # dispatch tokens to experts
            dispatched_input, tokens_per_expert = (
                self.token_dispatcher.token_permutation(
                    hidden_states, probs, routing_map
                )
            )

            # experts forward
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)

            # MoPE extra visualization data
            def _collect_covariance_payload(segment_idx):
                if hasattr(self, "moe_sigma") and self.moe_sigma.numel() > 0:
                    return {"cov_matrix": self.moe_sigma.detach().cpu().numpy()}
                return None

            extra_msg = f"capture_next={self._moe_capture_next}"
            self._handle_moe_stats_plotting(
                model_type="mope",
                device=hidden_states.device,
                extra_message=extra_msg,
                extra_rank0_dump_fn=_collect_covariance_payload,
            )

            # token unpermutation
            output, mlp_bias = self.token_dispatcher.token_unpermutation(
                expert_output, mlp_bias
            )

            # shared expert forward
            # if shared_expert_overlap is True, expert computation has already been overlapped in dispatcher
            if self.use_shared_expert and not self.shared_expert_overlap:
                if self.shared_experts is None:
                    raise RuntimeError("Shared expert module is not initialized.")
                output = output + self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            result = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
            if result is None:
                raise RuntimeError("MoPE checkpoint returned no outputs.")
            output, mlp_bias = result
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
