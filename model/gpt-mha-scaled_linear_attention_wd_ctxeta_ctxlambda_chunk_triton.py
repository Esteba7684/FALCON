r"""Scaled linear attention (MHA) with chunk-wise ctxeta+ctxlambda decay (Triton + fallback).

This module matches `gpt-mha-scaled_linear_attention_wd_ctxeta_ctxlambda_chunk.py`
mathematically with recurrence:

    S_t = \gamma_t S_{t-1} + \eta_t k_write,t v_t^T
    y_t = S_t^T q_t

where:

    \beta_t = sigmoid(W_\beta x_t) * (2 if beta_scale_by_2 else 1)
    \lambda_t = sigmoid(W_\lambda x_t) * (lambda_scale * ||k_{t-1}||_2^2)
    \eta_t = \beta_t / (||k_write,t||_2^2 + \lambda_t + fast_weight_eta_eps)
    \gamma_t = clamp_min(1 - \eta_t \lambda_t, fast_weight_decay_floor)

Implementation path:
1. Prefer vendored LASD Triton (`ops/lasd_parallel_triton.py`) when CUDA+Triton
   are available.
2. Fall back to the chunk-local renormalization reference path from
   `gpt-mha-scaled_linear_attention_wd_ctxeta_ctxlambda_chunk.py` when Triton
   cannot run.

Training config selector:
    model_type = "gpt-mha-scaled_linear_attention_wd_ctxeta_ctxlambda_chunk_triton"
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from .activations import ActivationName
from .attention_dtype import AttentionDType
from .gpt_base import GPTBase
from .lasd_triton_utils import (
    LinearAttentionTritonDType,
    get_vendored_lasd_parallel_triton,
    is_triton_unavailable_error,
    resolve_triton_qkv_dtype,
    validate_linear_attention_triton_dtype,
)
from .pydantic_config import validate_pretrained_config_kwargs


def _load_base_module() -> ModuleType:
    return importlib.import_module("model.gpt-mha-scaled_linear_attention_wd_ctxeta_ctxlambda_chunk")


_BASE_MODULE = _load_base_module()
_BaseLinearSelfAttention = cast(type[nn.Module], getattr(_BASE_MODULE, "LinearSelfAttention"))
_BASE_GPT = cast(type[GPTBase], getattr(_BASE_MODULE, "GPT"))
_INIT_EXCLUDE_SUFFIXES = tuple(cast(tuple[str, ...], getattr(_BASE_GPT, "init_exclude_suffixes", ())))


class LinearSelfAttention(_BaseLinearSelfAttention):
    """Triton-enabled variant of the corresponding chunk attention."""

    def __init__(self, config: "GPTConfig") -> None:
        super().__init__(config)
        self.use_triton_lasd = bool(getattr(config, "use_triton_lasd", True))
        self.linear_attention_triton_dtype: LinearAttentionTritonDType = validate_linear_attention_triton_dtype(
            getattr(config, "linear_attention_triton_dtype", None)
        )
        self.linear_attention_triton_use_chunk_loop = bool(
            getattr(config, "linear_attention_triton_use_chunk_loop", True)
        )
        # Saving LASD states in autograd improves backward speed but increases peak memory.
        self.linear_attention_triton_save_states = bool(getattr(config, "linear_attention_triton_save_states", False))

    def _run_lasd_triton(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta_prime: torch.Tensor,
        log_gamma: torch.Tensor,
        cu_seqlens: torch.LongTensor | None,
    ) -> torch.Tensor | None:
        if not self.use_triton_lasd:
            return None
        if q.device.type != "cuda":
            return None

        lasd_parallel_triton = get_vendored_lasd_parallel_triton()
        if lasd_parallel_triton is None:
            return None

        triton_dtype = resolve_triton_qkv_dtype(
            q=q,
            linear_attention_triton_dtype=self.linear_attention_triton_dtype,
        )

        eta = beta_prime.to(dtype=torch.float32) * torch.exp(log_gamma.to(dtype=torch.float32))

        q_run = q
        k_run = k * eta.unsqueeze(-1)
        v_run = v
        if triton_dtype != q.dtype:
            q_run = q_run.to(dtype=triton_dtype)
            k_run = k_run.to(dtype=triton_dtype)
            v_run = v_run.to(dtype=triton_dtype)

        try:
            o, _ = lasd_parallel_triton(
                q=q_run.contiguous(),
                k=k_run.contiguous(),
                v=v_run.contiguous(),
                ld=log_gamma.contiguous(),
                initial_state=None,
                cu_seqlens=cu_seqlens,
                save_states=self.linear_attention_triton_save_states,
                use_chunk_loop=self.linear_attention_triton_use_chunk_loop,
            )
        except RuntimeError as exc:
            if not is_triton_unavailable_error(exc):
                raise
            return None

        return o.to(dtype=q.dtype)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.LongTensor | None = None) -> torch.Tensor:
        """Run Triton LASD core when available; otherwise use chunk fallback."""
        if not self.use_triton_lasd and cu_seqlens is None:
            return cast(torch.Tensor, super().forward(x))

        original = cast(Any, getattr(_BASE_MODULE, "run_chunk_local_renorm_with_runner"))

        def _patched_run_chunk_local_renorm_with_runner(*args: Any, **kwargs: Any) -> torch.Tensor:
            q = cast(torch.Tensor, kwargs["q"])
            k = cast(torch.Tensor, kwargs["k"])
            v = cast(torch.Tensor, kwargs["v"])
            beta = cast(torch.Tensor, kwargs["beta"])
            log_gamma = cast(torch.Tensor, kwargs["log_gamma"])

            o = self._run_lasd_triton(
                q=q,
                k=k,
                v=v,
                beta_prime=beta,
                log_gamma=log_gamma,
                cu_seqlens=cu_seqlens,
            )
            if o is not None:
                return o

            if cu_seqlens is not None:
                kwargs["cu_seqlens"] = cu_seqlens
            return cast(torch.Tensor, original(*args, **kwargs))

        setattr(_BASE_MODULE, "run_chunk_local_renorm_with_runner", _patched_run_chunk_local_renorm_with_runner)
        try:
            return cast(torch.Tensor, super().forward(x))
        finally:
            setattr(_BASE_MODULE, "run_chunk_local_renorm_with_runner", original)


@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "nanogpt-pro"
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_size: int = 768
    head_dim: int = 128
    block_size: int = 1024
    bias: bool = False
    dropout: float = 0.0
    using_groupnorm: bool = True

    # QKV activation knobs (applied by attention impls when supported)
    q_activation: ActivationName | None = "silu"
    k_activation: ActivationName | None = "silu"
    v_activation: ActivationName | None = "silu"
    attention_dtype: AttentionDType = "auto"

    # Linear attention knobs
    use_rope: bool = False
    rope_ratio: float = 1.0  # Apply RoPE on the first rope_ratio*head_dim dimensions (must be in [0, 1])
    rope_base: float = 10000.0
    shortconvq: bool = True
    shortconvk: bool = True
    shortconvv: bool = True
    shortconv_kernel_size: int = 4
    shortconv_shift_right1_k: bool = True
    fast_weight_next_latent: bool = True
    q_rmsnorm: bool = True
    k_rmsnorm: bool = True
    v_rmsnorm: bool = False
    q_rmsnorm_learnable: bool = False
    k_rmsnorm_learnable: bool = False
    v_rmsnorm_learnable: bool = False

    # Fast weight / decay knobs
    fast_weight_beta: float = 0.999  # initialization for beta logits (post-sigmoid)
    beta_scale_by_2: bool = True
    fast_weight_lambda_min: float = 0.01
    fast_weight_lambda_max: float = 0.01  # initialization range for lambda logits (post-sigmoid, pre-scale)
    fast_weight_logit_eps: float = 1e-5
    lambda_scale: float = 0.95
    fast_weight_eta_eps: float = 1e-5
    fast_weight_decay_floor: float = 0.01
    linear_attention_chunk_size: int = 128
    attention_norm_eps: float = 1e-5
    use_output_gate: bool = True

    # Triton knobs
    use_triton_lasd: bool = True
    linear_attention_triton_dtype: LinearAttentionTritonDType = "auto"
    linear_attention_triton_use_chunk_loop: bool = True
    linear_attention_triton_save_states: bool = True

    # SwiGLU hidden dimension multiplier (LLaMA uses 8/3).
    mlp_hidden_mult: float = 8 / 3

    # Init knobs
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.5

    # muP knobs (forward/logits scaling + optimizer LR grouping)
    mup: bool = False
    hidden_size_base: int = 1024
    embedding_lr_multiplier: float = 1.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**validate_pretrained_config_kwargs(type(self), kwargs))


class GPT(GPTBase):
    config_class = GPTConfig
    attention_cls = LinearSelfAttention
    init_exclude_suffixes = _INIT_EXCLUDE_SUFFIXES


__all__ = ["GPT", "GPTConfig", "LinearSelfAttention"]
