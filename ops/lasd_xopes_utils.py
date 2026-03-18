"""Minimal xopes utility subset for LASD Triton kernels.

This vendors only the helpers used by `ops/lasd_parallel_triton.py` and its
kernel dependencies.
"""

from __future__ import annotations

import functools
import itertools
import os
from functools import lru_cache
from typing import Any, Callable, TypeVar

import torch

from .triton_compat import triton

F = TypeVar("F", bound=Callable[..., Any])

MIN_BLOCK: int = 16
_VALID_NUM_WARPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@lru_cache(maxsize=None)
def _get_sm_count() -> int:
    if not torch.cuda.is_available():
        return 1
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return 1
    return int(props.multi_processor_count)


SM_COUNT: int = _get_sm_count()


@lru_cache(maxsize=None)
def _get_xopes_debug() -> bool:
    return bool(eval(os.environ.get("XOPES_DEBUG", default="False")))


XOPES_DEBUG: bool = _get_xopes_debug()


@lru_cache(maxsize=None)
def _get_compute_capability_major() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return 0
    return int(major)


@lru_cache(maxsize=None)
def _get_lasd_allow_large_blocks() -> bool:
    raw = os.environ.get("LASD_ALLOW_LARGE_BLOCKS")
    if raw is None:
        return False
    token = raw.strip().lower()
    return token in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def lasd_block_c_options() -> list[int]:
    """Return autotune BLOCK_C candidates for LASD kernels.

    The conservative default avoids large-tile fp32 kernels that can exceed
    shared-memory limits on modern GPUs (for example A100/H200).
    Set `LASD_ALLOW_LARGE_BLOCKS=1` to opt in to larger search spaces.
    """
    if _get_lasd_allow_large_blocks():
        return [16, 128]
    return [16, 64]


@lru_cache(maxsize=None)
def lasd_block_de_options() -> tuple[list[int], list[int]]:
    """Return autotune BLOCK_D/BLOCK_E candidates for LASD kernels."""
    if _get_lasd_allow_large_blocks():
        return [64, 128], [64, 128]
    return [64], [64]


def _parse_num_warps_override(var_name: str) -> list[int] | None:
    raw = os.environ.get(var_name)
    if raw is None:
        return None

    parsed: list[int] = []
    for piece in raw.split(","):
        token = piece.strip()
        if token == "":
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value in _VALID_NUM_WARPS and value not in parsed:
            parsed.append(value)

    return parsed if parsed else None


@lru_cache(maxsize=None)
def lasd_intra_inter_num_warps() -> list[int]:
    """Return autotune warps for `_lasd_parallel_intra_inter`.

    SM90 GPUs are pinned to `[4]` because larger warps can trigger Triton
    compiler aborts in this kernel. Set `LASD_INTRA_INTER_NUM_WARPS` (e.g.
    `"4,8,16,32"`) to override this behavior explicitly.
    """
    override = _parse_num_warps_override("LASD_INTRA_INTER_NUM_WARPS")
    if override is not None:
        return override
    if _get_compute_capability_major() >= 9:
        return [4]
    return [4, 8, 16, 32]


def lasd_intra_inter_block_caps(*, q_dtype: torch.dtype) -> tuple[int, int, int]:
    """Return block-size caps for `_lasd_parallel_intra_inter`.

    Large fp32 tiles are the primary source of shared-memory OOR failures on
    Ampere/Hopper. Keep the fp32 path conservative by default.
    """
    if _get_lasd_allow_large_blocks():
        return 128, 128, 128
    if q_dtype == torch.float32:
        return 64, 64, 64
    return 128, 128, 128


def contiguous(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        return fn(
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
        )

    return wrapper  # pyright: ignore[reportReturnType]


def generate_configs(input_dict: dict[str, list[int]]) -> list[triton.Config]:
    configs_dict = dict(input_dict)
    num_stages_list = configs_dict.pop("num_stages", [2])
    num_warps_list = configs_dict.pop("num_warps", [4])

    keys = list(configs_dict.keys())
    values = list(configs_dict.values())
    combinations = list(itertools.product(*values))

    configs: list[triton.Config] = []
    for num_stages in num_stages_list:
        for num_warps in num_warps_list:
            for combo in combinations:
                meta = {keys[i]: combo[i] for i in range(len(keys))}
                configs.append(
                    triton.Config(
                        meta,
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )

    if XOPES_DEBUG:
        return configs[:1]
    return configs


def prod(shape: tuple[int, ...] | list[int] | torch.Size, start_dim: int = 0, end_dim: int | None = None) -> int:
    if end_dim is None:
        end_dim = len(shape) - 1

    lo = max(0, start_dim)
    hi = min(len(shape) - 1, end_dim)
    if hi < lo:
        return 1

    out = 1
    for idx in range(lo, hi + 1):
        out *= int(shape[idx])
    return out


__all__ = [
    "MIN_BLOCK",
    "SM_COUNT",
    "XOPES_DEBUG",
    "contiguous",
    "generate_configs",
    "lasd_block_c_options",
    "lasd_block_de_options",
    "lasd_intra_inter_block_caps",
    "lasd_intra_inter_num_warps",
    "prod",
]
