"""Log-decay derivative helpers used by LASD Triton backward.

Vendored from xopes (`log_decay_with_cumsum/dld_with_cumsum_torch.py`) with a
single exported selector `compute_dld_with_cumsum_fn`.
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_dld_with_cumsum_torch(
    dld_q: torch.Tensor,  # B N H F
    dld_k: torch.Tensor,  # B N H F
    final_state: Optional[torch.Tensor] = None,  # B H D E
    dfinal_state: Optional[torch.Tensor] = None,  # B H D E
    cu_seqlens: Optional[torch.Tensor] = None,  # M
    sum_option: Optional[int] = -1,
):
    del cu_seqlens
    dtype = dld_q.dtype
    dld_q = dld_q.to(torch.float32)
    dld_k = dld_k.to(torch.float32)
    if final_state is not None:
        final_state = final_state.to(torch.float32)
    if dfinal_state is not None:
        dfinal_state = dfinal_state.to(torch.float32)

    dld = dld_q - dld_k
    if sum_option == -1:
        dld = dld.sum(dim=-1)
    dld = torch.flip(dld, dims=[1])
    dld = torch.cumsum(dld, dim=1)
    dld = torch.flip(dld, dims=[1])
    dld = dld.to(dtype)

    if dfinal_state is not None and final_state is not None:
        dld_state = (final_state * dfinal_state).unsqueeze(1)
        if sum_option == -1:
            dld_state = dld_state.sum(dim=-1).sum(dim=-1)
        elif sum_option == 0:
            dld_state = dld_state.sum(dim=-1)
        else:
            dld_state = dld_state.sum(dim=-2)
        dld = dld + dld_state

    return dld.to(dtype)


compute_dld_with_cumsum_fn = compute_dld_with_cumsum_torch

__all__ = ["compute_dld_with_cumsum_fn", "compute_dld_with_cumsum_torch"]
