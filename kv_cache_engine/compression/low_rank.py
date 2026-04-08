from __future__ import annotations

import torch

from kv_cache_engine.types import LowRankTensor


def compress_tensor_low_rank(tensor: torch.Tensor, rank_ratio: float) -> LowRankTensor:
    if tensor.dim() != 4:
        raise ValueError("Low-rank KV compression expects a [B, H, S, D] tensor.")
    batch, heads, seq_len, hidden = tensor.shape
    rank = max(1, min(seq_len, hidden, int(min(seq_len, hidden) * rank_ratio)))
    matrices = tensor.reshape(batch * heads, seq_len, hidden).to(torch.float32)
    u_parts = []
    s_parts = []
    vh_parts = []
    for matrix in matrices:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        u_parts.append(u[:, :rank].to(dtype=tensor.dtype))
        s_parts.append(s[:rank].to(dtype=tensor.dtype))
        vh_parts.append(vh[:rank, :].to(dtype=tensor.dtype))
    u_tensor = torch.stack(u_parts, dim=0)
    s_tensor = torch.stack(s_parts, dim=0)
    vh_tensor = torch.stack(vh_parts, dim=0)
    return LowRankTensor(
        u=u_tensor,
        s=s_tensor,
        vh=vh_tensor,
        original_shape=tuple(int(dim) for dim in tensor.shape),
        original_dtype=tensor.dtype,
    )


def decompress_low_rank_tensor(low_rank_tensor: LowRankTensor) -> torch.Tensor:
    restored = torch.matmul(
        low_rank_tensor.u * low_rank_tensor.s.unsqueeze(1), low_rank_tensor.vh
    )
    batch, heads, seq_len, hidden = low_rank_tensor.original_shape
    return restored.reshape(batch, heads, seq_len, hidden).to(low_rank_tensor.original_dtype)
