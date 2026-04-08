from __future__ import annotations

import math

import torch

from kv_cache_engine.types import QuantizedTensor


def _symmetric_scale(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 127.0 if bits == 8 else 7.0
    return tensor.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / qmax


def _affine_params(
    tensor: torch.Tensor,
    bits: int,
    reduce_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = float((1 << bits) - 1)
    min_value = tensor.amin(dim=reduce_dims, keepdim=True)
    max_value = tensor.amax(dim=reduce_dims, keepdim=True)
    scale = (max_value - min_value).clamp_min(1e-8) / qmax
    return min_value, scale


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    flattened = values.reshape(-1).to(torch.int16)
    unsigned = torch.clamp(flattened + 8, 0, 15).to(torch.uint8)
    if unsigned.numel() % 2:
        unsigned = torch.cat(
            [unsigned, torch.full((1,), 8, device=unsigned.device, dtype=torch.uint8)]
        )
    low = unsigned[0::2]
    high = unsigned[1::2] << 4
    return (low | high).contiguous()


def pack_uint4(values: torch.Tensor) -> torch.Tensor:
    flattened = torch.clamp(values.reshape(-1), 0, 15).to(torch.uint8)
    if flattened.numel() % 2:
        flattened = torch.cat(
            [flattened, torch.zeros((1,), device=flattened.device, dtype=torch.uint8)]
        )
    low = flattened[0::2]
    high = flattened[1::2] << 4
    return (low | high).contiguous()


def unpack_int4(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    stacked = torch.stack((low, high), dim=1).reshape(-1)
    needed = math.prod(shape)
    signed = stacked[:needed].to(torch.int16) - 8
    return signed.reshape(shape).to(torch.int8)


def unpack_uint4(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    stacked = torch.stack((low, high), dim=1).reshape(-1)
    needed = math.prod(shape)
    return stacked[:needed].reshape(shape).to(torch.uint8)


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    scheme: str = "symmetric_per_head",
) -> QuantizedTensor:
    if bits not in {4, 8}:
        raise ValueError("Only int4 and int8 quantization are supported.")
    if scheme == "symmetric_per_head":
        scale = _symmetric_scale(tensor, bits).to(dtype=torch.float32)
        qmax = 127 if bits == 8 else 7
        qmin = -127 if bits == 8 else -8
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)
        storage = quantized if bits == 8 else pack_int4(quantized)
        offset = None
    elif scheme == "affine_per_channel":
        offset, scale = _affine_params(tensor, bits, reduce_dims=(2,))
        quantized = torch.clamp(
            torch.round((tensor - offset) / scale),
            0,
            (1 << bits) - 1,
        ).to(torch.uint8)
        storage = quantized if bits == 8 else pack_uint4(quantized)
        scale = scale.to(dtype=torch.float32)
        offset = offset.to(dtype=torch.float32)
    elif scheme == "affine_per_token":
        offset, scale = _affine_params(tensor, bits, reduce_dims=(3,))
        quantized = torch.clamp(
            torch.round((tensor - offset) / scale),
            0,
            (1 << bits) - 1,
        ).to(torch.uint8)
        storage = quantized if bits == 8 else pack_uint4(quantized)
        scale = scale.to(dtype=torch.float32)
        offset = offset.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported quantization scheme: {scheme}")
    return QuantizedTensor(
        data=storage,
        scale=scale,
        offset=offset,
        bits=bits,
        shape=tuple(int(dim) for dim in tensor.shape),
        original_dtype=tensor.dtype,
        scheme=scheme,
    )


def dequantize_tensor(quantized: QuantizedTensor) -> torch.Tensor:
    if quantized.scheme == "symmetric_per_head":
        if quantized.bits == 8:
            restored = quantized.data.to(torch.float32)
        else:
            restored = unpack_int4(quantized.data, quantized.shape).to(torch.float32)
        return (restored * quantized.scale).to(dtype=quantized.original_dtype)

    if quantized.bits == 8:
        restored = quantized.data.to(torch.float32)
    else:
        restored = unpack_uint4(quantized.data, quantized.shape).to(torch.float32)
    if quantized.offset is None:
        raise ValueError(f"Affine quantization scheme {quantized.scheme} requires an offset tensor.")
    return (restored * quantized.scale + quantized.offset).to(dtype=quantized.original_dtype)
