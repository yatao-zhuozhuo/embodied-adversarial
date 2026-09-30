#!/usr/bin/env python3
"""Run a minimal FLA gated-delta forward/backward compatibility probe."""

from __future__ import annotations

import importlib.metadata

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def main() -> None:
    torch.manual_seed(0)
    device = "cuda:0"
    batch, seq_len, heads, value_heads, key_dim, value_dim = 1, 64, 4, 4, 32, 32
    q = torch.randn(
        batch, seq_len, heads, key_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    k = F.normalize(
        torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.bfloat16),
        p=2,
        dim=-1,
    ).requires_grad_(True)
    v = torch.randn(
        batch, seq_len, value_heads, value_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    g = F.logsigmoid(
        torch.randn(batch, seq_len, value_heads, device=device, dtype=torch.bfloat16)
    ).requires_grad_(True)
    beta = torch.sigmoid(
        torch.randn(batch, seq_len, value_heads, device=device, dtype=torch.bfloat16)
    ).requires_grad_(True)

    output, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    loss = output.float().square().mean()
    loss.backward()
    grads_finite = all(torch.isfinite(tensor.grad).all().item() for tensor in (q, k, v, g, beta))
    print(
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "triton": importlib.metadata.version("triton"),
            "tilelang": importlib.metadata.version("tilelang"),
            "loss": float(loss.detach()),
            "grads_finite": grads_finite,
        }
    )
    if not grads_finite:
        raise RuntimeError("FLA probe produced non-finite gradients")


if __name__ == "__main__":
    main()
