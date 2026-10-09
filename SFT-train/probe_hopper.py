#!/usr/bin/env python3
"""Check the corrected Hopper gated-delta kernel's forward/backward."""
import torch
import triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

assert triton.__version__ == '3.7.1'
torch.manual_seed(42)
q, k, v = [torch.randn(1, 64, 2, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
g = torch.full((1, 64, 2), -0.1, device='cuda', dtype=torch.float32, requires_grad=True)
beta = torch.full((1, 64, 2), 0.5, device='cuda', dtype=torch.bfloat16, requires_grad=True)
output, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
output.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v, g, beta))
print('Hopper gated-delta forward/backward passed; Triton', triton.__version__, flush=True)
