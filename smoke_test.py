import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask = mask)

n = 4096
x = torch.randn(n, device = 'cuda')
y = torch.randn(n, device = 'cuda')
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, 1024)
print("PASS" if torch.allclose(out, x+y) else "FAIL")

