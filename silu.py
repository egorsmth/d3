import torch


import os


import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def torch_silu(w, up):
    return torch.nn.functional.silu(w) * up

# N = 10000
# M = 200
# w_shape = (N, M)
# w = torch.rand(w_shape, dtype=torch.float32, device=DEVICE, requires_grad=True)
# up = torch.rand(w_shape, dtype=torch.float32, device=DEVICE, requires_grad=True)

# torch_res = torch_silu(w, up)
# print(torch_res)

@triton.jit
def triton_silu(n_rows, n_cols, w, up, y, stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    w += row * stride
    y += row * stride
    up += row * stride
    for offs in range(0, n_cols, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        a = tl.load(w + cols, mask=mask, other=.0).to(tl.float32)
        b = tl.load(up + cols, mask=mask, other=.0).to(tl.float32)
        tl.store(y + cols, (a * tl.sigmoid(a)) * b, mask=mask)

def tri_easy(n_rows, n_cols, w, up):
    y = torch.empty_like(w)
    x_arg = w.reshape(-1, w.shape[-1])
    MAX_FUSED_SIZE = 65536 // w.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
# heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    triton_silu[(n_rows, )](n_rows, n_cols, w, up, y, x_arg.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    return y

#for i in range(N):
#    for j in range(M):
#        if (not torch.isclose(torch_res[i][j], y[i][j], atol=1e-2, rtol=0)):
#            print(f'i, j {i}, {j}, {torch_res[i][j]}, {y[i][j]}')
#            assert False
# assert torch.allclose(torch_res, y, atol=1e-2, rtol=0)

@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names = ['n_rows'],
            x_vals = [512 * i for i in range(2, 40)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
            ylabel='GB/s',
            plot_name='silu',
            args={'n_cols': 4096, 'dtype': torch.float32}
        ))
def bench_silu(n_rows, n_cols, dtype, provider, device=DEVICE):
    w_shape = (n_rows, n_cols)
    w = torch.rand(w_shape, dtype=dtype,device=device)
    up = torch.rand(w_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == 'triton':
            print('tri')
            return tri_easy(n_rows, n_cols, w, up)
        if provider == 'torch':
            print('tor')
            return torch_silu(w, up)
    gbps = lambda ms: 2 * w.numel() * w.element_size() * 1e-9 / (ms * 1e-3)
    ms ,min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=1)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

bench_silu.run(save_path='.', print_data=True)
