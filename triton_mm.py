import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def triton_matmul_naive(A, B, C, M, N, K):
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N

    if i >= M or j >= N:
        return
    c_val = tl.zeros((), dtype=tl.float32)

    for k in range(K):
        a_val = tl.load(A + i * K + k)
        b_val = tl.load(B + k * N + j)
        c_val += a_val * b_val

    tl.store(C + i * N + j, c_val)

@triton.jit
def triton_matmul_rowed(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N

   # c_val = tl.zeros((), dtype=tl.float16)
    a_row = i * K + tl.arange(0, K)[None, :]
    b_col = tl.arange(0, K)[:, None] * N + j
    a_val = tl.load(A + a_row, mask=a_row < M*K, other=.0)
    b_val = tl.load(B + b_col, mask=b_col < K*N, other=.0)

    c_val = tl.dot(a_val, b_val)
    tl.store(C + i * N + j + tl.arange(0,1)[:,None], c_val)

@triton.jit
def triton_matmul_grouped(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE
    j = tl.program_id(1) * BLOCK_SIZE

    c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float16)
    a_block = i * K + tl.arange(0, K)[None, :] + tl.arange(0, BLOCK_SIZE)[:, None] * K
    a = tl.load(A + a_block, mask=a_block < M*K, other=.0)
    
    b_block = j + tl.arange(0, BLOCK_SIZE)[None, :] + tl.arange(0, K)[:, None] * N
    b = tl.load(B + b_block, mask=b_block < K*N, other=.0)

    c = tl.dot(a, b)
    tl.store(C + (i * N) + (j + tl.arange(0, BLOCK_SIZE)[None, :]) + tl.arange(0, BLOCK_SIZE)[:, None] * N, c)


def run():
    M, N, K = 512, 512, 1024

    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)

    perfect = torch.matmul(a, b)

    grid = (M * N,)

    triton_matmul_naive[grid](a, b, o, M, N, K)
    if torch.allclose(perfect, o, atol=0.125, rtol=0):
        print("✅ matmul naive and Torch match")
    else:
        print(perfect[:5, :5])
        print(o[:5, :5])
    o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)
    triton_matmul_rowed[grid](a, b, o, M, N, K)
    if torch.allclose(perfect, o, atol=0.125, rtol=0):
        print("✅ matmul rowed and Torch match")
    
    o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)
    GROUP_SIZE = 8
    triton_matmul_grouped[(M // GROUP_SIZE, N // GROUP_SIZE)](a, b, o, M, N, K, GROUP_SIZE)
    if torch.allclose(perfect, o, atol=0.125, rtol=0):
        print("✅ matmul grouped and Torch match")



run()



configs = []
for i in [2, 8, 16]:
    configs.append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[128 * i for i in [2, 4, 8]],
                line_arg="provider",
                line_vals=['naive', 'rowed', 'grouped'],
                line_names=['naive', 'rowed', 'grouped'],
                ylabel='TFLOPS',
                plot_name=f'matmul-perf-gr{i}-no-torch',
                args={"GROUP_SIZE": i},
                )
            )

@triton.testing.perf_report(configs)
def benchmark(M, N, K, GROUP_SIZE, provider):

    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)

    rep = 10
    qs = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=qs, rep=rep)
    if provider == 'naive':
        grid = (M * N,)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul_naive[grid](a, b, o, M, N, K), quantiles=qs, rep=rep)
    if provider == 'rowed':
        grid = (M * N,)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul_rowed[grid](a, b, o, M, N, K), quantiles=qs, rep=rep)
    if provider == 'grouped':
        grid = (M // GROUP_SIZE, N // GROUP_SIZE)
        print(grid)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul_grouped[grid](a, b, o, M, N, K, GROUP_SIZE), quantiles=qs, rep=rep)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

# benchmark.run(save_path='./mm', print_data=True)
