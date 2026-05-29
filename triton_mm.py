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
    c_val = tl.zeros((), dtype=tl.float16)

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
    
    b_block = j + tl.arange(0, BLOCK_SIZE)[None, :] + tl.arange(0, N)[:, None] * N
    b = tl.load(B + b_block, mask=b_block < K*N, other=.0)

    c = tl.dot(a, b)
    tl.store(C + (i * K) + (j + tl.arange(0, BLOCK_SIZE)[None, :]) + tl.arange(0, BLOCK_SIZE)[:, None] * K, c)


def run():
    M, N, K = 16, 16, 16

    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)

    print(a)
    print(b)
    print(torch.matmul(a, b))
   # grid = (16 * 16,)

    #triton_matmul_naive[grid](a, b, o, M, N, K)
   # print(o)
   # o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)
   # triton_matmul_rowed[grid](a, b, o, M, N, K)
   # print(o)
    
    o = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)
    GROUP_SIZE = 2
    triton_matmul_grouped[(8, 8)](a, b, o, M, N, K, GROUP_SIZE)
    print(o)



run()

