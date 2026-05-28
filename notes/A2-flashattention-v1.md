---
type: paper-notes
paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré"
venue: "NeurIPS 2022"
arxiv: "2205.14135"
read_date: 2026-05-25
wp: WP-25 (A2 FP16 matmul deep — prerequisite reading)
status: in_progress
---

# FlashAttention v1 — рабочие заметки

> Источник для A2 §4 Engineering layer (tensor cores + roofline + numerical analysis).
> Конспект перенесён из `DS-strategy/inbox/fleeting-notes.md` 25 мая после чтения.

## 1. GPU memory hierarchy

GPU. Huge amount of thread where every thread has an SRAM (shared) memory. HBM (high bandwidth) memory is much larger that SRAM but demands more time to move data. Memory-bound operations are low in compute compared to memory access: sum, softmax, batchnorm, layernorm, activation, dropout. Compute-bound operations are high in computations compared to memory access: conv with large number of channels, matrix multiplication with high number of inner dimension.

Kernel fusion most common approach to accelerate memory-bound operations. Compilers could fuse automatically, however intermediate values should be saved to HBM for backward pass.

## 2. Attention setup + Algorithm 0 (standard implementation)

Attention. Q, K, V ∈ R^{n × d}. Where n is sequence length, d is head dimension. O ∈ R^{n × d} output of attention.

S = QK^T ∈ R^{n × n}, P = Softmax(S), O = PV, where softmax is applied row-wise.

GPT-2: N=1024, d=64.

**Algorithm 0 — Standard Attention Implementation:**
- Require: Matrices Q, K, V ∈ R^{N × d} in HBM.
- 1. Load Q, K by blocks from HBM, compute S = QK^T, write S to HBM.
- 2. Read S from HBM, compute P = softmax(S), write P to HBM.
- 3. Load P and V by blocks from HBM, compute O = PV, write O to HBM.
- 4. Return O.

## 3. Algorithm 1 (FlashAttention forward pass)

Require: Matrices Q, K, V ∈ R^{N × d} in HBM, on-chip SRAM of size M, softmax scaling constant τ ∈ R, masking function mask, dropout probability p.

1. Initialize the pseudo-random number generator state R and save to HBM.
2. Set block sizes B_c = ⌈M/(4d)⌉, B_r = min(⌈M/(4d)⌉, d).
3. Initialize O = (0)_{N×d} ∈ R^{N×d}, ℓ = (0)_{N} ∈ R^N, m = (−∞)_{N} ∈ R^N in HBM.
4. Divide Q into T_r = ⌈N/B_r⌉ blocks Q_1, ..., Q_{T_r} of size B_r × d each; divide K, V into T_c = ⌈N/B_c⌉ blocks K_1, ..., K_{T_c} and V_1, ..., V_{T_c}, of size B_c × d each.
5. Divide O into T_r blocks O_i, ..., O_{T_r} of size B_r × d each; divide ℓ into T_r blocks ℓ_i, ..., ℓ_{T_r} of size B_r each; divide m into T_r blocks m_1, ..., m_{T_r} of size B_r each.
6. for 1 ≤ j ≤ T_c do
7.   Load K_j, V_j from HBM to on-chip SRAM.
8.   for 1 ≤ i ≤ T_r do
9.     Load Q_i, O_i, ℓ_i, m_i from HBM to on-chip SRAM.
10.    On chip, compute S_ij = τQ_i K^T_j ∈ R^{B_r × B_c}.
11.    On chip, compute S^masked_ij = mask(S_ij).
12.    On chip, compute m̃_ij = rowmax(S^masked_ij) ∈ R^{B_r}, P̃_ij = exp(S^masked_ij − m̃_ij) ∈ R^{B_r × B_c} (pointwise), ℓ̃_ij = rowsum(P̃_ij) ∈ R^{B_r}.
13.    On chip, compute m^new_i = max(m_i, m̃_ij) ∈ R^{B_r}, ℓ^new_i = e^{m_i − m^new_i} ℓ_i + e^{m̃_ij − m^new_i} ℓ̃_ij ∈ R^{B_r}.
14.    On chip, compute P̃^dropped_ij = dropout(P̃_ij, p_drop).
15.    Write O_i ← diag(ℓ^new_i)^{−1} (diag(ℓ_i) e^{m_i − m^new_i} O_i + e^{m̃_ij − m^new_i} P̃^dropped_ij V_j) to HBM.
16.    Write ℓ_i ← ℓ^new_i, m_i ← m^new_i to HBM.
17.  end for
18. end for
19. Return O, ℓ, m, R.

**Главное:** на каждом outer iteration j загружаем K_j, V_j в SRAM один раз → reuse через все inner iterations i. На inner — Q_i, O_i, ℓ_i, m_i загружаются → online softmax обновление → запись обратно. S и P **не материализуются в HBM** (тут главная экономия памяти и времени).

## 4. GPU threading model (для §3 и kernel design)

- На GPU потоки группируются в **thread blocks**.
- Внутри thread block потоки группируются в **warps** (32 потока обычно).
- Внутри одного warp потоки могут быстро коммуницировать через **SRAM (shared memory)** — read/write одной памяти.
- Между thread blocks коммуникация только через HBM (медленно).

→ Алгоритм FlashAttention использует это: outer loop по j мапится на thread blocks, inner loop по i мапится на warps внутри thread block, на-chip вычисления над S_ij, P̃_ij идут через SRAM.

## 5. Tensor cores

Tensor cores на GPU — специализированные блоки для **efficient low-precision (fp16/bf16) matrix multiplication**.

- Один tensor core выполняет fused matrix multiply-accumulate (FMA) за такт на матрицах 4×4 или 16×8×8.
- На Turing (RTX 2080 Super) — 64 tensor cores per SM × 48 SM = ~3000 cores. Peak ~58 TFLOPS fp16.
- На Ampere/Hopper — больше + поддержка bf16, sparsity, fp8 (Hopper).

→ A2 FP16 matmul deep = использовать tensor cores через `wmma` (CUDA) или соответствующие Triton intrinsics. Это **компилятор-инструкция**, не auto-applied.

## 6. Softmax (стандартный + online)

### 6.1 Standard softmax (на GPU)

Превращает вектор логитов в вектор вероятностей:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}$$

Дана матрица S ∈ R^{N × d}. Разделим: [S^(1), S^(2)] = S, где S^(i) ∈ R^{B_r × B_c}. B_r — number of rows in block, B_c — number of columns.

Нужно вычислить softmax row-wise по S и умножить на V, V ∈ R^{N × d}.

$$m = \max(\text{rowmax}(S^{(1)}), \text{rowmax}(S^{(2)})) \in R^{B_r} \text{ — максимум каждой строки в блоке}$$

$$\ell = \text{rowsum}(e^{S^{(1)} - m}) + \text{rowsum}(e^{S^{(2)}-m}) \in R^{B_r}$$

$$P = [P^{(1)}, P^{(2)}] = \text{diag}(\ell)^{-1}[e^{S^{(1)}-m}, e^{S^{(2)}-m}] \in R^{B_r \times d}$$

$$O = [P^{(1)}, P^{(2)}] \begin{bmatrix} V^{(1)} \\ V^{(2)} \end{bmatrix} = \text{diag}(\ell)^{-1} (e^{S^{(1)}-m} V^{(1)} + e^{S^{(2)}-m} V^{(2)}) \in R^{B_r \times d}$$

**Проблема:** требует двух полных проходов по S — один для max + sum, второй для нормализованного экспонента + multiply на V. Материализация S промежуточно.

### 6.2 Online softmax

Вычисляет «локальный» softmax по каждому блоку и потом rescales для правильного output:

$$m^{(1)} = \text{rowmax}(S^{(1)}) \in R^{B_r}$$
$$\ell^{(1)} = \text{rowsum}(e^{S^{(1)}-m^{(1)}}) \in R^{B_r}$$
$$\hat{P}^{(1)} = \text{diag}(\ell^{(1)})^{-1} e^{S^{(1)}-m^{(1)}} \in R^{B_r \times B_c}$$
$$O^{(1)} = \hat{P}^{(1)} V^{(1)} = \text{diag}(\ell^{(1)})^{-1} e^{S^{(1)} - m^{(1)}} V^{(1)} \in R^{B_r \times d}$$

После второго блока — корректировка:
$$m^{(2)} = \max(m^{(1)}, \text{rowmax}(S^{(2)})) = m$$
$$\ell^{(2)} = e^{m^{(1)}-m^{(2)}} \ell^{(1)} + \text{rowsum}(e^{S^{(2)} - m^{(2)}}) = \text{rowsum}(e^{S^{(1)}-m}) + \text{rowsum}(e^{S^{(2)} - m}) = \ell$$
$$\hat{P}^{(2)} = \text{diag}(\ell^{(2)})^{-1} e^{S^{(2)} - m}$$
$$O^{(2)} = \text{diag}(\ell^{(1)} / \ell^{(2)})^{-1} O^{(1)} + \hat{P}^{(2)} V^{(2)} = \text{diag}(\ell^{(2)})^{-1} (e^{S^{(1)}-m} V^{(1)} + e^{S^{(2)}-m} V^{(2)}) = O$$

**Ключ:** S не материализуется глобально — обновление O идёт инкрементально по блокам с корректировкой через текущие m и ℓ. Это базис FlashAttention forward.

## 7. FlashAttention-2 — differences

FlashAttention-2 вычисляет online softmax **немного иначе** — использует **log-sum-exp (LSE)** чтобы лучше использовать **thread blocks и warps**:

- В v1 outer loop по K/V, inner loop по Q — это создавало serialization внутри thread block (каждая warp ждёт соседних).
- В v2 swap loop order: outer по Q, inner по K/V → каждая warp независимо обрабатывает свой Q block без synchronization.
- LSE-параметризация (log ℓ + m) вместо raw m, ℓ — численная стабильность + удобство для backward (один scalar per row вместо двух).

→ Результат: ~2× ускорение vs v1 на типичных размерах, лучшее использование SM. Архитектурно ближе к стандартному GEMM kernel pattern.

## 8. Backward pass + recomputation — TBD

> Из v1 paper: backward требует пересчитать S, P в момент backward (recomputation), потому что они не сохранены в HBM. Tradeoff = +20-30% compute, но огромная экономия памяти.

## 9. Roofline analysis — TBD

> Связь с A2 §4 Engineering layer. Для RTX 2080 Super: peak ~3 TFLOPS fp32, ~58 TFLOPS fp16, peak bandwidth ~496 GB/s. Operational intensity attention зависит от N и d (small N → memory-bound, large N → compute-bound).

## Outstanding questions

- В v2 LSE-параметризация — точно как меняет backward? (нужно прочитать backward раздел v2 paper)
- Tensor cores на Turing (sm_75) — поддержка fp16 matmul + accumulator fp32. Поддерживает ли Triton автоматическую генерацию tensor core инструкций?
- В Algorithm 1 v1 — почему B_r = min(⌈M/(4d)⌉, d)? Откуда константа 4 в делителе?

## Hooks for A2 writeup

- §1 (GPU memory) → A2 §4 Engineering: HBM vs SRAM hierarchy + memory-bound vs compute-bound classification.
- §2 (Algorithm 0) → A2 §3 SwiGLU math layer baseline (показать что мы оптимизируем относительно).
- §3 (Algorithm 1) → A2 §4 Engineering + §5 Numerical layers (tiling + recomputation).
- §4 (Threading) → A2 §4 Engineering kernel structure.
- §5 (Tensor cores) → A2 §4 Engineering: tensor core utilization + §6 Roofline upper bound.
- §6 (Online softmax) → A2 §5 Numerical layer (стабильность через LSE/m-tracking).
- §7 (v2 differences) → A2 §4 Engineering: parallelism patterns, warp utilization.
