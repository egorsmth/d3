# A1: RMSNorm + Fused SiLU × gate — Math, Engineering, Narrative

not only shift from GELU to SiLU is important but, more importantly shift from two-layer MLPs to gated feed-forward blocks like SwiGLU. SiLU matters because it is part of the feed-forward recipe in string current models. SiLU is smooth activation that is slightly cheaper than GELU. But performance change in models are not dramatic. Choosing of activation function was more popular topic before decoder LLMs, and researchers and engineers looked for simple functions that tradoffs effeciency and computation cost. SiLU ended ap being a very workable choice. SwiGLU matter more than SiLU alone. SiLU is used in modern LLMs not as a drop-in replacement in MLP, but inside a gated GLU-style feed-forward-block, usually SwiGLU. So SiLU is the activation inide default gated MLP recipe, not just free-standing choice between two scala functions.

ReLU is more eficcent for computation, it is just max(0, x), but it is not differentable and have a dying neirons problem, when they stuck at zero, SiLU don't have such problems. SiLU has non zero gradient event on large inputs, which helps with vanishing gradient problem.

non-monotonicity of SiLU helps with small nuances in learning.

## 1. RMSNorm — math layer
### 1.1 definition
LayerNorm normilize x across sample fature dimension, so batch size is irrelevant. It prevents from gradient exploding or vanishing.
$LayerNorm(x) = \frac{x- E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$
$\epsilon$ tiny constant to prevent division by zero.
$\gamma \beta$ are learnable parameters for scale and shift for network to retain expressive power.

RMSNorm removes centering operation and leaves only scaling, because it is more important factor in gradient stability in modern LLMs.

$RMSNorm(x_i) = \frac{x_i}{RMS(x)}*\gamma_i$
$RMS(x)= \sqrt{\frac{1}{n} \sum^n_{i=1}x^2_i + \epsilon}$

Less computation cost compared to LayerNorm.

### 1.2 first derivative

Gradient $\partial L / \partial x_i$ throught chain rule:
  $$
  \frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial 
  y_j}{\partial x_i}
  = \frac{1}{RMS(x)} \left[ \frac{\partial L}{\partial y_i} \gamma_i - \frac{x_i}{n \cdot RMS(x)^2} \sum_j 
  \frac{\partial L}{\partial y_j} \gamma_j x_j \right]
  $$

Gradient $\partial L / \partial \gamma_i$:
  $$
  \frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \cdot \frac{x_i}{RMS(x)}
  $$

### 1.3 Numerical stability
- $\epsilon$ added before taking square root to prevent division by zero when x activations are all zeros.

- FP32 accumulators (mixed precision). Calculating $\sum x_i^2$ across large feature dimension overflow risky. Modern GPU kernels have float32 accumulator during reduction phase.

- **Inverse rsqrt instead of sqrt+div.** Many GPU (including NVIDIA with CUDA
  `__frsqrt_rn`) have hardware rsqrt fast intrinsic. Instead of $y = x \cdot
  \gamma / r$ → $y = x \cdot \gamma \cdot \text{rsqrt}(\frac{1}{n}\sum
  x_i^2 + \varepsilon)$. Single fused opeartion, economy ~30-50% latency on this step.

- **ε placement matters.** $\sqrt{m + \varepsilon}$ vs $\sqrt{m} +
  \varepsilon$ vs $\sqrt{\max(m, \varepsilon)}$ — three different options with different gradient behavior with $x \to 0$. Standard — first (inside
  radical), it is smooth and differentiable everywhere.

## 2. SiLU — math layer
### 2.1 definition
SiLU (Sigmoid Linear Unit, aka Swish):

$SiLU(x) = (SiLU(x_0), ..., SiLU(x_n))$
where $x \in \mathbb{R}^n$ and
$SiLU(x) = x * \sigma(x) = x * \frac{1}{1+e^{-x}}$

$SwiGLU⁡(x) = SiLU⁡(x⁢W+b) * (x⁢V+c)$

Because of additional parameters in SwiGLU it could learn more nuanced patterns helping models learn more complex patterns without additional layers. Parameters W, b, V, c are learned. Biases b and c usually dropped.

### 2.2 first derivative
gradient:
$\Delta_x SiLU(x) = \frac{ e^{-x}(x+1)+1}{(1+e^{-x})^2}$

In computation of forward backward pass with autograd saved $\sigma(x)$ could reyused with compated formula
$\text{SiLU}'(x) = \sigma(x) \cdot \left(1 + x \cdot (1 - \sigma(x))\right)$
so in forward there will be division, addition, exponent operations and in backward there will be two multiplications, addition and substraction.

### 2.3 assymptotic behavior
$\lim_{x \to \infty}SiLU(x) = \infty $
$\lim_{x \to \infty}(SiLU(x) - x) = 0 $
$\lim_{x \to -\infty}SiLU(x) = 0 $
local minima at $x \approx -1.278$

### 2.4 numerical stability
SiLU numerical stability problems arise from $e^{-x}$ where large negative numbers could cause overflow and large positive numbers could cause underflow.

To prevent numerical instability, frameworks and production models rely on a few standard mitigations:
- The Log-Sum-Exp Trick / Numerically Stable Sigmoid: Frameworks like PyTorch compute the sigmoid function using stable mathematical branches:
    - For positive inputs: \(\sigma(x) = \frac{1}{1 + e^{-x}}\)
    - For negative inputs: \(\sigma(x) = \frac{e^{x}}{e^{x} + 1}\)
    
- Gradient Clipping & Weight Initialization: Because SiLU is unbounded for positive inputs, large activations can occasionally trigger exploding gradients. Standard implementations use Kaiming/Xavier weight initializations and gradient clipping to keep pre-activation values bounded.
- Mixed Precision (AMP) Scaling: During deep learning training with 16-bit floats (float16), the risk of overflow/underflow is vastly higher. Automatic Mixed Precision (AMP) uses dynamic loss scaling to keep intermediate tensor values within the representable range.Hardware - Look-Up Tables (LUTs): In highly optimized or embedded deep learning accelerators (e.g., edge devices or specialized ASICs), hardware implementations pre-compute SiLU over a defined range (e.g., \([-8, 8]\)) to completely sidestep floating-point exponential math and guarantee zero runtime exceptions.

Optimized ImplementationsFor the highest levels of performance and stability, it is best to avoid writing a custom SiLU function from scratch. Instead, utilize the optimized, hardware-fused operations provided by major machine learning libraries: 
- Use the native torch.nn.functional.silu or torch.nn.SiLU in PyTorch, which handles internal precision truncation.
- For hardware like NVIDIA GPUs, consider fused engines (such as cuDNN Frontend's Fused RMSNorm + SiLU) for joint execution efficiency and numerical consistency

## 3. Fused SiLU × gate (SwiGLU) — math layer
TBD W21.
## 4. Engineering — bench, hardware, kernel design
TBD W21.
## 5. Narrative — what this says about hardware-algorithm gap
TBD после W21.
