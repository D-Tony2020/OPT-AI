# Lecture 3 — January 28, 2026 — 忠实转制笔记

---

## 第 ① 页：Deep Learning（回顾）

Consider a multi-class classification problem with data points
{(xⱼ, yⱼ) : j = 1, ..., M} where xⱼ ∈ ℝⁿ and yⱼ ∈ {1, 2, ..., K}

> **[图示说明]** 深度神经网络层级结构图（同 Lecture 2 第⑦页），从左到右：
>
> **Layer 1:** 输入 xⱼ → (Q¹, g¹)，Q¹ ∈ ℝ^(d₁×n), g¹ ∈ ℝ^(d₁) → 输出 aⱼ¹
>
> **Layer 2:** (Q², g²)，Q² ∈ ℝ^(d₂×d₁), g² ∈ ℝ^(d₂) → 输出 aⱼ²
>
> **Layer L:** (Qᴸ, gᴸ)，Qᴸ ∈ ℝ^(dₗ×dₗ₋₁), gᴸ ∈ ℝ^(dₗ) → 输出 aⱼᴸ

$$a_j^1 = \sigma_1(Q^1 x_j + g^1) \qquad a_j^2 = \sigma_2(Q^2 a_j^1 + g^2) \qquad \cdots \qquad a_j^L = \sigma_L(Q^L a_j^{L-1} + g^L)$$

维度标注：Q¹ 为 d₁×n，xⱼ 为 n×1，输出 d₁×1；Q² 为 d₂×d₁，aⱼ¹ 为 d₁×1，输出 d₂×1；Qᴸ 为 dₗ×dₗ₋₁，输出 dₗ×1。

Each element of σ₁, or σ₂, ... or σₗ could be:
- σ(t) = 1/(1 + eᵗ) (sigmoid)
- σ(t) = max{t, 0} (ReLU)

---

## 第 ② 页：深度网络整体表达式

If we use a neural net with parameters (Q¹, g¹, Q², g², ..., Qᴸ, gᴸ), then corresponding to the input xⱼ, the output aⱼᴸ is given by the function:

$$z(x_j, Q^1, g^1, Q^2, g^2, \ldots, Q^L, g^L) =$$

$$\sigma_L\left(Q^L \, \sigma_{L-1}\left(\ldots \sigma_2\left(Q^2 \, \sigma_1(Q^1 x_j + g^1) + g^2\right) \ldots \right) + g^L\right)$$

> **[图示说明]** 嵌套结构标注（花括号由内到外）：
> - "output from layer 1"
> - "output from layer 2"
> - "output from layer L−1"
> - "output from layer L"

Solve a multi-class classification problem by using z(xⱼ, Q¹, g¹, ..., Qᴸ, gᴸ) as the feature vector for data point j.

---

## 第 ③ 页：深度学习的似然函数构建

| data point # | Feature vector | label | likelihood |
|:---:|:---:|:---:|:---:|
| 1 | z(x₁, Q¹, g¹, ..., Qᴸ, gᴸ) = z₁ | 2 | e^(w₂ᵀz₁) / ∑ₗ₌₁ᴷ e^(wₗᵀz₁) |
| 2 | z(x₂, Q¹, g¹, ..., Qᴸ, gᴸ) = z₂ | 4 | e^(w₄ᵀz₂) / ∑ₗ₌₁ᴷ e^(wₗᵀz₂) |
| 3 | z(x₃, Q¹, g¹, ..., Qᴸ, gᴸ) = z₃ | 1 | e^(w₁ᵀz₃) / ∑ₗ₌₁ᴷ e^(wₗᵀz₃) |
| ⋮ | ⋮ | ⋮ | ⋮ |
| M | z(xₘ, Q¹, g¹, ..., Qᴸ, gᴸ) = zₘ | 4 | e^(w₄ᵀzₘ) / ∑ₗ₌₁ᴷ e^(wₗᵀzₘ) |

Say (w₁, w₂, ..., wₖ) are the parameters for each class, where w₁, w₂, ..., wₖ ∈ ℝ^(dₗ)

Likelihood of the whole dataset:

$$L(Q^1, g^1, \ldots, Q^L, g^L, w_1, w_2, \ldots, w_K) = \frac{e^{w_2^\top z(x_1, Q^1, g^1, \ldots, Q^L, g^L)}}{\sum_{\ell=1}^{K} e^{w_\ell^\top z(x_1, Q^1, g^1, \ldots, Q^L, g^L)}} \cdots \frac{e^{w_4^\top z(x_M, Q^1, g^1, \ldots, Q^L, g^L)}}{\sum_{\ell=1}^{K} e^{w_\ell^\top z(x_M, Q^1, g^1, \ldots, Q^L, g^L)}}$$

---

## 第 ④ 页：深度学习的最大似然优化问题

$$\ln L(Q^1, g^1, \ldots, Q^L, g^L, w_1, w_2, \ldots, w_K)$$

$$= -\sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top z(x_j, Q^1, g^1, \ldots, Q^L, g^L)}\right)$$

$$+ \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbb{1}(y_j = k) \, w_k^\top \cdot z(x_j, Q^1, g^1, \ldots, Q^M, g^M)$$

The maximum likelihood problem we solve is:

$$\max \left\{ -\sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top z(x_j, Q^1, g^1, \ldots, Q^L, g^L)}\right) + \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbb{1}(y_j = k) \, w_k^\top \cdot z(x_j, Q^1, g^1, \ldots, Q^M, g^M) \right\}$$

over:
- (Q¹, g¹) ∈ ℝ^(d₁×(n+1))
- (Q², g²) ∈ ℝ^(d₂×(d₁+1))
- ⋮
- (Qᴸ, gᴸ) ∈ ℝ^(dₗ×(dₗ₋₁+1))
- (w₁, w₂, ..., wₖ) ∈ ℝ^(K×dₗ)

---

## 第 ⑤ 页：Basics of Gradient Descent Methods

Let f: ℝⁿ → ℝ be a function that maps n-dimensional vectors to a scalar. The gradient of f at point x⁰ is the n-dimensional vector given by

$$\nabla f(x^0) = \left(\left.\frac{\partial f(x)}{\partial x_1}\right|_{x=x^0}, \quad \left.\frac{\partial f(x)}{\partial x_2}\right|_{x=x^0}, \quad \ldots, \quad \left.\frac{\partial f(x)}{\partial x_n}\right|_{x=x^0}\right)$$

- The general principle is that if we start at point x⁰ and move in the direction of the gradient ∇f(x⁰) for a small step size, the value of the function gets larger.

- If we start at some point x⁰ and move against the direction of the gradient ∇f(x⁰) for a small step size, the value of the function gets smaller.

> **[图示说明]** 一维示意图：在数轴上标出点 x⁰，向左箭头标记 ∇f(x⁰)（梯度方向，函数值增大），向右箭头标记 −∇f(x⁰)（负梯度方向，函数值减小）。新点 x⁰ − α∇f(x⁰) 在负梯度方向上。步长 α 标注在箭头上。图示目的：直观说明梯度上升（沿梯度）和梯度下降（逆梯度）的方向选择。

---

## 第 ⑥ 页：梯度下降的数值示例

If α is small enough then

$$f(x^0 - \alpha \nabla f(x^0)) \leq f(x^0)$$

**Example:** Consider the function: f(x) = x₁² + 4x₁x₂ + 3x₂² − 5

$$\nabla f(x) = \begin{pmatrix} 2x_1 + 4x_2 \\ 4x_1 + 6x_2 \end{pmatrix}$$

- Let's start at point x⁰ = (1, 1)ᵀ and move against the gradient for a step size of 0.1.

$$\nabla f((1,1)^\top) = \begin{pmatrix} 6 \\ 10 \end{pmatrix}$$

The new point we reach is:

$$\begin{pmatrix} 1 \\ 1 \end{pmatrix} - 0.1 \begin{pmatrix} 6 \\ 10 \end{pmatrix} = \begin{pmatrix} 0.4 \\ 0 \end{pmatrix}$$

f(x⁰) = f((1,1)ᵀ) = 3

f(x⁰ − α∇f(x⁰)) = f((0.4, 0)ᵀ) = −4.84

Indeed f(x⁰) = 3 ≥ −4.84 = f(x⁰ − α∇f(x⁰))

---

## 第 ⑦ 页：步长过大的反例

- Let's start at the point x⁰ = (1, 1)ᵀ and move against the gradient for a step size of 0.5.

$$\nabla f((1,1)^\top) = \begin{pmatrix} 6 \\ 10 \end{pmatrix}$$

The new point we reach is:

$$\begin{pmatrix} 1 \\ 1 \end{pmatrix} - 0.5 \begin{pmatrix} 6 \\ 10 \end{pmatrix} = \begin{pmatrix} -2 \\ -4 \end{pmatrix}$$

f(x⁰) = f((1,1)ᵀ) = 3

f(x⁰ − α∇f(x⁰)) = f((−2, −4)ᵀ) = 79

We do not have f(x⁰) = 3 ≥ 79 = f(x⁰ − α·∇f(x⁰))

---

## 第 ⑧ 页：步长选择的图示

> **[图示说明]** 一维凸函数 f(x) 的曲线图，横轴为 x，纵轴为 f(x)。
>
> 标注了两个场景：
> 1. **步长足够小时**（标注 "x⁰ − α∇f(x⁰) step size is small enough"）：新点落在函数曲线的下降段，函数值确实减小。
> 2. **步长过大时**（标注 "x⁰ − α∇f(x⁰) step size is too large"）：新点越过了最小值，落在另一侧的上升段，函数值反而增大。
>
> 在函数递减区域标注 ∇f(x⁰) ≤ 0。
>
> 图示底部：Because ∇f(x⁰) ≤ 0, x⁰ − α∇f(x⁰) ≥ x⁰ （即逆梯度方向使 x 增大，向右移动向最小值靠近）。
>
> 图示目的：直观说明步长α过大会导致"过冲"（overshoot），梯度下降失败。

---

## 第 ⑨ 页：梯度下降算法

- We immediately get a crude algorithm to minimize a function. Start at any point x⁰ and generate a sequence of points x¹, x², x⁵, ... by moving against the gradient for small step sizes.

- At iteration k of the algorithm, we are at point xᵏ. Say we use the step size αᵏ. Then the next point we reach is

$$x^{k+1} = x^k - \alpha^k \nabla f(x^k)$$

- Find αᵏ by "trial and error". In particular, pick any value for αᵏ. Check if f(xᵏ − αᵏ ∇f(xᵏ)) < f(xᵏ)
  - if yes, take the step
  - if no, halve αᵏ and try again.
