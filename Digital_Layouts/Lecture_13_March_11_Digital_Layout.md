# Lecture 13 — March 11, 2026 — 忠实转制笔记

---

## 第 ① 页：Projected Gradient Descent

For convex function f: ℝⁿ → ℝ and convex set 𝕏 ⊆ ℝⁿ, consider the constrained minimization problem:

$$\min \quad f(x)$$
$$\text{s.t.} \quad x \in \mathbb{X}$$

Projected gradient descent works as follows:

Start at some initial point x⁰, generate a sequence of points y¹, x¹, y², x², y³, x³, ... as:

$$y^{k+1} = x^k - \gamma \, \nabla f(x^k)$$

$$x^{k+1} = \Pi_\mathbb{X}(y^{k+1})$$

---

## 第 ② 页：投影梯度下降示意图与投影的重要性质引入

> **[图示说明]** 一个凸集 𝕏（封闭区域）。从集合内部的点 xᵏ 出发，沿负梯度方向走一步（红色线段），到达 yᵏ⁺¹ = xᵏ − γ∇f(xᵏ)（可能在 𝕏 外部），然后投影回 𝕏 的边界上，得到 xᵏ⁺¹ = Π_𝕏(yᵏ⁺¹)（箭头指向边界上的点）。

**Definition:** The **projection** of y onto the convex set 𝕏 is

$$\Pi_\mathbb{X}(y) = \arg\min_{x \in \mathbb{X}} \|x - y\|^2$$

Let's show an important property of projections.

**Recall:** If g is convex,

$$g(w) = g(z) + \nabla g(z)^\top (w - z) + o(\|w - z\|) \quad \text{(Taylor)}$$

**Recall:** xᵀy = ‖x‖·‖y‖·cos θ, where θ is the angle between x and y.

Thus, if xᵀy ≤ 0, then the angle between x and y is obtuse.

---

## 第 ③ 页：∇‖x‖² 的计算与投影的重要性质

**Note:** Let g(x) = ‖x‖², compute ∇g(x).

$$g(x) = \sum_i (x_i)^2, \quad \frac{\partial g(x)}{\partial x_i} = 2x_i$$

$$\nabla g(x) = (2x_1, \ldots, 2x_n) = 2x$$

**Important property of projections:** Let 𝕏 be a convex set, x ∈ 𝕏, and y ∈ ℝⁿ. We have:

$$(x - \Pi_\mathbb{X}(y))^\top (y - \Pi_\mathbb{X}(y)) \leq 0$$

> **[图示说明]** 凸集 𝕏（封闭区域）。点 y 在集合外部，Π_𝕏(y) 在集合边界上，x 在集合内部。从 Π_𝕏(y) 分别指向 x 的向量 (x − Π_𝕏(y)) 和指向 y 的向量 (y − Π_𝕏(y)) 之间的夹角是钝角（≥ 90°）。

---

## 第 ④ 页：投影钝角性质的证明（上）

**Proof:** We want to show (x − π)ᵀ(y − π) ≤ 0 where π = Π_𝕏(y).

> **[图示说明]** 凸集 𝕏，y 在集合外部，π = Π_𝕏(y) 在边界上，x 在集合内部。

For any λ ∈ [0, 1), because 𝕏 is convex, x ∈ 𝕏, π ∈ 𝕏, we have

$$\lambda x + (1 - \lambda) \pi \in \mathbb{X}$$

Because π = argmin_{z∈𝕏} ‖z − y‖², and λx + (1−λ)π ∈ 𝕏, we must have:

$$\|y - \pi\|^2 \leq \|y - (\lambda x + (1 - \lambda)\pi)\|^2 = \|y - \pi + \lambda(\pi - x)\|^2 \quad (\ddagger)$$

Let's apply Taylor's theorem with w = y − π + λ(π − x), z = y − π, g(x) = ‖x‖².

$$g(y - \pi + \lambda(\pi - x)) = g(y - \pi) + \nabla g(y - \pi)^\top \big(y - \pi + \lambda(\pi - x) - (y - \pi)\big) + o\big(\|y - \pi + \lambda(\pi - x) - (y - \pi)\|\big)$$

$$\|y - \pi + \lambda(\pi - x)\|^2 = \|y - \pi\|^2 + 2\lambda(y - \pi)^\top(\pi - x) + o(\lambda \|\pi - x\|)$$

By (‡): ‖y − π‖² ≤ this expression, so ‖y − π‖² appears on both sides.

---

## 第 ⑤ 页：投影钝角性质的证明（下）

Continuing:

$$\|y - \pi\|^2 \leq \|y - \pi\|^2 + 2\lambda(y - \pi)^\top(\pi - x) + o(\lambda \|\pi - x\|) \quad \text{for any } \lambda \in [0, 1)$$

$$\Longrightarrow \quad 0 \leq 2\lambda(y - \pi)^\top(\pi - x) + o(\lambda \|\pi - x\|)$$

$$\Longrightarrow \quad 0 \leq 2(y - \pi)^\top(\pi - x) + \frac{o(\lambda \|\pi - x\|)}{\lambda}$$

$$\Longrightarrow \quad 0 \leq 2(y - \pi)^\top(\pi - x) + \frac{o(\lambda \|\pi - x\|)}{\lambda \cdot \|\pi - x\|} \cdot \|\pi - x\|$$

As λ → 0:

$$\lim_{\lambda \to 0} \frac{o(\lambda \|\pi - x\|)}{\lambda \cdot \|\pi - x\|} = 0$$

$$\Longrightarrow \quad 0 \leq 2(y - \pi)^\top(\pi - x)$$

$$\Longrightarrow \quad (y - \pi)^\top(x - \pi) \leq 0$$

Done! ∎

---

## 第 ⑥ 页：投影的勾股定理型不等式

Here's an immediate implication of property of projections:

Let 𝕏 be convex, x ∈ 𝕏 and y ∈ ℝⁿ. Then:

$$\|x - y\|^2 \geq \|x - \Pi_\mathbb{X}(y)\|^2 + \|y - \Pi_\mathbb{X}(y)\|^2$$

> **[图示说明]** 三角形：顶点分别为 y（集合外）、Π_𝕏(y)（集合边界）、x（集合内）。凸集 𝕏 包含 Π_𝕏(y) 和 x。不等式类似勾股定理但取 ≥，因为 Π_𝕏(y) 处的角为钝角。

**Recall:** ‖a + b‖² = ‖a‖² + ‖b‖² + 2aᵀb.

**Proof:** Let π = Π_𝕏(y).

$$\|x - y\|^2 = \|x - \pi + \pi - y\|^2 = \|x - \pi\|^2 + \|\pi - y\|^2 + 2(x - \pi)^\top(\pi - y)$$

$$= \|x - \pi\|^2 + \|y - \pi\|^2 - 2(x - \pi)^\top(y - \pi)$$

From the previous result, (x − π)ᵀ(y − π) ≤ 0, so −2(x − π)ᵀ(y − π) ≥ 0.

$$\|x - y\|^2 \geq \|x - \pi\|^2 + \|y - \pi\|^2$$

Done! ∎

---

## 第 ⑦ 页：Convergence of Projected Gradient Descent with Bounded Gradients

Projected gradient descent is:

$$y^{k+1} = x^k - \gamma \, \nabla f(x^k), \quad x^{k+1} = \Pi_\mathbb{X}(y^{k+1})$$

Let's refine the preliminary analysis we did when we proved convergence for gradient descent without projections.

Let x\* = argmin_{x∈𝕏} f(x), we want to bound f(xᵏ) − f(x\*).

By first order characterization of convexity:

$$f(x^*) \geq f(x^k) + \nabla f(x^k)^\top (x^* - x^k)$$

$$\Longrightarrow \quad f(x^k) - f(x^*) \leq \nabla f(x^k)^\top (x^k - x^*)$$

So it is enough to upper bound ∇f(xᵏ)ᵀ(xᵏ − x\*).

$$\nabla f(x^k)^\top (x^k - x^*) = \frac{1}{\gamma} (x^k - y^{k+1})^\top (x^k - x^*)$$

**Recall:** ‖a − b‖² = ‖a‖² + ‖b‖² − 2aᵀb ⟹ 2aᵀb = ‖a‖² + ‖b‖² − ‖a − b‖².

---

## 第 ⑧ 页：收敛分析的展开（续）

Thus,

$$\nabla f(x^k)^\top (x^k - x^*) = \frac{1}{2\gamma} \left( \|x^k - y^{k+1}\|^2 + \|x^k - x^*\|^2 - \|(x^k - y^{k+1}) - (x^k - x^*)\|^2 \right)$$

$$= \frac{1}{2\gamma} \left( \|x^k - y^{k+1}\|^2 + \|x^k - x^*\|^2 - \|y^{k+1} - x^*\|^2 \right)$$
