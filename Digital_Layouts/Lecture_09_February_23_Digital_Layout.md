# Lecture 9 — February 23, 2026 — 忠实转制笔记

---

## 第 ① 页：复习与凸函数一阶刻画

- A function g(t) is called o(t) if it satisfies

$$\lim_{t \to 0} \frac{g(t)}{t} = 0$$

- Implication of Taylor's Theorem:

$$f(y) = f(x) + \nabla^T f(x) \cdot (y - x) + o(\|y - x\|)$$

- Definition of a convex function:
  f: ℝⁿ → ℝ is convex if and only if

$$f\left(\lambda x + (1 - \lambda) y\right) \leq \lambda \, f(x) + (1 - \lambda) \, f(y)$$

- **First-order Characterization of Convexity:**

  The function f: ℝⁿ → ℝ is convex if and only if

$$f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x) \quad \text{for all } x, y \in \mathbb{R}^n.$$

> **[图示说明]** 一维凸函数曲线。在点 x 处画出切线（绿色），切线始终在函数曲线下方。在点 y 处标注 f(y)，显示 f(y) 高于切线在 y 处的值。

---

## 第 ② 页：一阶刻画定理的证明（⟹ 方向）

> **[图示说明——左图]** 凸函数：两条切线均在曲线下方，交叉形成"X"形。
>
> **[图示说明——右图]** 非凸函数：切线在某些点穿过函数曲线，不构成全局下界。

**Theorem:** The function f: ℝⁿ → ℝ is convex if and only if

$$f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x) \quad \text{for all } x, y \in \mathbb{R}^n$$

**Pf (⟹):** If f is convex, then f(y) ≥ f(x) + ∇ᵀf(x)·(y−x) for all x, y ∈ ℝⁿ.

Because f is convex:

$$f\left(x + \lambda(y - x)\right) = f\left((1 - \lambda)x + \lambda y\right) \leq (1 - \lambda) f(x) + \lambda f(y)$$

$$\implies f\left(x + \lambda(y - x)\right) \leq (1 - \lambda) f(x) + \lambda f(y)$$

$$\implies \lambda f(y) \geq \lambda f(x) + f\left(x + \lambda(y - x)\right) - f(x)$$

$$\implies f(y) \geq f(x) + \frac{f\left(x + \lambda(y - x)\right) - f(x)}{\lambda}$$

---

## 第 ③ 页：一阶刻画定理的证明（⟹ 方向续）

$$= f(x) + \frac{\nabla^T f(x) \cdot \left(\cancel{x} + \lambda(y - x) - \cancel{x}\right) + o(\|\cancel{x} + \lambda(y - x) - x\|)}{\lambda}$$

(implication of Taylor's theorem)

$$= f(x) + \frac{\lambda \, \nabla^T f(x) \cdot (y - x) + o(\lambda \|y - x\|)}{\lambda}$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \frac{o(\lambda \|y - x\|)}{\lambda}$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \frac{o(\lambda \|y - x\|)}{\lambda \|y - x\|} \cdot \|y - x\| \qquad (\ddagger)$$

Note:

$$\lim_{\lambda \to 0} \frac{o(\lambda \|y - x\|)}{\lambda \|y - x\|} = 0$$

holds for all x, y and λ.

By (‡), for all x, y ∈ ℝⁿ, λ ∈ (0, 1), we have

$$f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x) + \frac{o(\lambda \|y - x\|)}{\lambda \|y - x\|} \cdot \|y - x\|$$

So,

$$f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x) + \lim_{\lambda \to 0} \frac{o(\lambda \|y - x\|)}{\lambda \|y - x\|} \cdot \|y - x\|$$

$$\leq \quad \text{If } f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x) \text{ for all } x, y \in \mathbb{R}^n, \text{ then f is convex.}$$

---

## 第 ④ 页：一阶刻画定理的证明（⟸ 方向）

Take any a, b ∈ ℝⁿ, and any λ ∈ [0, 1].

$$f(a) \geq f\left(\lambda a + (1 - \lambda) b\right) + \nabla^T f\left(\lambda a + (1 - \lambda) b\right) \cdot \left(a - (\lambda a + (1 - \lambda) b)\right)$$

$$f(b) \geq f\left(\lambda a + (1 - \lambda) b\right) + \nabla^T f\left(\lambda a + (1 - \lambda) b\right) \cdot \left(b - (\lambda a + (1 - \lambda) b)\right)$$

Multiply first inequality by λ, second by (1−λ):

$$f(a) \geq f\left(\lambda a + (1 - \lambda) b\right) + (1 - \lambda) \, \nabla^T f\left(\lambda a + (1 - \lambda) b\right) \cdot (a - b) \quad \times \lambda$$

$$f(b) \geq f\left(\lambda a + (1 - \lambda) b\right) - \lambda \, \nabla^T f\left(\lambda a + (1 - \lambda) b\right) \cdot (a - b) \quad \times (1 - \lambda)$$

Adding:

$$\lambda f(a) + (1 - \lambda) f(b) \geq \lambda \, f\left(\lambda a + (1 - \lambda) b\right) + (1 - \lambda) \, f\left(\lambda a + (1 - \lambda) b\right)$$

$$= f\left(\lambda a + (1 - \lambda) b\right)$$

So we have: λf(a) + λf(b) ≥ f(λa + (1−λ)b). Done. ∎

---

## 第 ⑤ 页：Minimizing Convex Functions

**Definition of a global minimum:**

The point x* is a global minimum of f if

$$f(x^*) \leq f(y) \quad \text{for all } y \in \mathbb{R}^n.$$

**Definition of a local minimum:**

The point x̂ is a local minimum of f if there exists some ε > 0 such that

$$f(\hat{x}) \leq f(y) \quad \text{for all } y \text{ that satisfies } \|y - \hat{x}\| \leq \varepsilon.$$

**Theorem:** Let f: ℝⁿ → ℝ be convex and x̂ be a local minimum of f. Then x̂ has to be a global minimum of f.

**Pf:** Fix any y ∈ ℝⁿ, we want to show that f(x̂) ≤ f(y).

Because x̂ is a local minimum of f, there exists some ε > 0, such that f(x̂) ≤ f(z) for any z that satisfies ‖z − x̂‖ ≤ ε.

Choose λ such that (1−λ) ≤ ε/‖x̂ − y‖, where y is as fixed at the beginning of the proof.

$$\|\lambda \hat{x} + (1 - \lambda) y - \hat{x}\| = \|(1 - \lambda)(y - \hat{x})\| = (1 - \lambda) \|y - \hat{x}\| \leq \varepsilon$$

---

## 第 ⑥ 页：局部最小值即全局最小值的证明完成

Thus λx̂ + (1−λ)y is within ε-neighborhood of local min x̂.

$$f(\hat{x}) \leq f\left(\lambda \hat{x} + (1 - \lambda) y\right)$$

Then, we get

$$f(\hat{x}) \leq f\left(\lambda \hat{x} + (1 - \lambda) y\right) \leq \lambda f(\hat{x}) + (1 - \lambda) f(y)$$

So f(x̂) ≤ λf(x̂) + (1−λ)f(y)

$$\implies (1 - \lambda) f(\hat{x}) \leq (1 - \lambda) f(y) \implies f(\hat{x}) \leq f(y) \quad \text{Done!}$$

---

**Theorem:** Let f: ℝⁿ → ℝ be a convex function. If ∇f(x̂) = 0, then x̂ is a global minimum of f.

**Pf:** Fix any point y ∈ ℝⁿ, we want to show f(x̂) ≤ f(y).

By first order characterization of convexity:

$$f(y) \geq f(\hat{x}) + \nabla^T f(\hat{x}) \cdot (y - \hat{x})$$

$$= f(\hat{x}) + \underbrace{\nabla^T f(\hat{x})}_{= 0} \cdot (y - \hat{x})$$

So we get f(y) ≥ f(x̂). Done! ∎
