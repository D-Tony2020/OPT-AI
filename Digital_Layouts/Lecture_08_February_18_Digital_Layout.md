# Lecture 8 — February 18, 2026 — 忠实转制笔记

---

## 第 ① 页：Taylor's Theorem（复习）

- We use the notation o(t) to denote any function g(t) such that

$$\lim_{t \to 0} \frac{g(t)}{t} = 0$$

- Recall x ∈ ℝⁿ, ‖x‖ = √(Σᵢ xᵢ²), xᵀx = (x₁, x₂, ..., xₙ)·(x₁, x₂, ..., xₙ)ᵀ = Σxᵢ² = ‖x‖².

- **Taylor's Theorem:** Let f: ℝⁿ → ℝ.

$$f(y) = f(x) + \nabla^T f\left(\gamma x + (1 - \gamma) y\right) \cdot (y - x), \quad \text{for some } \gamma \in (0, 1)$$

> **[图示说明]** 一维函数曲线（蓝色）。标出点 x（左）和 y（右），以及中间点 γx+(1−γ)y。在 x 处画 f(x)，在 y 处画 f(y)。红色直线为连接 (x, f(x)) 和 (y, f(y)) 的割线。蓝色虚线为某水平参考线。
>
> 函数曲线在 x 和 y 之间经历起伏，但中间点处的切线（标记为 t）与割线平行，体现中值定理的含义。

---

## 第 ② 页：Implication of Taylor's Theorem

- For any x, y ∈ ℝⁿ, we have

$$f(y) = f(x) + \nabla^T f(x) \cdot (y - x) + o(\|y - x\|) \qquad (\ddagger)$$

- Why does Taylor's theorem imply ‡?

By Taylor's theorem, for any x, y,

$$f(y) = f(x) + \nabla^T f\left(\gamma x + (1 - \gamma) y\right) \cdot (y - x), \quad \text{for some } \gamma \in (0, 1)$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \left(\nabla^T f\left(\gamma x + (1 - \gamma) y\right) - \nabla^T f(x)\right)(y - x)$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \left\|\nabla^T f\left(\gamma x + (1 - \gamma) y\right) - \nabla^T f(x)\right\| \; \|y - x\| \; \cos \theta,$$

where θ is the angle between ∇f(γx+(1−γ)y) − ∇f(x) and y − x.

We will be done if we can show

$$\left\|\nabla f\left(\gamma x + (1 - \gamma) y\right) - \nabla f(x)\right\| \; \|y - x\| \; \cos \theta \quad \text{is} \quad o(\|y - x\|)$$

---

## 第 ③ 页：Taylor 推论的证明完成与应用

Check:

$$\lim_{\|y - x\| \to 0} \frac{\left\|\nabla f\left(\gamma x + (1 - \gamma) y\right) - \nabla f(x)\right\| \; \|y - x\| \; \cos \theta}{\|y - x\|} = 0$$

Because ‖y − x‖ → 0 ⟺ y − x → 0 ⟺ γx + (1−γ)y → x, so the gradient difference → 0. ∎

**Why is the implication of Taylor's theorem important?**

Say we are at point x̂ with ∇f(x̂) ≠ 0. We have been saying that if we move against ∇f(x̂) for a small step size, the value of the function gets smaller. This result follows from the implication of Taylor's theorem. Why?

$$f(\hat{x} - \alpha \, \nabla f(\hat{x})) = f(\hat{x}) + \nabla^T f(\hat{x}) \cdot (\hat{x} - \alpha \, \nabla f(\hat{x}) - \hat{x}) + o(\|\hat{x} - \alpha \, \nabla f(\hat{x}) - \hat{x}\|)$$

$$= f(\hat{x}) - \alpha \, \|\nabla f(\hat{x})\|^2 + o(\alpha \, \|\nabla f(\hat{x})\|)$$

$$= f(\hat{x}) + \alpha \, \|\nabla f(\hat{x})\| \left(-\|\nabla f(\hat{x})\| + \frac{o(\alpha \, \|\nabla f(\hat{x})\|)}{\alpha \, \|\nabla f(\hat{x})\|}\right) \qquad (\dagger)$$

Note:

$$\lim_{\alpha \to 0} \frac{o(\alpha \, \|\nabla f(\hat{x})\|)}{\alpha \, \|\nabla f(\hat{x})\|} = 0,$$

this means that we can pick

---

## 第 ④ 页：梯度下降函数值下降的严格证明

α > 0 small enough such that

$$\frac{o(\alpha \, \|\nabla f(\hat{x})\|)}{\alpha \, \|\nabla f(\hat{x})\|} \leq \frac{1}{2} \|\nabla f(\hat{x})\|$$

Then by (†) we get

$$f(\hat{x} - \alpha \, \nabla f(\hat{x})) \leq f(\hat{x}) + \alpha \, \|\nabla f(\hat{x})\| \cdot \left(-\|\nabla f(\hat{x})\| + \frac{1}{2}\|\nabla f(\hat{x})\|\right)$$

$$= f(\hat{x}) - \frac{1}{2} \alpha \, \|\nabla f(\hat{x})\|^2 \leq f(\hat{x})$$

Thus, f(x̂ − α∇f(x̂)) ≤ f(x̂).

---

## 第 ⑤ 页：Convex Functions

**Def:** The function f: ℝⁿ → ℝ is **convex** if it satisfies:

$$f\left(\lambda x + (1 - \lambda) y\right) \leq \lambda \, f(x) + (1 - \lambda) \, f(y) \quad \text{for all } x, y \in \mathbb{R}^n, \; \lambda \in [0, 1]$$

> **[图示说明]** 一维凸函数曲线（蓝色）。标出点 x 和 y，连接 (x, f(x)) 和 (y, f(y)) 的红色直线段（弦）。在凸组合点 λx+(1−λ)y 处：
>
> - 函数值 f(λx+(1−λ)y)（曲线上的点）低于
> - 弦上的值 λf(x)+(1−λ)f(y)（红色直线上的点）
>
> 蓝色虚线标注各个函数值的水平。
>
> 图示目的：凸函数的定义——弦在曲线上方（"弦在上"）。

---

## 第 ⑥ 页：Jensen's Inequality

Let f: ℝⁿ → ℝ be convex, x¹, x², ..., xᵏ ∈ ℝⁿ, λ¹, λ², ..., λᵏ ∈ ℝ₊ such that Σλⁱ = 1. Then

$$f\left(\sum_{\ell=1}^{k} \lambda^\ell x^\ell\right) \leq \sum_{\ell=1}^{k} \lambda^\ell f(x^\ell)$$

**Pf:**

$$f\left(\sum_{\ell=1}^{k} \lambda^\ell x^\ell\right) = f\left(\lambda^1 x^1 + (1 - \lambda^1)\sum_{\ell=2}^{k} \frac{\lambda^\ell}{1 - \lambda^1} x^\ell\right)$$

$$\leq \lambda^1 f(x^1) + (1 - \lambda^1) \, f\left(\sum_{\ell=2}^{k} \frac{\lambda^\ell}{1 - \lambda^1} x^\ell\right) \quad \text{(f is convex)}$$

$$= \lambda^1 f(x^1) + (1 - \lambda^1) \, f\left(\frac{\lambda^2}{1 - \lambda^1} x^2 + \left(1 - \frac{\lambda^2}{1 - \lambda^1}\right) \sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)$$

$$\leq \lambda^1 f(x^1) + (1 - \lambda^1)\left(\frac{\lambda^2}{1 - \lambda^1} f(x^2) + \left(1 - \frac{\lambda^2}{1 - \lambda^1}\right) f\left(\sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)\right) \quad \text{(f is convex)}$$

$$= \lambda^1 f(x^1) + \lambda^2 f(x^2) + (1 - \lambda^1 - \lambda^2) \, f\left(\sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)$$

Continue similarly...

---

## 第 ⑦ 页：凸函数的梯度刻画

$$= \lambda^1 f(x^1) + \lambda^2 f(x^2) + (1 - \lambda^1 - \lambda^2) \, f\left(\sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)$$

Continue similarly.

**Gradient Based Characterization of Convexity**

**Theorem:** The function f: ℝⁿ → ℝ is convex if and only if

$$f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x)$$

> **[图示说明]** 一维凸函数 f（蓝色曲线）。在点 x 处画切线（红色直线）f(x) + ∇ᵀf(x)·(y−x)。
>
> 切线始终在函数曲线的**下方**——即 f(y) ≥ f(x) + ∇f(x)ᵀ(y−x) 对所有 y 成立。
>
> 切线在 x 处与曲线相切（等号成立），但在其他所有点上严格位于曲线下方。
>
> 图示目的：凸函数的梯度刻画——切线（一阶 Taylor 近似）是全局下界。
