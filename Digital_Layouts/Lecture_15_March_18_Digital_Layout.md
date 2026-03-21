# Lecture 15 — March 18, 2026 — 忠实转制笔记

---

## 第 ① 页：Carrying out projections on to specially structured sets

- To do projections on to sets other than a box, we need some special tools. Consider the problem:

$$\min \quad f(x) \quad \text{s.t.} \quad g(x) = b, \; x \in \mathbb{X} \tag{‡}$$

where f: ℝⁿ → ℝ, g: ℝⁿ → ℝ.

- For fixed λ ∈ ℝ, consider the problem:

$$\min \quad f(x) + \lambda\bigl(g(x) - b\bigr) \quad \text{s.t.} \quad x \in \mathbb{X} \tag{*}$$

- Let x\*(λ) be a unique optimal solution to problem (\*).

- Find value of λ\* such that g(x\*(λ\*)) = b.

- Then x\*(λ\*) is an optimal solution to problem (‡).

---

## 第 ② 页：拉格朗日方法的正确性证明

**Pf:** Let x̂ be an optimal solution to (‡). So, g(x̂) = b, x̂ ∈ 𝕏.

Also, x\*(λ\*) satisfies g(x\*(λ\*)) = b, x\*(λ\*) ∈ 𝕏.

So we get f(x̂) ≤ f(x\*(λ\*)).

x\*(λ\*) is an opt. solution to (\*) with λ = λ\*, so x\*(λ\*) ∈ 𝕏, g(x\*(λ\*)) = b.

Also we have x̂ ∈ 𝕏, g(x̂) = b.

Because x\*(λ\*) is opt. solution to (\*), we get:

$$f(x^*(\lambda^*)) + \lambda^*\bigl(g(x^*(\lambda^*)) - b\bigr) \leq f(\hat{x}) + \lambda^*\bigl(g(\hat{x}) - b\bigr)$$

Both g terms are 0:

$$f(x^*(\lambda^*)) \leq f(\hat{x})$$

Thus, f(x\*(λ\*)) = f(x̂).

Also x\*(λ\*) ∈ 𝕏, g(x\*(λ\*)) = b. So x\*(λ\*) is a feasible solution to (‡) and gives an objective value equal to the opt. obj. value.

Thus x\*(λ\*) must be optimal to (‡). Done! ∎

---

## 第 ③ 页：Projections on to the Euclidean Ball

Consider the case

$$\mathbb{X} = \left\{ x \in \mathbb{R}^n : \sum_{i} x_i^2 \leq R^2 \right\}$$

> **[图示说明]** 二维坐标系（x₁, x₂ 轴），原点为圆心、半径为 R 的圆形区域即为欧几里得球 𝕏。

- We want to find Π_𝕏(ŷ), where ŷ is a fixed point.

- If ŷ ∈ 𝕏, then Π_𝕏(ŷ) = ŷ, so consider the case ŷ ∉ 𝕏, which means

$$\sum_{i} \hat{y}_i^2 > R^2$$

---

## 第 ④ 页：欧几里得球投影——拉格朗日求解

We want to solve

$$\min \quad \sum_{i} (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_{i} x_i^2 = R^2 \tag{‡}$$

For fixed λ, consider the problem:

$$\min_x \left\{ \sum_{i} (x_i - \hat{y}_i)^2 + \lambda\left(\sum_{i} x_i^2 - R^2\right) \right\}$$

The last problem is equivalent to:

$$\sum_{i} \min_{x_i} \left\{ (x_i - \hat{y}_i)^2 + \lambda \, x_i^2 \right\} - \lambda R^2$$

The opt. solution to each minimization problem above is given by:

$$2(x_i - \hat{y}_i) + 2\lambda x_i = 0 \implies x_i = \frac{1}{1+\lambda}\,\hat{y}_i$$

$$\implies x_i^*(\lambda) = \frac{1}{1+\lambda}\,\hat{y}_i$$

Find λ\* such that g(x\*(λ\*)) = b, which means $\sum_{i} (x_i^*(\lambda^*))^2 = R^2$. So we want:

$$\sum_{i} \frac{1}{(1+\lambda^*)^2}\,\hat{y}_i^2 = R^2 \implies \frac{1}{(1+\lambda^*)^2}\,\|\hat{y}\|^2 = R^2 \implies \frac{1}{1+\lambda^*} = \frac{R}{\|\hat{y}\|}$$

---

## 第 ⑤ 页：欧几里得球投影公式

Therefore

$$x_i^*(\lambda^*) = \frac{1}{1+\lambda^*}\,\hat{y}_i = \frac{R}{\|\hat{y}\|}\,\hat{y}_i$$

In this case, the opt. solution to (‡) is given by x\*, where

$$x_i^* = \frac{R}{\|\hat{y}\|}\,\hat{y}_i$$

Thus, the projection of ŷ on to Euclidean ball is given by π, where

$$\pi_i = \frac{R}{\|\hat{y}\|}\,\hat{y}_i \qquad \left(\pi = \frac{R}{\|\hat{y}\|}\,\hat{y}\right)$$

> **[图示说明]** 二维坐标系中的圆（欧几里得球）。点 ŷ 在圆外。从原点指向 ŷ 的方向上的单位向量为 ŷ/‖ŷ‖。投影点 Π_𝕏(ŷ) = R·ŷ/‖ŷ‖ 位于圆的边界上，即沿 ŷ 方向、距原点 R 的点。

- Projection on to Euclidean ball is useful for ridge regression.

---

## 第 ⑥ 页：Projection on to a Probability Simplex

Consider the case

$$\mathbb{X} = \left\{ x \in \mathbb{R}^n : \sum_{i} x_i = 1, \; x_i \geq 0, \; \forall\, i = 1, \ldots, n \right\}$$

> **[图示说明]** 二维坐标系（x₁, x₂ 轴），第一象限内从 (1,0) 到 (0,1) 的线段即为概率单纯形。

- We want to find Π_𝕏(ŷ), where ŷ is a fixed point.

---

## 第 ⑦ 页：概率单纯形投影——拉格朗日求解

We want to solve

$$\min \quad \sum_{i} (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_{i} x_i = 1, \; x_i \geq 0, \; \forall\, i = 1, \ldots, n$$

For fixed λ, consider the problem:

$$\min_{x_i \geq 0} \left\{ \sum_{i} (x_i - \hat{y}_i)^2 + \lambda\left(\sum_{i} x_i - 1\right) \right\}$$

The last problem is equivalent to:

$$\sum_{i} \min_{x_i \geq 0} \left\{ (x_i - \hat{y}_i)^2 + \lambda \, x_i \right\} - \lambda$$

To solve each minimization problem above:

$$2(x_i - \hat{y}_i) + \lambda = 0 \implies x_i = \hat{y}_i - \frac{\lambda}{2}$$

Thus, we get

$$x_i^*(\lambda) = \left[\hat{y}_i - \frac{\lambda}{2}\right]^+$$

---

## 第 ⑧ 页：分段最优解的图示

> **[图示说明——左图]** 绘制 $(x_i - \hat{y}_i)^2 + \lambda x_i$ 关于 $x_i$ 的图像，横轴 $x_i \geq 0$。当 $\hat{y}_i - \lambda/2 > 0$ 时，无约束最小值点 $\hat{y}_i - \lambda/2$ 在可行域内，因此最优解为 $x_i^* = \hat{y}_i - \lambda/2$。
>
> **[图示说明——右图]** 同样的函数图像，当 $\hat{y}_i - \lambda/2 < 0$ 时，无约束最小值点在可行域外（$x_i < 0$），因此在约束 $x_i \geq 0$ 下最优解为 $x_i^* = 0$。

---

## 第 ⑨ 页：概率单纯形投影公式

Find λ\* such that g(x\*(λ\*)) = b, which means $\sum_{i} x_i^*(\lambda^*) = 1$. So we want:

$$\sum_{i} \left[\hat{y}_i - \frac{\lambda^*}{2}\right]^+ = 1$$

Find λ\* such that

$$\sum_{i} \left[\hat{y}_i - \frac{\lambda^*}{2}\right]^+ = 1$$

(which is always possible).

Then $x_i^*(\lambda^*) = \left[\hat{y}_i - \lambda^*/2\right]^+$.

Thus, to project ŷ on to the probability simplex, find λ\* such that $\sum_{i} [\hat{y}_i - \lambda^*/2]^+ = 1$, then the projection π is given by

$$\pi_i = \left[\hat{y}_i - \frac{\lambda^*}{2}\right]^+$$

---

## 第 ⑩ 页：概率单纯形投影的几何示意

> **[图示说明——上方]** 二维概率单纯形（从 (1,0) 到 (0,1) 的线段）。点 ŷ 位于第一象限中单纯形上方，投影 Π_𝕏(ŷ) 在单纯形边上，为 ŷ 到单纯形的垂直投影。
>
> **[图示说明——下方]** 点 ŷ 在 x₁ 轴附近（靠近 (1,0) 端），投影 Π_𝕏(ŷ) 在单纯形靠近 x₁ 顶点的位置。
