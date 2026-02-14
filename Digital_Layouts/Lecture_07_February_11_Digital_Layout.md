# Lecture 7 — February 11, 2026 — 忠实转制笔记

---

## 第 ① 页：Barrier Function Method for Constrained Optimization

- Applicable to problems with only inequality constraints.

$$\min \quad f(x)$$
$$\text{s.t.} \quad g_1(x) \leq 0$$
$$\quad\quad\quad \vdots$$
$$\quad\quad\quad g_n(x) \leq 0$$

- When we are close to the boundary of first constraints within inside, gᵢ(x) is a very small, negative number.

- So −gᵢ(x) is a very small, positive number.

- Then, 1/(−gᵢ(x)) is a very large positive number.

- Solve the problem:

$$\min_x \quad f(x) - \sum_{i=1}^{n} \mu_i \frac{1}{g_i(x)},$$

where μ₁, ..., μₙ are small barrier multipliers.

---

## 第 ② 页：障碍函数法的几何直觉

> **[图示说明]** 三维示意图。水平面为决策变量空间（x₁, x₂ 两个轴），垂直轴为函数值。
>
> 可行域（feasible region）标注在水平面上，由约束 gᵢ(x̂) < 0 定义。约束边界 gᵢ(x̂) = 0 也标出。
>
> 碗形等高线曲面表示目标函数 f(x)，最小值点 x* 在可行域内部。
>
> 垂直方向上标注 −Σμᵢ · 1/gᵢ(x)——这个障碍项在接近可行域边界时趋于无穷大（形成"高墙"），阻止迭代点离开可行域。
>
> 图示目的：说明障碍函数在可行域边界处形成一道"墙"，使得迭代点始终留在可行域内部。

---

## 第 ③ 页：障碍函数法的完整算法

① Choose the initial barrier parameters μ₁, ..., μₙ, pick a stopping tolerance ε. Pick shrinkage factor β ∈ (0, 1). Set iteration counter k = 1. Pick x⁰ strictly inside feasible region, gᵢ(x⁰) < 0, i = 1, ..., n.

② Solve the problem:

$$x^k = \arg\min \quad f(x) - \sum_{i=1}^{n} \mu_i^k \frac{1}{g_i(x)}$$

by using gradient descent by starting from the initial point xᵏ⁻¹.

③ If

$$-\sum_{i=1}^{n} \mu_i^k \frac{1}{g_i(x^k)} \leq \varepsilon \quad \text{then stop.}$$

Else

$$\mu_i^{k+1} = \beta \, \mu_i^k, \quad i = 1, \ldots, n,$$

increase iteration counter by 1, and go to Step 2.

- **Example:** min (x₁ − 2)⁴ + (x₁ − 2x₂)² s.t. x₁² − x₂ ≤ 0

---

## 第 ④ 页：Taylor's Theorem（预备知识）

- We will use the notation o(t) to denote any single-dimensional function g: ℝ → ℝ that satisfies

$$\lim_{t \to 0} \frac{g(t)}{t} = 0$$

- For example, g(t) = t². Is g(t) o(t) function?

$$\lim_{t \to 0} \frac{g(t)}{t} = \lim_{t \to 0} \frac{t^2}{t} = \lim_{t \to 0} t = 0 \quad \text{Yes}$$

> **[图示]** t² 的曲线（红色），在原点附近增长比 t 慢。

- For example, g(t) = 2t. Is g(t) o(t) function? No.

$$\lim_{t \to 0} \frac{2t}{t} = 2$$

> **[图示]** 2t 的直线（红色），斜率不为零。

- For example, g(t) = √t. Is g(t) o(t) function? No.

$$\lim_{t \to 0} \frac{\sqrt{t}}{t} = \lim_{t \to 0} \frac{1}{\sqrt{t}} = \infty$$

> **[图示]** √t 的曲线（红色），在原点附近增长比 t 快。

---

## 第 ⑤ 页：Taylor's Theorem 的陈述

- Let's focus on infinitely differentiable function f: ℝⁿ → ℝ. This means that

$$\frac{\partial^k f(x)}{\partial x_{i_1} \, \partial x_{i_2} \cdots \partial x_{i_k}}$$

exists for all k = 1, 2, ...

- **Taylor's Theorem:**

$$f(y) = f(x) + \nabla^T f\left(\gamma x + (1 - \gamma) y\right) \cdot (y - x) \quad \text{for some } \gamma \in (0, 1)$$

> **[图示说明]** 一维函数图。横轴标出 x 和 y，以及中间点 γx + (1−γ)y。在 x 处画出函数值 f(x)，在 y 处画出 f(y)。从 x 点处的切线延伸到某处。在中间点 γx+(1−γ)y 处画出切线（红色点标注）。
>
> 蓝色虚线连接 (x, f(x)) 和 (y, f(y))。图示表明存在某个中间点的梯度使得 Taylor 定理成立——类似中值定理的几何含义。

---

## 第 ⑥ 页：Taylor's Theorem 的推论

- Recall: ‖x‖ = √(Σxᵢ²), xᵀy = ‖x‖ ‖y‖ cos θ, where θ is the angle between x and y.

- **Implication of Taylor's Theorem:**

$$f(y) = f(x) + \nabla^T f(x) \cdot (y - x) + o(\|y - x\|)$$

**Pf:** By Taylor's Thm:

$$f(y) = f(x) + \nabla^T f\left(\theta x + (1 - \theta) y\right) \cdot (y - x) \quad \text{for some } \theta \in (0, 1)$$

$$f(y) = f(x) + \nabla^T f\left(\theta x + (1 - \theta) y\right) \cdot (y - x)$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \left(\nabla^T f\left(\theta x + (1 - \theta) y\right) - \nabla^T f(x)\right)(y - x)$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \left\|\nabla^T f\left(\theta x + (1 - \theta) y\right) - \nabla^T f(x)\right\| \; \|y - x\| \; \cos \phi$$

where φ is the angle between ∇ᵀf(θx + (1−θ)y) − ∇ᵀf(x) and y − x.

If we can show that ‖∇ᵀf(θx + (1−θ)y) − ∇ᵀf(x)‖ · ‖y−x‖ cos φ is o(‖y−x‖) then we are done.

---

## 第 ⑦ 页：Taylor's Theorem 推论的证明完成

Check:

$$\lim_{\|y - x\| \to 0} \frac{\left\|\nabla^T f\left(\gamma x + (1 - \gamma) y\right) - \nabla^T f(x)\right\| \; \|y - x\| \; \cos \phi}{\|y - x\|} = 0$$

because if ‖y − x‖ → 0, then y − x → 0, then

$$\nabla^T f\left(\gamma x + (1 - \gamma) y\right) \longrightarrow \nabla^T f(x)$$

so the numerator's gradient difference → 0. ∎
