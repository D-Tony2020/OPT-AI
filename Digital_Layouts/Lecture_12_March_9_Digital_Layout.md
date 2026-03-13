# Lecture 12 — March 9, 2026 — 忠实转制笔记

---

## 第 ① 页：Convex Sets

**Definition:** The set C ⊆ ℝⁿ is **convex** if for all x, y ∈ C and λ ∈ [0, 1], we have

$$\lambda x + (1 - \lambda) y \in C$$

> **[图示说明——左图]** 一个椭圆形集合（凸集），内部取两点 x 和 y，连线段完全在集合内部，凸组合点 λx+(1−λ)y 在集合中。标注 "convex"。
>
> **[图示说明——右图]** 一个弯月形/豆形集合（非凸集），内部取两点 x 和 y，连线段部分在集合外部，凸组合点 λx+(1−λ)y 在集合外。标注 "not convex"。

- Intuitively, for a convex set, we have direct line of sight from any point in the set to any other point in the set.

---

## 第 ② 页：凸集的交集性质

- If C₁, C₂, ..., Cₗ are all convex, then C₁ ∩ C₂ ∩ ... ∩ Cₗ is also convex.

> **[图示说明]** 三个凸集 C₁、C₂、C₃ 的 Venn 图。C₁ 和 C₂ 各为椭圆形，C₃ 在下方。三者的交集 C₁ ∩ C₂ ∩ C₃（红色斜线标注区域）也是凸的。
>
> 旁注（红色）：the intersection of bunch of convex sets is still convex.

**Pf:** To show C₁ ∩ C₂ ∩ ... ∩ Cₗ is convex, take any x, y ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ and any λ ∈ [0, 1], we want to show λx + (1−λ)y ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ.

Because x ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ, we have x ∈ C₁.
y ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ, we have y ∈ C₁.
Because C₁ is convex, x ∈ C₁ and y ∈ C₁, we have λx + (1−λ)y ∈ C₁.

Similarly, because x ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ, we have x ∈ C₂.
y ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ, we have y ∈ C₂.
Because C₂ is convex, x ∈ C₂, and y ∈ C₂, we have λx + (1−λ)y ∈ C₂.

Continuing similarly: λx + (1−λ)y ∈ C₃, ..., λx + (1−λ)y ∈ Cₗ.

Then we must have λx + (1−λ)y ∈ C₁ ∩ C₂ ∩ ... ∩ Cₗ. Done! ∎

---

## 第 ③ 页：凸函数的下水平集是凸集

- Let f: ℝⁿ → ℝ be convex, define C = {x ∈ ℝⁿ : f(x) ≤ 0}.

  Then C is convex.

**Pf:** Take any x, y ∈ C and λ ∈ [0, 1]. We want to show λx + (1−λ)y ∈ C.

Because x ∈ C, we have f(x) ≤ 0.
y ∈ C, we have f(y) ≤ 0.

Thus,

$$f(x) \leq 0 \quad \times \lambda$$
$$f(y) \leq 0 \quad \times (1 - \lambda)$$
$$\lambda f(x) + (1 - \lambda) f(y) \leq 0$$

Also, because f is convex:

$$f\left(\lambda x + (1 - \lambda) y\right) \leq \lambda f(x) + (1 - \lambda) f(y)$$

So we get:

$$f\left(\lambda x + (1 - \lambda) y\right) \leq \lambda f(x) + (1 - \lambda) f(y) \leq 0$$

Because f(λx + (1−λ)y) ≤ 0, we have λx + (1−λ)y ∈ C. Done! ∎

---

## 第 ④ 页：约束优化问题的可行域是凸集

- Why are these two results useful? Consider the constrained optimization problem:

$$\min \quad f(x)$$
$$\text{s.t.} \quad g_1(x) \leq 0, \; g_2(x) \leq 0, \; \ldots, \; g_m(x) \leq 0,$$

where g₁, g₂, ..., gₘ are convex functions.

- The set of feasible solutions for this problem is

$$C = \{x \in \mathbb{R}^n : g_1(x) \leq 0, \; g_2(x) \leq 0, \; \ldots, \; g_m(x) \leq 0\}$$

- Define Cᵢ = {x ∈ ℝⁿ : gᵢ(x) ≤ 0}.

  Because gᵢ is convex, Cᵢ is convex for all i = 1, ..., m.

- Also C = C₁ ∩ C₂ ∩ ... ∩ Cₘ.

  Because C₁, ..., Cₘ are all convex, C is convex as well because C is the intersection of a bunch of convex sets.

- Thus, if we have an optimization problem with left side of all ≤ constraints are convex functions, then the set of feasible solutions is a convex set.

---

## 第 ⑤ 页：Projected Gradient Descent

**Definition:** The function f: ℝⁿ → ℝ is **strictly convex** if

$$f\left(\lambda x + (1 - \lambda) y\right) < \lambda f(x) + (1 - \lambda) f(y) \quad \text{for all } x, y \in \mathbb{R}^n, \; x \neq y, \; \lambda \in (0, 1)$$

- Let g: ℝⁿ → ℝ be a strictly convex function and 𝕏 be a convex set. Consider the constrained optimization problem:

$$\min \quad g(x)$$
$$\text{s.t.} \quad x \in \mathbb{X}$$

  Then, there is a **unique** optimal solution to this problem.

  (proof on next page)

---

## 第 ⑥ 页：严格凸函数在凸集上有唯一最优解的证明

**Pf:** To get a contradiction, let x\* and y\* with x\* ≠ y\* be two optimal solutions to the problem above. Let z\* be the opt. obj. value. So, z\* = min_{x∈𝕏} g(x).

- Because x\* is an opt. solution, we have x\* ∈ 𝕏 and g(x\*) = z\*.
  Because y\* is an opt. solution, we have y\* ∈ 𝕏 and g(y\*) = z\*.

- Because x\* ∈ 𝕏, y\* ∈ 𝕏 and 𝕏 is convex, we have λx\* + (1−λ)y\* ∈ 𝕏.

- Because g is strictly convex, g(x\*) = z\* and g(y\*) = z\*, we have

$$g\left(\lambda x^* + (1 - \lambda) y^*\right) < \lambda \, g(x^*) + (1 - \lambda) \, g(y^*) = \underbrace{\lambda z^*}_{} + \underbrace{(1 - \lambda) z^*}_{} = z^*$$

- Thus, λx\* + (1−λ)y\* ∈ 𝕏, so λx\* + (1−λ)y\* is feasible to the opt. problem.
  Also, g(λx\* + (1−λ)y\*) < z\*, so λx\* + (1−λ)y\* is giving an obj. value strictly better than z\*, which is the obj. value from x\* or y\*.

So x\* and y\* cannot be an optimal solution. Contradiction!

Done! ∎

---

## 第 ⑦ 页：投影梯度下降的动机

- When we apply gradient descent for a constrained opt. problem, moving against the gradient can take us outside the set of feasible solutions.

> **[图示说明——左图]** 凸集 𝕏（蓝色椭圆）。在集合内部点 x 处，梯度 ∇f(x) 指向某方向。沿负梯度方向走步长 γ 后到达点 x − γ∇f(x)，该点在集合**外部**（红色标注 ✗）。
>
> **[图示说明——右图]** 同样的凸集 𝕏。从 x 处沿负梯度走到 x − γ∇f(x)（在集合外部），然后将其**投影**回集合表面（蓝色虚线箭头），到达集合边界上的点。

- Projected gradient descent is useful for solving constrained optimization problems. Assume that we are currently at point x ∈ 𝕏 and move against the gradient for a step size of γ to reach the point x − γ∇f(x).

- If x − γ∇f(x) is not in the feasible region 𝕏, then we project x − γ∇f(x) back on to 𝕏. That is, we find the point in 𝕏 that is closest to x − γ∇f(x).

---

## 第 ⑧ 页：投影算子与投影梯度下降算法

- Define the **projection** of x on to the convex set 𝕏 as

$$\Pi_\mathbb{X}(x) = \arg\min_{y \in \mathbb{X}} \|y - x\|^2$$

> **[图示说明]** 一个圆形凸集 𝕏。点 x 在集合外部。从 x 向集合作垂线，垂足即为投影 Π_𝕏(x)——集合中距 x 最近的点。

- Note: if x ∈ 𝕏, then Π_𝕏(x) = x.

- The projected gradient descent works as follows. We start with some initial point x⁰. At any iteration k, we do the following:

$$y^{k+1} = x^k - \gamma \, \nabla f(x^k)$$

$$x^{k+1} = \Pi_\mathbb{X}(y^{k+1})$$
