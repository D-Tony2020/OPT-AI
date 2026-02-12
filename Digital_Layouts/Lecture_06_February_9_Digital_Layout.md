# Lecture 6 — February 9, 2026 — 忠实转制笔记

---

## 第 ① 页：Using Single-Dimensional Optimization to Find Best Step Size During Gradient Descent

- We are minimizing f: ℝⁿ → ℝ by using gradient descent. At iteration k of gradient descent we are at point xᵏ.

- We compute ∇f(xᵏ) and we want to move against the gradient for a certain step size α.

> **[图示说明]** 从点 xᵏ 出发，沿负梯度方向 ∇f(xᵏ)（红色虚线箭头）移动步长 α，到达点 xᵏ − α∇f(xᵏ)。蓝色虚线标注方向，红色实线标注梯度向量。

- Choose the step size α such that the value of the function at the next point we reach is as small as possible.

- Trying to find:

$$\min_{\alpha \geq 0} f\left(x^k - \alpha \, \nabla f(x^k)\right)$$

where xᵏ is known and ∇f(xᵏ) is known.

- This is a single dim. optimization problem.

---

## 第 ② 页：用二分搜索选步长的具体示例

- Say we are at iteration k of gradient descent and we are at point xᵏ (known).

- Compute ∇f(xᵏ) (known).

- Let's use single-dimensional optimization to find best step size.

- Think about what bisection search would look like to find the best step size.

> **[图示说明]** 将 f(xᵏ − α∇f(xᵏ)) 视为 α 的一元函数。在区间 [a, b] 上画出关于 α 的函数曲线（红色）。中点为 (a+b)/2，λ = (a+b)/2 − ε，ρ = (a+b)/2 + ε。对 f(xᵏ − λ∇f(xᵏ)) 与 f(xᵏ − ρ∇f(xᵏ)) 进行比较。

- Moreover:

$$f(x) = 4 \ln(e^{2x_1} + e^{4x_2}) - 6x_1 - 4x_2$$

using gradient descent.

$$\nabla f(x) = \left(\frac{8 e^{2x_1}}{e^{2x_1} + e^{4x_2}} - 6, \quad \frac{16 e^{4x_2}}{e^{2x_1} + e^{4x_2}} - 4\right)$$

- Use bisection search to find the best step size at each iteration.

---

## 第 ③ 页：Constrained Optimization

- Consider a constrained optimization problem of the form:

$$\min \quad f(x)$$
$$\text{s.t.} \quad g_1(x) \leq 0$$
$$\quad\quad\quad \vdots$$
$$\quad\quad\quad g_n(x) \leq 0$$
$$\quad\quad\quad h_1(x) = 0$$
$$\quad\quad\quad \vdots$$
$$\quad\quad\quad h_m(x) = 0,$$

where f, g₁, ..., gₙ, h₁, ..., hₘ : ℝⁿ → ℝ.

- In all the crude constrained optimization methods we will consider, we will convert the constrained optimization problem into an equivalent unconstrained optimization problem and use our standard gradient descent algorithm for unconstrained optimization.

---

## 第 ④ 页：Penalty Function Method

- Move the constraints into the objective function by largely penalizing the violations of the constraints.

- 原约束问题：

$$\min \quad f(x)$$
$$\text{s.t.} \quad g_1(x) \leq 0$$
$$\quad\quad\quad \vdots$$
$$\quad\quad\quad g_n(x) \leq 0$$
$$\quad\quad\quad h_1(x) = 0$$
$$\quad\quad\quad \vdots$$
$$\quad\quad\quad h_m(x) = 0$$

- Solve the unconstrained problem:

$$\min_x \quad f(x) + \sum_{i=1}^{n} \theta_i \left(\max\{0, \, g_i(x)\}\right)^2 + \sum_{i=1}^{m} \gamma_i \left(h_i(x)\right)^2,$$

where θ₁, ..., θₙ, γ₁, ..., γₘ are large penalty parameters.

---

## 第 ⑤ 页：为什么对不等式约束的惩罚项取平方？

- Why did we square the penalty term of the inequality constraint?

> **[图示说明——上左]** 函数 g(x) 的曲线，在 a 和 b 两个零点之间取负值（可行域），两侧取正值（不可行域）。
>
> **[图示说明——上右]** max{0, g(x)} 的曲线。可行域内为 0，不可行域内等于 g(x)。标注：max{0, g(x)} is not differentiable（在零点 a 和 b 处不可微）。
>
> **[图示说明——下]** (max{0, g(x)})² = r(x) 的曲线。函数变得光滑，在 a 和 b 处可微。在 b 附近标注 b⁺ 处的行为。

- r'(b⁻) = 0. r(x) to the right of b behaves like g²(x).

- r'(x) to the right of b behaves like 2g(x)·g'(x).

- r'(b⁺) = 2g(b⁺)·g'(b⁺) = 0 when b⁺ is very close to b.

---

## 第 ⑥ 页：惩罚函数法的完整算法与示例

① Choose the initial penalty parameters θ₁, ..., θₙ, γ₁, ..., γₘ, tolerance parameter ε, magnification parameter β > 1, set k = 1.

② Solve the problem:

$$\min_x \left\{ f(x) + \sum_{i=1}^{n} \theta_i^k \left(\max\{0, \, g_i(x)\}\right)^2 + \sum_{i=1}^{m} \gamma_i^k \left(h_i(x)\right)^2 \right\}$$

Let xᵏ be the optimal solution.

③ If

$$\sum_{i=1}^{n} \theta_i^k \left(\max\{0, \, g_i(x^k)\}\right)^2 + \sum_{i=1}^{m} \gamma_i^k \left(h_i(x^k)\right)^2 \leq \varepsilon, \quad \text{then stop.}$$

Else set

$$\theta_i^{k+1} = \beta \, \theta_i^k, \quad i = 1, \ldots, n \qquad \gamma_i^{k+1} = \beta \, \gamma_i^k, \quad i = 1, \ldots, m$$

and go to Step 2.

**Example:**

$$\min \quad (x_1 - 2)^4 + (x_1 - 2x_2)^2$$
$$\text{s.t.} \quad x_1^2 - x_2 \leq 0$$
