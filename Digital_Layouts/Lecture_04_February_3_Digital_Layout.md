# Lecture 4 — February 3, 2026 — 忠实转制笔记

---

## 第 ① 页：A Crude Gradient Descent Algorithm for Minimizing a Function

- Let f: ℝⁿ → ℝ be a function we want to minimize.

- Start with any point x⁰. Generate a sequence of points x¹, x², ... by moving against the gradient for small step sizes.

- Say at iteration k, we are at point xᵏ. The next point is computed as

$$x^{k+1} = x^k - \alpha^k \cdot \nabla f(x^k)$$

where αᵏ is a "small" step size.

- How small is small? Do trial and error. Pick any αᵏ and check if

$$f(x^k - \alpha^k \nabla f(x^k)) < f(x^k)$$

- If so, take the step.
- If not, halve αᵏ and try again.

---

## 第 ② 页：梯度计算示例与梯度下降的期望

- Say we want to minimize

$$f(x) = 4 \cdot \ln(e^{2x_1} + e^{4x_2}) - 6x_1 - 4x_2$$

$$\nabla f(x) = \left( 4 \cdot \frac{2 e^{2x_1}}{e^{2x_1} + e^{4x_2}} - 6, \quad 4 \cdot \frac{4 e^{4x_2}}{e^{2x_1} + e^{4x_2}} - 4 \right)$$

- What is the best we can expect from gradient descent?

> **[图示说明]** 两幅一维函数曲线图并排：
>
> **左图：** 凸函数曲线，xᵏ 在曲线左侧（递减段）。∇f(xᵏ) = f'(xᵏ) ≤ 0，负梯度方向向右。新点 xᵏ − α∇f(xᵏ) ≥ xᵏ，即向右移动，朝最小值靠近。
>
> **右图：** 凸函数曲线，xᵏ 在曲线右侧（递增段）。∇f(xᵏ) = f'(xᵏ) > 0，负梯度方向向左。新点 xᵏ − α∇f(xᵏ) ≤ xᵏ，即向左移动，朝最小值靠近。
>
> 图示目的：说明无论起点在最小值的左边还是右边，沿负梯度方向移动总是朝着最小值的方向。

---

## 第 ③ 页：局部最小值 vs 全局最小值

> **[图示说明]** 一维非凸函数 f(x) = (x²−2)²·(x−1)² 的曲线图。函数有多个极值点：
> - 左侧有一个**局部最小值**（红色标注 "local minimum"），梯度下降从附近的 xᵏ 出发，沿负梯度方向移动到 xᵏ⁺¹，陷入该局部最小值。
> - 右侧有一个**全局最小值**（红色标注 "global minimum"，记为 x\*），从另一个起点 xᵏ 出发可到达。
>
> 图示目的：说明梯度下降可能陷入局部最小值而非全局最小值。

- **global minimum** x\*: x\* satisfies f(x\*) ≤ f(x) for any x

- **local minimum** x̂: x̂ satisfies f(x̂) ≤ f(x) for any x in a small neighborhood of x̂

> **[补充图示]** 两个小图：左侧为一个有明确谷底的曲线，标注 "local minimum"；右侧为一个鞍点（非局部最小值），标注 "not a local minimum"。

- **Best we can expect from gradient descent is we get to a local minimum.**

---

## 第 ④ 页：凸函数——局部最小值即全局最小值

- If we are minimizing a convex function, then there is no distinction between global and local minimum. All local minima are also global minima.

> **[图示说明]** 三幅图：
>
> **左上（一维凸函数）：** 标准 U 形曲线，只有一个最低点，标注 "global minimum, also only local minimum"。
>
> **右上（二维凸函数）：** 碗形三维曲面（等高线为同心椭圆），标注 "global, only local minimum"。从侧面看如一个碗。
>
> **下方（非严格凸函数）：** 曲线有一段平坦的底部（多个点取得同一最小值），标注 "global and local minima"（多个全局最小值点）。
>
> 图示目的：说明凸函数的不同情形——唯一最小值、碗形曲面、平坦底部——但都满足"局部即全局"。

---

## 第 ⑤ 页：凸性保证与精确线搜索

- If we a priori know that the objective function we are minimizing is convex, then we can be sure about the success of the gradient descent algorithm.

- When the obj. function is not convex, a heuristic method to try to find a global minimum is to start gradient descent with different initial points.

- Let's focus on the step size selection problem: Currently we are at point xᵏ. We computed ∇f(xᵏ).

- Choose the step size α such that the value of the function at the next point we reach is as small as possible. In other words, solve the optimization problem:

$$\min_{\alpha \geq 0} f\left(x^k - \alpha \, \nabla f(x^k)\right)$$

其中 xᵏ 和 ∇f(xᵏ) 均为已知量。

This optimization problem is a **single-dimensional optimization problem** over α.

---

## 第 ⑥ 页：Single Dimensional Optimization

- We have a function f: ℝ → ℝ and we want to minimize f.

- We will construct iterative algorithms to minimize f. We will start with an interval of uncertainty [a¹, b¹] over which we want to find the minimizer of f.

- Iteratively, we will reduce the width of the interval of uncertainty.

> **[图示说明]** 区间嵌套示意图：
> - [a¹, b¹]：初始最宽的区间
> - [a², b²]：缩小后的区间
> - [a³, b³]：进一步缩小
> - [a⁴, b⁴]：再缩小
> - [a⁵, b⁵]：最小的区间
>
> 每一步区间都严格包含在前一步之内，宽度逐步缩小，逼近最小值点。

---

## 第 ⑦ 页：迭代缩减不确定区间

- Here's how we will reduce the interval of uncertainty iteratively:

- Say we are at iteration k and the current interval of uncertainty is [aᵏ, bᵏ].

- Take two points in the interval of uncertainty, a left point λᵏ and a right point ρᵏ.

- Compute f(λᵏ) and f(ρᵏ).

---

## 第 ⑧ 页：区间更新规则

> **[图示说明——情况 ①]** 一维单峰函数曲线。区间 [aᵏ, bᵏ] 中取两个试探点 λᵏ（左）和 ρᵏ（右）。
>
> 当 **f(λᵏ) ≤ f(ρᵏ)** 时：最小值在 ρᵏ 的左侧，因此可以丢弃 ρᵏ 右边的区间。
> - aᵏ⁺¹ = aᵏ
> - bᵏ⁺¹ = ρᵏ
>
> 红色标注新区间 [aᵏ⁺¹, bᵏ⁺¹] = [aᵏ, ρᵏ]。

> **[图示说明——情况 ②]** 同一结构，但函数形态不同。
>
> 当 **f(λᵏ) > f(ρᵏ)** 时：最小值在 λᵏ 的右侧，因此可以丢弃 λᵏ 左边的区间。
> - aᵏ⁺¹ = λᵏ
> - bᵏ⁺¹ = bᵏ
>
> 红色标注新区间 [aᵏ⁺¹, bᵏ⁺¹] = [λᵏ, bᵏ]。

---

## 第 ⑨ 页：Bisection Search for Single Dimensional Minimization

- Different ways of choosing λᵏ and ρᵏ gives us different algorithms.

- Here's one way of picking λᵏ and ρᵏ: Place λᵏ and ρᵏ symmetrically around the midpoint of the interval of uncertainty.

> **[图示说明]** 数轴上标出区间 [aᵏ, bᵏ]，中点为 ½(aᵏ + bᵏ)。λᵏ 在中点左侧距离 ε 处，ρᵏ 在中点右侧距离 ε 处。双向红色箭头标注距离 ε。

- If we choose ε very small, we need to compare f(λᵏ) and f(ρᵏ), but when λᵏ and ρᵏ are very close, we run into numerical difficulties when comparing f(λᵏ) with f(ρᵏ).

- If we choose ε very large, then the interval of uncertainty will shrink slowly from one iteration to the next.
