# Lecture 5 — February 4, 2026 — 忠实转制笔记

---

## 第 ① 页：Single Dimensional Optimization（续）

> **[图示说明——情况 ①]** 一维单峰函数曲线。区间 [aᵏ, bᵏ] 中取两个试探点 λᵏ（左）和 ρᵏ（右）。
>
> 当 **f(λᵏ) ≤ f(ρᵏ)** 时：then
> - aᵏ⁺¹ = aᵏ
> - bᵏ⁺¹ = ρᵏ
>
> 红色标注新区间 [aᵏ⁺¹, bᵏ⁺¹]。

> **[图示说明——情况 ②]** 同一结构。
>
> 当 **f(ρᵏ) < f(λᵏ)** 时：then
> - aᵏ⁺¹ = λᵏ
> - bᵏ⁺¹ = bᵏ
>
> 红色标注新区间 [aᵏ⁺¹, bᵏ⁺¹]。

---

## 第 ② 页：Bisection Search for Single-Dimensional Optimization

> **[图示说明]** 数轴上标出区间 [aᵏ, bᵏ]，中点为 (aᵏ + bᵏ)/2。λᵏ 在中点左侧距离 ε 处，ρᵏ 在中点右侧距离 ε 处。双向箭头标注距离 ε。

算法步骤：

① Choose the initial interval of uncertainty (a¹, b¹) over which we want to find the minimum of f. Pick stopping tolerance θ. Pick iteration counter k = 1.

② Set

$$\lambda^k = \frac{a^k + b^k}{2} - \varepsilon, \qquad \rho^k = \frac{a^k + b^k}{2} + \varepsilon$$

③ If f(λᵏ) ≤ f(ρᵏ), then aᵏ⁺¹ = aᵏ, bᵏ⁺¹ = ρᵏ.
   If f(λᵏ) > f(ρᵏ), then aᵏ⁺¹ = λᵏ, bᵏ⁺¹ = bᵏ.

④ If bᵏ⁺¹ − aᵏ⁺¹ ≤ θ then stop.
   Else increase k by 1 and go to Step 2.

---

## 第 ③ 页：二分搜索示例与动机

- Use bisection search to minimize x² + 2x over [−3, 5].

- Bisection search requires making two function evaluations at each iteration: f(λᵏ) and f(ρᵏ).

- Can we get away with one function computation at each iteration?

---

## 第 ④ 页：Golden Section Search for One-Dimensional Optimization

- Can we set up the left and right points in interval of uncertainty such that either λᵏ⁺¹ matches ρᵏ, or ρᵏ⁺¹ matches λᵏ?

> **[图示说明——情况 (★)]** 数轴上区间 [aᵏ, bᵏ]，标出 λᵏ 和 ρᵏ。更新后新区间 [aᵏ⁺¹, bᵏ⁺¹] = [aᵏ, ρᵏ]（情况①发生时），在新区间中 λᵏ⁺¹ 已知、只需计算 f(ρᵏ⁺¹)。标注 "only compute f(ρᵏ⁺¹)"。

> **[图示说明——情况 (★★)]** 同一结构。更新后新区间 [aᵏ⁺¹, bᵏ⁺¹] = [λᵏ, bᵏ]（情况②发生时），在新区间中 ρᵏ⁺¹ 已知、只需计算 f(λᵏ⁺¹)。标注 "only compute f(λᵏ⁺¹)"。

- In addition to this alignment, let's impose the condition that the interval of uncertainty will shrink by a factor of β at every iteration. So,

$$b^{k+1} - a^{k+1} = \beta \cdot (b^k - a^k)$$

---

## 第 ⑤ 页：黄金分割比推导（情况 ★）

- If (★) happens, shrink interval of uncertainty by a factor of β:

$$b^{k+1} - a^{k+1} = \beta(b^k - a^k) \implies b^k - \lambda^k = \beta(b^k - a^k)$$

$$\implies \lambda^k = \beta \cdot a^k + (1 - \beta) \cdot b^k$$

- If (★★) happens, shrink interval of uncertainty by a factor of β:

$$b^{k+1} - a^{k+1} = \beta(b^k - a^k) \implies \rho^k - a^k = \beta(b^k - a^k)$$

$$\implies \rho^k = (1 - \beta) \cdot a^k + \beta \cdot b^k$$

- If (★) happens, make sure to match λᵏ⁺¹ with ρᵏ:

$$\rho^k = \lambda^{k+1} = \beta \cdot a^{k+1} + (1 - \beta) \cdot b^{k+1} = \beta \cdot a^k + (1 - \beta) \cdot b^k \qquad (\ddagger)$$

Remember:

$$\lambda^k = \beta \cdot a^k + (1 - \beta) \cdot b^k, \qquad \rho^k = (1 - \beta) \cdot a^k + \beta \cdot b^k$$

To have (‡):

$$(1 - \beta) \cdot a^k + \beta \cdot b^k = \beta\left(\beta \cdot a^k + (1 - \beta) \cdot b^k\right) + (1 - \beta) \cdot b^k$$

$$(1 - \beta) \cdot a^k + \beta \cdot b^k = \beta^2 \cdot a^k + (\beta - \beta^2) \cdot b^k + (1 - \beta) \cdot b^k$$

$$(1 - \beta)(a^k - b^k) = \beta^2(a^k - b^k) \implies 1 - \beta = \beta^2 \implies \beta^2 + \beta - 1 = 0$$

$$\beta = \frac{-1 + \sqrt{5}}{2}$$

---

## 第 ⑥ 页：黄金分割比推导（情况 ★★）

- If (★★) happens, make sure to match ρᵏ⁺¹ with λᵏ:

$$\lambda^k = \rho^{k+1} = (1 - \beta) \cdot a^{k+1} + \beta \cdot b^{k+1} = (1 - \beta) \cdot a^k + \beta \cdot b^k \qquad (\ddagger)$$

Remember:

$$\lambda^k = \beta \cdot a^k + (1 - \beta) \cdot b^k, \qquad \rho^k = (1 - \beta) \cdot a^k + \beta \cdot b^k$$

To have (‡):

$$\beta \cdot a^k + (1 - \beta) \cdot b^k = (1 - \beta) \cdot a^k + \beta \cdot \left((1 - \beta) \cdot a^k + \beta \cdot b^k\right)$$

$$\beta \cdot a^k + (1 - \beta) \cdot b^k = (1 - \beta) \cdot a^k + (\beta - \beta^2) \cdot a^k + \beta^2 \cdot b^k$$

$$(1 - \beta)(b^k - a^k) = \beta^2(b^k - a^k) \implies 1 - \beta = \beta^2 \implies \beta^2 + \beta - 1 = 0$$

$$\beta = \frac{-1 + \sqrt{5}}{2}$$

---

## 第 ⑦ 页：Golden Section Search 算法与数值示例

> **[图示说明]** 三层嵌套区间的数值示例：
> - 第一层：[0, 10]，试探点在 3.82 和 6.18
> - 第二层：[3.82, 10]，试探点在 6.18 和 7.64
> - 第三层：[3.82, 7.64]，试探点在 5.28 和 6.18
>
> 每层区间宽度缩减为上一层的 β ≈ 0.618 倍。

算法步骤：

① Choose (a¹, b¹), pick stopping tolerance θ, set k = 1.

② Set λᵏ = β·aᵏ + (1−β)·bᵏ, ρᵏ = (1−β)·aᵏ + β·bᵏ, where β = (−1+√5)/2.

③ If f(λᵏ) ≤ f(ρᵏ), then aᵏ⁺¹ = aᵏ, bᵏ⁺¹ = ρᵏ.
   If f(λᵏ) > f(ρᵏ), then aᵏ⁺¹ = λᵏ, bᵏ⁺¹ = bᵏ.

④ If bᵏ⁺¹ − aᵏ⁺¹ ≤ θ then stop.
   Otherwise increase k by 1 and go to Step 2.

- Don't forget to either borrow f(λᵏ) from f(ρᵏ⁻¹) or f(ρᵏ) from f(λᵏ⁻¹).

---

## 第 ⑧ 页：利用导数的单维优化

- These single-dim optimization algorithms only use function evaluations. They do not use the derivative of the function. If we have access to the derivatives of f, we can be a bit more efficient.

> **[图示说明——左图]** 一维凸函数曲线，在区间 [aᵏ, bᵏ] 中点 (aᵏ+bᵏ)/2 处求导。
>
> 当 **f'((aᵏ+bᵏ)/2) ≤ 0** 时（中点处导数为负，函数仍在递减），最小值在中点右侧：
> - aᵏ⁺¹ = (aᵏ+bᵏ)/2
> - bᵏ⁺¹ = bᵏ
>
> 新区间为右半部分。

> **[图示说明——右图]** 同一结构。
>
> 当 **f'((aᵏ+bᵏ)/2) > 0** 时（中点处导数为正，函数在递增），最小值在中点左侧：
> - aᵏ⁺¹ = aᵏ
> - bᵏ⁺¹ = (aᵏ+bᵏ)/2
>
> 新区间为左半部分。
