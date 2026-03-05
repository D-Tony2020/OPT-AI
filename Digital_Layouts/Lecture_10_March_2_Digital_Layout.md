# Lecture 10 — March 2, 2026 — 忠实转制笔记

---

## 第 ① 页：Convergence of Gradient Descent for Convex Functions

- We want to minimize a convex function f: ℝⁿ → ℝ by using gradient descent with constant step size δ.

- We generate a sequence of points x⁰, x¹, x², ... by using

$$x^{k+1} = x^k - \gamma \, \nabla f(x^k)$$

where x⁰ is the initial point we pick.

- With a constant step size, we will not be able to establish convergence to a minimum. The best we can accomplish is that we will bounce around a minimum. Even so, our convergence result will be a meaningful result.

---

## 第 ② 页：常步长梯度下降的弹跳示例

> **[图示说明]** 函数 f(x) = (x−1)² 的曲线，最小值在 x = 1 处。f'(x) = 2(x−1)。

- Minimize f(x) = (x−1)² starting with x⁰ = 2 with constant step size γ = 1.

| 迭代 | 梯度 | 更新 |
|---|---|---|
| f'(x⁰) = f'(2) = 2 | x¹ = x⁰ − γf'(x⁰) = 2 − 1·2 = 0 |
| f'(x¹) = f'(0) = −2 | x² = x¹ − γf'(x¹) = 0 − 1·(−2) = 2 |
| f'(x²) = f'(2) = 2 | x³ = 0, x⁴ = 2, x⁵ = 0, ... |

- So we will be bouncing around the minimum x* = 1.

---

## 第 ③ 页：Preliminary Analysis

- Recall f: ℝⁿ → ℝ is convex iff f(y) ≥ f(x) + ∇ᵀf(x)·(y−x) for all x, y ∈ ℝⁿ.

- Another useful equality is:

$$\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2x^T y \qquad (\ddagger)$$

**Pf:** Remember ‖z‖² = zᵀz and

$$\|x - y\|^2 = (x - y)^T(x - y) = (x^T - y^T)(x - y) = x^T x - x^T y - y^T x + y^T y$$

$$= \|x\|^2 - 2x^T y + \|y\|^2$$

- Here is the preliminary analysis we will use in our convergence proof.

  We have xᵏ⁺¹ = xᵏ − γ∇f(xᵏ), denote gᵏ = ∇f(xᵏ).

- For simplicity of notation, set gᵏ = ∇f(xᵏ), so xᵏ⁺¹ = xᵏ − γ·gᵏ.

- Let x* = argmin_{x∈ℝⁿ} f(x) ← x* is the minimum of f.

- We want to make sure that we get close to the value of f at x*.

---

## 第 ④ 页：上界推导

- We want to upper bound f(xᵏ) − f(x*).

- Let's start with first order characterization of convexity:

$$f(x^*) \geq f(x^k) + \nabla^T f(x^k) \cdot (x^* - x^k)$$

$$\implies f(x^k) - f(x^*) \leq \nabla^T f(x^k) \cdot (x^k - x^*) = (g^k)^T(x^k - x^*) \qquad (\star)$$

- Thus, to upper bound f(xᵏ) − f(x*), it is enough to upper bound (gᵏ)ᵀ(xᵏ − x*).

- From gradient descent: xᵏ⁺¹ = xᵏ − γgᵏ ⟹ gᵏ = (1/γ)·(xᵏ − xᵏ⁺¹)

- Let's go ahead and upper bound (gᵏ)ᵀ(xᵏ − x*).

---

## 第 ⑤ 页：展开内积并利用伸缩和

- $(g^k)^T(x^k - x^*) = \frac{1}{\gamma} \cdot (x^k - x^{k+1})^T \cdot (x^k - x^*)$

  From (‡):

$$= \frac{1}{2\gamma}\left(\|x^k - x^{k+1}\|^2 + \|x^k - x^*\|^2 - \|(x^k - x^{k+1}) - (x^k - x^*)\|^2\right)$$

$$= \frac{1}{2\gamma}\left(\gamma^2\|g^k\|^2 + \|x^k - x^*\|^2 - \|x^{k+1} - x^*\|^2\right)$$

$$= \frac{\gamma}{2}\|g^k\|^2 + \frac{1}{2\gamma}\left(\|x^k - x^*\|^2 - \|x^{k+1} - x^*\|^2\right)$$

- Add the inequality above over k = 0, 1, 2, ..., K−1:

$$\sum_{k=0}^{K-1} (g^k)^T(x^k - x^*) = \frac{\gamma}{2} \sum_{k=0}^{K-1} \|g^k\|^2 + \frac{1}{2\gamma}\left(\|x^0 - x^*\|^2 - \|x^K - x^*\|^2\right)$$

(by telescoping sums)

$$\leq \frac{\gamma}{2} \sum_{k=0}^{K-1} \|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

- Using (★), we get:

$$\sum_{k=0}^{K-1} \left(f(x^k) - f(x^*)\right) \leq \sum_{k=0}^{K-1} (g^k)^T(x^k - x^*) \leq \frac{\gamma}{2} \sum_{k=0}^{K-1} \|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

End of preliminary analysis.

---

## 第 ⑥ 页：预备分析的总结与解读

- Out line of preliminary analysis:
  - Start with first-order characterization of convexity.
  - Use ‖x−y‖² = ‖x‖² + ‖y‖² − 2xᵀy to convert scalar product to norm differences.
  - Use a telescoping sum.

- The last inequality has some meaning:

  - Σ(f(xᵏ) − f(x*)) is the cumulative "error" over the first K iterations.

  - If δ is small, the upper bound on the error is dominated by (1/(2δ))‖x⁰−x*‖².

    If step size is small, we will get stuck close to the initial point x⁰, so the error will be dominated by how close the initial point is to x*.

  - If δ is large, then the upper bound on the error is dominated by (γ/2)Σ‖gᵏ‖².

    If the step size is large, whenever the gradient is non-zero, we keep on overshooting.

---

## 第 ⑦ 页：Convergence of Gradient Descent Under Bounded Gradients

- Assume f is convex and ‖∇f(x)‖ ≤ B for all x ∈ ℝⁿ.

**Theorem:** Let f be convex with minimum x\*. Assume that ‖∇f(x)‖ ≤ B for all x ∈ ℝⁿ and we start gradient descent at point x⁰ with ‖x⁰ − x\*‖ = R.

If we choose the step size as

$$\gamma = \frac{R}{B\sqrt{K}},$$

then the K iterates of gradient descent satisfy:

$$\frac{1}{K} \sum_{k=0}^{K-1} \left(f(x^k) - f(x^*)\right) \leq \frac{RB}{\sqrt{K}}$$

So the average "error" of gradient descent diminishes at rate 1/√K after K iterations.

---

## 第 ⑧ 页：收敛定理的证明

**Pf:** From our preliminary analysis:

$$\sum_{k=0}^{K-1} \left(f(x^k) - f(x^*)\right) \leq \frac{\gamma}{2} \sum_{k=0}^{K-1} \|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

$$\leq \frac{\gamma}{2} K B^2 + \frac{1}{2\gamma} R^2$$

So the cumulative error over K iterations is upper bounded by (γ/2)KB² + (1/(2γ))R².

Choose γ to make this upper bound as small as possible:

$$H(\gamma) = \frac{1}{2}\left(K B^2 \gamma + R^2 \frac{1}{\gamma}\right), \quad H'(\gamma) = \frac{1}{2}\left(K B^2 - R^2 \frac{1}{\gamma^2}\right) = 0$$

$$\implies \gamma = \sqrt{\frac{R^2}{KB^2}} \implies \gamma = \frac{R}{B\sqrt{K}}$$

- With this step size:

$$\sum_{k=0}^{K-1} \left(f(x^k) - f(x^*)\right) \leq \frac{1}{2}\left(\frac{R}{B\sqrt{K}} \cdot K \cdot B^2 + \frac{B\sqrt{K}}{R} \cdot R^2\right) = RB\sqrt{K}$$

- Thus,

$$\frac{1}{K} \sum_{k=0}^{K-1} \left(f(x^k) - f(x^*)\right) \leq \frac{1}{\sqrt{K}} \, RB \quad \text{Done!}$$
