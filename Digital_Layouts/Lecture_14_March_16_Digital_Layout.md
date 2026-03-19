# Lecture 14 — March 16, 2026 — 忠实转制笔记

---

## 第 ① 页：Two Properties of Projection

**Property 1:** For 𝕏 convex, y ∈ ℝⁿ, x ∈ 𝕏:

$$(x - \Pi_\mathbb{X}(y))^\top (y - \Pi_\mathbb{X}(y)) \leq 0$$

> **[图示说明]** 凸集 𝕏，点 y 在集合外部，Π_𝕏(y) 在集合边界上，x 在集合内部。从 Π_𝕏(y) 分别画向量指向 x 和 y，两个向量之间的夹角为钝角（内积 ≤ 0）。

**Property 2:** For 𝕏 convex, y ∈ ℝⁿ, x ∈ 𝕏:

$$\|x - y\|^2 \geq \|y - \Pi_\mathbb{X}(y)\|^2 + \|x - \Pi_\mathbb{X}(y)\|^2$$

---

## 第 ② 页：Projected Gradient Descent

$$y^{k+1} = x^k - \gamma \, \nabla f(x^k)$$

$$x^{k+1} = \Pi_\mathbb{X}(y^{k+1})$$

We will use the following preliminary analysis to give a convergence result for projected gradient descent.

Let $x^* = \arg\min_{x \in \mathbb{X}} f(x)$. The error at each iteration is $f(x^k) - f(x^*)$. We want to bound error.

By first-order characterization of convexity:

$$f(x^*) \geq f(x^k) + \nabla f(x^k)^\top (x^* - x^k)$$

$$\implies f(x^k) - f(x^*) \leq \nabla f(x^k)^\top (x^k - x^*)$$

So it is enough to upper bound $\nabla f(x^k)^\top (x^k - x^*)$.

Let $g^k = \nabla f(x^k)$.

---

## 第 ③ 页：展开内积的上界

$$\nabla f(x^k)^\top (x^k - x^*) = (g^k)^\top (x^k - x^*) = \frac{1}{\gamma}(x^k - y^{k+1})^\top (x^k - x^*)$$

$$\star = \frac{1}{2\gamma}\left(\|x^k - y^{k+1}\|^2 + \|x^k - x^*\|^2 - \|(x^k - y^{k+1}) - (x^k - x^*)\|^2\right)$$

$$= \frac{1}{2\gamma}\left(\gamma^2 \|g^k\|^2 + \|x^k - x^*\|^2 - \|y^{k+1} - x^*\|^2\right) \quad (\ddagger)$$

$\star$ follows by $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b$, so $a^\top b = \frac{1}{2}(\|a\|^2 + \|b\|^2 - \|a - b\|^2)$.

We know from second property of projections:

$$\|y^{k+1} - x^*\|^2 \geq \|x^{k+1} - x^*\|^2 + \|y^{k+1} - x^{k+1}\|^2 \geq \|x^{k+1} - x^*\|^2 \quad (\ddagger\ddagger)$$

> **[图示说明]** 凸集 𝕏，y^{k+1} 在集合外部，x^{k+1} = Π_𝕏(y^{k+1}) 在集合边界上，x* 在集合内部。展示了 Property 2 的几何关系。

By $(\ddagger)$ and $(\ddagger\ddagger)$:

$$\nabla f(x^k)^\top (x^k - x^*) \leq \frac{\gamma}{2}\|g^k\|^2 + \frac{1}{2\gamma}\left(\|x^k - x^*\|^2 - \|x^{k+1} - x^*\|^2\right)$$

Add this inequality from $k = 0$ to $k = K-1$ to get:

---

## 第 ④ 页：求和与伸缩消去

$$\sum_{k=0}^{K-1} \nabla f(x^k)^\top (x^k - x^*) \leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\|g^k\|^2 + \frac{1}{2\gamma}\left(\|x^0 - x^*\|^2 - \|x^K - x^*\|^2\right)$$

$$\leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

Thus, we get:

$$\sum_{k=0}^{K-1}\left(f(x^k) - f(x^*)\right) \leq \sum_{k=0}^{K-1}\nabla f(x^k)^\top (x^k - x^*) \leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

---

## 第 ⑤ 页：Convergence of Projected Gradient Descent with Bounded Gradients

**Theorem:** Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex and $\mathbb{X}$ be a convex set. $x^* = \arg\min_{x \in \mathbb{X}} f(x)$. Assume $\|x^0 - x^*\| \leq R$ and $\|\nabla f(x)\| \leq B$ for all $x \in \mathbb{X}$. Then, choosing the step size as $\gamma = \frac{R}{B\sqrt{K}}$, $K$ iterations of projected gradient descent satisfy:

$$\frac{1}{K}\sum_{k=0}^{K-1}\left(f(x^k) - f(x^*)\right) \leq \frac{RB}{\sqrt{K}}$$

**Proof:** By our preliminary analysis:

$$\sum_{k=0}^{K-1}\left(f(x^k) - f(x^*)\right) \leq \frac{\gamma}{2} K \cdot B^2 + \frac{1}{2\gamma} R^2$$

$$\frac{1}{K}\sum_{k=0}^{K-1}\left(f(x^k) - f(x^*)\right) \leq \frac{\gamma}{2}B^2 + \frac{1}{2\gamma}\frac{R^2}{K}$$

$$= \frac{1}{2}\cdot\frac{R}{B\sqrt{K}}\cdot B^2 + \frac{B\sqrt{K}}{2R}\cdot\frac{R^2}{K} = \frac{RB}{2\sqrt{K}} + \frac{RB}{2\sqrt{K}} = \frac{RB}{\sqrt{K}}$$

Done! ∎

---

## 第 ⑥ 页：关于有界梯度假设与计算投影

- Note that $\|\nabla f(x)\| \leq B$ for all $x \in \mathbb{X}$ is not a horrible assumption when $\mathbb{X}$ is bounded.

**Computing Projections**

- At each iteration of projected gradient descent, we need to compute $\Pi_\mathbb{X}(y^{k+1})$.

- Computing this projection is equivalent to solving:

$$\min_{x \in \mathbb{X}} \sum_i (x_i - y_i^{k+1})^2$$

- This is yet another constrained optimization problem!

- By exploiting the structure of the norm function $\sum(x_i - y_i)^2$, we can compute projections onto $\mathbb{X}$ for many different feasible sets $\mathbb{X}$.

- Here are some feasible sets $\mathbb{X}$ for which projections are simple to compute.

---

## 第 ⑦ 页：Projections on a Box

Consider the case $\mathbb{X} = \{x \in \mathbb{R}^n : 0 \leq x_i \leq U_i, \; \forall \, i = 1, \ldots, n\}$.

For example, $n = 2$.

> **[图示说明]** 三维图，水平轴为 x₁ 和 x₂，纵轴为目标函数值。可行域为 x₁-x₂ 平面上的矩形 [0, U₁]×[0, U₂]。

We want to find $\Pi_\mathbb{X}(\hat{y})$, where $\hat{y}$ is a fixed point.

We want to solve $\min_{x \in \mathbb{X}} \|x - \hat{y}\|^2$.

---

## 第 ⑧ 页：Box 投影的可分离性与 Case 1

We want to solve

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad 0 \leq x_i \leq U_i, \; \forall \, i = 1, \ldots, n$$

This problem is equivalent to

$$\sum_i \min_{0 \leq x_i \leq U_i} (x_i - \hat{y}_i)^2$$

Let's look at 3 cases:

**Case 1:** $0 \leq \hat{y}_i \leq U_i$.

> **[图示说明]** 函数 $(x_i - \hat{y}_i)^2$ 的图像，最小值点 $\hat{y}_i$ 在 $[0, U_i]$ 内部。最小值在可行域内取到。

$$\hat{y}_i = \arg\min_{0 \leq x_i \leq U_i} (x_i - \hat{y}_i)^2$$

---

## 第 ⑨ 页：Case 2 与 Case 3

**Case 2:** $\hat{y}_i \leq 0$.

> **[图示说明]** 函数 $(x_i - \hat{y}_i)^2$ 的图像，最小值点 $\hat{y}_i$ 在 0 的左侧。在 $[0, U_i]$ 上，函数在 $x_i = 0$ 处取最小值。

$$0 = \arg\min_{0 \leq x_i \leq U_i} (x_i - \hat{y}_i)^2$$

**Case 3:** $\hat{y}_i \geq U_i$.

> **[图示说明]** 函数 $(x_i - \hat{y}_i)^2$ 的图像，最小值点 $\hat{y}_i$ 在 $U_i$ 的右侧。在 $[0, U_i]$ 上，函数在 $x_i = U_i$ 处取最小值。

$$U_i = \arg\min_{0 \leq x_i \leq U_i} (x_i - \hat{y}_i)^2$$

---

## 第 ⑩ 页：Box 投影的闭式解

Thus,

$$\arg\min_{0 \leq x_i \leq U_i} (x_i - \hat{y}_i)^2 = \min\{U_i, \; [\hat{y}_i]^+\}$$

where $[a]^+ = \max\{0, a\}$.

Thus, the projection of $\hat{y}$ on to $\mathbb{X}$ is given by $\pi = (\pi_1, \ldots, \pi_n)$, where

$$\pi_i = \min\{U_i, \; [\hat{y}_i]^+\}$$
