# Lecture 16 — March 23, 2026 — 忠实转制笔记

---

## 第 ① 页：Projections onto the L1-Ball

Consider the case

$$\mathbb{X} = \left\{ x \in \mathbb{R}^n : \sum_i |x_i| \leq R \right\}$$

> **[图示说明]** 二维 L1-ball（菱形/钻石形），顶点分别在 (R,0)、(0,R)、(−R,0)、(0,−R)。四条边界线分别为 x₁+x₂=R、−x₁+x₂=R、−x₁−x₂=R、x₁−x₂=R。

We want to find $\Pi_\mathbb{X}(\hat{y})$, where $\hat{y}$ is a fixed point.

---

## 第 ② 页：问题建模与 Observation 1

We want to solve the problem:

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

Assume that $\hat{y} \notin \mathbb{X}$.

**Observation 1:** We can assume $\hat{y}_i \geq 0$, $i = 1, \ldots, n$.

> **[图示说明]** L1-ball 中，$\hat{y}$ 位于第二象限（x₁ 为负），$\hat{y}'$ 是 $\hat{y}$ 关于 y 轴的镜像反射，位于第一象限。投影 $\hat{y}$ 等价于投影 $\hat{y}'$（分量取绝对值后投影），再将结果的 x₁ 分量取负。

---

## 第 ③ 页：Observation 2

**Observation 2:** Given $\hat{y}$ such that $\hat{y}_i \geq 0$, $i = 1, \ldots, n$, the projection of $\hat{y}$ on to the L1-ball has all non-negative components.

> **[图示说明]** L1-ball 中，$\hat{y}$ 位于第一象限，$\Pi_\mathbb{X}(\hat{y})$ 位于 L1-ball 边界上的第一象限部分。

Let's mathematically establish that both of these observations are indeed correct.

---

## 第 ④ 页：Observation 1 的证明

**Observation 1 proof:** When projecting $\hat{y}$ onto L1-Ball, we can assume $\hat{y}_i \geq 0$, $\forall i = 1, \ldots, n$.

**Why?** Assume $\hat{y}_j < 0$. Consider the problem:

$$\min \sum_{i \neq j} (x_i - \hat{y}_i)^2 + (x_j - \hat{y}_j)^2 \quad (*) \quad \text{s.t.} \quad \sum_{i \neq j} |x_i| + |x_j| \leq R$$

Consider instead:

$$\min \sum_{i \neq j} (x_i - \hat{y}_i)^2 + (x_j - (-\hat{y}_j))^2 \quad (\ddagger) \quad \text{s.t.} \quad \sum_{i \neq j} |x_i| + |x_j| \leq R$$

Let's argue both problems have the same opt. obj. value.

Let $x^*$ be an opt. solution to $(*)$. Define $\tilde{x}$ such that $\tilde{x}_i = x_i^*$ for $i \neq j$, $\tilde{x}_j = -x_j^*$.

Because $x^*$ is opt. to $(*)$, $\sum_i |x_i^*| \leq R$, but also $|x_i^*| = |\tilde{x}_i|$ for all $i$. Thus $\tilde{x}$ is feasible to $(\ddagger)$.

---

## 第 ⑤ 页：Observation 1 证明（续）

The obj. value that $\tilde{x}$ provides for problem $(\ddagger)$ is:

$$\sum_{i \neq j} (\tilde{x}_i - \hat{y}_i)^2 + (\tilde{x}_j - (-\hat{y}_j))^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (-x_j^* + \hat{y}_j)^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (x_j^* - \hat{y}_j)^2$$

which is the opt. objective value of problem $(*)$.

Thus, given an opt. solution to $(*)$, we can construct a feasible solution to $(\ddagger)$ with obj. value equal to the opt. obj. value of $(*)$. This means that the opt. obj. value of $(\ddagger)$ is less than or equal to the opt. obj. value of $(*)$.

We can follow precisely the reverse argument to show that given an optimal solution to $(\ddagger)$, we can construct a feasible solution to $(*)$ with obj. value equal to opt. obj. value of $(\ddagger)$. This means opt. obj. value of $(*)$ ≤ opt. obj. value of $(\ddagger)$.

Thus, the two problems have the same opt. obj. values. Done! ∎

---

## 第 ⑥ 页：Observation 2 的证明

**Observation 2 proof:** Let $\hat{y}$ be such that $\hat{y}_i \geq 0$, $i = 1, \ldots, n$. Then the projection of $\hat{y}$ on to the L1-ball has all non-negative components.

**Why?** To get a contradiction, let $\hat{y}$ be such that $\hat{y}_i \geq 0$, $\forall i = 1, \ldots, n$, but the $j$-th component of the projection of $\hat{y}$ on to L1-ball is negative.

$$\min \sum_{i \neq j} (x_i - \hat{y}_i)^2 + (x_j - \hat{y}_j)^2 \quad (**) \quad \text{s.t.} \quad \sum_{i \neq j} |x_i| + |x_j| \leq R$$

Let $x^*$ be an opt. solution to $(**)$ and assume $x_j^* < 0$.

Define $\tilde{x}$ to $(**)$ as $\tilde{x}_i = x_i^*$ for $i \neq j$, $\tilde{x}_j = 0$.

Because $x^*$ is feasible to $(**)$, we have $\sum_{i \neq j} |x_i^*| + |x_j^*| \leq R$, but then $\sum_{i \neq j} |\tilde{x}_i| + |\tilde{x}_j| = \sum_{i \neq j} |x_i^*| + 0 \leq R$.

So $\tilde{x}$ is feasible to $(**)$.

---

## 第 ⑦ 页：Observation 2 证明（续）

The solution $\tilde{x}$ provides the following objective value for problem $(**)$:

$$\sum_{i \neq j} (\tilde{x}_i - \hat{y}_i)^2 + (\tilde{x}_j - \hat{y}_j)^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (\hat{y}_j)^2 < \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (\hat{y}_j - x_j^*)^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (x_j^* - \hat{y}_j)^2$$

Thus, $\tilde{x}$ gives a strictly better obj. value for problem $(**)$ than the solution $x^*$. So the solution $x^*$ cannot be an opt. solution to problem $(**)$. Contradiction! Done! ∎

---

## 第 ⑧ 页：利用两个 Observation 化简问题

We want to solve the problem:

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

By observation 1, we can equivalently solve:

$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

By observation 2, we can further equivalently solve:

$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i x_i \leq R, \quad x_i \geq 0, \; \forall i = 1, \ldots, n$$

---

## 第 ⑨ 页：归结为概率单纯形投影

When $\hat{y} \notin \mathbb{X}$ (which is the interesting case), the last problem is equivalent to:

$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i x_i = R, \quad x_i \geq 0, \; \forall i = 1, \ldots, n$$

By the detour to last lecture, to solve the problem above, find $\lambda^*$ such that

$$\sum_i \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ = R$$

Then the opt. solution to the problem above is $x_i^* = \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+$.

---

## 第 ⑩ 页：回顾上节课——概率单纯形投影

Detour to the last lecture: Projecting on to the probability simplex.

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad (***) \quad \text{s.t.} \quad \sum_i x_i = R, \quad x_i \geq 0, \; \forall i = 1, \ldots, n$$

To solve the problem above, find $\lambda^*$ such that

$$\sum_i \left[ \hat{y}_i - \frac{\lambda^*}{2} \right]^+ = R$$

Then the opt. solution to problem $(***) $ is given by $x_i^* = \left[ \hat{y}_i - \frac{\lambda^*}{2} \right]^+$.

---

## 第 ⑪ 页：L1-Ball 投影的最终公式

To conclude, let $\hat{y}$ have components with arbitrary signs and $\hat{y} \notin \mathbb{X}$, we want to solve:

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

Find $\lambda^*$ such that

$$\sum_i \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ = R$$

Thus, the opt. solution $x^*$ to the last problem above is given by:

$$x_i^* = \begin{cases} \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ & \text{if } \hat{y}_i \geq 0 \\ -\left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ & \text{if } \hat{y}_i < 0 \end{cases}$$
