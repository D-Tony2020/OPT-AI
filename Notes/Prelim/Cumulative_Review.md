# Exam Scope

## Scope of Examination

**Lec1 – Lec12**

- Projected Gradient Descent is **NOT** included
- Definition of a convex set **IS** included

## Exam Logistics

| Item | Detail |
|------|--------|
| Date | March 17, 2026 |
| Time | 7:30 PM – 9:30 PM (2 hours) |
| Location | Bloomberg 081 |
| Format | Paper and pencil, 3 problems (each with parts A and B, but not C) |
| Allowed materials | Lecture notes as prepared in class **OR** professor's posted handwritten notes (**not both**). No additional notes, no homework solutions, no outside work. |
| iPad notes | Must be printed; iPads not allowed during exam |
| Calculator | Non-communicating calculator allowed, but numbers will be easy enough without one |

## Specific Exam Topics

1. High level description of gradient descent idea
2. Bisection and golden section search for single dimensional optimization
3. Penalty function method, barrier function method
4. Taylor's theorem and its implication
5. Definition of a convex function
6. First order characterization of convexity
7. Convergence of gradient descent with fixed step size for functions with bounded gradients
8. Preliminary analysis / convergence of gradient descent with fixed step size for smooth convex functions
9. Convex sets

## Professor's Exam Hints

**Problem 1 (Committed)**:
- "I'm going to give you a function. And I'm going to ask you, start from this point and do two iterations of gradient descent, like manually."
- Numbers will be easy — no calculator needed
- May choose step size by bisection or golden section
- Could be gradient descent on a penalized objective function (penalty function method)

**Problems 2 & 3 (Roughly)**:
- "Show that in this type of situation, if you give me a convex function, it's still going to be a convex function type of derivations."
- Small pieces of convergence analysis, like HW3 Q3
- "I would never ask you a proof along those lines, like a full-blown proof, but I may ask for a small little piece."

**Other Notes**:
- Basic algebraic factorizations expected (e.g., $x^2 - y^2 = (x-y)(x+y)$), but not polynomial long division
- Can ask the professor during the exam if a shortcut/theorem is allowed (e.g., "sum of convex functions is convex")
- Small step size definition: output of bisection/golden section search (layperson optimization), or $\gamma = \frac{R}{B\sqrt{K}}$ (fixed step size analysis)

## Review Session Practice Problem

> This is highly exam-relevant — the professor demonstrated this as the exact type of Problem 1.

**Problem:** Minimize $e^{x_1} + 100(x_2 + 2)^2$ subject to $x_1 + x_2 \leq 10$, $x_1 \geq 7$.

Use penalty parameters $\theta_1 = \theta_2 = 100$. Start from initial point $x^0 = (5, 6)$. Do one iteration of gradient descent with step size $\gamma = 0.01$.

**Step 1: Rewrite constraints as $g(x) \leq 0$**

$$g_1(x) = 10 - x_1 - x_2 \leq 0$$

$$g_2(x) = 7 - x_1 \leq 0$$

**Step 2: Write penalized objective function**

$$h(x) = e^{x_1} + 100(x_2 + 2)^2 + \theta_1 \cdot (\max\lbrace g_1(x), 0\rbrace)^2 + \theta_2 \cdot (\max\lbrace g_2(x), 0\rbrace)^2$$

**Step 3: Determine which penalty terms are active at $(5, 6)$**

- $g_1(5, 6) = 10 - 5 - 6 = -1 \leq 0$ → constraint satisfied, penalty term = 0
- $g_2(5, 6) = 7 - 5 = 2 > 0$ → constraint violated, penalty term active

**Step 4: Simplify $h(x)$ around neighborhood of $(5, 6)$**

Around a small neighborhood of $(5, 6)$:

$$h(x) \approx e^{x_1} + 100(x_2 + 2)^2 + 100(7 - x_1)^2$$

**Step 5: Compute gradient of $h$ at $(5, 6)$**

$$\nabla h(x) = \begin{pmatrix} e^{x_1} - 200(7 - x_1) \\ 200(x_2 + 2) \end{pmatrix}$$

$$\nabla h(5, 6) = \begin{pmatrix} e^5 - 200(7 - 5) \\ 200(6 + 2) \end{pmatrix} = \begin{pmatrix} e^5 - 400 \\ 1600 \end{pmatrix} \approx \begin{pmatrix} -251.57 \\ 1600 \end{pmatrix}$$

**Step 6: One iteration of gradient descent**

$$x^1 = x^0 - \gamma \nabla h(x^0) = \begin{pmatrix} 5 \\ 6 \end{pmatrix} - 0.01 \begin{pmatrix} -251.57 \\ 1600 \end{pmatrix} = \begin{pmatrix} 7.5157 \\ -10 \end{pmatrix}$$

**Important Note (If doing a second iteration):**

At the new point $(7.5157, -10)$:
- $g_1 = 10 - 7.5157 - (-10) = 12.49 > 0$ → **first constraint now violated**, this penalty term becomes active
- $g_2 = 7 - 7.5157 = -0.5157 \leq 0$ → second constraint now satisfied, penalty term = 0

So the penalized objective around $(7.5157, -10)$ changes to:

$$h(x) \approx e^{x_1} + 100(x_2 + 2)^2 + 100(10 - x_1 - x_2)^2$$

Which penalty terms survive depends on which point we're sitting at.

---

# Course Content Review (Lec 1–12)

本章按 Lecture 顺序复习 ORIE 5320 的全部课程内容（Lec 1–12）。每个 Lecture 开头标注其覆盖的考点编号。考点内容用 📌 标记，非考点内容用 [非考点] 标记。

---

## Lecture 1 — ML/AI Optimization & Logistic Regression

> [非考点 · 背景知识] 本讲建立优化问题的基本框架，并引入机器学习中的优化动机。

### 1.1 优化问题的基本形式

一个优化问题由三个要素组成：**决策变量 (decision variables)**、**目标函数 (objective function)**、**约束 (constraints)**。

**Box Optimization 例子：** 设计一个无顶正方形底面盒子，底面边长 $x$，高 $y$，使体积最大且表面积不超过 500 cm²：

$$\max \quad x^2 y$$

$$\text{s.t.} \quad x^2 + 4xy \leq 500, \quad x \geq 0, \quad y \geq 0$$

最优解 $(\hat{x}, \hat{y}) = (12.91, 6.45)$，最优体积 $1075.82$。

**关键观察：** 非线性优化问题中，优化软件可能轻易失败（如初始猜测 $x=0, y=0$ 时 Excel 求解失败）。

### 1.2 ML/AI 中的优化

数据集 $\lbrace(x_j, y_j) : j = 1, \ldots, M\rbrace$，其中 $x_j \in \mathbb{R}^n$ 是特征向量，$y_j \in \mathbb{R}$ 是标签。

目标：找到参数化预测函数 $\phi(\cdot, w): \mathbb{R}^n \to \mathbb{R}$，使得 $\phi(x_j, w) \approx y_j$。

ML/AI 中的通用优化问题：

$$\min_{w \in \mathbb{R}^p} \frac{1}{M} \sum_{j=1}^{M} \ell(x_j, y_j, w)$$

### 1.3 Least Squares, Ridge, Lasso

- **Least Squares:** $\min_{w} \frac{1}{M} \sum_{j=1}^{M} (w^{\top} x_j - y_j)^2$

- **Ridge Regression:** $\min_{w} \frac{1}{M} \sum_{j=1}^{M} (w^{\top} x_j - y_j)^2 + \lambda \lVert w \rVert_2^2$
  - 避免过拟合，增强对 outlier 的鲁棒性

- **Lasso:** $\min_{w} \frac{1}{M} \sum_{j=1}^{M} (w^{\top} x_j - y_j)^2 + \lambda \lVert w \rVert_1$
  - 用于特征选择

### 1.4 Logistic Regression

二分类问题，$y_j \in \lbrace+1, -1\rbrace$，预测函数：

$$\phi(x, w) = \frac{1}{1 + e^{w^{\top} x}}$$

- $\phi(x,w)$ 解释为属于 $+1$ 类的概率
- $1 - \phi(x,w) = \frac{e^{w^{\top} x}}{1 + e^{w^{\top} x}}$ 为属于 $-1$ 类的概率

**最大似然估计（MLE）：** 数据集的似然函数：

$$L(w) = \prod_{j=1}^{M} \left(\frac{1}{1 + e^{w^{\top} x_j}}\right)^{\mathbf{1}(y_j=+1)} \left(\frac{e^{w^{\top} x_j}}{1 + e^{w^{\top} x_j}}\right)^{\mathbf{1}(y_j=-1)}$$

---

## Lecture 2 — Multi-class Logistic Regression & Deep Learning Intro

> [非考点 · 背景知识] 从二分类推广到多分类，并引入深度学习。

### 2.1 对数似然推导

取对数得到：

$$\ln L(w) = -\sum_{j=1}^{M} \ln(1 + e^{w^{\top} x_j}) + \sum_{j=1}^{M} \mathbf{1}(y_j = -1) \cdot w^{\top} x_j$$

最大化问题：

$$\max_{w \in \mathbb{R}^n} \left\lbrace -\sum_{j=1}^{M} \ln(1 + e^{w^{\top} x_j}) + \sum_{j=1}^{M} \mathbf{1}(y_j = -1) \cdot w^{\top} x_j \right\rbrace$$

### 2.2 Multi-class Logistic Regression

$K$ 类分类，$y_j \in \lbrace 1, 2, \ldots, K\rbrace$。Softmax 预测函数：

$$\phi_k(x_j, w_1, \ldots, w_K) = \frac{e^{w_k^{\top} x_j}}{\sum_{\ell=1}^{K} e^{w_\ell^{\top} x_j}}$$

对数似然：

$$\ln L(w_1, \ldots, w_K) = \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbf{1}(y_j = k) \, w_k^{\top} x_j - \sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^{\top} x_j}\right)$$

### 2.3 Deep Learning 引入

$L$ 层神经网络，每层参数 $(Q^\ell, g^\ell)$：

$$a_j^1 = \sigma_1(Q^1 x_j + g^1), \quad a_j^2 = \sigma_2(Q^2 a_j^1 + g^2), \quad \ldots, \quad a_j^L = \sigma_L(Q^L a_j^{L-1} + g^L)$$

维度：$Q^\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$（$d_0 = n$），$g^\ell \in \mathbb{R}^{d_\ell}$。

激活函数：Sigmoid $\sigma(t) = \frac{1}{1+e^t}$，ReLU $\sigma(t) = \max\lbrace t, 0\rbrace$。

整体表达式：

$$z(x_j, Q^1, g^1, \ldots, Q^L, g^L) = \sigma_L\left(Q^L \, \sigma_{L-1}\left(\ldots \sigma_2\left(Q^2 \, \sigma_1(Q^1 x_j + g^1) + g^2\right) \ldots \right) + g^L\right)$$

---

## Lecture 3 — Deep Learning (Continued) & Gradient Descent Basics

> 📌 **考点 1: High level description of gradient descent idea**
>
> [非考点] 深度学习的 MLE 框架（续）

### 3.1 [非考点] 深度学习的最大似然优化

将神经网络输出 $z_j = z(x_j, Q^1, g^1, \ldots, Q^L, g^L)$ 作为 softmax 的输入特征：

$$\max \left\lbrace -\sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^{\top} z_j}\right) + \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbf{1}(y_j = k) \, w_k^{\top} z_j \right\rbrace$$

优化变量为所有层的参数 $(Q^1, g^1, \ldots, Q^L, g^L, w_1, \ldots, w_K)$。

### 3.2 📌 梯度的定义

设 $f: \mathbb{R}^n \to \mathbb{R}$，$f$ 在点 $x^0$ 处的**梯度** (gradient) 为：

$$\nabla f(x^0) = \left(\frac{\partial f(x)}{\partial x_1}\bigg|_{x=x^0}, \; \frac{\partial f(x)}{\partial x_2}\bigg|_{x=x^0}, \; \ldots, \; \frac{\partial f(x)}{\partial x_n}\bigg|_{x=x^0}\right)$$

### 3.3 📌 梯度下降的核心思想

- 沿梯度方向走小步 → 函数值**增大**
- 沿**负**梯度方向走小步 → 函数值**减小**

如果步长 $\alpha$ 足够小：

$$f(x^0 - \alpha \nabla f(x^0)) \leq f(x^0)$$

GD 更新规则：

$$x^{k+1} = x^k - \alpha^k \nabla f(x^k)$$

### 3.4 📌 数值例子

$f(x) = x_1^2 + 4x_1 x_2 + 3x_2^2 - 5$，$\nabla f(x) = (2x_1 + 4x_2, \; 4x_1 + 6x_2)^{\top}$

从 $x^0 = (1, 1)^{\top}$ 出发：$\nabla f((1,1)^{\top}) = (6, 10)^{\top}$

- **步长 $\alpha = 0.1$（合适）：** 新点 $(1,1)^{\top} - 0.1(6,10)^{\top} = (0.4, 0)^{\top}$。$f(x^0) = 3 \geq -4.84 = f(0.4, 0)$ ✓

- **步长 $\alpha = 0.5$（过大）：** 新点 $(1,1)^{\top} - 0.5(6,10)^{\top} = (-2, -4)^{\top}$。$f(x^0) = 3 \ngeq 79 = f(-2,-4)$ ✗ 过冲！

**关键教训：** 步长太大会导致过冲 (overshoot)，函数值反而增大。

---

## Lecture 4 — GD Algorithm, Local/Global Minimum, Line Search

> 📌 **考点 1: High level description of gradient descent idea**
>
> 📌 **考点 2 引入: Bisection search 初步**

### 4.1 📌 梯度下降的完整算法

输入：函数 $f: \mathbb{R}^n \to \mathbb{R}$，初始点 $x^0$。

在第 $k$ 次迭代，当前点为 $x^k$，计算：

$$x^{k+1} = x^k - \alpha^k \nabla f(x^k)$$

**步长选择（试错法）：** 选择 $\alpha^k$，检查 $f(x^k - \alpha^k \nabla f(x^k)) < f(x^k)$？
- 若是 → 接受该步
- 若否 → 将 $\alpha^k$ 减半，重试

### 4.2 📌 梯度计算示例

$$f(x) = 4 \ln(e^{2x_1} + e^{4x_2}) - 6x_1 - 4x_2$$

$$\nabla f(x) = \left( \frac{8 e^{2x_1}}{e^{2x_1} + e^{4x_2}} - 6, \quad \frac{16 e^{4x_2}}{e^{2x_1} + e^{4x_2}} - 4 \right)$$

### 4.3 📌 Local Minimum vs Global Minimum

- **全局最小值** $x^{\ast}$：$f(x^{\ast}) \leq f(x)$ 对所有 $x$ 成立
- **局部最小值** $\hat{x}$：$f(\hat{x}) \leq f(x)$ 对 $\hat{x}$ 的小邻域内所有 $x$ 成立

**梯度下降的最好期望是到达一个局部最小值。**

### 4.4 凸函数保证局部即全局

如果目标函数是**凸函数**，则局部最小值和全局最小值没有区别——所有局部最小值都是全局最小值。

当目标函数不是凸的时，启发式方法是从**不同初始点**多次运行 GD。

### 4.5 📌 精确线搜索 (Exact Line Search)

当前在点 $x^k$，已算出 $\nabla f(x^k)$。选择步长使下一步函数值尽可能小：

$$\min_{\alpha \geq 0} f\left(x^k - \alpha \, \nabla f(x^k)\right)$$

这是一个关于 $\alpha$ 的**单维优化问题** (single-dimensional optimization)。

### 4.6 📌 单维优化与区间缩减

给定函数 $f: \mathbb{R} \to \mathbb{R}$，从初始不确定区间 $[a^1, b^1]$ 出发，通过迭代缩小区间逼近最小值点。

**区间更新规则：** 取两个试探点 $\lambda^k$（左）和 $\rho^k$（右）：
- 若 $f(\lambda^k) \leq f(\rho^k)$：$a^{k+1} = a^k$，$b^{k+1} = \rho^k$（丢弃右边）
- 若 $f(\lambda^k) > f(\rho^k)$：$a^{k+1} = \lambda^k$，$b^{k+1} = b^k$（丢弃左边）

### 4.7 📌 Bisection Search 初步

将 $\lambda^k$ 和 $\rho^k$ 对称放置在中点两侧：

$$\lambda^k = \frac{a^k + b^k}{2} - \varepsilon, \qquad \rho^k = \frac{a^k + b^k}{2} + \varepsilon$$

$\varepsilon$ 太小 → 数值比较困难；$\varepsilon$ 太大 → 区间缩减慢。

---

## Lecture 5 — Bisection Search & Golden Section Search

> 📌 **考点 2: Bisection and golden section search for single dimensional optimization**

### 5.1 📌 Bisection Search 完整算法

① 选择初始不确定区间 $(a^1, b^1)$，停止容差 $\theta$，令 $k = 1$。

② 设：
$$\lambda^k = \frac{a^k + b^k}{2} - \varepsilon, \qquad \rho^k = \frac{a^k + b^k}{2} + \varepsilon$$

③ 若 $f(\lambda^k) \leq f(\rho^k)$，则 $a^{k+1} = a^k$，$b^{k+1} = \rho^k$。
　若 $f(\lambda^k) > f(\rho^k)$，则 $a^{k+1} = \lambda^k$，$b^{k+1} = b^k$。

④ 若 $b^{k+1} - a^{k+1} \leq \theta$ 则停止。否则 $k \leftarrow k+1$，回到 ②。

**每次迭代需要 2 次函数评估。**

### 5.2 📌 Golden Section Search 推导

**核心问题：** 能否让每次迭代只需 1 次函数评估？

**思路：** 让 $\lambda^{k+1}$ 与 $\rho^k$ 重合（或 $\rho^{k+1}$ 与 $\lambda^k$ 重合），同时要求区间每次缩减同一比例 $\beta$：

$$b^{k+1} - a^{k+1} = \beta \cdot (b^k - a^k)$$

由此推导出：
$$\lambda^k = \beta \cdot a^k + (1-\beta) \cdot b^k, \qquad \rho^k = (1-\beta) \cdot a^k + \beta \cdot b^k$$

**对齐条件：** 要求 $\rho^k = \lambda^{k+1}$（当情况 ★ 发生时），代入化简：

$$(1-\beta)(a^k - b^k) = \beta^2(a^k - b^k) \implies 1 - \beta = \beta^2$$

$$\beta^2 + \beta - 1 = 0 \implies \beta = \frac{-1 + \sqrt{5}}{2} \approx 0.618$$

情况 ★★ 同理得到相同的 $\beta$。

### 5.3 📌 Golden Section Search 完整算法

① 选择 $(a^1, b^1)$，停止容差 $\theta$，令 $k = 1$。

② 设 $\lambda^k = \beta \cdot a^k + (1-\beta) \cdot b^k$，$\rho^k = (1-\beta) \cdot a^k + \beta \cdot b^k$，其中 $\beta = \frac{-1+\sqrt{5}}{2}$。

③ 若 $f(\lambda^k) \leq f(\rho^k)$，则 $a^{k+1} = a^k$，$b^{k+1} = \rho^k$。
　若 $f(\lambda^k) > f(\rho^k)$，则 $a^{k+1} = \lambda^k$，$b^{k+1} = b^k$。

④ 若 $b^{k+1} - a^{k+1} \leq \theta$ 则停止。否则 $k \leftarrow k+1$，回到 ②。

**关键：** 每次迭代只需 1 次新函数评估（复用上一轮的另一个点）。区间每次缩减为原来的 $\beta \approx 0.618$ 倍。

### 5.4 📌 基于导数的单维优化

若可以计算 $f'$，则在区间中点求导：
- $f'((a^k+b^k)/2) \leq 0$（函数递减）→ 最小值在右半：$a^{k+1} = (a^k+b^k)/2$，$b^{k+1} = b^k$
- $f'((a^k+b^k)/2) > 0$（函数递增）→ 最小值在左半：$a^{k+1} = a^k$，$b^{k+1} = (a^k+b^k)/2$

每次迭代区间减半，效率更高。

---

## Lecture 6 — Line Search for GD Step Size & Penalty Function Method

> 📌 **考点 2: 用 Bisection 做 GD 步长搜索**
>
> 📌 **考点 3: Penalty function method**

### 6.1 📌 用单维优化选择 GD 步长

GD 第 $k$ 步在点 $x^k$，需求解：

$$\min_{\alpha \geq 0} f\left(x^k - \alpha \, \nabla f(x^k)\right)$$

$x^k$ 和 $\nabla f(x^k)$ 已知，这是关于 $\alpha$ 的单维优化问题。可以用 bisection search 或 golden section search 求解。

### 6.2 📌 约束优化问题

$$\min \quad f(x) \qquad \text{s.t.} \quad g_1(x) \leq 0, \ldots, g_n(x) \leq 0, \quad h_1(x) = 0, \ldots, h_m(x) = 0$$

**基本策略：** 将约束优化转化为无约束优化，再用标准 GD 求解。

### 6.3 📌 Penalty Function Method

**核心思想：** 将约束违反程度作为惩罚项加入目标函数。

$$\min_x \quad f(x) + \sum_{i=1}^{n} \theta_i \left(\max\lbrace 0, g_i(x)\rbrace\right)^2 + \sum_{i=1}^{m} \gamma_i \left(h_i(x)\right)^2$$

其中 $\theta_1, \ldots, \theta_n, \gamma_1, \ldots, \gamma_m$ 为大的惩罚参数。

**为什么对不等式约束取平方？** $\max\lbrace 0, g(x)\rbrace$ 在零点处不可微，但 $(\max\lbrace 0, g(x)\rbrace)^2$ 是光滑的，可以用 GD。

### 6.4 📌 Penalty Function Method 完整算法

① 选择初始惩罚参数 $\theta_1, \ldots, \theta_n$，$\gamma_1, \ldots, \gamma_m$，容差 $\varepsilon$，放大因子 $\beta > 1$，令 $k = 1$。

② 求解无约束问题：

$$x^k = \arg\min_x \left\lbrace f(x) + \sum_{i=1}^{n} \theta_i^k \left(\max\lbrace 0, g_i(x)\rbrace\right)^2 + \sum_{i=1}^{m} \gamma_i^k \left(h_i(x)\right)^2 \right\rbrace$$

③ 若 $\sum_{i=1}^{n} \theta_i^k (\max\lbrace 0, g_i(x^k)\rbrace)^2 + \sum_{i=1}^{m} \gamma_i^k (h_i(x^k))^2 \leq \varepsilon$，则停止。

否则 $\theta_i^{k+1} = \beta \, \theta_i^k$，$\gamma_i^{k+1} = \beta \, \gamma_i^k$，$k \leftarrow k+1$，回到 ②。

**例子：** $\min (x_1-2)^4 + (x_1-2x_2)^2$ s.t. $x_1^2 - x_2 \leq 0$

---

## Lecture 7 — Barrier Function Method & Taylor's Theorem

> 📌 **考点 3: Barrier function method**
>
> 📌 **考点 4: Taylor's theorem and its implication**

### 7.1 📌 Barrier Function Method

**适用范围：** 仅限不等式约束问题。

$$\min \quad f(x) \qquad \text{s.t.} \quad g_1(x) \leq 0, \ldots, g_n(x) \leq 0$$

**核心思想：** 在可行域边界处设置"障碍墙"——当接近边界时 $g_i(x) \to 0^-$，$-1/g_i(x) \to +\infty$：

$$\min_x \quad f(x) - \sum_{i=1}^{n} \mu_i \frac{1}{g_i(x)}$$

其中 $\mu_1, \ldots, \mu_n$ 为小的 barrier 参数。

### 7.2 📌 Barrier Function Method 完整算法

① 选择初始 barrier 参数 $\mu_1, \ldots, \mu_n$，容差 $\varepsilon$，缩减因子 $\beta \in (0,1)$，令 $k = 1$。选择 $x^0$ 严格在可行域内部（$g_i(x^0) < 0$）。

② 从 $x^{k-1}$ 出发，用 GD 求解：

$$x^k = \arg\min_x \left\lbrace f(x) - \sum_{i=1}^{n} \mu_i^k \frac{1}{g_i(x)} \right\rbrace$$

③ 若 $-\sum_{i=1}^{n} \mu_i^k \frac{1}{g_i(x^k)} \leq \varepsilon$，则停止。否则 $\mu_i^{k+1} = \beta \, \mu_i^k$，$k \leftarrow k+1$，回到 ②。

**Penalty vs Barrier 对比：**

| 特性 | Penalty Method | Barrier Method |
|------|---------------|----------------|
| 约束类型 | 等式 + 不等式 | 仅不等式 |
| 初始点 | 任意 | 必须严格可行 |
| 参数变化 | 增大（$\beta > 1$） | 缩小（$\beta \in (0,1)$） |
| 迭代点 | 可能不可行 | 始终严格可行 |

### 7.3 📌 小 $o$ 记号

$g(t)$ 是 $o(t)$ 函数，如果：

$$\lim_{t \to 0} \frac{g(t)}{t} = 0$$

**例子：**
- $g(t) = t^2$：$\lim_{t\to 0} t^2/t = 0$ → ✓ 是 $o(t)$
- $g(t) = 2t$：$\lim_{t\to 0} 2t/t = 2$ → ✗ 不是 $o(t)$
- $g(t) = \sqrt{t}$：$\lim_{t\to 0} \sqrt{t}/t = \infty$ → ✗ 不是 $o(t)$

**直观理解：** $o(t)$ 表示"比 $t$ 趋于 0 更快的函数"——即 $t^2, t^3, t^{3/2}$ 等。

### 7.4 📌 Taylor's Theorem

设 $f: \mathbb{R}^n \to \mathbb{R}$ 无穷可微，则：

$$f(y) = f(x) + \nabla^{\top} f\left(\gamma x + (1-\gamma)y\right) \cdot (y - x)$$

其中 $\gamma \in (0, 1)$。

这本质上是**多维中值定理**：存在 $x$ 和 $y$ 之间的某个中间点，其梯度的投影等于函数值之差。

---

## Lecture 8 — Taylor's Theorem Implication & Convex Functions

> 📌 **考点 4: Taylor's theorem and its implication**
>
> 📌 **考点 5: Definition of a convex function**

### 8.1 📌 Taylor 定理的推论（Implication）

$$f(y) = f(x) + \nabla^{\top} f(x) \cdot (y - x) + o(\lVert y - x \rVert)$$

**证明：** 由 Taylor 定理：

$$f(y) = f(x) + \nabla^{\top} f(\gamma x + (1-\gamma)y) \cdot (y-x)$$

$$= f(x) + \nabla^{\top} f(x) \cdot (y-x) + \underbrace{\left(\nabla^{\top} f(\gamma x + (1-\gamma)y) - \nabla^{\top} f(x)\right)(y-x)}_{(\star)}$$

其中 $(\star)$ 需证明为 $o(\lVert y-x \rVert)$。

用内积展开：

$$= f(x) + \nabla^{\top} f(x)(y-x) + \lVert \nabla f(\gamma x + (1-\gamma)y) - \nabla f(x) \rVert \cdot \lVert y-x \rVert \cdot \cos\phi$$

验证 $o(\lVert y-x \rVert)$：

$$\lim_{\lVert y-x \rVert \to 0} \frac{\lVert \nabla f(\gamma x + (1-\gamma)y) - \nabla f(x) \rVert \cdot \lVert y-x \rVert \cdot \cos\phi}{\lVert y-x \rVert} = 0$$

因为 $\lVert y-x \rVert \to 0$ 时 $\gamma x + (1-\gamma)y \to x$，故梯度差 $\to 0$。$\blacksquare$

### 8.2 📌 Taylor 推论的重要应用：GD 使函数值下降

设 $\nabla f(\hat{x}) \neq 0$，则存在足够小的 $\alpha > 0$ 使得 $f(\hat{x} - \alpha \nabla f(\hat{x})) < f(\hat{x})$。

**证明：** 令 $y = \hat{x} - \alpha \nabla f(\hat{x})$，由推论：

$$f(\hat{x} - \alpha \nabla f(\hat{x})) = f(\hat{x}) - \alpha \lVert \nabla f(\hat{x}) \rVert^2 + o(\alpha \lVert \nabla f(\hat{x}) \rVert)$$

$$= f(\hat{x}) + \alpha \lVert \nabla f(\hat{x}) \rVert \left(-\lVert \nabla f(\hat{x}) \rVert + \frac{o(\alpha \lVert \nabla f(\hat{x}) \rVert)}{\alpha \lVert \nabla f(\hat{x}) \rVert}\right)$$

因为 $\lim_{\alpha \to 0} \frac{o(\alpha \lVert \nabla f(\hat{x}) \rVert)}{\alpha \lVert \nabla f(\hat{x}) \rVert} = 0$，可选 $\alpha$ 足够小使：

$$\frac{o(\alpha \lVert \nabla f(\hat{x}) \rVert)}{\alpha \lVert \nabla f(\hat{x}) \rVert} \leq \frac{1}{2}\lVert \nabla f(\hat{x}) \rVert$$

代入得：

$$f(\hat{x} - \alpha \nabla f(\hat{x})) \leq f(\hat{x}) - \frac{1}{2}\alpha \lVert \nabla f(\hat{x}) \rVert^2 < f(\hat{x}) \quad \blacksquare$$

### 8.3 📌 凸函数的定义

$f: \mathbb{R}^n \to \mathbb{R}$ 是凸的，当且仅当：

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y) \quad \forall\, x,y \in \mathbb{R}^n,\; \lambda \in [0,1]$$

**几何含义：** 函数曲线上任意两点的连线段（弦）在函数曲线的**上方**。

### 8.4 📌 Jensen 不等式

设 $f$ 凸，$x^1, \ldots, x^k \in \mathbb{R}^n$，$\lambda^1, \ldots, \lambda^k \geq 0$，$\sum_{\ell=1}^{k} \lambda^\ell = 1$，则：

$$f\left(\sum_{\ell=1}^{k} \lambda^\ell x^\ell\right) \leq \sum_{\ell=1}^{k} \lambda^\ell f(x^\ell)$$

**证明思路：** 反复使用凸函数的二点定义。将 $\sum_{\ell=1}^{k} \lambda^\ell x^\ell$ 写成 $\lambda^1 x^1 + (1-\lambda^1) \cdot$ [其余的凸组合]，然后递归展开。

---

## Lecture 9 — First-Order Characterization of Convexity & Local = Global

> 📌 **考点 5: Definition of a convex function**（续）
>
> 📌 **考点 6: First order characterization of convexity**

### 9.1 📌 一阶特征化定理（First-Order Characterization）

$f: \mathbb{R}^n \to \mathbb{R}$ 凸，当且仅当：

$$f(y) \geq f(x) + \nabla^{\top} f(x) \cdot (y - x) \quad \forall\, x, y \in \mathbb{R}^n$$

**几何含义：** 凸函数在任意点的切线（一阶 Taylor 近似）是函数的**全局下界**。

### 9.2 📌 证明（⟹ 方向：凸 → 一阶条件）

设 $f$ 凸，由凸性：

$$f(x + \lambda(y-x)) \leq (1-\lambda)f(x) + \lambda f(y)$$

$$\implies f(y) \geq f(x) + \frac{f(x + \lambda(y-x)) - f(x)}{\lambda}$$

由 Taylor 推论展开分子：

$$= f(x) + \frac{\nabla^{\top} f(x) \cdot \lambda(y-x) + o(\lambda\lVert y-x \rVert)}{\lambda}$$

$$= f(x) + \nabla^{\top} f(x)(y-x) + \frac{o(\lambda\lVert y-x \rVert)}{\lambda\lVert y-x \rVert} \cdot \lVert y-x \rVert$$

对所有 $\lambda \in (0,1)$ 成立，令 $\lambda \to 0$，余项趋于 0：

$$f(y) \geq f(x) + \nabla^{\top} f(x)(y-x) \quad \blacksquare$$

### 9.3 📌 证明（⟸ 方向：一阶条件 → 凸）

取任意 $a, b \in \mathbb{R}^n$，$\lambda \in [0,1]$，令 $c = \lambda a + (1-\lambda)b$。

由一阶条件分别以 $c$ 为切点：

$$f(a) \geq f(c) + \nabla^{\top} f(c) \cdot (a - c) \qquad \times \lambda$$

$$f(b) \geq f(c) + \nabla^{\top} f(c) \cdot (b - c) \qquad \times (1-\lambda)$$

两式加权求和：

$$\lambda f(a) + (1-\lambda)f(b) \geq f(c) + \nabla^{\top} f(c) \cdot \underbrace{[\lambda(a-c) + (1-\lambda)(b-c)]}_{= \lambda a + (1-\lambda)b - c = 0}$$

$$= f(c) = f(\lambda a + (1-\lambda)b) \quad \blacksquare$$

### 9.4 📌 局部最小值 = 全局最小值（凸函数）

**定理：** 设 $f: \mathbb{R}^n \to \mathbb{R}$ 凸，$\hat{x}$ 是 $f$ 的局部最小值，则 $\hat{x}$ 是全局最小值。

**证明：** 对任意 $y \in \mathbb{R}^n$，因为 $\hat{x}$ 是局部最小，存在 $\varepsilon > 0$ 使得 $f(\hat{x}) \leq f(z)$ 对 $\lVert z - \hat{x} \rVert \leq \varepsilon$ 成立。

选 $\lambda$ 使得 $(1-\lambda)\lVert y - \hat{x} \rVert \leq \varepsilon$，则 $\lambda\hat{x} + (1-\lambda)y$ 在 $\hat{x}$ 的 $\varepsilon$-邻域内：

$$f(\hat{x}) \leq f(\lambda\hat{x} + (1-\lambda)y) \leq \lambda f(\hat{x}) + (1-\lambda)f(y)$$

$$\implies (1-\lambda)f(\hat{x}) \leq (1-\lambda)f(y) \implies f(\hat{x}) \leq f(y) \quad \blacksquare$$

### 9.5 📌 梯度为零 → 全局最小

**定理：** 设 $f$ 凸，若 $\nabla f(\hat{x}) = 0$，则 $\hat{x}$ 是全局最小值。

**证明：** 对任意 $y$，由一阶特征化：

$$f(y) \geq f(\hat{x}) + \underbrace{\nabla^{\top} f(\hat{x})}_{= 0} \cdot (y - \hat{x}) = f(\hat{x}) \quad \blacksquare$$

---

## Lecture 10 — GD Convergence: Bounded Gradients

> 📌 **考点 7: Convergence of gradient descent with fixed step size for functions with bounded gradients**

### 10.1 📌 问题设置

最小化凸函数 $f: \mathbb{R}^n \to \mathbb{R}$，使用常步长 $\gamma$ 的 GD：

$$x^{k+1} = x^k - \gamma \, \nabla f(x^k) = x^k - \gamma \, g^k$$

其中 $g^k = \nabla f(x^k)$，$x^{\ast}$ 为全局最小点。

**常步长的局限：** 无法收敛到最小值，只能在最小值附近**弹跳** (bounce)。

**例子：** $f(x) = (x-1)^2$，$x^0 = 2$，$\gamma = 1$：
- $x^1 = 2 - 1 \cdot 2(2-1) = 0$
- $x^2 = 0 - 1 \cdot 2(0-1) = 2$
- $x^3 = 0, x^4 = 2, \ldots$ → 在 $x^{\ast} = 1$ 两侧弹跳

### 10.2 📌 预备分析 (Preliminary Analysis)

**有用等式：**

$$\lVert x - y \rVert^2 = \lVert x \rVert^2 + \lVert y \rVert^2 - 2x^{\top} y$$

**关键推导步骤：**

**Step 1：** 由一阶特征化：$f(x^{\ast}) \geq f(x^k) + (g^k)^{\top}(x^{\ast} - x^k)$，故：

$$f(x^k) - f(x^{\ast}) \leq (g^k)^{\top}(x^k - x^{\ast}) \qquad (\star)$$

**Step 2：** 由 GD 规则 $g^k = \frac{1}{\gamma}(x^k - x^{k+1})$，展开内积：

$$(g^k)^{\top}(x^k - x^{\ast}) = \frac{1}{\gamma}(x^k - x^{k+1})^{\top}(x^k - x^{\ast})$$

$$= \frac{1}{2\gamma}\left(\lVert x^k - x^{k+1} \rVert^2 + \lVert x^k - x^{\ast} \rVert^2 - \lVert x^{k+1} - x^{\ast} \rVert^2\right)$$

$$= \frac{\gamma}{2}\lVert g^k \rVert^2 + \frac{1}{2\gamma}\left(\lVert x^k - x^{\ast} \rVert^2 - \lVert x^{k+1} - x^{\ast} \rVert^2\right)$$

**Step 3：** 对 $k = 0, 1, \ldots, K-1$ 求和（**伸缩和** telescoping sums）：

$$\sum_{k=0}^{K-1}(g^k)^{\top}(x^k - x^{\ast}) = \frac{\gamma}{2}\sum_{k=0}^{K-1}\lVert g^k \rVert^2 + \frac{1}{2\gamma}\left(\lVert x^0 - x^{\ast} \rVert^2 - \lVert x^K - x^{\ast} \rVert^2\right)$$

$$\leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\lVert g^k \rVert^2 + \frac{1}{2\gamma}\lVert x^0 - x^{\ast} \rVert^2$$

**Step 4：** 由 $(\star)$ 代入得：

$$\sum_{k=0}^{K-1}\left(f(x^k) - f(x^{\ast})\right) \leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\lVert g^k \rVert^2 + \frac{1}{2\gamma}\lVert x^0 - x^{\ast} \rVert^2$$

**直觉解读：**
- $\gamma$ 太小 → 被 $\frac{1}{2\gamma}\lVert x^0 - x^{\ast} \rVert^2$ 主导（步太小，卡在初始点附近）
- $\gamma$ 太大 → 被 $\frac{\gamma}{2}\sum\lVert g^k \rVert^2$ 主导（步太大，不断过冲）

### 10.3 📌 收敛定理（有界梯度）

**定理：** 设 $f$ 凸，$\lVert \nabla f(x) \rVert \leq B$ 对所有 $x$ 成立，$\lVert x^0 - x^{\ast} \rVert = R$。若选步长：

$$\gamma = \frac{R}{B\sqrt{K}}$$

则 $K$ 次迭代后：

$$\frac{1}{K}\sum_{k=0}^{K-1}\left(f(x^k) - f(x^{\ast})\right) \leq \frac{RB}{\sqrt{K}}$$

**证明：** 由预备分析结果，代入 $\lVert g^k \rVert \leq B$：

$$\sum_{k=0}^{K-1}(f(x^k) - f(x^{\ast})) \leq \frac{\gamma}{2}KB^2 + \frac{R^2}{2\gamma}$$

最优化步长 $\gamma$：令 $H'(\gamma) = \frac{1}{2}(KB^2 - R^2/\gamma^2) = 0$，得 $\gamma = R/(B\sqrt{K})$。

代入：$\sum(f(x^k) - f(x^{\ast})) \leq RB\sqrt{K}$

除以 $K$：$\frac{1}{K}\sum(f(x^k) - f(x^{\ast})) \leq RB/\sqrt{K}$ $\blacksquare$

**收敛速率：** $O(1/\sqrt{K})$。迭代复杂度：要达精度 $\varepsilon$，需 $K = (RB/\varepsilon)^2$ 次迭代。

---

## Lecture 11 — GD Convergence: Smooth Convex Functions

> 📌 **考点 8: Preliminary analysis / convergence of gradient descent with fixed step size for smooth convex functions**

### 11.1 📌 光滑凸函数的定义

**定义：** 凸函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是**光滑的** (smooth)，参数为 $L$，如果：

$$f(y) \leq f(x) + \nabla f(x)^{\top}(y-x) + \frac{L}{2}\lVert y-x \rVert^2 \quad \forall\, x, y \in \mathbb{R}^n$$

**直观理解：** 光滑凸函数被夹在：
- **下界：** 切线（一阶近似）$f(x) + \nabla f(x)^{\top}(y-x)$（一阶特征化）
- **上界：** 二次近似 $f(x) + \nabla f(x)^{\top}(y-x) + \frac{L}{2}\lVert y-x \rVert^2$

### 11.2 📌 预备结论 1：GD 单步下降量

设 $f$ 光滑（参数 $L$），步长 $\gamma$ 的 GD 满足：

$$f(x^{k+1}) \leq f(x^k) - \gamma\left(1 - \frac{\gamma L}{2}\right)\lVert g^k \rVert^2$$

**证明：** 由光滑性定义（令 $y = x^{k+1} = x^k - \gamma g^k$）：

$$f(x^{k+1}) \leq f(x^k) + (g^k)^{\top}(-\gamma g^k) + \frac{L}{2}\gamma^2\lVert g^k \rVert^2 = f(x^k) - \gamma\left(1 - \frac{L\gamma}{2}\right)\lVert g^k \rVert^2 \quad \blacksquare$$

**特别地，取 $\gamma = 1/L$：**

$$f(x^{k+1}) \leq f(x^k) - \frac{1}{2L}\lVert g^k \rVert^2$$

这意味着 $f(x^0) \geq f(x^1) \geq f(x^2) \geq \cdots$（函数值单调递减）。

### 11.3 📌 预备结论 2（复用 Lecture 10 的预备分析）

$$\sum_{k=0}^{K-1}(f(x^k) - f(x^{\ast})) \leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\lVert g^k \rVert^2 + \frac{1}{2\gamma}\lVert x^0 - x^{\ast} \rVert^2$$

### 11.4 📌 收敛定理（光滑凸函数）

**定理：** 设 $f: \mathbb{R}^n \to \mathbb{R}$ 光滑凸（参数 $L$），$\lVert x^0 - x^{\ast} \rVert \leq R$。取步长 $\gamma = 1/L$，则：

$$f(x^K) - f(x^{\ast}) \leq \frac{LR^2}{2K}$$

**证明：**

由结论 1：$\frac{1}{2L}\lVert g^k \rVert^2 \leq f(x^k) - f(x^{k+1})$

对 $k = 0, \ldots, K-1$ 求和（telescoping）：

$$\frac{1}{2L}\sum_{k=0}^{K-1}\lVert g^k \rVert^2 \leq f(x^0) - f(x^K) \qquad (\ast)$$

由结论 2（取 $\gamma = 1/L$）：

$$\sum_{k=0}^{K-1}(f(x^k) - f(x^{\ast})) \leq \frac{1}{2L}\sum_{k=0}^{K-1}\lVert g^k \rVert^2 + \frac{L}{2}R^2$$

由 $(\ast)$ 代入：

$$\sum_{k=0}^{K-1}(f(x^k) - f(x^{\ast})) \leq f(x^0) - f(x^K) + \frac{L}{2}R^2$$

整理：

$$\sum_{k=1}^{K}(f(x^k) - f(x^{\ast})) \leq \frac{L}{2}R^2$$

由结论 1，$f(x^K)$ 是最小的，故：

$$f(x^K) - f(x^{\ast}) \leq \frac{1}{K}\sum_{k=1}^{K}(f(x^k) - f(x^{\ast})) \leq \frac{LR^2}{2K} \quad \blacksquare$$

### 11.5 📌 迭代复杂度与渐近收敛

**迭代复杂度：** 要达精度 $\varepsilon$：

$$K = \frac{LR^2}{2\varepsilon}$$

对比有界梯度情形 $K = (RB/\varepsilon)^2$，光滑凸函数收敛**快得多**（$O(1/K)$ vs $O(1/\sqrt{K})$）。

**渐近收敛：**

$$0 \leq f(x^K) - f(x^{\ast}) \leq \frac{LR^2}{2K} \xrightarrow{K \to \infty} 0$$

$$\implies \lim_{K \to \infty} f(x^K) = f(x^{\ast})$$

长时间运行 GD，最后一个迭代点的函数值可以任意接近最优值。

---

## Lecture 12 — Convex Sets & Projected Gradient Descent

> 📌 **考点 9: Convex sets**
>
> [非考点] Projected Gradient Descent（不考，但了解概念）

### 12.1 📌 凸集的定义

$C \subseteq \mathbb{R}^n$ 是凸集，当且仅当：

$$\forall\, x, y \in C,\; \lambda \in [0,1]: \quad \lambda x + (1-\lambda)y \in C$$

**直观理解：** 凸集内任意两点的连线段完全在集合内部——从任一点到任一点都有"直接视线"。

### 12.2 📌 凸集的交仍是凸集

**定理：** 若 $C_1, C_2, \ldots, C_\ell$ 均为凸集，则 $C_1 \cap C_2 \cap \cdots \cap C_\ell$ 也是凸集。

**证明：** 取 $x, y \in C_1 \cap \cdots \cap C_\ell$，$\lambda \in [0,1]$。

对每个 $i$：$x \in C_i$ 且 $y \in C_i$，因为 $C_i$ 凸，故 $\lambda x + (1-\lambda)y \in C_i$。

对所有 $i$ 均成立，故 $\lambda x + (1-\lambda)y \in C_1 \cap \cdots \cap C_\ell$。$\blacksquare$

### 12.3 📌 凸函数的下水平集是凸集

**定理：** 设 $f: \mathbb{R}^n \to \mathbb{R}$ 凸，定义 $C = \lbrace x \in \mathbb{R}^n : f(x) \leq 0\rbrace$，则 $C$ 是凸集。

**证明：** 取 $x, y \in C$（即 $f(x) \leq 0$，$f(y) \leq 0$），$\lambda \in [0,1]$：

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y) \leq \lambda \cdot 0 + (1-\lambda) \cdot 0 = 0$$

故 $\lambda x + (1-\lambda)y \in C$。$\blacksquare$

### 12.4 📌 约束优化的可行域是凸集

考虑约束问题 $\min f(x)$ s.t. $g_1(x) \leq 0, \ldots, g_m(x) \leq 0$，其中 $g_i$ 均凸。

- 定义 $C_i = \lbrace x : g_i(x) \leq 0\rbrace$，由上述定理 $C_i$ 是凸集
- 可行域 $C = C_1 \cap C_2 \cap \cdots \cap C_m$ 是凸集的交，故 $C$ 也是凸集

### 12.5 [非考点] 严格凸函数与唯一最优解

**定义：** $f$ **严格凸** (strictly convex) 如果对 $x \neq y$，$\lambda \in (0,1)$：

$$f(\lambda x + (1-\lambda)y) < \lambda f(x) + (1-\lambda)f(y)$$

**定理：** 设 $g$ 严格凸，$\mathbb{X}$ 为凸集，则 $\min_{x \in \mathbb{X}} g(x)$ 有**唯一**最优解。

**证明（反证法）：** 假设 $x^{\ast} \neq y^{\ast}$ 均为最优解，$g(x^{\ast}) = g(y^{\ast}) = z^{\ast}$。因为 $\mathbb{X}$ 凸，$\lambda x^{\ast} + (1-\lambda)y^{\ast} \in \mathbb{X}$。由严格凸性：

$$g(\lambda x^{\ast} + (1-\lambda)y^{\ast}) < \lambda z^{\ast} + (1-\lambda)z^{\ast} = z^{\ast}$$

这给出了严格更好的可行解，与 $z^{\ast}$ 的最优性矛盾。$\blacksquare$

### 12.6 [非考点] Projected Gradient Descent

当 GD 的更新 $x^k - \gamma \nabla f(x^k)$ 可能落在可行域 $\mathbb{X}$ 外部时，需要**投影**回去。

**投影算子：**

$$\Pi_\mathbb{X}(x) = \arg\min_{y \in \mathbb{X}} \lVert y - x \rVert^2$$

**Projected GD 算法：**

$$y^{k+1} = x^k - \gamma \nabla f(x^k), \qquad x^{k+1} = \Pi_\mathbb{X}(y^{k+1})$$

注意：若 $x \in \mathbb{X}$，则 $\Pi_\mathbb{X}(x) = x$（无需投影）。

---

## 考点覆盖总结

| 考点 | 主要 Lecture | 关键内容 |
|------|-------------|---------|
| 1. GD idea | Lec 3, 4 | 梯度定义、GD 更新规则、步长选择 |
| 2. Bisection & Golden Section | Lec 4, 5, 6 | 两种算法、$\beta = (-1+\sqrt{5})/2$、步长线搜索 |
| 3. Penalty & Barrier | Lec 6, 7 | 两种方法的完整算法、对比 |
| 4. Taylor's Theorem | Lec 7, 8 | 定理陈述、推论、GD 函数值下降证明 |
| 5. Convex Function Def | Lec 8, 9 | 定义、Jensen 不等式 |
| 6. First-Order Characterization | Lec 9 | 双向证明、局部=全局、$\nabla f = 0$ 充分性 |
| 7. Convergence (Bounded Grad) | Lec 10 | 预备分析、$O(1/\sqrt{K})$ 收敛 |
| 8. Convergence (Smooth Convex) | Lec 11 | 光滑性定义、$O(1/K)$ 收敛 |
| 9. Convex Sets | Lec 12 | 定义、交集、下水平集 |

---

# Typical Exam Questions

## Code & Excel

### Topic 2 — Bisection and Golden Section Search

#### Bisection Search

**Target function:** $f(x) = 6.4x^5 + 20x^4 + 0.9x^3 - 19.6x^2 + 2.1x + 8.2$

```python
import numpy as np
from matplotlib import pyplot as plt

def function(x):
    ## this computes the target function in question 1
    return 6.4 * x**5 + 20 * x**4 + 0.9 * x **3 - 19.6 * x**2 + 2.1 * x + 8.2

def bisection_search(func, a, b, eps, theta):
    ## This function implements a binary search .
    ## func: the target function
    ## a, b: the initial search interval
    ## eps: the perturbation parameter used for comparison
    ## theta: the stopping tolerance on the interval length
    k = 0
    while np.abs(b - a) > theta:
        mid = (a + b)/2
        lam, rho = mid - eps, mid + eps
        f_lam, f_rho = func(lam), func(rho)
        print(f"Iter {k}: [{a:.6f}, {b:.6f}] | lambda ={lam:.6f}, rho={rho:.6f} "
              f"| f(lambda)={f_lam:.6f}, f(rho)={f_rho:.6f}")
        if f_lam <= f_rho:
            b = rho
        else:
            a = lam
        k += 1
    return (a + b)/2
```

**Results (interval $[-2, 1]$):**

```
Local minimum at x = -0.936, objective value = -0.9221429421238625
```

**Results (interval $[-1.25, 2]$):**

```
Local minimum at x = 0.588, objective value = 5.681785651717276
```

**Plot:**

![HW1 Bisection Search Plot](assets/homework_1_solution_p3_img1.png)

**Discussion:** The function has at least two local minima: one near $x \approx -0.94$ and another near $x \approx 0.59$. The minimum near $-0.94$ has a smaller objective value and is therefore the better minimum. The bisection search method may converge to different local minima depending on the initial interval.

#### Golden Section Search

```python
def golden_section_search(func, a, b, theta):
    ## This function implements the golden section search
    ## func: the target function
    ## a, b: the initial search interval
    ## theta: the stopping tolerance on the interval length
    beta = (-1 + np.sqrt(5))/2
    k = 0
    lam = beta * a + (1-beta) * b
    rho = (1-beta) * a + beta * b
    f_lam, f_rho = func(lam), func(rho)
    while np.abs(b - a) > theta:
        print(f"Iter {k}: [{a:.6f}, {b:.6f}] | lambda ={lam:.6f}, rho={rho:.6f} "
              f"| f(lambda)={f_lam:.6f}, f(rho)={f_rho:.6f}")
        if f_lam <= f_rho:
            # Update interval
            b = rho
            rho = lam
            lam = beta * a + (1-beta) * b
            # Update function value
            f_rho = f_lam
            f_lam = func(lam)
        else:
            # Update interval
            a = lam
            lam = rho
            rho = (1-beta) * a + beta * b
            # Update function value
            f_lam = f_rho
            f_rho = func(rho)
        k += 1
    return (a+b)/2
```

**Results:**

```
Local minimum at x = -0.938, objective value = -0.9222264671231972
```

---

### Topic 1 — Gradient Descent

#### Gradient Descent with Bisection Line Search

**Function:** $f(x_1, x_2) = (x_1 - 4x_2 + 4)^2 + (x_1 + 3x_2 + 2)^4$

**Gradient:**

$$\nabla f(x_1, x_2) = \begin{pmatrix} 2(x_1 - 4x_2 + 4) + 4(x_1 + 3x_2 + 2)^3 \\ -8(x_1 - 4x_2 + 4) + 12(x_1 + 3x_2 + 2)^3 \end{pmatrix}$$

```python
import numpy as np

def function(x):
    ## this computes the target function in question 1
    x1, x2 = x[0], x[1]
    return (x1 - 4 * x2 + 4)**2 + (x1 + 3 * x2 + 2)**4

def gradient(x):
    ## this computes the gradient of the target function
    x1, x2 = x[0], x[1]
    g1 = 2 * (x1 - 4 * x2 + 4) + 4 * (x1 + 3 * x2 + 2)**3
    g2 = -8 * (x1 - 4 * x2 + 4) + 12 * (x1 + 3 * x2 + 2)**3
    return np.array([g1, g2])

def bisection_search(func, a, b, eps, theta, max_iter=1000):
    ## This function implements a binary search for step size
    ## func: the one-dimensional function along the search direction
    ## a, b: the initial search interval
    ## eps: the perturbation parameter used for comparison
    ## theta: the stopping tolerance on the interval length
    k = 0
    while np.abs(b - a) > theta and k < max_iter:
        mid = (a + b) / 2
        lam, rho = mid - eps, mid + eps
        f_lam, f_rho = func(lam), func(rho)
        if f_lam <= f_rho:
            b = rho
        else:
            a = lam
        k += 1
    return (a + b) / 2

def gradient_descent(x0, eps, theta, max_iter = 1000):
    ## This function implements gradient descent with bisection line search
    ## x0: initial point
    ## eps, theta: parameters for bisection search
    ## max_iter: maximum number of iterations for gradient descent
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        f_val = function(x)
        print(f"Iter {k}: x = {x}, f(x) = {f_val:.10f}")
        g = gradient(x)
        if np.linalg.norm(g) <= 1e-8:
            break
        d = -g
        def line_function(alpha):
            return function(x + alpha * d)
        alpha_star = bisection_search(
            line_function, 0, 1, eps, theta
        )
        x = x + alpha_star * d
    return x
```

**Results:** Starting from $(1, 1)$ with $\epsilon = 10^{-4}$, $\theta = 10^{-5}$:

```
x = [-2.84263012  0.28933933], f(x) = 4.16e-07
```

**Analytical verification:** The function is a sum of two nonnegative terms: $(x_1 - 4x_2 + 4)^2 \geq 0$ and $(x_1 + 3x_2 + 2)^4 \geq 0$. Hence $f(x_1, x_2) \geq 0$ for all $(x_1, x_2) \in \mathbb{R}^2$. The minimum value is 0, attained when $x_1 - 4x_2 + 4 = 0$ and $x_1 + 3x_2 + 2 = 0$. Solving: $x^{\ast} = \left(-\frac{20}{7}, \frac{2}{7}\right)$, $f(x^{\ast}) = 0$.

---

### Topic 3 — Penalty Function Method

#### Penalty Function Method

**Problem:** Minimize $f(x) = (x_1 - 2)^2 + 2(x_2 - 4)^2 + x_1 x_2$ subject to $g_1(x) = x_1 + x_2 - 2 \leq 0$, $g_2(x) = x_1^2 + x_2^2 - 3 \leq 0$.

**Part (a): Code**

```python
import numpy as np

# objective f(x)
def function(x):
    x1, x2 = x[0], x[1]
    return (x1 - 2)**2 + 2*(x2 - 4)**2 + x1*x2

def gradient(x):
    x1, x2 = x[0], x[1]
    dfdx1 = 2*(x1 - 2) + x2
    dfdx2 = 4*(x2 - 4) + x1
    return np.array([dfdx1, dfdx2])

# constraints: g_i(x) <= 0
def constraint1(x):
    return x[0] + x[1] - 2
def constraint1_grad(x):
    return np.array([1.0, 1.0])
def constraint2(x):
    return x[0]**2 + x[1]**2 - 3
def constraint2_grad(x):
    return np.array([2*x[0], 2*x[1]])
```

```python
# penalized objective
def penalized_obj(x, th1, th2):
    val = function(x)
    g1 = constraint1(x)
    g2 = constraint2(x)
    val += th1 * (max(0, g1))**2 + th2 * (max(0, g2))**2
    return val

def penalized_gradient(x, th1, th2):
    gr = gradient(x).copy()
    g1 = constraint1(x)
    g2 = constraint2(x)
    # from the hint: d/dx (max{0,g(x)})^2 = 2*g(x)*dg/dx if g(x)>0, else 0
    if g1 > 0:
        gr += th1 * 2 * g1 * constraint1_grad(x)
    if g2 > 0:
        gr += th2 * 2 * g2 * constraint2_grad(x)
    return gr
```

```python
def gd_minimize(x0, th1, th2, maxiter=10000, tol=1e-10):
    x = x0.copy()
    for _ in range(maxiter):
        g = penalized_gradient(x, th1, th2)
        # backtracking to pick step size, otherwise blows up for big theta
        lr = 1.0
        fval = penalized_obj(x, th1, th2)
        for __ in range(50):
            xtry = x - lr * g
            if penalized_obj(xtry, th1, th2) < fval - 1e-12 * lr * np.dot(g, g):
                break
            lr *= 0.5
        xnew = x - lr * g
        if np.linalg.norm(xnew - x) < tol:
            break
        x = xnew
    return x
```

```python
def penalty_method(x0, theta_init=10.0, scale=10.0, max_outer=12):
    th = theta_init
    x = x0.copy()
    prev_x = x.copy()

    print(f"{'Iter':>4}  {'x1':>10}  {'x2':>10}  {'f(x)':>12}  {'h(x)':>12}  {'g1(x)':>10}  {'g2(x)':>10}")
    print("-" * 82)

    for k in range(1, max_outer + 1):
        x = gd_minimize(x, th, th)
        fval = function(x)
        hval = penalized_obj(x, th, th)
        g1val = constraint1(x)
        g2val = constraint2(x)
        print(f"{k:4d}  {x[0]:10.6f}  {x[1]:10.6f}  {fval:12.6f}  {hval:12.6f}  {g1val:10.6f}  {g2val:10.6f}")

        # stop when solution stops moving
        if k > 1 and np.linalg.norm(x - prev_x) < 1e-5:
            print(f"\nConverged at outer iteration {k}.")
            break

        prev_x = x.copy()
        th *= scale

    print(f"\nSolution: x = ({x[0]:.6f}, {x[1]:.6f})")
    print(f"Objective value: f(x) = {function(x):.6f}")
    print(f"g1(x) = x1+x2-2 = {constraint1(x):.12f}")
    print(f"g2(x) = x1^2+x2^2-3 = {constraint2(x):.10f}")
    return x
```

```python
x_init = np.array([0.0, 0.0])
x_sol = penalty_method(x_init)
```

**Output:**

```
Iter        x1        x2        f(x)        h(x)     g1(x)     g2(x)
----------------------------------------------------------------------------------
   1    0.274724    1.744780    13.627947    13.775103    0.019503    0.119730
   2    0.289955    1.711290    13.896839    13.912837    0.001245    0.012587
   3    0.292581    1.707531    13.925696    13.927314    0.000112    0.001267
   4    0.292862    1.707149    13.928608    13.928770    0.000011    0.000127
   5    0.292890    1.707111    13.928900    13.928916    0.000001    0.000013
   6    0.292893    1.707107    13.928929    13.928931    0.000000    0.000001

Converged at outer iteration 6.

Solution: x = (0.292893, 1.707107)
Objective value: f(x) = 13.928929
g1(x) = x1+x2-2 = 0.000000110386
g2(x) = x1^2+x2^2-3 = 0.0000012677
```

**Converged solution:** $x^{\ast} \approx (0.2929, 1.7071)$, $f(x^{\ast}) \approx 13.929$. Both constraints are active at the solution ($g_1(x) \approx 0$, $g_2(x) \approx 0$).

**Convergence visualization:**

![Penalty Method Convergence Path](assets/hw3_fig1.png)

**Part (c): Excel Solver Verification**

Excel Solver (GRG Nonlinear / 非线性 GRG) setup:
- Decision variables: $x_1, x_2$ (cells B1:B2, starting from 0)
- Objective cell: B4 = `=(B1-2)^2 + 2*(B2-4)^2 + B1*B2` → minimize
- Constraints: `B5 <= 0` ($x_1 + x_2 - 2 \leq 0$) and `B6 <= 0` ($x_1^2 + x_2^2 - 3 \leq 0$)

![Excel Solver Setup and Results](assets/hw3_p5_img1.jpeg)

**Excel result:** $x_1 = 0.2929$, $x_2 = 1.7071$, $f = 13.929$

This matches the penalty method result. Both find the same optimum at the intersection of the two constraint boundaries.

---

## Deduction Problem

### Topic 4 — Taylor's Theorem

#### Proof that $h(x, y) = o(\lVert y - x \rVert)$

Let $f(x) = x^3 + 4x^2 + 5x + 1$. Then $f'(x) = 3x^2 + 8x + 5$.

Define $h(x, y) = f(y) - f(x) - f'(x)(y - x)$.

**Goal:** Show that $h(x, y) = o(\lVert y - x \rVert)$.

**Proof:** First, expand $f(y) - f(x)$:

$$f(y) - f(x) = (y^3 - x^3) + 4(y^2 - x^2) + 5(y - x)$$

Using the standard factorizations $y^3 - x^3 = (y - x)(y^2 + xy + x^2)$ and $y^2 - x^2 = (y - x)(y + x)$:

$$f(y) - f(x) = (y - x)(y^2 + xy + x^2) + 4(y - x)(y + x) + 5(y - x)$$

Substituting into $h(x, y)$:

$$h(x, y) = (y - x)(y^2 + xy + x^2 + 4y + 4x + 5) - (3x^2 + 8x + 5)(y - x)$$

Factoring out $(y - x)$:

$$h(x, y) = (y - x)\left(y^2 + xy + x^2 + 4y + 4x + 5 - (3x^2 + 8x + 5)\right)$$

Simplify: $y^2 + xy + x^2 + 4y + 4x + 5 - (3x^2 + 8x + 5) = y^2 + xy - 2x^2 + 4y - 4x$.

Therefore: $h(x, y) = (y - x)(y^2 + xy - 2x^2 + 4y - 4x)$.

$$\lim_{\lVert y - x \rVert \to 0} \frac{h(x, y)}{\lVert y - x \rVert} = \lim_{x \to y} \frac{(y - x)(y^2 + xy - 2x^2 + 4y - 4x)}{|y - x|}$$

$$= \begin{cases} \lim_{x \to y}(y^2 + xy - 2x^2 + 4y - 4x) & \text{if } x < y \\ \lim_{x \to y} -(y^2 + xy - 2x^2 + 4y - 4x) & \text{if } x > y \end{cases}$$

Because $y^2 + yy - 2y^2 + 4y - 4y = 0$, both limits above are zero. 

---

### Topic 5 — Definition of a Convex Function

#### Show $f(x) = x^2$ is convex

![](assets/Pasted%20image%2020260407164927.png)

---

####  Show $f(x) = (x - a)^2$ is convex

![](assets/Pasted%20image%2020260407164949.png)

---

#### Show $f(\mathbf{x}) = \lVert \mathbf{x} - \mathbf{a} \rVert^2$ is convex

![](assets/Pasted%20image%2020260407165023.png)
![](assets/Pasted%20image%2020260407165038.png)

---

#### Convex Combination (Jensen's Inequality for 4 Points)

Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex. Let $x_1, x_2, x_3, x_4 \in \mathbb{R}^n$ and $\lambda_1, \lambda_2, \lambda_3, \lambda_4 \geq 0$ with $\sum_{i=1}^{4} \lambda_i = 1$. We want to prove that $f\left(\sum_{i=1}^{4} \lambda_i x_i\right) \leq \sum_{i=1}^{4} \lambda_i f(x_i)$.

**Proof:** First, rewrite the convex combination by separating the first term:

$$\sum_{i=1}^{4} \lambda_i x_i = \lambda_1 x_1 + (1 - \lambda_1) \sum_{i=2}^{4} \frac{\lambda_i}{1 - \lambda_1} x_i$$

Since $f$ is convex:

$$f\left(\sum_{i=1}^{4} \lambda_i x_i\right) \leq \lambda_1 f(x_1) + (1 - \lambda_1) f\left(\sum_{i=2}^{4} \frac{\lambda_i}{1 - \lambda_1} x_i\right) \quad (1)$$

Next rewrite the remaining convex combination:

$$\sum_{i=2}^{4} \frac{\lambda_i}{1-\lambda_1} x_i = \frac{\lambda_2}{1-\lambda_1} x_2 + \left(1 - \frac{\lambda_2}{1-\lambda_1}\right) \sum_{i=3}^{4} \frac{\lambda_i}{1-\lambda_1-\lambda_2} x_i$$

Because $\lambda_2 \leq \lambda_2 + \lambda_3 + \lambda_4 = 1 - \lambda_1$, we have $\frac{\lambda_2}{1-\lambda_1} \in [0, 1]$. Applying convexity again:

$$f\left(\sum_{i=2}^{4} \frac{\lambda_i}{1-\lambda_1} x_i\right) \leq \frac{\lambda_2}{1-\lambda_1} f(x_2) + \left(1 - \frac{\lambda_2}{1-\lambda_1}\right) f\left(\sum_{i=3}^{4} \frac{\lambda_i}{1-\lambda_1-\lambda_2} x_i\right)$$

Substituting into inequality (1):

$$f\left(\sum_{i=1}^{4} \lambda_i x_i\right) \leq \lambda_1 f(x_1) + \lambda_2 f(x_2) + (1 - \lambda_1 - \lambda_2) f\left(\sum_{i=3}^{4} \frac{\lambda_i}{1 - \lambda_1 - \lambda_2} x_i\right) \quad (2)$$

Now expand the last convex combination:

$$\sum_{i=3}^{4} \frac{\lambda_i}{1-\lambda_1-\lambda_2} x_i = \frac{\lambda_3}{1-\lambda_1-\lambda_2} x_3 + \left(1 - \frac{\lambda_3}{1-\lambda_1-\lambda_2}\right) x_4$$

Because $\lambda_3 \leq \lambda_3 + \lambda_4 = 1 - \lambda_1 - \lambda_2$, we have $\frac{\lambda_3}{1-\lambda_1-\lambda_2} \in [0, 1]$. Applying convexity once again:

$$f\left(\sum_{i=3}^{4} \frac{\lambda_i}{1-\lambda_1-\lambda_2} x_i\right) \leq \frac{\lambda_3}{1-\lambda_1-\lambda_2} f(x_3) + \frac{\lambda_4}{1-\lambda_1-\lambda_2} f(x_4)$$

Substituting back into inequality (2):

$$f\left(\sum_{i=1}^{4} \lambda_i x_i\right) \leq \lambda_1 f(x_1) + \lambda_2 f(x_2) + \lambda_3 f(x_3) + \lambda_4 f(x_4) = \sum_{i=1}^{4} \lambda_i f(x_i)$$


---

#### Positive Combination of Convex Functions is Convex

Let $h(x) = \alpha f(x) + \beta g(x)$, where $\alpha, \beta \geq 0$, and suppose that both $f$ and $g$ are convex. We want to show that $h$ is also convex.

**Proof:** Take any $x, y$ and any $\lambda \in [0, 1]$.

$$h(\lambda x + (1-\lambda)y) = \alpha f(\lambda x + (1-\lambda)y) + \beta g(\lambda x + (1-\lambda)y)$$

Since $f$ is convex: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$.

Since $g$ is convex: $g(\lambda x + (1-\lambda)y) \leq \lambda g(x) + (1-\lambda)g(y)$.

Because $\alpha, \beta \geq 0$, multiplying preserves the inequality signs:

$$\alpha f(\lambda x + (1-\lambda)y) \leq \alpha\lambda f(x) + \alpha(1-\lambda)f(y)$$

$$\beta g(\lambda x + (1-\lambda)y) \leq \beta\lambda g(x) + \beta(1-\lambda)g(y)$$

Adding these two inequalities:

$$h(\lambda x + (1-\lambda)y) \leq \lambda(\alpha f(x) + \beta g(x)) + (1-\lambda)(\alpha f(y) + \beta g(y))$$

Recognizing that $\alpha f(x) + \beta g(x) = h(x)$ and $\alpha f(y) + \beta g(y) = h(y)$:

$$h(\lambda x + (1-\lambda)y) \leq \lambda h(x) + (1-\lambda)h(y)$$

This is exactly the definition of convexity. Hence, $h$ is convex. $\blacksquare$

---

### Topic 7/8 — GD Convergence (Strongly Convex Functions)

####  Strongly Convex GD Convergence Bound

![](assets/Pasted%20image%2020260407165214.png)
![](assets/Pasted%20image%2020260407165231.png)
---

### Appendix — Not in 9 Exam Topics

####  Logistic Regression Gradient

Define $x_0^j = 1$ for all $j$. Then for $\ell \in \lbrace 0, 1, 2, 3\rbrace$:

$$\frac{\partial L(w_0, w_1, w_2, w_3)}{\partial w_\ell} = \sum_{j=1}^{m} \mathbf{1}(y^j = 1) \, x_\ell^j - \sum_{j=1}^{m} \frac{e^{w_0 + w_1 x_1^j + w_2 x_2^j + w_3 x_3^j}}{1 + e^{w_0 + w_1 x_1^j + w_2 x_2^j + w_3 x_3^j}} \, x_\ell^j$$

In particular, when $\ell = 0$, since $x_0^j = 1$:

$$\frac{\partial L}{\partial w_0} = \sum_{j=1}^{m} \mathbf{1}(y^j = 1) - \sum_{j=1}^{m} \frac{e^{w_0 + w_1 x_1^j + w_2 x_2^j + w_3 x_3^j}}{1 + e^{w_0 + w_1 x_1^j + w_2 x_2^j + w_3 x_3^j}}$$
