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