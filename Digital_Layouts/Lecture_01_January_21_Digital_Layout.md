# Lecture 1 — January 21, 2026 — 忠实转制笔记

---

## 第 ① 页：What do we mean by solving an optimization problem?

- Let's design a box with largest possible volume such that
  - The base of the box is a square
  - The surface area of the box will be at most 500 cm²
  - The box will not have a top.

> **[图示说明]** 三维立体图：一个无顶的正方形底面长方体盒子。底面边长标记为 x，高度标记为 y。图示目的：直观展示决策变量 x（底面边长）和 y（高度）的几何含义。

Decision vars, objective func, constraints.

- Decision vars: x = length/width of base of the box
  - y = height of the box

- Obj. function: x²y

- Constraints: x² + 4xy ≤ 500
  - x ≥ 0, y ≥ 0

opt. problem:

$$\max \quad x^2 y$$
$$\text{s.t.} \quad x^2 + 4xy \leq 500$$
$$x \geq 0, \quad y \geq 0$$

According to Excel, the opt. solution is (x̂, ŷ) = (12.91, 6.45) with the opt volume 1075.82

---

## 第 ② 页：非线性优化问题的挑战

- This optimization problem is non linear because either the obj. function is nonlinear in the decision vars, or the left sides of constraints are non linear in the decision vars or both.

- It turns out opt. software may very easily fail when we solve an nonlinear optimization problem.

- E.g. Try to use Excel for the previous opt. problem with the initial guess x = 0, y = 0.

---

## 第 ③ 页：Role of Optimization in Data Analysis, ML and AI

- Typical data set in an ML/AI application looks like
  {(xⱼ, yⱼ) : j = 1, ..., M}, where xⱼ ∈ ℝⁿ is a vector of features and yⱼ ∈ ℝ is the label corresponding to data point j.

- The goal is to find a prediction function ϕ(·, w) : ℝⁿ → ℝ parameterized by w ∈ ℝᵖ such that ϕ(xⱼ, w) ≈ yⱼ on a large fraction of data points.

- Let ℓ(xⱼ, yⱼ, w) be the loss or "error" we incur when we use the prediction function parameterized by w to predict the label for the j-th data point.

- E.g. for regression, ℓ(xⱼ, yⱼ, w) = (ϕ(xⱼ, w) − yⱼ)²

- The opt. problem in ML/AI context looks like

$$\min_{w \in \mathbb{R}^p} \frac{1}{M} \sum_{j=1}^{M} \ell(x_j, y_j, w)$$

---

## 第 ④ 页：Least Squares Regression

- We predict the label for data point j as wᵀ·xⱼ, where w ∈ ℝⁿ is the vector of parameters we need to estimate.

- In this case the opt. we solve is of the form

$$\min_{w \in \mathbb{R}^n} \frac{1}{M} \sum_{j=1}^{M} (w^\top x_j - y_j)^2$$

- **Ridge regression:**

$$\min_{w \in \mathbb{R}^n} \frac{1}{M} \sum_{j=1}^{M} (w^\top x_j - y_j)^2 + \lambda \|w\|_2^2 \quad \downarrow \quad \sum_{i=1}^{n} w_i^2$$

useful for avoiding over fitting, ensuring robustness against outliers

- **Lasso:**

$$\min_{w \in \mathbb{R}^n} \frac{1}{M} \sum_{j=1}^{M} (w^\top x_j - y_j)^2 + \lambda \|w\|_1 \quad \downarrow \quad \sum_{i=1}^{n} |w_i|$$

useful for feature selection

---

## 第 ⑤ 页：Logistic Regression

- Consider a two-class classification problem with the dataset
  {(xⱼ, yⱼ) : j = 1, ..., M}, where xⱼ ∈ ℝⁿ and yⱼ ∈ {+1, −1}

- Our prediction function is of the form

$$\phi(x, w) = \frac{1}{1 + e^{w^\top x}}$$

where ϕ(x, w) can be interpreted as the probability that a data point with features x belongs to class +1.

- 1 − ϕ(x, w) = eʷᵀˣ / (1 + eʷᵀˣ) is the probability that a data point with feature vector x belongs to class −1.

---

## 第 ⑥ 页：最大似然估计拟合逻辑回归

To fit a logistic regression model, we use maximum likelihood estimation.

| data point # | feature vector | label | likelihood |
|:---:|:---:|:---:|:---:|
| 1 | x₁ | +1 | 1 / (1 + eʷᵀˣ¹) |
| 2 | x₂ | −1 | eʷᵀˣ² / (1 + eʷᵀˣ²) |
| 3 | x₃ | +1 | 1 / (1 + eʷᵀˣ³) |
| ⋮ | ⋮ | ⋮ | ⋮ |
| M | xₘ | −1 | eʷᵀˣᴹ / (1 + eʷᵀˣᴹ) |

Likelihood of the data set:

$$L(w) = \frac{1}{1 + e^{w^\top x_1}} \cdot \frac{e^{w^\top x_2}}{1 + e^{w^\top x_2}} \cdot \frac{1}{1 + e^{w^\top x_3}} \cdots \frac{e^{w^\top x_M}}{1 + e^{w^\top x_M}}$$
