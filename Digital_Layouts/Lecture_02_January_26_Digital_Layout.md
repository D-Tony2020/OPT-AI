# Lecture 2 — January 26, 2026 — 忠实转制笔记

---

## 第 ① 页：Logistic Regression（回顾）

- Consider a two-class classification problem with the data set
  {(xⱼ, yⱼ) : j = 1, ..., M}, where xⱼ ∈ ℝⁿ and yⱼ ∈ {+1, −1}

- Our prediction function is of the form

$$\phi(x_j, w) = \frac{1}{1 + e^{w^\top x_j}}$$

where ϕ(x, w) can be interpreted as the probability that a data point with feature vector xⱼ belongs to class +1.

- 1 − ϕ(xⱼ, w) = eʷᵀˣʲ / (1 + eʷᵀˣʲ) is then the probability that a data point with feature vector xⱼ belongs to class −1.

---

## 第 ② 页：对数似然推导

| data point # | feature vect | labels | likelihood |
|:---:|:---:|:---:|:---:|
| 1 | x₁ | +1 | 1 / (1 + eʷᵀˣ¹) |
| 2 | x₂ | −1 | eʷᵀˣ² / (1 + eʷᵀˣ²) |
| 3 | x₃ | −1 | eʷᵀˣ³ / (1 + eʷᵀˣ³) |
| ⋮ | ⋮ | ⋮ | ⋮ |
| M | xₘ | +1 | 1 / (1 + eʷᵀˣᴹ) |

- Likelihood of the data set:

$$L(w) = \frac{1}{1 + e^{w^\top x_1}} \cdot \frac{e^{w^\top x_2}}{1 + e^{w^\top x_2}} \cdot \frac{e^{w^\top x_3}}{1 + e^{w^\top x_3}} \cdots \frac{1}{1 + e^{w^\top x_M}}$$

- ln L(w) = −ln(1 + eʷᵀˣ¹) + wᵀx₂ − ln(1 + eʷᵀˣ²) + wᵀx₃ − ln(1 + eʷᵀˣ³) − ... − ln(1 + eʷᵀˣᴹ)

$$= -\sum_{j=1}^{M} \ln(1 + e^{w^\top x_j}) + \sum_{j=1}^{M} \mathbb{1}(y_j = -1) \cdot w^\top x_j$$

- 最大化对数似然：

$$\max_{w \in \mathbb{R}^n} \left\{ -\sum_{j=1}^{M} \ln(1 + e^{w^\top x_j}) + \sum_{j=1}^{M} \mathbb{1}(y_j = -1) \cdot w^\top x_j \right\}$$

---

## 第 ③ 页：Multi-Class Logistic Regression

- Consider a multi-class classification problem where the dataset is of the form
  {(xⱼ, yⱼ) : j = 1, ..., M} where xⱼ ∈ ℝⁿ and yⱼ ∈ {1, ..., K}

- Our prediction function has the form:

$$\phi_k(x_j, w_1, \ldots, w_K) = \frac{e^{w_k^\top x_j}}{\sum_{\ell=1}^{K} e^{w_\ell^\top x_j}}$$

where ϕₖ(xⱼ, w₁, ..., wₖ) is the probability that a data point with feature vector xⱼ belongs to class k.

- Let's use maximum likelihood estimation to estimate the parameters (w₁, w₂, ..., wₖ) ∈ ℝⁿˣᴷ

---

## 第 ④ 页：多类逻辑回归的似然与对数似然

| Data point | Feature vect. | Label | Likelihood |
|:---:|:---:|:---:|:---:|
| 1 | x₁ | 3 | e^(w₃ᵀx₁) / ∑ₗ₌₁ᴷ e^(wₗᵀx₁) |
| 2 | x₂ | 1 | e^(w₁ᵀx₂) / ∑ₗ₌₁ᴷ e^(wₗᵀx₂) |
| 3 | x₃ | 4 | e^(w₄ᵀx₃) / ∑ₗ₌₁ᴷ e^(wₗᵀx₃) |
| ⋮ | ⋮ | ⋮ | ⋮ |
| M | xₘ | 3 | e^(w₃ᵀxₘ) / ∑ₗ₌₁ᴷ e^(wₗᵀxₘ) |

$$L(w_1, w_2, \ldots, w_K) = \frac{e^{w_3^\top x_1}}{\sum_{\ell=1}^{K} e^{w_\ell^\top x_1}} \cdot \frac{e^{w_1^\top x_2}}{\sum_{\ell=1}^{K} e^{w_\ell^\top x_2}} \cdot \frac{e^{w_4^\top x_3}}{\sum_{\ell=1}^{K} e^{w_\ell^\top x_3}} \cdots \frac{e^{w_3^\top x_M}}{\sum_{\ell=1}^{K} e^{w_\ell^\top x_M}}$$

$$\ln L(w_1, w_2, \ldots, w_K) = w_3^\top x_1 - \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_1}\right) + w_1^\top x_2 - \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_2}\right) + w_4^\top x_3 - \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_3}\right) - \cdots + w_3^\top x_M - \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_M}\right)$$

$$= \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbb{1}(y_j = k) \, w_k^\top x_j - \sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_j}\right)$$

---

## 第 ⑤ 页：多类逻辑回归最终优化问题

$$\max_{(w_1, \ldots, w_K) \in \mathbb{R}^{n \times K}} \left\{ \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbb{1}(y_j = k) \, w_k^\top x_j - \sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_j}\right) \right\}$$

---

## 第 ⑥ 页：数值示例

Features for each data point (income, # of years in school)

| data point | feature vect | label | likelihood |
|:---:|:---:|:---:|:---:|
| 1 | (50K, 11) | A | e^(0.1·50 + 0.5·11) / (e^(0.1·50 + 0.5·11) + e^(0.4·50 + 0.1·11)) |
| 2 | (150K, 16) | B | e^(0.4·150 + 0.1·16) / (e^(0.1·150 + 0.5·16) + e^(0.4·150 + 0.1·16)) |

Parameter vectors:

(wₐ, wᵦ) = ((wₐ,income, wₐ,#years), (wᵦ,income, wᵦ,#years))
= ((0.1, 0.5), (0.4, 0.1))

---

## 第 ⑦ 页：Deep Learning

Consider a multi-class classification problem with datapoints
{(xⱼ, yⱼ) : j = 1, ..., M}, xⱼ ∈ ℝⁿ, yⱼ ∈ {1, 2, ..., K}

> **[图示说明]** 深度神经网络的层级结构图（流程图形式），从左到右依次为：
>
> **Layer 1:** 输入 xⱼ → 参数 (Q¹, g¹)，其中 Q¹ ∈ ℝ^(d₁×n), g¹ ∈ ℝ^(d₁) → 输出 aⱼ¹
>
> **Layer 2:** 输入 aⱼ¹ → 参数 (Q², g²)，其中 Q² ∈ ℝ^(d₂×d₁), g² ∈ ℝ^(d₂) → 输出 aⱼ²
>
> **...→ Layer L:** 输入 aⱼᴸ⁻¹ → 参数 (Qᴸ, gᴸ)，其中 Qᴸ ∈ ℝ^(dₗ×dₗ₋₁), gᴸ ∈ ℝ^(dₗ) → 输出 aⱼᴸ
>
> 图示目的：展示深度网络的前向传播过程及每层参数的维度关系。

各层计算公式：

$$a_j^1 = \sigma_1(Q^1 x_j + g^1)$$

其中维度标注：Q¹ 为 d₁×n，xⱼ 为 n×1，g¹ 为 d₁×1，输出 aⱼ¹ 为 d₁×1。

$$a_j^2 = \sigma_2(Q^2 a_j^1 + g^2)$$

其中维度标注：Q² 为 d₂×d₁，aⱼ¹ 为 d₁×1，g² 为 d₂×1，输出 aⱼ² 为 d₂×1。

Both component of σ₁ cause sigmoid σ(t) = 1/(1+eᵗ), rectified linear unit σ(t) = max{t, 0}

---

## 第 ⑧ 页：深度网络的整体表达式

Given the parameters (Q¹, g¹, Q², g², ..., Qᴸ, gᴸ) of the neural network, corresponding to the input xⱼ in the first layer, the output we get from the last layer is given by

$$\sigma_L\left(Q^L \, \sigma_{L-1}\left(\ldots \sigma_2\left(Q^2 \, \sigma_1(Q^1 x_j + g^1) + g^2\right) \ldots \right) + g^L\right)$$

> **[图示说明]** 嵌套函数结构标注：
> - 最内层花括号标注 "output from layer 1"
> - 次内层花括号标注 "output from second layer"
> - 最外层花括号标注 "output from last layer"
>
> 整个表达式等记为 z(xⱼ, Q¹, g¹, Q², g², ..., Qᴸ, gᴸ)

Solve a multi-class logistic regression problem by using z(xⱼ, Q¹, g¹, Q², g², ..., Qᴸ, gᴸ) as the feature for datapoint j and yⱼ as the label for data point j.
