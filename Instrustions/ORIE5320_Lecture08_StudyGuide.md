# ORIE5320 Lecture 08 — Taylor 定理的应用、凸函数的梯度刻画与 Jensen 不等式

> **课程**: ORIE5320 Optimization for AI · Cornell Tech
> **日期**: February 18, 2026
> **关键词**: Taylor's Theorem Application, Gradient Descent Descent Proof, Convex Functions, First-Order Characterization, Jensen's Inequality, Tangent Plane Lower Bound

---

## 0 · 本讲在课程中的位置

```
Lec 01 (Intro: 优化在ML中的角色)
  +-- Lec 02 (Convex Sets: 凸集的定义与性质)
        +-- Lec 03 (Convex Functions: 凸函数定义与判别)
              +-- Lec 04 (Gradient Descent Intro: 梯度下降的基本框架)
                    +-- Lec 05 (单维优化: 二分搜索与黄金分割法)
                          +-- Lec 06 (线搜索步长选择 + 约束优化：惩罚函数法)
                                +-- Lec 07 (障碍函数法 + Taylor 定理)
                                      +-- * Lec 08 (Taylor 应用 + 凸性梯度刻画 + Jensen 不等式) *
                                            +-- Lec 09 (SGD, Lipschitz 连续, Descent Lemma)
```

本讲是课程的**里程碑讲**，完成了三件大事。第一，用 Taylor 定理的推论严格证明了"沿负梯度方向走，函数值确实下降"——这是 Lec 04 以来一直在直觉层面使用但未证明的事实。第二，回顾凸函数的定义并证明了 Jensen 不等式——这个不等式在后续收敛速率分析中反复出现。第三，给出了凸函数的**一阶梯度刻画**——切线是全局下界——这是凸优化理论最深刻的结论之一，直接决定了梯度下降在凸函数上能找到全局最优。

---

## 1 · 全局叙事线

| 板块 | 页码 | 主题 | 核心结果 |
|------|------|------|----------|
| **I** | 1--2 | Taylor 定理复习与推论回顾 | $f(y) = f(x) + \nabla f(x)^T(y-x) + o(\|y-x\|)$ |
| **II** | 3--4 | Taylor 推论的应用：梯度下降降低函数值 | $f(\hat{x} - \alpha\nabla f(\hat{x})) \leq f(\hat{x}) - \frac{1}{2}\alpha\|\nabla f(\hat{x})\|^2$ |
| **III** | 5 | 凸函数定义复习 | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ |
| **IV** | 6 | Jensen 不等式 | $f(\sum \lambda^\ell x^\ell) \leq \sum \lambda^\ell f(x^\ell)$ |
| **V** | 7 | 凸函数的梯度刻画（一阶条件） | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ |

**核心叙事**：板块 I--II 完成了从 Lec 07 延续的"梯度下降为什么有效"的严格证明。板块 III--V 建立了凸函数的核心理论工具——Jensen 不等式和梯度刻画。这两大板块看似独立，实则紧密关联：Taylor 推论说"局部线性近似有小误差"，凸性梯度刻画说"局部线性近似是全局下界"。后者远比前者强大——它保证了梯度下降在凸函数上不仅"走的方向对"，而且"一定能走到全局最优"。

---

### 第 1 页：Taylor's Theorem（复习）

**原文翻译**：

- 我们用符号 $o(t)$ 表示任何函数 $g(t)$ 满足 $\lim_{t \to 0} g(t)/t = 0$。

- 回忆 $x \in \mathbb{R}^n$，$\|x\| = \sqrt{\sum_i x_i^2}$，$x^T x = (x_1, x_2, \ldots, x_n) \cdot (x_1, x_2, \ldots, x_n)^T = \sum x_i^2 = \|x\|^2$。

- **Taylor 定理**：设 $f: \mathbb{R}^n \to \mathbb{R}$。

$$f(y) = f(x) + \nabla^T f\left(\gamma x + (1 - \gamma) y\right) \cdot (y - x), \quad \text{对某个 } \gamma \in (0, 1)$$

[图示] 一维函数曲线（蓝色）。标出点 $x$（左）和 $y$（右），以及中间点 $\gamma x + (1-\gamma)y$。函数曲线在 $x$ 和 $y$ 之间经历起伏，但中间点处的切线与割线平行，体现中值定理的含义。

**精讲内容**：

这是对 Lec 07 内容的简洁复习。由于 Lec 07 和 Lec 08 之间可能隔了春假或其他日程，教授在这里做了一次完整的回顾。

**关键公式复习**：

1. **范数公式**：$\|x\| = \sqrt{\sum x_i^2}$，$\|x\|^2 = x^T x$。这个等式在后面证明梯度下降函数值下降时会被直接使用——当 $y - x = -\alpha \nabla f(x)$ 时，$(y-x)^T \nabla f(x) = -\alpha \|\nabla f(x)\|^2$。

2. **Taylor 定理**：$f(y) = f(x) + \nabla f(c)^T(y-x)$，$c = \gamma x + (1-\gamma)y$。精确等式，但 $\gamma$ 未知。

3. **图示的几何含义**：割线（连接 $(x, f(x))$ 和 $(y, f(y))$ 的直线）的斜率等于某个中间点的切线斜率。这和一维微积分的中值定理完全一样——只不过"斜率"变成了"方向导数"。

**为什么先复习 $\|x\|^2 = x^T x$？** 因为后面几页要大量使用"$\nabla f(x)^T \nabla f(x) = \|\nabla f(x)\|^2$"这个恒等式。教授提前把这个工具亮出来，避免推导中途停下来解释。

**教授口吻讲解**：

开学后的第一讲总是先复习上次的内容。Taylor 定理是我们后面所有分析的起点——如果这一点不清楚，后面的推导你会完全跟不上。

我再强调一次 $\|x\|^2 = x^T x$ 这个关系。看起来是废话对不对？但它在推导中出现的频率极高——你会看到 $\nabla f^T(y-x)$ 这种内积形式反复出现，而当 $y - x$ 和 $\nabla f$ 方向一致时，内积就变成了范数的平方。这个"内积 = 范数平方"的变换是后面所有计算的核心技巧。

---

### 第 2 页：Implication of Taylor's Theorem

**原文翻译**：

- 对任意 $x, y \in \mathbb{R}^n$，有

$$f(y) = f(x) + \nabla^T f(x) \cdot (y - x) + o(\|y - x\|) \qquad (\ddagger)$$

- 为什么 Taylor 定理蕴含 $(\ddagger)$？

由 Taylor 定理，对任意 $x, y$：

$$f(y) = f(x) + \nabla^T f(\gamma x + (1 - \gamma) y) \cdot (y - x), \quad \text{对某个 } \gamma \in (0, 1)$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \left(\nabla^T f(\gamma x + (1 - \gamma) y) - \nabla^T f(x)\right)(y - x)$$

$$= f(x) + \nabla^T f(x) \cdot (y - x) + \|\nabla f(\gamma x + (1 - \gamma) y) - \nabla f(x)\| \|y - x\| \cos \theta$$

其中 $\theta$ 是 $\nabla f(\gamma x + (1-\gamma)y) - \nabla f(x)$ 和 $y - x$ 的夹角。

如果我们能证明 $\|\nabla f(\gamma x + (1-\gamma)y) - \nabla f(x)\| \|y-x\| \cos\theta$ 是 $o(\|y-x\|)$，就完成了。

**精讲内容**：

这一页再次展示了 Lec 07 的推论及其证明——为第 3--4 页的关键应用做准备。由于这个推论是接下来证明的唯一数学工具，教授花了整整一页来回顾它的证明逻辑。

**推论的核心信息**：

$$\underbrace{f(y)}_{\text{精确函数值}} = \underbrace{f(x) + \nabla f(x)^T(y-x)}_{\text{一阶线性近似}} + \underbrace{o(\|y-x\|)}_{\text{可忽略的误差}}$$

这个公式的每一项都有清晰的几何含义：
- $f(x)$：当前点的函数值（已知）
- $\nabla f(x)^T(y-x)$：沿 $y-x$ 方向的一阶变化（线性项）
- $o(\|y-x\|)$：高阶修正（当 $y$ 接近 $x$ 时可忽略）

**证明的代数技巧回顾**：

核心操作是"加零减零"：$\nabla f(c) = \nabla f(x) + (\nabla f(c) - \nabla f(x))$。

这种技巧在优化证明中非常常见——当你有一个"不方便"的量（$\nabla f(c)$，$c$ 未知），就把它拆成"方便的量"（$\nabla f(x)$）加"差值"，然后证明差值是小量。

内积拆为 $\|a\| \|b\| \cos\theta$ 的用途：将一个内积分解为"大小"（$\|a\|, \|b\|$）和"方向"（$\cos\theta$）。$\cos\theta$ 有界，所以只需分析大小即可。当 $\|y-x\| \to 0$ 时，$c \to x$，梯度连续性给出 $\|\nabla f(c) - \nabla f(x)\| \to 0$。

**教授口吻讲解**：

我知道这和上一讲几乎一样。但我故意重复，因为下面两页才是这个推论的"杀手级应用"。如果你不完全理解 $(\ddagger)$ 这个公式，后面的推导就没法跟。

记住三个关键信息：(1) 等号是精确的（不是近似）；(2) $o(\|y-x\|)$ 意味着误差比步长还小；(3) 整个推论只需要 $f$ 连续可微。把这三条刻在脑子里，翻到下一页。

---

### 第 3 页：Taylor 推论的证明完成与应用

**原文翻译**：

验证：

$$\lim_{\|y - x\| \to 0} \frac{\|\nabla f(\gamma x + (1 - \gamma) y) - \nabla f(x)\| \|y - x\| \cos \theta}{\|y - x\|} = 0$$

因为 $\|y - x\| \to 0$ 等价于 $y - x \to 0$ 等价于 $\gamma x + (1-\gamma)y \to x$，所以梯度差 $\to 0$。$\square$

**为什么 Taylor 定理的推论重要？**

假设我们在点 $\hat{x}$ 处且 $\nabla f(\hat{x}) \neq 0$。我们一直在说，沿负梯度方向走一小步，函数值会变小。这个结论来自 Taylor 定理的推论。为什么？

$$f(\hat{x} - \alpha \nabla f(\hat{x})) = f(\hat{x}) + \nabla^T f(\hat{x}) \cdot (\hat{x} - \alpha \nabla f(\hat{x}) - \hat{x}) + o(\|\hat{x} - \alpha \nabla f(\hat{x}) - \hat{x}\|)$$

$$= f(\hat{x}) - \alpha \|\nabla f(\hat{x})\|^2 + o(\alpha \|\nabla f(\hat{x})\|)$$

$$= f(\hat{x}) + \alpha \|\nabla f(\hat{x})\| \left(-\|\nabla f(\hat{x})\| + \frac{o(\alpha \|\nabla f(\hat{x})\|)}{\alpha \|\nabla f(\hat{x})\|}\right) \qquad (\dagger)$$

注意：

$$\lim_{\alpha \to 0} \frac{o(\alpha \|\nabla f(\hat{x})\|)}{\alpha \|\nabla f(\hat{x})\|} = 0$$

这意味着我们可以选择...

**精讲内容**：

这一页是本讲的**高潮**——用 Taylor 推论证明梯度下降确实降低函数值。

**逐行推导解读**：

**第一步**：将 Taylor 推论 $(\ddagger)$ 中的 $x = \hat{x}$，$y = \hat{x} - \alpha \nabla f(\hat{x})$ 代入：

$$f(\hat{x} - \alpha \nabla f(\hat{x})) = f(\hat{x}) + \nabla f(\hat{x})^T \underbrace{(-\alpha \nabla f(\hat{x}))}_{y - x} + o(\underbrace{\alpha \|\nabla f(\hat{x})\|}_{\|y-x\|})$$

**第二步**：计算内积 $\nabla f(\hat{x})^T (-\alpha \nabla f(\hat{x}))$：

$$= -\alpha \nabla f(\hat{x})^T \nabla f(\hat{x}) = -\alpha \|\nabla f(\hat{x})\|^2$$

这里用到了 $x^T x = \|x\|^2$。注意**负号**——因为我们走的是负梯度方向，内积为负，函数值下降。

**第三步**：提取公因子 $\alpha \|\nabla f(\hat{x})\|$：

$$f(\hat{x} - \alpha \nabla f(\hat{x})) = f(\hat{x}) + \alpha \|\nabla f(\hat{x})\| \left(-\|\nabla f(\hat{x})\| + \frac{o(\alpha \|\nabla f(\hat{x})\|)}{\alpha \|\nabla f(\hat{x})\|}\right)$$

**关键观察**：当 $\alpha \to 0$ 时，$o(\cdot)/(\cdot) \to 0$（小 $o$ 的定义）。所以括号内的量趋向 $-\|\nabla f(\hat{x})\| + 0 = -\|\nabla f(\hat{x})\| < 0$（只要 $\nabla f(\hat{x}) \neq 0$）。

这意味着：对于足够小的 $\alpha$，括号内是负数，乘以正数 $\alpha \|\nabla f(\hat{x})\|$ 仍然是负数。因此：

$$f(\hat{x} - \alpha \nabla f(\hat{x})) < f(\hat{x})$$

**函数值确实下降了！**

**教授口吻讲解**：

这一页我要你一行一行地跟。第一步是"代入"——把 $y = \hat{x} - \alpha \nabla f(\hat{x})$ 代进 Taylor 推论。第二步是"化简"——内积变成范数平方。第三步是"提取"——把 $\alpha \|\nabla f\|$ 提出来。最后是"取极限"——当 $\alpha$ 很小时，$o$ 项可忽略，剩下的是一个负数。

整个证明的精髓在于那个 $-\alpha \|\nabla f(\hat{x})\|^2$ 项。这就是梯度下降每步的"保底下降量"——只要梯度不为零，函数值就会下降。梯度越大，下降越多。梯度为零时？下降量为零——你已经在驻点了（对于凸函数，驻点就是全局最优）。

---

### 第 4 页：梯度下降函数值下降的严格证明

**原文翻译**：

$\alpha > 0$ 足够小使得

$$\frac{o(\alpha \|\nabla f(\hat{x})\|)}{\alpha \|\nabla f(\hat{x})\|} \leq \frac{1}{2} \|\nabla f(\hat{x})\|$$

则由 $(\dagger)$ 得

$$f(\hat{x} - \alpha \nabla f(\hat{x})) \leq f(\hat{x}) + \alpha \|\nabla f(\hat{x})\| \left(-\|\nabla f(\hat{x})\| + \frac{1}{2}\|\nabla f(\hat{x})\|\right)$$

$$= f(\hat{x}) - \frac{1}{2} \alpha \|\nabla f(\hat{x})\|^2 \leq f(\hat{x})$$

因此 $f(\hat{x} - \alpha \nabla f(\hat{x})) \leq f(\hat{x})$。

**精讲内容**：

这一页完成了梯度下降函数值下降的**严格量化**证明。

**证明的最后一步**：

上一页我们知道 $o$ 项在 $\alpha \to 0$ 时趋于零。现在我们选择一个具体的"足够小"的 $\alpha$，使得 $o$ 项的比值不超过 $\frac{1}{2}\|\nabla f(\hat{x})\|$。

这个选择的精妙之处在于：
- 括号内变成 $-\|\nabla f\| + \frac{1}{2}\|\nabla f\| = -\frac{1}{2}\|\nabla f\|$
- 整体下降量变成 $-\frac{1}{2}\alpha \|\nabla f(\hat{x})\|^2$

**下降不等式**：

$$\boxed{f(\hat{x} - \alpha \nabla f(\hat{x})) \leq f(\hat{x}) - \frac{1}{2}\alpha \|\nabla f(\hat{x})\|^2}$$

这个不等式极其重要，它提供了**定量的下降保证**：
- 每步至少下降 $\frac{1}{2}\alpha \|\nabla f(\hat{x})\|^2$
- 下降量与梯度范数的平方成正比
- 下降量与步长 $\alpha$ 成正比（在步长足够小的前提下）

**为什么选 $\frac{1}{2}$？**

$\frac{1}{2}$ 不是唯一的选择——任何小于 1 的正常数都可以。选 $\frac{1}{2}$ 是为了得到一个简洁的表达式。实际上，如果我们选常数 $c \in (0, 1)$，下降量变为 $(1-c)\alpha \|\nabla f\|^2$。这个常数的选择在 Armijo 线搜索条件中会再次出现。

**与后续课程的衔接**：

这个证明有一个重要的局限：它只保证了"存在足够小的 $\alpha$ 使得函数值下降"，但没有告诉我们 $\alpha$ 应该多大。具体来说：
- $\alpha$ 太小：下降量 $\frac{1}{2}\alpha \|\nabla f\|^2$ 接近零，收敛极慢
- $\alpha$ 太大：$o$ 项不再可忽略，可能反而上升

**Lec 09+ 将引入 Lipschitz 连续梯度条件**来精确量化 $\alpha$ 的选择范围——Descent Lemma 给出 $\alpha \leq 1/L$ 的明确上界（$L$ 是梯度的 Lipschitz 常数）。这将 $o(\cdot)$ 的模糊表述替换为精确的二次上界。

**教授口吻讲解**：

这一页的结论只有一句话但价值千金：**梯度下降在小步长下保证函数值下降，每步至少下降 $\frac{1}{2}\alpha \|\nabla f\|^2$**。

为什么这个结论这么重要？因为从 Lec 04 开始，我们一直在"凭直觉"说"沿负梯度走函数值会下降"。现在我们第一次给出了严格的数学证明。而且证明非常优美——只用了 Taylor 推论和一个简单的不等式估计。

但不要以为这就完了。这个证明告诉你"能下降"，但没有告诉你"以什么速度收敛到最优"。要回答收敛速度的问题，我们需要两个额外工具：(1) Lipschitz 条件来确定最优步长，(2) 凸性条件来保证全局最优。第一个工具在 Lec 09 以后会讲，第二个工具——凸函数的梯度刻画——就在下面几页。

---

### 第 5 页：Convex Functions

**原文翻译**：

**定义**：函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是**凸的**，如果满足：

$$f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y) \quad \text{对所有 } x, y \in \mathbb{R}^n, \lambda \in [0, 1]$$

[图示] 一维凸函数曲线（蓝色）。标出点 $x$ 和 $y$，连接 $(x, f(x))$ 和 $(y, f(y))$ 的红色直线段（弦）。在凸组合点 $\lambda x + (1-\lambda)y$ 处：
- 函数值 $f(\lambda x + (1-\lambda)y)$（曲线上的点）低于
- 弦上的值 $\lambda f(x) + (1-\lambda)f(y)$（红色直线上的点）

**精讲内容**：

这是 Lec 03 内容的简洁回顾。凸函数定义的几何含义是"弦在曲线上方"——连接函数图像上任意两点的线段不低于函数曲线本身。

**凸性定义的三种等价理解**：

1. **几何理解**（弦在曲线上方）：连接任意两点的弦始终在曲线上方
2. **代数理解**（Jensen 型不等式）：凸组合的函数值不超过函数值的凸组合
3. **集合理解**（上图集是凸集）：epigraph $\{(x, t) : t \geq f(x)\}$ 是凸集

**为什么在这里复习凸性？**

上面几页我们证明了"梯度下降在小步长下函数值下降"。但这不够——它只保证每步下降，不保证收敛到全局最优。为什么？

- 对于一般（非凸）函数：梯度下降可能收敛到局部极小值或鞍点
- 对于凸函数：每个局部极小值都是全局极小值，而且梯度为零的点就是全局最优

凸性是从"每步下降"到"全局收敛"的桥梁。第 7 页的梯度刻画定理将精确阐述这个关系。

**凸函数的常见例子**（ML中）：

| 函数 | 应用 | 凸性 |
|------|------|------|
| $\|x\|^2$ | L2 正则化 | 严格凸 |
| $\|x\|_1$ | L1 正则化 | 凸（但不严格凸） |
| $\log(1 + e^{-x})$ | Logistic 损失 | 凸 |
| $-\log(x)$ | 对数障碍 | 严格凸（$x > 0$） |
| $\max_i x_i$ | 分段线性 | 凸 |
| $\ln(\sum e^{x_i})$ | Log-sum-exp | 凸 |

**教授口吻讲解**：

凸函数你在 Lec 03 就学过了，这里不过是快速回忆。但我要强调一个你可能没有注意到的要点：凸性定义中的 $\lambda \in [0, 1]$ 和 $\lambda x + (1-\lambda)y$ 的形式。这不是随便写的——$\lambda x + (1-\lambda)y$ 是 $x$ 和 $y$ 的凸组合，它正好落在 $x$ 和 $y$ 的连线上。凸性定义说的就是"函数在连线上的值不超过端点值的加权平均"。

你会问：为什么只看两个点的凸组合？如果有三个、四个、$k$ 个点呢？下一页的 Jensen 不等式就回答了这个问题——凸性定义可以推广到任意有限个点的凸组合。

---

### 第 6 页：Jensen's Inequality

**原文翻译**：

设 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数，$x^1, x^2, \ldots, x^k \in \mathbb{R}^n$，$\lambda^1, \lambda^2, \ldots, \lambda^k \in \mathbb{R}_+$ 且 $\sum \lambda^\ell = 1$。则

$$f\left(\sum_{\ell=1}^{k} \lambda^\ell x^\ell\right) \leq \sum_{\ell=1}^{k} \lambda^\ell f(x^\ell)$$

**证明**：

$$f\left(\sum_{\ell=1}^{k} \lambda^\ell x^\ell\right) = f\left(\lambda^1 x^1 + (1 - \lambda^1)\sum_{\ell=2}^{k} \frac{\lambda^\ell}{1 - \lambda^1} x^\ell\right)$$

$$\leq \lambda^1 f(x^1) + (1 - \lambda^1) f\left(\sum_{\ell=2}^{k} \frac{\lambda^\ell}{1 - \lambda^1} x^\ell\right) \quad \text{（$f$ 凸）}$$

$$= \lambda^1 f(x^1) + (1 - \lambda^1) f\left(\frac{\lambda^2}{1 - \lambda^1} x^2 + \left(1 - \frac{\lambda^2}{1 - \lambda^1}\right) \sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)$$

$$\leq \lambda^1 f(x^1) + \lambda^2 f(x^2) + (1 - \lambda^1 - \lambda^2) f\left(\sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)$$

继续类推...

**精讲内容**：

Jensen 不等式是凸函数定义从两点到多点的自然推广。

**定理内容**：凸函数在凸组合处的值不超过函数值的凸组合。

$$f\left(\sum_\ell \lambda^\ell x^\ell\right) \leq \sum_\ell \lambda^\ell f(x^\ell) \qquad (\lambda^\ell \geq 0, \sum \lambda^\ell = 1)$$

**证明思路**：数学归纳法。
- $k = 2$ 时就是凸函数的定义
- $k \to k+1$ 时，把第一个点"分离"出来，对剩余 $k$ 个点应用归纳假设

具体操作：将 $\sum_{\ell=1}^k \lambda^\ell x^\ell$ 写成

$$\lambda^1 x^1 + (1-\lambda^1) \cdot \underbrace{\sum_{\ell=2}^k \frac{\lambda^\ell}{1-\lambda^1} x^\ell}_{\text{剩余点的凸组合}}$$

注意 $\frac{\lambda^\ell}{1-\lambda^1}$ 的和等于 $\frac{1-\lambda^1}{1-\lambda^1} = 1$，所以它确实是一个凸组合。然后对这个两点凸组合应用凸性定义，再对剩余部分递归处理。

**Jensen 不等式的重要应用**：

1. **期望形式**：如果 $X$ 是随机变量，$f$ 是凸函数，则 $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$。这是有限 Jensen 不等式的连续推广——用积分替代求和。

2. **ML 中的应用**：
   - **KL 散度非负性**：$\text{KL}(p\|q) \geq 0$ 的证明用到 $-\log$ 是凸函数
   - **EM 算法**：E 步的 lower bound 来自 Jensen 不等式
   - **变分推断**：ELBO (Evidence Lower Bound) 的推导基于 Jensen 不等式
   - **损失函数分析**：证明某些损失函数是凸的

3. **优化中的应用**：Jensen 不等式在收敛速率分析中频繁出现——例如证明梯度下降的函数值收敛速率时，需要对凸组合施加 Jensen 不等式来获得上界。

**教授口吻讲解**：

Jensen 不等式的证明方法——"分离第一个点，对剩余递归"——是数学归纳法的经典套路。如果你理解了 $k=2$ 的情况（凸性定义），$k$ 个点的情况就是"剥洋葱"——一层一层剥开，每一层都用一次凸性定义。

Jensen 不等式在整个机器学习理论中的地位极其重要。KL 散度为什么非负？因为 $-\log$ 是凸函数，Jensen 不等式一用就出来了。EM 算法为什么有效？因为 Jensen 不等式提供了 log-likelihood 的下界。变分推断的 ELBO 是什么？Jensen 不等式的直接应用。你在后面的课程中会反复看到它的身影。

---

### 第 7 页：凸函数的梯度刻画

**原文翻译**：

（接上页 Jensen 不等式证明的收尾）

$$= \lambda^1 f(x^1) + \lambda^2 f(x^2) + (1 - \lambda^1 - \lambda^2) f\left(\sum_{\ell=3}^{k} \frac{\lambda^\ell}{1 - \lambda^1 - \lambda^2} x^\ell\right)$$

继续类推。

**凸函数的梯度刻画 (Gradient Based Characterization of Convexity)**

**定理**：函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸的当且仅当

$$f(y) \geq f(x) + \nabla^T f(x) \cdot (y - x)$$

[图示] 一维凸函数 $f$（蓝色曲线）。在点 $x$ 处画切线（红色直线）$f(x) + \nabla f(x)^T(y-x)$。

切线始终在函数曲线的**下方**——即 $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ 对所有 $y$ 成立。

切线在 $x$ 处与曲线相切（等号成立），但在其他所有点上严格位于曲线下方。

**精讲内容**：

这个定理是凸优化理论中**最深刻、最有用**的结论之一。

**定理内容**：$f$ 是凸函数 $\iff$ 对所有 $x, y$，切线是全局下界。

$$\boxed{f(y) \geq f(x) + \nabla f(x)^T(y-x) \quad \forall x, y}$$

**与 Taylor 推论的对比**：

| 性质 | Taylor 推论 | 凸函数梯度刻画 |
|------|-----------|--------------|
| 公式 | $f(y) = f(x) + \nabla f(x)^T(y-x) + o(\|y-x\|)$ | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ |
| 精确 or 不等式 | 等式（精确） | 不等式（下界） |
| 适用范围 | 任何可微函数 | 仅凸函数 |
| 余项 | $o(\|y-x\|)$（符号不定） | $\geq 0$（非负） |
| 有效范围 | 局部（$y$ 接近 $x$） | **全局**（任意 $y$） |
| 核心意义 | 局部线性近似 | 切线是全局下界 |

**关键区别**：Taylor 推论中的余项 $o(\|y-x\|)$ 可以是正的也可以是负的——它只告诉你"误差很小"，不告诉你方向。而凸函数的梯度刻画直接说"函数值 $\geq$ 线性近似"——余项不仅小，而且**非负**。这是一个极其强大的条件。

**为什么切线下界如此重要？**

1. **全局最优性条件**：如果 $\nabla f(x^*) = 0$，则对所有 $y$：
   $$f(y) \geq f(x^*) + 0 = f(x^*)$$
   即 $x^*$ 是全局最优解！**驻点 = 全局最优** 对凸函数成立。

2. **收敛性分析**：设 $x^*$ 是最优解，在梯度刻画中取 $x = x^k$（当前迭代点），$y = x^*$：
   $$f(x^*) \geq f(x^k) + \nabla f(x^k)^T(x^* - x^k)$$
   这给出 $f(x^k) - f(x^*) \leq \nabla f(x^k)^T(x^k - x^*)$——即次优性差距被梯度控制。这个不等式是后续收敛速率证明的起点。

3. **拟合下界**：每个点的切线都是函数的全局下界。所有切线的上确界（maximum of all tangent planes）就精确等于函数本身——这就是凸共轭（Fenchel conjugate）的几何含义。

**图示的核心信息**：

蓝色曲线（凸函数）始终在红色直线（切线）上方。在切点 $x$ 处两者相切（等号），在其他地方曲线严格高于切线。这意味着：如果你只知道某一点的函数值和梯度，你可以画一条切线，**保证**函数在任何地方都不低于这条线。这个"保证"对非凸函数不成立——非凸函数的切线可能在某些地方穿过函数曲线。

**教授口吻讲解**：

这个定理你必须记住——它是凸优化的"灵魂"。

让我把它和 Taylor 推论对比一下。Taylor 推论说：$f(y) \approx f(x) + \nabla f(x)^T(y-x)$，误差是 $o(\|y-x\|)$。这是一个**近似**——误差可正可负。

凸函数的梯度刻画说：$f(y) \geq f(x) + \nabla f(x)^T(y-x)$。这不是近似，是一个**下界**——而且是全局的，对任意远的 $y$ 都成立！

这个下界为什么这么有用？因为它意味着梯度为零的点就是全局最优。想想看：$\nabla f(x^*) = 0$ 代入定理，得到 $f(y) \geq f(x^*)$ 对所有 $y$ 成立。就这么简单！没有凸性条件的话，$\nabla f = 0$ 只能保证局部最优或鞍点——你永远不知道远处是不是有更低的点。凸性给你的承诺是：局部最优就是全局最优，切线下界就是全球保险。

这就是为什么机器学习中大家拼命设计凸损失函数（logistic loss、hinge loss、squared loss）——因为凸性保证你能用梯度下降找到全局最优，而非凸损失（比如神经网络的损失函数）只能祈祷你不被困在不好的局部极小值里。

---

## 跨讲交叉引用

| 本讲概念 | 相关前序/后续知识 | Lecture |
|---------|-----------------|---------|
| Taylor 推论 $f(y) = f(x) + \nabla f(x)^T(y-x) + o(\|y-x\|)$ | Taylor 定理的陈述 (Lec 07) | Lec 07 |
| 梯度下降保证下降 $f(x-\alpha\nabla f) \leq f(x) - \frac{1}{2}\alpha\|\nabla f\|^2$ | Descent Lemma (更精确的版本) | **Lec 09+** |
| $o(\|y-x\|)$ 余项 | Lipschitz 连续 $\to$ 二次上界替代 $o$ 记号 | **Lec 09+** |
| 凸函数定义 | 凸集、凸函数基础 | Lec 02--03 |
| Jensen 不等式 $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ | KL散度非负性, EM 算法, ELBO | ML 基础, **Lec 09+** |
| 凸函数梯度刻画 $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ | 梯度下降全局收敛性的证明 | **Lec 09+** |
| 驻点 = 全局最优（凸函数） | KKT 条件 | 优化理论进阶 |
| 切线是全局下界 | Fenchel 对偶, 凸共轭 | 优化理论进阶 |
| $\|\nabla f\|^2$ 下降量 | 收敛速率 $O(1/k)$ (凸), $O(\rho^k)$ (强凸) | **Lec 09+** |

---

## 核心收获

**收获一：Taylor 推论严格证明了梯度下降的有效性——"沿负梯度走函数值下降"不再是直觉**

通过将 $y = x - \alpha \nabla f(x)$ 代入 Taylor 推论，我们得到 $f(x - \alpha\nabla f(x)) \leq f(x) - \frac{1}{2}\alpha\|\nabla f(x)\|^2$。这是自 Lec 04 引入梯度下降以来的首次严格证明。关键洞察：下降量与 $\|\nabla f\|^2$ 成正比——梯度越大下降越快，梯度为零时已达驻点。但此证明尚未回答"步长该多大"和"收敛多快"，这些将在后续引入 Lipschitz 条件后解决。

**收获二：Jensen 不等式是凸性定义从两点到多点的推广——ML理论的基础工具**

Jensen 不等式 $f(\sum \lambda^\ell x^\ell) \leq \sum \lambda^\ell f(x^\ell)$ 将凸函数的"弦在上"性质推广到任意有限个（进而无穷个）点。其期望形式 $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ 在 KL 散度非负性证明、EM 算法的 E-step、变分推断的 ELBO 中无处不在。

**收获三：凸函数的梯度刻画是凸优化的"灵魂"——切线是全局下界保证了全局最优性**

$f(y) \geq f(x) + \nabla f(x)^T(y-x)$ 说明凸函数的切线（一阶近似）是全局下界。这比 Taylor 推论（局部近似）强得多——它保证了驻点就是全局最优点，使得梯度下降在凸函数上能收敛到全局最优。这个性质是凸优化相比非凸优化的根本优势，也是为什么 ML 社区偏好凸损失函数的数学原因。
