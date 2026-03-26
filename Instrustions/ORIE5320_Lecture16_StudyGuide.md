# ORIE5320 Lecture 16 — L1-Ball 投影：对称归约与稀疏优化

> **课程**: ORIE5320 Optimization for AI · Cornell Tech
> **日期**: March 23, 2026
> **关键词**: L1-Ball Projection, Symmetry Reduction, Sparsity, LASSO, Soft-thresholding, Probability Simplex Reduction, Sign Preservation

---

## 0 · 本讲在课程中的位置

```
Lec 01-04 (基础：优化建模, 凸集, 凸函数, GD 入门)
  +-- Lec 05-08 (GD 理论：收敛性, Lipschitz, 步长选择, 强凸性)
        +-- Lec 09-12 (收敛速率, SGD 基础)
              +-- Lec 13 (投影 GD 引入, 投影性质, 收敛分析预备)
                    +-- Lec 14 (收敛定理完成, Box 投影)
                          +-- Lec 15 (拉格朗日方法, 欧几里得球 / 概率单纯形投影)
                                +-- * Lec 16 (L1-Ball 投影) *
```

这是 ORIE 5320 的最后一讲。Lecture 13-15 依次建立了投影梯度下降的理论框架（收敛分析）和三种投影的计算方法（Box、欧几里得球、概率单纯形）。本讲处理课程中最复杂的投影——L1-ball 投影。L1-ball 对应 LASSO/L1 正则化这一 ML 中最重要的稀疏诱导技术。L1-ball 投影的推导巧妙地利用了两个对称性观察（Observation 1 和 2），将问题归结为上一讲已经解决的概率单纯形投影——展示了数学中"化归"思想的强大威力。作为课程的收官之讲，它将 Lecture 13-16 的投影理论串联为一条完整的因果链：理论（收敛定理）→ 简单投影（Box）→ 中等投影（球、单纯形）→ 复杂投影（L1-ball，归结为单纯形）。

---

## 1 · 全局叙事线

本讲 11 页 slides 分为四大板块：

| 板块 | 页码 | 主题 | 核心内容 |
|------|------|------|----------|
| **I** | 1 | L1-Ball 投影问题引入 | L1-ball 的定义与几何 |
| **II** | 2--7 | 两个关键 Observation 的陈述与证明 | 符号对称性、非负性保持 |
| **III** | 8--10 | 利用 Observation 化简并归结为单纯形投影 | 绝对值替换、不等式变等式、调用单纯形公式 |
| **IV** | 11 | L1-Ball 投影最终公式 | 综合所有结果的闭式表达 |

**核心叙事**：L1-ball 投影比前面所有投影都复杂——约束 $\sum|x_i| \leq R$ 涉及绝对值，不是光滑的，也不能直接用拉格朗日方法。解决策略是"化归"：通过两个巧妙的对称性观察，先消除绝对值（Observation 1：可以假设所有分量非负），再消除不等式约束（Observation 2：非负分量的投影保持非负），最终将 L1-ball 投影归结为概率单纯形投影——而后者在 Lecture 15 中已经解决了。

---

### 第 ① 页：Projections onto the L1-Ball

**原文翻译**：

考虑

$$\mathbb{X} = \left\{ x \in \mathbb{R}^n : \sum_i |x_i| \leq R \right\}$$

> **[图示]** 二维 L1-ball（菱形/钻石形），顶点分别在 $(R,0)$、$(0,R)$、$(-R,0)$、$(0,-R)$。四条边界线分别为 $x_1+x_2=R$、$-x_1+x_2=R$、$-x_1-x_2=R$、$x_1-x_2=R$。

我们要找 $\Pi_\mathbb{X}(\hat{y})$，其中 $\hat{y}$ 是给定点。

**精讲内容**：

这一页引入了本讲的主角——L1-ball，也是整门课程中最后一个也是最复杂的投影对象。

**L1-ball 的定义**。$\mathbb{X} = \{x \in \mathbb{R}^n : \|x\|_1 \leq R\}$，其中 $\|x\|_1 = \sum_i |x_i|$ 是 L1 范数。在二维中，L1-ball 的形状是菱形（或正方形旋转 45 度）；在高维中，它是交叉多面体（cross-polytope）。

**L1-ball vs 欧几里得球的几何对比**。欧几里得球（L2-ball）是"圆的"——所有方向等价。L1-ball 是"尖的"——它在坐标轴方向上有尖角。这种"尖角"结构正是 L1 范数诱导稀疏性的几何原因：在优化中，等高线与 L1-ball 的交点更容易落在坐标轴上（即某些坐标为零），产生稀疏解。

**L1-ball 在 ML 中的核心地位**：
- **LASSO**（Least Absolute Shrinkage and Selection Operator）：$\min_{\|w\|_1 \leq R} \|Xw - y\|^2$——这是统计学和 ML 中最重要的稀疏估计方法。
- **压缩感知**（Compressed Sensing）：利用 L1 约束从少量测量中恢复稀疏信号。
- **特征选择**：L1 正则化自动将不重要的特征权重设为 0。
- **稀疏编码**：信号处理中用 L1 约束学习稀疏表示。

**为什么 L1-ball 投影更难**？对比之前的投影：
| 约束集 | 约束表达式 | 关键难点 |
|--------|----------|---------|
| Box | $0 \leq x_i \leq U_i$ | 无难点，完全可分离 |
| L2-ball | $\sum x_i^2 \leq R^2$ | 耦合约束，但光滑 → 拉格朗日直接求解 |
| 单纯形 | $\sum x_i = 1, x_i \geq 0$ | 耦合等式 + 非负不等式 → 拉格朗日 + 截断 |
| L1-ball | $\sum\|x_i\| \leq R$ | **绝对值不光滑** + 耦合不等式 |

绝对值 $|x_i|$ 在 $x_i = 0$ 处不可微——这使得不能直接对 $\sum|x_i| = R$ 用拉格朗日方法（因为它不是光滑等式约束）。解决方案是通过对称性论证消除绝对值。

**教授口吻讲解**：

最后一讲了，我们来挑战课程中最难的投影——L1-ball。为什么难？因为 $\sum|x_i|$ 里有绝对值，不光滑。你不能像欧几里得球那样直接求导。但别担心，我们不会硬碰硬。接下来的策略是"化归"——通过两个巧妙的观察，把 L1-ball 投影变成你们上节课已经学会的单纯形投影。这就是数学的艺术：不解决难题，而是把难题变成已解决的简单题。

---

### 第 ② 页：问题建模与 Observation 1

**原文翻译**：

我们要求解：

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

假设 $\hat{y} \notin \mathbb{X}$。

**Observation 1**：我们可以假设 $\hat{y}_i \geq 0$，$i = 1, \ldots, n$。

> **[图示]** L1-ball 中，$\hat{y}$ 位于第二象限（$x_1$ 为负），$\hat{y}'$ 是 $\hat{y}$ 关于 $y$ 轴的镜像反射，位于第一象限。投影 $\hat{y}$ 等价于投影 $\hat{y}'$（分量取绝对值后投影），再将结果的 $x_1$ 分量取负。

**精讲内容**：

这一页陈述了 L1-ball 投影的第一个关键简化——利用 L1-ball 的坐标轴对称性。

**问题设置**。给定 $\hat{y} \notin \mathbb{X}$（即 $\|\hat{y}\|_1 > R$），求解：
$$\min_{x} \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

当 $\hat{y} \in \mathbb{X}$ 时投影是 $\hat{y}$ 本身，所以只需考虑 $\hat{y} \notin \mathbb{X}$ 的情况。

**Observation 1 的含义**：无论 $\hat{y}$ 的各坐标是正是负，我们都可以先把所有坐标取绝对值，求出绝对值版本的投影，然后再恢复原来的符号。

**直觉**：L1-ball 关于每个坐标轴都是对称的——$|x_i|$ 在 $x_i$ 和 $-x_i$ 处取相同的值。所以如果 $\hat{y}_j < 0$，把 $\hat{y}_j$ "翻转"为 $|\hat{y}_j|$ 不会改变问题的本质（因为约束 $\sum|x_i| \leq R$ 关于翻转是不变的），只需在最后把对应的投影坐标也翻转回去。

**图示的含义**：在二维中，L1-ball 是菱形。如果 $\hat{y}$ 在第二象限，将 $\hat{y}_1$ 翻转为 $|\hat{y}_1|$，得到第一象限的 $\hat{y}'$。在第一象限投影 $\hat{y}'$ 到 L1-ball 上，得到投影点 $\pi'$。然后把 $\pi'$ 的第一个坐标翻转回负数，得到原问题的投影 $\pi$。

这个 Observation 使我们可以不失一般性地假设 $\hat{y}_i \geq 0$——大大简化了问题，因为它消除了绝对值中的符号复杂性。

**数学上的严格含义**：我们将在第 ④-⑤ 页严格证明：将 $\hat{y}$ 的第 $j$ 个分量从 $\hat{y}_j$ 变为 $-\hat{y}_j$，两个投影问题的最优目标值相同——因此可以互相转化。

**教授口吻讲解**：

Observation 1 的核心思想：L1-ball 在每个坐标轴上是对称的——翻转 $x_j \to -x_j$ 后 $\sum|x_i|$ 不变。所以如果 $\hat{y}_j$ 是负的，你先把它翻正，做完投影再翻回去。这让我们可以假设所有 $\hat{y}_i \geq 0$——去掉绝对值的符号复杂性。这是第一步简化。

---

### 第 ③ 页：Observation 2

**原文翻译**：

**Observation 2**：给定 $\hat{y}$ 满足 $\hat{y}_i \geq 0$，$i = 1, \ldots, n$，$\hat{y}$ 到 L1-ball 的投影的所有分量都是非负的。

> **[图示]** L1-ball 中，$\hat{y}$ 位于第一象限，$\Pi_\mathbb{X}(\hat{y})$ 位于 L1-ball 边界上的第一象限部分。

让我们用数学方法严格证明这两个 Observation 确实是正确的。

**精讲内容**：

这一页陈述了第二个关键简化——如果被投影的点是非负的，那么投影结果也是非负的。

**Observation 2 的含义**：结合 Observation 1（可以假设 $\hat{y}_i \geq 0$），Observation 2 告诉我们投影结果 $\pi_i \geq 0$。这意味着在投影问题中，约束 $\sum|x_i| \leq R$ 可以简化为 $\sum x_i \leq R$（因为 $x_i \geq 0$ 时 $|x_i| = x_i$）。

这是至关重要的简化！它将带绝对值的 L1 约束转化为不带绝对值的线性约束——加上非负约束，这正是概率单纯形（缩放版本）的约束结构。

**直觉**：如果 $\hat{y}$ 在第一象限（所有坐标非负），L1-ball 上离 $\hat{y}$ 最近的点当然也在第一象限——因为绕到其他象限只会增加距离。

**两步归约的逻辑链**：
1. Observation 1：一般 $\hat{y}$ → 非负 $|\hat{y}|$（通过坐标翻转）。
2. Observation 2：非负 $\hat{y}$ → 非负投影（$\sum|x_i| = \sum x_i$）。
3. 问题变为：$\min \sum(x_i - |\hat{y}_i|)^2$ s.t. $\sum x_i \leq R, x_i \geq 0$。
4. 当 $\hat{y} \notin \mathbb{X}$ 时不等式在最优解处取等号：$\sum x_i = R$。
5. 这就是一个**缩放概率单纯形投影**（$R$ 替代 1）！

这个归约是本讲最精妙的地方——通过两个简单的对称性观察，把一个看似棘手的非光滑投影问题完全化归为已经解决的问题。

**教授口吻讲解**：

Observation 2 告诉我们：非负的 $\hat{y}$ 投影后还是非负的。这跟 Observation 1 配合起来，威力巨大——绝对值可以去掉了！因为当所有 $x_i \geq 0$ 时，$|x_i| = x_i$。这样 L1-ball 约束 $\sum|x_i| \leq R$ 就变成了 $\sum x_i \leq R$，加上 $x_i \geq 0$——这不就是上节课的单纯形吗（除了总和是 $R$ 不是 1）？看到了吧，困难的问题不需要困难的解法——你需要的是正确的观察。

---

### 第 ④ 页：Observation 1 的证明

**原文翻译**：

**Observation 1 的证明**：投影 $\hat{y}$ 到 L1-Ball 上时，可以假设 $\hat{y}_i \geq 0$，$\forall i = 1, \ldots, n$。

**为什么？** 假设 $\hat{y}_j < 0$。考虑问题：

$$\min \sum_{i \neq j} (x_i - \hat{y}_i)^2 + (x_j - \hat{y}_j)^2 \quad (*) \quad \text{s.t.} \quad \sum_{i \neq j} |x_i| + |x_j| \leq R$$

再考虑：

$$\min \sum_{i \neq j} (x_i - \hat{y}_i)^2 + (x_j - (-\hat{y}_j))^2 \quad (\ddagger) \quad \text{s.t.} \quad \sum_{i \neq j} |x_i| + |x_j| \leq R$$

让我们论证两个问题有相同的最优目标值。

设 $x^*$ 是 $(*)$ 的最优解。定义 $\tilde{x}$：$\tilde{x}_i = x_i^*$ 对 $i \neq j$，$\tilde{x}_j = -x_j^*$。

因为 $x^*$ 对 $(*)$ 是最优的，$\sum_i |x_i^*| \leq R$，但 $|x_i^*| = |\tilde{x}_i|$ 对所有 $i$。因此 $\tilde{x}$ 对 $(\ddagger)$ 是可行的。

**精讲内容**：

这一页开始严格证明 Observation 1——通过构造可行解来证明两个问题等价。

**证明策略**：要证明"投影 $\hat{y}$"和"投影 $\hat{y}'$（将 $\hat{y}_j$ 翻转为 $-\hat{y}_j$）"给出相同的最优目标值。方法是：从 $(*)$ 的最优解 $x^*$ 出发，构造 $(\ddagger)$ 的一个可行解 $\tilde{x}$，使其目标值等于 $x^*$ 在 $(*)$ 中的目标值。

**构造的可行解**：$\tilde{x}_j = -x_j^*$，其他坐标不变。

**可行性验证**：$\sum_i |\tilde{x}_i| = \sum_{i \neq j} |x_i^*| + |-x_j^*| = \sum_{i \neq j} |x_i^*| + |x_j^*| = \sum_i |x_i^*| \leq R$。关键是 $|-x_j^*| = |x_j^*|$——绝对值对翻转不变。

**目标值比较**（在下一页完成）：$\tilde{x}$ 在 $(\ddagger)$ 中的目标值等于 $x^*$ 在 $(*)$ 中的目标值。这需要验证 $(x_j - (-\hat{y}_j))^2|_{x_j = \tilde{x}_j} = (x_j - \hat{y}_j)^2|_{x_j = x_j^*}$。

$$(\tilde{x}_j - (-\hat{y}_j))^2 = (-x_j^* + \hat{y}_j)^2 = (x_j^* - \hat{y}_j)^2$$

所以 $\tilde{x}$ 在 $(\ddagger)$ 中的目标值 $=$ $x^*$ 在 $(*)$ 中的最优目标值。由于 $\tilde{x}$ 只是 $(\ddagger)$ 的一个可行解（不一定是最优解），$(\ddagger)$ 的最优目标值 $\leq$ $(*)$ 的最优目标值。

**这个证明方法的普遍性**。"构造可行解 + 比较目标值"是优化中证明两个问题等价（或一个不弱于另一个）的标准手法。在 LP 对偶理论中（弱对偶定理的证明）、在半定规划中都广泛使用。

**教授口吻讲解**：

证明的思路很标准：给了你 $(*)$ 的最优解 $x^*$，我来构造 $(\ddagger)$ 的一个可行解——就是把第 $j$ 个坐标翻转。可行性不变（因为绝对值对翻转不变），目标值也不变（因为 $(-x_j^* + \hat{y}_j)^2 = (x_j^* - \hat{y}_j)^2$）。所以 $(\ddagger)$ 的最优值 $\leq$ $(*)$ 的最优值。反过来同理。两边一夹——相等。

---

### 第 ⑤ 页：Observation 1 证明（续）

**原文翻译**：

$\tilde{x}$ 对问题 $(\ddagger)$ 提供的目标值为：

$$\sum_{i \neq j} (\tilde{x}_i - \hat{y}_i)^2 + (\tilde{x}_j - (-\hat{y}_j))^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (-x_j^* + \hat{y}_j)^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (x_j^* - \hat{y}_j)^2$$

这就是问题 $(*)$ 的最优目标值。

因此，给定 $(*)$ 的最优解，我们可以构造 $(\ddagger)$ 的一个可行解，其目标值等于 $(*)$ 的最优目标值。这意味着 $(\ddagger)$ 的最优目标值 $\leq$ $(*)$ 的最优目标值。

我们可以精确地按相反的论证来证明：给定 $(\ddagger)$ 的最优解，可以构造 $(*)$ 的一个可行解，其目标值等于 $(\ddagger)$ 的最优目标值。这意味着 $(*)$ 的最优目标值 $\leq$ $(\ddagger)$ 的最优目标值。

因此，两个问题有相同的最优目标值。证毕！$\blacksquare$

**精讲内容**：

这一页完成了 Observation 1 的证明，通过对称的双向论证建立了两个问题的等价性。

**目标值的计算**。$\tilde{x}$ 在 $(\ddagger)$ 中的目标值：

$$\sum_{i \neq j}(\tilde{x}_i - \hat{y}_i)^2 + (\tilde{x}_j + \hat{y}_j)^2 = \sum_{i \neq j}(x_i^* - \hat{y}_i)^2 + (-x_j^* + \hat{y}_j)^2$$

关键的代数等式：$(-x_j^* + \hat{y}_j)^2 = (\hat{y}_j - x_j^*)^2 = (x_j^* - \hat{y}_j)^2$。这是平方的"翻转不变性"——$(a-b)^2 = (b-a)^2$。

因此 $\tilde{x}$ 在 $(\ddagger)$ 中的目标值 $= x^*$ 在 $(*)$ 中的目标值 = $(*)$ 的最优目标值。

**双向论证**。
- 方向 1：$(*)$ 的最优解 → $(\ddagger)$ 的可行解，目标值相等 → $\text{OPT}(\ddagger) \leq \text{OPT}(*)$。
- 方向 2：$(\ddagger)$ 的最优解 → $(*)$ 的可行解，目标值相等 → $\text{OPT}(*) \leq \text{OPT}(\ddagger)$。
- 合并：$\text{OPT}(*) = \text{OPT}(\ddagger)$。

**从等价到化归**。Observation 1 的实用意义是：要投影 $\hat{y}$（可能有负分量），可以先把所有分量取绝对值得到 $|\hat{y}|$（所有分量非负），投影 $|\hat{y}|$ 到 L1-ball，得到投影 $\pi'$，然后恢复原来的符号：$\pi_i = \text{sign}(\hat{y}_i) \cdot \pi'_i$。

但等等——这个推论需要 Observation 2 的配合。Observation 1 只证明了两个问题的**最优目标值**相同，但投影是**最优解**——我们还需要知道 $\pi'$ 的各分量是非负的（否则恢复符号时可能出错）。这正是 Observation 2 的作用。

**教授口吻讲解**：

双向论证——从 $(*)$ 到 $(\ddagger)$ 和从 $(\ddagger)$ 到 $(*)$——得到两个问题的最优值相等。翻转操作不改变可行性（$|x|$ 不变），也不改变目标值（$(a-b)^2 = (b-a)^2$）。简洁有力。但要注意，这只证明了目标值相等，要完全做到"取绝对值后投影再恢复符号"，还需要 Observation 2 来保证投影结果是非负的。接下来就证明它。

---

### 第 ⑥ 页：Observation 2 的证明

**原文翻译**：

**Observation 2 的证明**：设 $\hat{y}$ 满足 $\hat{y}_i \geq 0$，$i = 1, \ldots, n$。则 $\hat{y}$ 到 L1-ball 的投影的所有分量都非负。

**为什么？** 为了推出矛盾，设 $\hat{y}$ 满足 $\hat{y}_i \geq 0$，$\forall i = 1, \ldots, n$，但 $\hat{y}$ 到 L1-ball 投影的第 $j$ 个分量为负。

$$\min \sum_{i \neq j} (x_i - \hat{y}_i)^2 + (x_j - \hat{y}_j)^2 \quad (**) \quad \text{s.t.} \quad \sum_{i \neq j} |x_i| + |x_j| \leq R$$

设 $x^*$ 是 $(**)$ 的最优解，假设 $x_j^* < 0$。

定义 $\tilde{x}$：$\tilde{x}_i = x_i^*$ 对 $i \neq j$，$\tilde{x}_j = 0$。

因为 $x^*$ 对 $(**)$ 可行，$\sum_{i \neq j} |x_i^*| + |x_j^*| \leq R$，但 $\sum_{i \neq j} |\tilde{x}_i| + |\tilde{x}_j| = \sum_{i \neq j} |x_i^*| + 0 \leq R$。

所以 $\tilde{x}$ 对 $(**)$ 可行。

**精讲内容**：

这一页开始了 Observation 2 的反证法证明——假设投影有负分量，构造严格更优的可行解以推出矛盾。

**反证法的设置**：假设 $\hat{y}_i \geq 0$ 对所有 $i$，但最优解 $x^*$ 的第 $j$ 个分量 $x_j^* < 0$。我们要导出矛盾——即 $x^*$ 不可能是最优解。

**构造竞争解**：$\tilde{x}$ 与 $x^*$ 的唯一区别是 $\tilde{x}_j = 0$（而非 $x_j^* < 0$）。

**可行性**：$\sum |\tilde{x}_i| = \sum_{i \neq j} |x_i^*| + 0 \leq \sum_{i \neq j} |x_i^*| + |x_j^*| \leq R$。即 $\tilde{x}$ 不仅可行，而且"更宽裕"——它用了更少的 L1"预算"。

**下一页将证明**：$\tilde{x}$ 的目标值严格小于 $x^*$ 的目标值——这与 $x^*$ 是最优解矛盾。

**直觉**：$\hat{y}_j \geq 0$ 但 $x_j^* < 0$，意味着 $x_j^*$ "跑到了 $\hat{y}_j$ 的对面"。将 $x_j$ 从负值拉回到 0，缩短了 $|x_j - \hat{y}_j|$ 的距离（因为 0 比任何负数都离非负的 $\hat{y}_j$ 更近），同时节省了 L1 预算——双重利好。

**教授口吻讲解**：

反证法：假设投影有个分量 $x_j^* < 0$，但 $\hat{y}_j \geq 0$。我构造一个"竞争者"$\tilde{x}$——把 $x_j$ 设成 0，其他不变。这个新的 $\tilde{x}$ 显然可行（L1 范数只会更小）。下一页我要证明它的目标值严格更好——这就矛盾了。直觉很简单：$\hat{y}_j \geq 0$，你跑到负数区域去干什么？0 比任何负数都离 $\hat{y}_j$ 更近。

---

### 第 ⑦ 页：Observation 2 证明（续）

**原文翻译**：

$\tilde{x}$ 对问题 $(**)$ 提供的目标值为：

$$\sum_{i \neq j} (\tilde{x}_i - \hat{y}_i)^2 + (\tilde{x}_j - \hat{y}_j)^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (\hat{y}_j)^2 < \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (\hat{y}_j - x_j^*)^2 = \sum_{i \neq j} (x_i^* - \hat{y}_i)^2 + (x_j^* - \hat{y}_j)^2$$

因此，$\tilde{x}$ 对问题 $(**)$ 给出了比 $x^*$ 严格更好的目标值。所以 $x^*$ 不可能是 $(**)$ 的最优解。矛盾！证毕！$\blacksquare$

**精讲内容**：

这一页完成了 Observation 2 的证明——通过目标值的严格比较推出矛盾。

**严格不等式的关键**。核心是比较 $(\tilde{x}_j - \hat{y}_j)^2 = \hat{y}_j^2$ 和 $(x_j^* - \hat{y}_j)^2$：

由 $\hat{y}_j \geq 0$ 和 $x_j^* < 0$：
$$|0 - \hat{y}_j| = \hat{y}_j < \hat{y}_j + |x_j^*| = \hat{y}_j - x_j^* = |x_j^* - \hat{y}_j|$$

所以 $\hat{y}_j^2 < (x_j^* - \hat{y}_j)^2$——严格不等式。这里的关键是 $\hat{y}_j \geq 0$ 和 $x_j^* < 0$ 保证了 $0$ 比 $x_j^*$ 离 $\hat{y}_j$ 严格更近。

如果 $\hat{y}_j = 0$ 且 $x_j^* < 0$，仍然有 $0^2 = 0 < x_j^{*2} = (x_j^* - 0)^2$。

**矛盾的建立**。$\tilde{x}$ 是 $(**)$ 的可行解，且目标值严格小于"最优解" $x^*$ 的目标值——这与 $x^*$ 的最优性矛盾。因此假设"$x_j^* < 0$"不成立，即投影的所有分量 $\geq 0$。

**Observation 2 与"稀疏性"的联系**。Observation 2 不仅证明了投影是非负的，它的证明方法还揭示了一个更深的结构：如果某个分量的投影是 0，那就是 0——不会是负数。这种"0 或正"的二分结构正是 L1 范数诱导稀疏性的微观机制——不需要的分量被精确地设为 0，而不是被设为某个小的非零值。

**证明方法的通用性**。这种"构造严格更优的可行解 → 矛盾"的反证法在组合优化和凸优化中非常常见。例如，证明 LP 的最优解在多面体的顶点取到，也用类似的论证——如果不在顶点，你可以沿某个方向移动而保持可行，但改善目标值。

**教授口吻讲解**：

关键的严格不等式：$\hat{y}_j^2 < (x_j^* - \hat{y}_j)^2$。为什么？因为 $\hat{y}_j \geq 0$ 而 $x_j^* < 0$，所以 $0$ 比 $x_j^*$ 离 $\hat{y}_j$ 更近——这是一维数轴上的显而易见的事实。0 在 $\hat{y}_j$ 和 $x_j^*$ 之间（或跟 $\hat{y}_j$ 在同一侧），而 $x_j^*$ 在 $\hat{y}_j$ 的另一侧。好，两个 Observation 都证完了。现在该享受成果了——看看它们如何把 L1-ball 投影变成单纯形投影。

---

### 第 ⑧ 页：利用两个 Observation 化简问题

**原文翻译**：

我们要求解：

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

由 Observation 1，可以等价地求解：

$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

由 Observation 2，可以进一步等价地求解：

$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i x_i \leq R, \quad x_i \geq 0, \; \forall i = 1, \ldots, n$$

**精讲内容**：

这一页是两个 Observation 的"收割"——将它们依次应用于 L1-ball 投影问题，逐步简化。

**第一步简化（Observation 1）**：将 $\hat{y}_i$ 替换为 $|\hat{y}_i|$。原问题是 $\min \sum(x_i - \hat{y}_i)^2$ s.t. $\sum|x_i| \leq R$。由 Observation 1，这等价于 $\min \sum(x_i - |\hat{y}_i|)^2$ s.t. $\sum|x_i| \leq R$。

等价性的精确含义是：两个问题的最优目标值相同，且从一个问题的最优解可以通过翻转符号得到另一个问题的最优解。

**第二步简化（Observation 2）**：消除绝对值约束。现在 $|\hat{y}_i| \geq 0$，由 Observation 2，投影结果 $x_i^* \geq 0$。因此 $|x_i^*| = x_i^*$，约束 $\sum|x_i| \leq R$ 可以替换为 $\sum x_i \leq R$ 加上 $x_i \geq 0$。

最终化简后的问题：
$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i x_i \leq R, \quad x_i \geq 0$$

**与之前投影的对比**：
- Box 投影：$\min \sum(x_i - \hat{y}_i)^2$ s.t. $0 \leq x_i \leq U_i$（完全可分离）。
- 单纯形投影：$\min \sum(x_i - \hat{y}_i)^2$ s.t. $\sum x_i = 1, x_i \geq 0$（等式 + 非负）。
- 化简后的 L1-ball：$\min \sum(x_i - |\hat{y}_i|)^2$ s.t. $\sum x_i \leq R, x_i \geq 0$（不等式 + 非负）。

化简后的 L1-ball 问题与单纯形投影非常相似！唯一的区别是：
1. 不等式 $\sum x_i \leq R$ vs 等式 $\sum x_i = 1$。
2. 常数 $R$ vs $1$。

下一页将处理区别 1（不等式变等式），然后直接调用 Lecture 15 的单纯形投影公式。

**教授口吻讲解**：

看到化归的威力了吗？两个 Observation 像两把钥匙，一把消除了符号问题（$\hat{y}_i \to |\hat{y}_i|$），一把消除了绝对值约束（$\sum|x_i| \to \sum x_i$）。现在问题变成了 $\sum x_i \leq R, x_i \geq 0$ 下的二次最小化——跟单纯形投影就差一个"不等式 vs 等式"的区别。下一页解决这最后一个差异。

---

### 第 ⑨ 页：归结为概率单纯形投影

**原文翻译**：

当 $\hat{y} \notin \mathbb{X}$（即有趣的情况）时，最后一个问题等价于：

$$\min \sum_i (x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum_i x_i = R, \quad x_i \geq 0, \; \forall i = 1, \ldots, n$$

由上节课的推导，求解上述问题，找 $\lambda^*$ 使得

$$\sum_i \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ = R$$

则最优解为 $x_i^* = \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+$。

**精讲内容**：

这一页完成了 L1-ball 投影到概率单纯形投影的归约。

**不等式变等式**。化简后的问题有不等式约束 $\sum x_i \leq R$。当 $\hat{y} \notin \mathbb{X}$（即 $\sum |\hat{y}_i| > R$）时，最优解必须满足 $\sum x_i = R$——即不等式在最优解处取等号。

**为什么不等式一定取等号**？反证法：如果 $\sum x_i^* < R$，那么 $x^*$ 在约束的严格内部。由于 $|\hat{y}_i| > 0$（至少对某些 $i$），目标函数 $\sum(x_i - |\hat{y}_i|)^2$ 可以通过增大某些 $x_i$（朝着 $|\hat{y}_i|$ 靠近）来减小，同时不违反 $\sum x_i \leq R$（因为有松弛量）。这与 $x^*$ 的最优性矛盾。

**更直观的论证**：$|\hat{y}_i| \geq 0$，所以 $\hat{y}'$ 在"正象限"内但在 L1-ball 外（$\sum |\hat{y}_i| > R$）。投影必然在 L1-ball 边界上（$\sum x_i = R$），因为 $\hat{y}'$ 比边界上任何点离中心都远。

**调用单纯形投影公式**。现在问题变为：
$$\min \sum(x_i - |\hat{y}_i|)^2 \quad \text{s.t.} \quad \sum x_i = R, x_i \geq 0$$

这就是 Lecture 15 第 ⑦-⑨ 页推导的概率单纯形投影问题（只是总和从 1 变为 $R$）。直接应用公式：

找 $\lambda^*$ 使得 $\sum_i [|\hat{y}_i| - \lambda^*/2]^+ = R$，则 $x_i^* = [|\hat{y}_i| - \lambda^*/2]^+$。

**化归的完成**。从 L1-ball 投影到单纯形投影，经历了：
1. Observation 1：$\hat{y}_i \to |\hat{y}_i|$
2. Observation 2：$\sum|x_i| \to \sum x_i + (x_i \geq 0)$
3. 不等式取等号：$\sum x_i \leq R \to \sum x_i = R$
4. 调用 Lecture 15 公式

**计算复杂度**：与单纯形投影相同，$O(n\log n)$（排序法）。

**教授口吻讲解**：

最后一个缺口补上了：当 $\hat{y}$ 在球外时，$\sum x_i$ 一定等于 $R$（不等式一定取等号）。这样问题就完全变成了单纯形投影——上节课已经解决了！$\sum[|\hat{y}_i| - \lambda^*/2]^+ = R$，找到这个 $\lambda^*$，投影就有了。你看，L1-ball 投影看似困难，其实就是"对称性化归 + 单纯形投影"的组合。

---

### 第 ⑩ 页：回顾上节课——概率单纯形投影

**原文翻译**：

回顾上节课：投影到概率单纯形上。

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad (***) \quad \text{s.t.} \quad \sum_i x_i = R, \quad x_i \geq 0, \; \forall i = 1, \ldots, n$$

求解上述问题，找 $\lambda^*$ 使得

$$\sum_i \left[ \hat{y}_i - \frac{\lambda^*}{2} \right]^+ = R$$

则问题 $(***)$ 的最优解为 $x_i^* = [\hat{y}_i - \lambda^*/2]^+$。

**精讲内容**：

这一页是对 Lecture 15 概率单纯形投影公式的回顾，确保学生能无缝衔接。

**公式回顾**。单纯形投影的核心公式已在 Lecture 15 第 ⑦-⑨ 页推导：
1. 用拉格朗日方法松弛等式约束 $\sum x_i = R$。
2. 逐坐标求解，得到 $x_i^*(\lambda) = [\hat{y}_i - \lambda/2]^+$。
3. 选择 $\lambda^*$ 使得 $\sum [\hat{y}_i - \lambda^*/2]^+ = R$。

**注意与标准单纯形的区别**。标准概率单纯形要求 $\sum x_i = 1$，这里要求 $\sum x_i = R$。形式上完全一样，只是常数从 1 变为 $R$。公式不变，只是 $\lambda^*$ 的值不同。

**为什么要单独列一页回顾**？因为这是两堂课之间的衔接点——Lecture 15 的概率单纯形投影公式是 Lecture 16 L1-ball 投影的子程序。确保学生清楚地看到"L1-ball 投影 → 单纯形投影"的归约，是本讲教学设计的重要环节。

**教授口吻讲解**：

这一页就是提醒你们：上节课的单纯形投影公式，我们现在要用它了。$x_i^* = [\hat{y}_i - \lambda^*/2]^+$，$\lambda^*$ 由 $\sum[\hat{y}_i - \lambda^*/2]^+ = R$ 确定。在 L1-ball 投影中，$\hat{y}_i$ 被替换为 $|\hat{y}_i|$。好，最后一页——我们把所有东西放在一起。

---

### 第 ⑪ 页：L1-Ball 投影的最终公式

**原文翻译**：

总结：设 $\hat{y}$ 的分量有任意符号，且 $\hat{y} \notin \mathbb{X}$，我们要求解：

$$\min \sum_i (x_i - \hat{y}_i)^2 \quad \text{s.t.} \quad \sum_i |x_i| \leq R$$

找 $\lambda^*$ 使得

$$\sum_i \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ = R$$

则最优解 $x^*$ 为：

$$x_i^* = \begin{cases} \left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ & \text{if } \hat{y}_i \geq 0 \\ -\left[ |\hat{y}_i| - \frac{\lambda^*}{2} \right]^+ & \text{if } \hat{y}_i < 0 \end{cases}$$

**精讲内容**：

这一页给出了 L1-ball 投影的完整最终公式——本讲的高潮，也是整门课程投影理论的最后一块拼图。

**公式的统一写法**。可以更简洁地写为：

$$x_i^* = \text{sign}(\hat{y}_i) \cdot \left[|\hat{y}_i| - \frac{\lambda^*}{2}\right]^+$$

其中 $\text{sign}(\hat{y}_i)$ 是 $\hat{y}_i$ 的符号（正为 +1，负为 -1，零为 0）。

**公式的三步运算**：
1. **取绝对值**：$|\hat{y}_i|$（消除符号，对应 Observation 1）。
2. **软阈值化**：$[|\hat{y}_i| - \tau]^+$，其中 $\tau = \lambda^*/2$（单纯形投影，对应 Lecture 15 公式）。
3. **恢复符号**：$\text{sign}(\hat{y}_i) \cdot [\cdot]^+$（对应 Observation 1 的逆操作）。

**与 soft-thresholding 的深层联系**。这个公式 $\text{sign}(\hat{y}_i) \cdot [|\hat{y}_i| - \tau]^+$ 正是**软阈值算子**（soft-thresholding operator），在信号处理和稀疏优化中具有核心地位。它是 LASSO 问题的 proximal operator——在 proximal gradient method（近端梯度法）中，每一步的"近端步"就是这个操作。

软阈值化的效果：
- 如果 $|\hat{y}_i| > \tau$：$x_i^* = \text{sign}(\hat{y}_i)(|\hat{y}_i| - \tau)$——绝对值缩小 $\tau$，方向不变。
- 如果 $|\hat{y}_i| \leq \tau$：$x_i^* = 0$——完全"压缩"到零。

这正是 L1 正则化诱导稀疏性的机制：小于阈值 $\tau$ 的分量被精确地设为 0，大于阈值的分量被均匀地缩小。

**计算复杂度**。
1. 取绝对值：$O(n)$。
2. 排序求 $\lambda^*$：$O(n \log n)$。
3. 计算各分量：$O(n)$。
总复杂度：$O(n \log n)$。

**四种投影的完整对比**：

| 约束集 | 投影公式 | 关键技巧 | 计算复杂度 |
|--------|---------|---------|----------|
| Box $[0, U_i]$ | $\text{clip}(\hat{y}_i, 0, U_i)$ | 可分离性 | $O(n)$ |
| L2-ball | $R\hat{y}/\|\hat{y}\|$ | 拉格朗日 + 旋转对称 | $O(n)$ |
| 单纯形 | $[\hat{y}_i - \lambda^*/2]^+$ | 拉格朗日 + 截断 | $O(n\log n)$ |
| L1-ball | $\text{sign}(\hat{y}_i)[|\hat{y}_i| - \lambda^*/2]^+$ | 对称化归 + 单纯形 | $O(n\log n)$ |

**课程回顾**。从 Lecture 13 到 Lecture 16，我们走过了一条完整的路：
- Lec 13：投影 GD 的算法框架和投影性质。
- Lec 14：收敛定理 + Box 投影。
- Lec 15：拉格朗日方法 + L2-ball / 单纯形投影。
- Lec 16：L1-ball 投影（归结为单纯形投影）。

加上 Lecture 1-12 建立的 GD 理论基础，整门课程构成了一个从**无约束凸优化**到**约束凸优化**的完整理论体系。

**教授口吻讲解**：

最终公式：$x_i^* = \text{sign}(\hat{y}_i) \cdot [|\hat{y}_i| - \lambda^*/2]^+$。三步：取绝对值，软阈值化，恢复符号。如果你学过 LASSO 的 proximal 方法，你会发现这就是 soft-thresholding——ML 里最重要的稀疏诱导操作之一。

好了同学们，这是我们这门课的最后一个公式了。回顾一下：从 Lecture 1 的"什么是优化"开始，经过凸集、凸函数、梯度下降、收敛理论、步长选择、强凸性、SGD，一直到现在的投影梯度下降和四种投影。你们手里现在有一套完整的工具箱——从最简单的无约束梯度下降到能处理 LASSO、岭回归、概率约束的投影梯度下降。这套工具覆盖了 ML 实践中你们会遇到的绝大多数凸优化问题。期末考试好好复习，祝大家考试顺利！

---

## 跨讲交叉引用

| 引用来源 | 引用内容 | 在本讲的用途 |
|----------|---------|-------------|
| Lec 01-04 | 凸集定义（L1-ball 是凸集）；凸函数一阶特征化 | L1-ball 的凸性是投影唯一性的前提 |
| Lec 05-08 | 无约束 GD 的收敛理论 | 投影 GD 是无约束 GD 的自然推广 |
| **Lec 13** | 投影的钝角性质和勾股定理型不等式 | 保证投影 GD 在 L1-ball 约束下的收敛性 |
| **Lec 14** | 投影 GD 的收敛定理（$O(RB/\sqrt{K})$）；Box 投影 | 本讲的投影嵌入到 Lec 14 的算法框架中 |
| **Lec 15** | 拉格朗日方法；概率单纯形投影公式 $[\hat{y}_i - \lambda^*/2]^+$ | L1-ball 投影的核心子程序 |

## 核心收获

1. **L1-ball 投影的最终公式**：$x_i^* = \text{sign}(\hat{y}_i) \cdot [|\hat{y}_i| - \lambda^*/2]^+$，其中 $\lambda^*$ 由 $\sum[|\hat{y}_i| - \lambda^*/2]^+ = R$ 确定。
2. **化归思想**：通过两个对称性观察（Observation 1: 符号翻转不变性；Observation 2: 非负保持性），将非光滑的 L1-ball 投影归结为已解决的概率单纯形投影。
3. **软阈值化**是 L1 正则化的核心算子：小分量被精确归零（稀疏诱导），大分量被均匀缩小。
4. **课程的完整弧线**：无约束 GD → 收敛理论 → 约束优化（投影 GD）→ 四种投影（Box → L2-ball → 单纯形 → L1-ball），复杂度递增但概念统一。
5. **"构造可行解 + 比较目标值"**是优化中证明问题等价性的标准范式，在 Observation 1 和 2 的证明中得到了教科书级别的展示。
6. **整门课程的大图景**：ORIE 5320 从"如何下降"（GD）到"如何保证有效"（收敛理论）到"如何处理约束"（投影 GD + 投影计算），构建了 AI/ML 优化的完整理论基础设施。
