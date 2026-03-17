# OPT-AI 课程核心词汇表（Glossary）

> 覆盖 Lecture 1–12 全部 Slides 涉及的关键术语，每词附课程语境下的例句。
> 考试时遇到这些词就不怕看不懂题目了！

---

## 一、优化问题基础（Optimization Basics）— L1–L4

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 1 | **Optimization problem** | 优化问题 | *We formulate the box design task as an optimization problem to find the largest volume.* |
| 2 | **Decision variable** | 决策变量 | *The decision variables are x (base length) and y (height of the box).* |
| 3 | **Objective function** | 目标函数 | *The objective function x²y represents the volume we want to maximize.* |
| 4 | **Constraint** | 约束（条件） | *The constraint x² + 4xy ≤ 500 limits the surface area of the box.* |
| 5 | **Feasible (solution/region)** | 可行的（解/域） | *A point x is feasible if it satisfies all the constraints of the problem.* |
| 6 | **Optimal solution** | 最优解 | *The optimal solution is (x̂, ŷ) = (12.91, 6.45) with the optimal volume 1075.82.* |
| 7 | **Maximize / Minimize** | 最大化 / 最小化 | *We want to maximize the volume subject to the surface area constraint.* |
| 8 | **Subject to (s.t.)** | 受限于；使得 | *max x²y subject to x² + 4xy ≤ 500.* |
| 9 | **Nonlinear** | 非线性的 | *This optimization problem is nonlinear because the objective function is nonlinear in the decision variables.* |
| 10 | **Initial guess / Initial point** | 初始猜测 / 初始点 | *Try to use Excel for the previous optimization problem with the initial guess x = 0, y = 0.* |
| 11 | **Surface area** | 表面积 | *The surface area of the box will be at most 500 cm².* |
| 12 | **Volume** | 体积 | *We want to design a box with the largest possible volume.* |

---

## 二、机器学习与 AI 语境（ML/AI Context）— L1–L3

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 13 | **Dataset / Data set** | 数据集 | *A typical dataset in an ML/AI application looks like {(xⱼ, yⱼ) : j = 1, ..., M}.* |
| 14 | **Data point** | 数据点 | *Each data point j has a feature vector xⱼ and a label yⱼ.* |
| 15 | **Feature (vector)** | 特征（向量） | *xⱼ ∈ ℝⁿ is a vector of features corresponding to data point j.* |
| 16 | **Label** | 标签 | *yⱼ ∈ ℝ is the label corresponding to data point j.* |
| 17 | **Prediction function** | 预测函数 | *The goal is to find a prediction function ϕ(·, w) such that ϕ(xⱼ, w) ≈ yⱼ.* |
| 18 | **Parameter(s)** | 参数 | *w ∈ ℝᵖ is the vector of parameters we need to estimate.* |
| 19 | **Parameterized (by)** | 由……参数化 | *The prediction function is parameterized by w ∈ ℝᵖ.* |
| 20 | **Loss (function)** | 损失（函数） | *Let ℓ(xⱼ, yⱼ, w) be the loss we incur when we use the prediction function to predict the label.* |
| 21 | **Error** | 误差 | *The loss ℓ(xⱼ, yⱼ, w) measures the error between prediction and true label.* |
| 22 | **Regression** | 回归 | *For regression, ℓ(xⱼ, yⱼ, w) = (ϕ(xⱼ, w) − yⱼ)².* |
| 23 | **Classification** | 分类 | *Consider a two-class classification problem where yⱼ ∈ {+1, −1}.* |
| 24 | **Predict / Prediction** | 预测 | *We predict the label for data point j as wᵀxⱼ.* |
| 25 | **Estimate** | 估计 | *w ∈ ℝⁿ is the vector of parameters we need to estimate.* |

---

## 三、回归模型（Regression Models）— L1

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 26 | **Least squares (regression)** | 最小二乘（回归） | *In least squares regression, we minimize (1/M) Σ(wᵀxⱼ − yⱼ)².* |
| 27 | **Ridge regression** | 岭回归 | *Ridge regression adds λ‖w‖₂² to avoid overfitting and ensure robustness against outliers.* |
| 28 | **Lasso** | Lasso 回归 | *Lasso adds λ‖w‖₁ and is useful for feature selection.* |
| 29 | **Overfitting** | 过拟合 | *Ridge regression is useful for avoiding overfitting.* |
| 30 | **Robustness** | 鲁棒性；稳健性 | *Ridge regression helps ensure robustness against outliers.* |
| 31 | **Outlier** | 异常值；离群点 | *Ridge regression is useful for ensuring robustness against outliers.* |
| 32 | **Feature selection** | 特征选择 | *Lasso is useful for feature selection because it drives some weights to zero.* |
| 33 | **Regularization (term)** | 正则化（项） | *The term λ‖w‖₂² is a regularization term that penalizes large weights.* |

---

## 四、逻辑回归与分类（Logistic Regression & Classification）— L1–L3

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 34 | **Logistic regression** | 逻辑回归 | *Consider a two-class classification problem using logistic regression.* |
| 35 | **Two-class / Multi-class** | 二分类 / 多分类 | *Consider a multi-class classification problem where yⱼ ∈ {1, ..., K}.* |
| 36 | **Probability** | 概率 | *ϕ(x, w) can be interpreted as the probability that a data point belongs to class +1.* |
| 37 | **Likelihood** | 似然（值） | *The likelihood of data point 1 with label +1 is 1/(1 + e^(wᵀx₁)).* |
| 38 | **Log-likelihood** | 对数似然 | *We maximize the log-likelihood ln L(w) instead of the likelihood L(w).* |
| 39 | **Maximum likelihood estimation (MLE)** | 最大似然估计 | *To fit a logistic regression model, we use maximum likelihood estimation.* |
| 40 | **Indicator function 𝟙(·)** | 指示函数 | *𝟙(yⱼ = −1) equals 1 when yⱼ = −1, and 0 otherwise.* |
| 41 | **Class** | 类别 | *ϕ(x, w) is the probability that a data point with features x belongs to class +1.* |
| 42 | **Belongs to** | 属于 | *The probability that a data point with feature vector x belongs to class −1.* |

---

## 五、深度学习（Deep Learning）— L2–L3

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 43 | **Neural network** | 神经网络 | *Given the parameters of the neural network, the output from the last layer is computed recursively.* |
| 44 | **Layer** | 层 | *Layer 1 takes input xⱼ and produces output aⱼ¹ = σ₁(Q¹xⱼ + g¹).* |
| 45 | **Sigmoid** | Sigmoid 函数 | *Each component of σ could be the sigmoid σ(t) = 1/(1 + eᵗ).* |
| 46 | **ReLU (Rectified Linear Unit)** | 修正线性单元 | *Another common activation is the rectified linear unit σ(t) = max{t, 0}.* |
| 47 | **Activation function** | 激活函数 | *Each element of σ₁ or σ₂ could be a sigmoid or ReLU activation function.* |
| 48 | **Forward propagation** | 前向传播 | *The output aⱼᴸ is computed by forward propagation through L layers.* |
| 49 | **Output** | 输出 | *The output we get from the last layer is σ_L(Qᴸ · σ_{L−1}(…) + gᴸ).* |
| 50 | **Input** | 输入 | *The input to the first layer is the feature vector xⱼ.* |
| 51 | **Weight (matrix)** | 权重（矩阵） | *Q¹ ∈ ℝ^(d₁×n) is the weight matrix of the first layer.* |
| 52 | **Bias (vector)** | 偏置（向量） | *g¹ ∈ ℝ^(d₁) is the bias vector of the first layer.* |

---

## 六、梯度下降（Gradient Descent）— L3–L4, L6

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 53 | **Gradient** | 梯度 | *The gradient of f at point x⁰ is the n-dimensional vector ∇f(x⁰).* |
| 54 | **Gradient descent** | 梯度下降 | *Start at any point x⁰ and generate a sequence by moving against the gradient.* |
| 55 | **Partial derivative** | 偏导数 | *∂f(x)/∂x₁ evaluated at x = x⁰ is the first component of the gradient.* |
| 56 | **Step size (learning rate)** | 步长（学习率） | *αᵏ is a "small" step size used at iteration k of gradient descent.* |
| 57 | **Iteration** | 迭代 | *At iteration k, we are at point xᵏ and the next point is xᵏ⁺¹ = xᵏ − αᵏ∇f(xᵏ).* |
| 58 | **Sequence (of points)** | （点的）序列 | *Generate a sequence of points x¹, x², ... by moving against the gradient.* |
| 59 | **Trial and error** | 试错法 | *How small is small? Do trial and error to pick the step size.* |
| 60 | **Halve** | 减半 | *If the step does not decrease f, halve αᵏ and try again.* |
| 61 | **Overshoot** | 越过；过冲 | *If the step size is too large, we overshoot the minimum.* |
| 62 | **Converge / Convergence** | 收敛 | *With a constant step size, we will not be able to establish convergence to a minimum.* |
| 63 | **Iterate (n.)** | 迭代点 | *The best iterate along the way will yield an objective value close to optimal.* |

---

## 七、极值与凸性基础（Extrema & Convexity Basics）— L4

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 64 | **Local minimum** | 局部最小值 | *x̂ is a local minimum if f(x̂) ≤ f(x) for all x in a small neighborhood of x̂.* |
| 65 | **Global minimum** | 全局最小值 | *x* is a global minimum if f(x*) ≤ f(x) for any x.* |
| 66 | **Minimizer** | 最小值点 | *We want to find the minimizer of f over the interval [a, b].* |
| 67 | **Neighborhood** | 邻域 | *f(x̂) ≤ f(x) for any x in a small neighborhood of x̂.* |
| 68 | **Convex (function)** | 凸函数 | *If we are minimizing a convex function, all local minima are also global minima.* |
| 69 | **A priori** | 先验地 | *If we a priori know that the objective function is convex, we can be sure about the success of gradient descent.* |
| 70 | **Heuristic** | 启发式的 | *A heuristic method to find a global minimum is to start gradient descent with different initial points.* |

---

## 八、单维优化（Single-Dimensional Optimization）— L4–L6

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 71 | **Single-dimensional optimization** | 单维优化；一维优化 | *Choosing the best step size is a single-dimensional optimization problem over α.* |
| 72 | **Interval of uncertainty** | 不确定区间 | *We start with an interval of uncertainty [a¹, b¹] and iteratively reduce its width.* |
| 73 | **Bisection search** | 二分搜索 | *Bisection search places λᵏ and ρᵏ symmetrically around the midpoint.* |
| 74 | **Golden section search** | 黄金分割搜索 | *Golden section search ensures the interval shrinks by a factor of β = (−1+√5)/2 at each iteration.* |
| 75 | **Midpoint** | 中点 | *Place λᵏ and ρᵏ symmetrically around the midpoint of the interval.* |
| 76 | **Stopping tolerance** | 停止容差 | *If bᵏ⁺¹ − aᵏ⁺¹ ≤ θ (stopping tolerance) then stop.* |
| 77 | **Shrink / Shrinkage factor** | 缩小 / 收缩因子 | *The interval of uncertainty shrinks by a factor of β at every iteration.* |
| 78 | **Function evaluation** | 函数求值 | *Bisection search requires making two function evaluations at each iteration.* |
| 79 | **Derivative** | 导数 | *If we have access to the derivatives of f, we can be more efficient.* |
| 80 | **Numerical difficulties** | 数值困难 | *When λᵏ and ρᵏ are very close, we run into numerical difficulties.* |
| 81 | **Exact line search** | 精确线搜索 | *Use single-dimensional optimization as an exact line search to find the best step size during gradient descent.* |

---

## 九、约束优化（Constrained Optimization）— L6–L7

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 82 | **Constrained optimization** | 约束优化 | *In constrained optimization, we minimize f(x) subject to gᵢ(x) ≤ 0 and hᵢ(x) = 0.* |
| 83 | **Unconstrained optimization** | 无约束优化 | *We convert the constrained problem into an equivalent unconstrained optimization problem.* |
| 84 | **Inequality constraint** | 不等式约束 | *gᵢ(x) ≤ 0 is an inequality constraint.* |
| 85 | **Equality constraint** | 等式约束 | *hᵢ(x) = 0 is an equality constraint.* |
| 86 | **Penalty function method** | 惩罚函数法 | *The penalty function method moves constraints into the objective by largely penalizing violations.* |
| 87 | **Penalty parameter** | 惩罚参数 | *θ₁, ..., θₙ are large penalty parameters.* |
| 88 | **Violation (of constraint)** | （约束的）违反 | *The penalty method penalizes violations of the constraints.* |
| 89 | **Magnification parameter** | 放大参数 | *β > 1 is the magnification parameter that increases the penalty at each iteration.* |
| 90 | **Barrier function method** | 障碍函数法 | *The barrier function method is applicable to problems with only inequality constraints.* |
| 91 | **Barrier multiplier** | 障碍乘子 | *μ₁, ..., μₙ are small barrier multipliers.* |
| 92 | **Boundary** | 边界 | *When we are close to the boundary of the feasible region, −gᵢ(x) is a very small positive number.* |
| 93 | **Feasible region** | 可行域 | *The barrier term forms a "wall" preventing the iterate from leaving the feasible region.* |
| 94 | **Differentiable** | 可微的 | *max{0, g(x)} is not differentiable at the zero points, so we square it.* |

---

## 十、泰勒定理与分析基础（Taylor's Theorem & Analysis）— L7–L9

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 95 | **Taylor's theorem** | 泰勒定理 | *By Taylor's theorem, f(y) = f(x) + ∇ᵀf(γx + (1−γ)y)·(y−x) for some γ ∈ (0,1).* |
| 96 | **Little-o notation o(t)** | 小 o 记号 | *A function g(t) is called o(t) if lim_{t→0} g(t)/t = 0.* |
| 97 | **Implication** | 推论；蕴含 | *The implication of Taylor's theorem is: f(y) = f(x) + ∇ᵀf(x)·(y−x) + o(‖y−x‖).* |
| 98 | **Infinitely differentiable** | 无穷次可微的 | *Let's focus on infinitely differentiable functions f: ℝⁿ → ℝ.* |
| 99 | **Norm** | 范数 | *‖x‖ = √(Σxᵢ²) is the Euclidean norm of vector x.* |
| 100 | **Angle (between vectors)** | （向量间的）夹角 | *θ is the angle between ∇f(γx+(1−γ)y) − ∇f(x) and y − x.* |
| 101 | **Proof (Pf)** | 证明 | *Pf: By Taylor's theorem, for any x, y, ...* |

---

## 十一、凸函数理论（Convex Function Theory）— L8–L9

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 102 | **Convex function** | 凸函数 | *f is convex if f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y) for all x, y and λ ∈ [0,1].* |
| 103 | **Convex combination** | 凸组合 | *λx + (1−λ)y is a convex combination of x and y.* |
| 104 | **Jensen's inequality** | 延森不等式 | *By Jensen's inequality, f(Σλⁱxⁱ) ≤ Σλⁱf(xⁱ).* |
| 105 | **First-order characterization (of convexity)** | 凸性的一阶刻画 | *The first-order characterization states: f is convex iff f(y) ≥ f(x) + ∇ᵀf(x)·(y−x).* |
| 106 | **Tangent (line/plane)** | 切线/切平面 | *For a convex function, the tangent line at any point is a global lower bound.* |
| 107 | **Lower bound / Upper bound** | 下界 / 上界 | *The linear approximation f(x) + ∇f(x)ᵀ(y−x) is a lower bound on f(y).* |
| 108 | **If and only if (iff)** | 当且仅当 | *f is convex if and only if f(y) ≥ f(x) + ∇ᵀf(x)·(y−x) for all x, y.* |
| 109 | **Contradiction** | 矛盾 | *To get a contradiction, assume x* and y* are two distinct optimal solutions.* |
| 110 | **Fix (a point)** | 固定（一个点） | *Fix any y ∈ ℝⁿ; we want to show f(x̂) ≤ f(y).* |
| 111 | **Satisfy / Satisfies** | 满足 | *x̂ satisfies f(x̂) ≤ f(x) for any x in a small neighborhood of x̂.* |

---

## 十二、梯度下降收敛分析（Convergence Analysis）— L10–L11

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 112 | **Convergence (of gradient descent)** | （梯度下降的）收敛 | *We analyze the convergence of gradient descent for convex functions.* |
| 113 | **Constant step size** | 常步长 | *We use gradient descent with constant step size γ.* |
| 114 | **Bounce around** | 在附近振荡 | *With a constant step size, we will bounce around the minimum.* |
| 115 | **Bounded gradient** | 有界梯度 | *Assume ‖∇f(x)‖ ≤ B for all x ∈ ℝⁿ (bounded gradient).* |
| 116 | **Cumulative error** | 累积误差 | *Σ(f(xᵏ) − f(x*)) is the cumulative error over the first K iterations.* |
| 117 | **Telescoping sum** | 伸缩和；望远镜求和 | *Use a telescoping sum to simplify ‖x⁰−x*‖² − ‖xᴷ−x*‖².* |
| 118 | **Preliminary analysis** | 预备分析 | *The preliminary analysis bounds the cumulative error using convexity and telescoping sums.* |
| 119 | **Convergence rate** | 收敛速率 | *The average error diminishes at rate 1/√K after K iterations.* |
| 120 | **Smooth (function)** | 光滑的（函数） | *f is smooth with parameter L if f(y) ≤ f(x) + ∇f(x)ᵀ(y−x) + (L/2)‖y−x‖².* |
| 121 | **Smoothness parameter (L)** | 光滑度参数 | *A smooth convex function has a smoothness parameter L that controls the quadratic upper bound.* |
| 122 | **Quadratic upper bound** | 二次上界 | *Smooth convex functions satisfy a quadratic upper bound around any point.* |
| 123 | **Iteration complexity** | 迭代复杂度 | *To achieve error ε, the iteration complexity is K = LR²/(2ε) for smooth convex functions.* |
| 124 | **Precision** | 精度 | *The number of iterations increases fast as we want more precision.* |
| 125 | **Diminish** | 递减；减小 | *The average error of gradient descent diminishes at rate 1/√K.* |
| 126 | **Deviate (from)** | 偏离 | *The best solution along the way will yield an objective value not deviating from optimal by at most RB/√K.* |
| 127 | **Arbitrarily close** | 任意接近 | *If we run gradient descent long enough, the last iterate will be arbitrarily close to optimal.* |

---

## 十三、凸集与投影梯度下降（Convex Sets & Projected GD）— L12

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 128 | **Convex set** | 凸集 | *The set C is convex if for all x, y ∈ C and λ ∈ [0,1], λx + (1−λ)y ∈ C.* |
| 129 | **Intersection** | 交集 | *The intersection of convex sets C₁ ∩ C₂ ∩ ... ∩ Cₗ is also convex.* |
| 130 | **Line of sight** | 视线（直线可达） | *For a convex set, we have direct line of sight from any point to any other point in the set.* |
| 131 | **Sublevel set** | 下水平集 | *If f is convex, then the sublevel set C = {x : f(x) ≤ 0} is convex.* |
| 132 | **Feasible solution** | 可行解 | *The set of feasible solutions for this constrained problem is convex.* |
| 133 | **Strictly convex** | 严格凸的 | *f is strictly convex if f(λx+(1−λ)y) < λf(x)+(1−λ)f(y) for x ≠ y and λ ∈ (0,1).* |
| 134 | **Unique optimal solution** | 唯一最优解 | *A strictly convex function over a convex set has a unique optimal solution.* |
| 135 | **Projected gradient descent** | 投影梯度下降 | *Projected gradient descent projects the iterate back onto the feasible set after each gradient step.* |
| 136 | **Projection (Π)** | 投影 | *The projection of x onto 𝕏 is Π_𝕏(x) = argmin_{y∈𝕏} ‖y−x‖².* |
| 137 | **Project (onto)** | 投影到 | *If x − γ∇f(x) is outside 𝕏, we project it back onto 𝕏.* |

---

## 十四、数学通用术语（General Mathematical Terms）— 贯穿全课程

| # | 英文术语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 138 | **Vector** | 向量 | *xⱼ ∈ ℝⁿ is a vector of features.* |
| 139 | **Scalar** | 标量 | *f: ℝⁿ → ℝ maps n-dimensional vectors to a scalar.* |
| 140 | **Matrix** | 矩阵 | *Q¹ ∈ ℝ^(d₁×n) is the weight matrix of the first layer.* |
| 141 | **Transpose (ᵀ)** | 转置 | *We predict the label as wᵀxⱼ, where wᵀ denotes the transpose of w.* |
| 142 | **Inner product / Dot product** | 内积 / 点积 | *wᵀx = Σwᵢxᵢ is the inner product of w and x.* |
| 143 | **Euclidean norm ‖·‖** | 欧几里得范数 | *‖x‖ = √(Σxᵢ²) is the Euclidean norm of x.* |
| 144 | **Summation (Σ)** | 求和 | *The loss is averaged over all data points: (1/M) Σℓ(xⱼ, yⱼ, w).* |
| 145 | **Denote** | 记作；表示 | *We denote the gradient at xᵏ as gᵏ = ∇f(xᵏ).* |
| 146 | **Such that** | 使得 | *Find a prediction function such that ϕ(xⱼ, w) ≈ yⱼ on a large fraction of data points.* |
| 147 | **For all / For any** | 对于所有 / 对于任意 | *f(y) ≥ f(x) + ∇ᵀf(x)·(y−x) for all x, y ∈ ℝⁿ.* |
| 148 | **There exists** | 存在 | *There exists some ε > 0 such that f(x̂) ≤ f(y) for all y with ‖y−x̂‖ ≤ ε.* |
| 149 | **Without loss of generality** | 不失一般性 | *Without loss of generality, assume λ ∈ (0, 1).* |
| 150 | **Argmin** | 取最小值的自变量 | *x* = argmin_{x∈ℝⁿ} f(x) is the point where f achieves its minimum.* |
| 151 | **Domain** | 定义域 | *The domain of f is ℝⁿ.* |
| 152 | **Map(s) to** | 映射到 | *f: ℝⁿ → ℝ is a function that maps n-dimensional vectors to a scalar.* |
| 153 | **Fraction** | 比例；部分 | *We want ϕ(xⱼ, w) ≈ yⱼ on a large fraction of data points.* |
| 154 | **Respectively** | 分别 | *x and y denote the base length and height, respectively.* |

---

## 十五、考试高频动词与短语（Exam Action Words）

| # | 英文短语 | 中文释义 | 课程语境例句 |
|---|--------|---------|-----------|
| 155 | **Formulate (as)** | 建模为 | *Formulate the problem as a constrained optimization problem.* |
| 156 | **Solve** | 求解 | *Solve the unconstrained optimization problem using gradient descent.* |
| 157 | **Compute** | 计算 | *Compute ∇f(xᵏ) at the current iterate.* |
| 158 | **Evaluate** | 求值 | *Evaluate f at the two test points λᵏ and ρᵏ.* |
| 159 | **Show / Prove** | 证明 | *Show that if f is convex, then every local minimum is a global minimum.* |
| 160 | **Assume / Suppose** | 假设 | *Assume that ‖∇f(x)‖ ≤ B for all x ∈ ℝⁿ.* |
| 161 | **Derive** | 推导 | *Derive the log-likelihood function for logistic regression.* |
| 162 | **Determine** | 确定；判断 | *Determine whether the given function is convex.* |
| 163 | **Verify** | 验证 | *Verify that the gradient descent update decreases the function value.* |
| 164 | **Obtain** | 得到 | *We obtain the update rule xᵏ⁺¹ = xᵏ − γ∇f(xᵏ).* |
| 165 | **Yield** | 产生；给出 | *The best iterate along the way will yield an objective value close to optimal.* |
| 166 | **Denotes / Stands for** | 表示；代表 | *gᵏ denotes the gradient ∇f(xᵏ).* |
| 167 | **Incur** | 招致；产生 | *ℓ(xⱼ, yⱼ, w) is the loss we incur when using the prediction function.* |
| 168 | **Ensure** | 确保 | *Ridge regression is useful for ensuring robustness against outliers.* |
| 169 | **Establish** | 建立 | *With a constant step size, we will not be able to establish convergence.* |
| 170 | **It turns out** | 结果是；事实上 | *It turns out optimization software may very easily fail for nonlinear problems.* |
| 171 | **Plug in / Substitute** | 代入 | *Plug in xᵏ⁺¹ = xᵏ − γgᵏ into the smoothness inequality.* |
| 172 | **Apply** | 应用 | *Apply the first-order characterization of convexity.* |
| 173 | **At most / At least** | 至多 / 至少 | *The surface area of the box will be at most 500 cm².* |
| 174 | **Equivalent** | 等价的 | *We convert the constrained problem into an equivalent unconstrained problem.* |

---

> **使用建议：** 考试前通读一遍，重点记忆每个词在课程语境中的含义。看到 "subject to" 就知道是约束条件，看到 "show that" 就知道要证明，看到 "bounded" 就知道是有界的。祝考试顺利！
