# Lec 1 - Jan 21 - Intro to Optimization

## what is an optimization problem

- prof started with a really simple example: design a box with maximum volume
  - box has square base, no top, surface area <= 500 cm²
  - decision variables: $x$ (base width), $y$ (height)
  - objective: maximize $x^2 y$
  - constraint: $x^2 + 4xy \leq 500$, and $x, y \geq 0$
  - surface area is $x^2 + 4xy$ not $2x^2 + 4xy$ because there is no top!
- basically every optimization problem has 3 things: decision variables, objective function, constraints
- prof showed this in Excel, got optimal solution (12.91, 6.45), volume = 1075.82
- I noticed the optimal height is roughly half the base width. apparently if you use Lagrange multipliers you can prove $y = x/2$ exactly
- I think this is a good way to think about it — what can we control, what do we want, what are the limits
<!-- 这三个要素的框架很好记：变量、目标、约束 -->

## nonlinear = trouble

- the box problem is nonlinear because $x^2 y$ and $x^2 + 4xy$ are nonlinear in the decision variables
- this matters because nonlinear problems can have multiple local optima, weird saddle points, etc.
- important point: if you start Excel solver at $x=0, y=0$, it totally fails
- why? at (0,0) the gradient is zero vector:
$$\nabla f = \begin{pmatrix} 2xy \\ x^2 \end{pmatrix}\bigg|_{(0,0)} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
- so the optimizer thinks it already found the best point, but volume is 0... clearly wrong
- this is the same reason you can't initialize all neural network weights to zero — gradient is zero, nothing moves
- prof basically saying: don't blindly trust software, you need to understand what's going on
- this is kinda the reason why this course exists i guess
- prof's message: understanding optimization theory = knowing when to trust your tool and when not to

## optimization in ML/AI — the ERM framework

- this part felt very natural to me since I already know ML basics
- typical ML dataset: $\{(x_j, y_j) : j = 1, \ldots, M\}$ where $x_j \in \mathbb{R}^n$ is features and $y_j$ is label
- we want a prediction function $\phi(x, w)$ parameterized by $w$ that predicts well
- loss function measures how bad our prediction is, e.g. squared error $(\phi(x_j, w) - y_j)^2$
- the whole ML training thing is just minimizing average loss:
$$\min_{w \in \mathbb{R}^p} \frac{1}{M} \sum_{j=1}^{M} \ell(x_j, y_j, w)$$
- this is called **Empirical Risk Minimization (ERM)** — pretty cool to see it formalized like this
- so ML training = solving an optimization problem. the entire course is about how to solve this efficiently
<!-- 以前只是调sklearn的API，现在终于知道背后在做什么了 -->

### mapping from box to ML

| Box Problem | ML Problem |
|---|---|
| decision variables $(x, y)$ | model parameters $w$ |
| objective $x^2 y$ | average loss $\frac{1}{M}\sum \ell$ |
| constraint: surface area | regularization (implicit constraint) |
| maximize volume | minimize loss |

## least squares regression

- simplest case: $\phi(x, w) = w^\top x$ (linear prediction)
- loss = squared error, so the problem becomes:
$$\min_{w \in \mathbb{R}^n} \frac{1}{M} \sum_{j=1}^{M} (w^\top x_j - y_j)^2$$
- can also write in matrix form: $\min_w \frac{1}{M} \|Xw - y\|_2^2$ where $X$ is the data matrix (rows = data points)
- this has a closed-form solution (normal equation): $\hat{w} = (X^\top X)^{-1} X^\top y$
- but when $M$ or $n$ is huge (images, NLP), matrix inverse costs $O(n^3)$ — too expensive, need iterative methods
- so even for the "easiest" ML problem, we might need optimization algorithms when scale is large

### ridge regression (L2)

- add $\lambda \|w\|_2^2$ to the objective (penalize large weights)
- closed-form: $\hat{w} = (X^\top X + \lambda M \cdot I)^{-1} X^\top y$
- the $\lambda M \cdot I$ term makes the matrix always positive definite — solution always exists and unique
- prevents overfitting, improves numerical stability

### lasso (L1)

- add $\lambda \|w\|_1$ (sum of absolute values)
- cool property: makes some $w_i$ exactly zero → automatic feature selection
- why L1 gives sparsity but L2 doesn't: L1 has diamond-shaped level sets, the contact point with the loss ellipse tends to hit the axis (= zero coordinates). L2 has circle level sets, contact usually not on axis
- but $|w_i|$ is not differentiable at $w_i = 0$, so standard gradient descent doesn't work directly
- need subgradient or proximal methods (apparently covered in Lec 13-16)
<!-- L1 vs L2的几何直觉很好理解，但数学上为什么L1不可微会带来这么大的影响，后面应该会讲 -->

## logistic regression

- now switching from regression to classification — labels are $+1$ or $-1$
- prediction uses sigmoid: $\phi(x, w) = \frac{1}{1 + e^{w^\top x}}$
  (note: prof uses $e^{w^\top x}$ not $e^{-w^\top x}$, just a sign convention — equivalent to flipping sign of $w$)
- output is between 0 and 1 → interpret as probability of class $+1$
- $1 - \phi(x, w) = \frac{e^{w^\top x}}{1 + e^{w^\top x}}$ = probability of class $-1$
- decision boundary at $w^\top x = 0$ where probability = 0.5
- the key property: sigmoid is smooth (infinitely differentiable), so gradient-based optimization is possible
  - if we used a hard threshold (step function) instead, it would not be differentiable — can't do gradient descent
  - sigmoid is the smooth approximation of the step function

## fitting logistic regression — MLE

- use Maximum Likelihood Estimation
- assume data points are i.i.d., so joint likelihood = product of individual likelihoods
- for $y_j = +1$: likelihood = $\frac{1}{1 + e^{w^\top x_j}}$
- for $y_j = -1$: likelihood = $\frac{e^{w^\top x_j}}{1 + e^{w^\top x_j}}$
- total likelihood $L(w) = \prod_j P(y_j | x_j, w)$
- take log to avoid numerical underflow and simplify derivatives:
  - product → sum
  - tiny numbers multiplied → manageable negative numbers added
- maximize log-likelihood = minimize negative log-likelihood (= cross-entropy loss)
- this fits perfectly into the ERM framework — negative log-likelihood is just the loss function $\ell$
- no closed-form solution here, need iterative methods like gradient descent
- not 100% sure about all the detailed math for MLE yet, but the high-level logic makes sense: pick $w$ that makes the observed data most probable

## my questions after this lecture

- how exactly do we choose the step size for gradient descent? (feels like this will be a big topic)
- when does gradient descent succeed vs fail? (the (0,0) failure example was concerning)
- is there a way to know in advance if a problem is "easy" or "hard" to optimize?

## takeaway from lec 1

- optimization is everywhere in ML — every model training is solving an optimization problem
- the ERM formula $\min_w \frac{1}{M}\sum \ell(x_j, y_j, w)$ unifies everything: regression, classification, deep learning
- nonlinear problems are hard, software can fail, initialization matters a lot
- we looked at regression (least squares, ridge, lasso) and classification (logistic reg)
- key insight: smooth vs non-smooth (ridge is smooth, lasso is not) is apparently a fundamental divide in optimization theory
- also: constrained vs unconstrained — regularization is like an implicit constraint
- rest of course will be about HOW to solve these problems efficiently and reliably
- I'm excited about this course — it feels like it will explain what's really happening when I call model.fit()

