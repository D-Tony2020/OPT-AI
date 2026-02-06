# Lec 4 - Feb 3 - Convexity, Local vs Global, and Line Search

## overview

This is a turning point in the course. Lec 3 showed gradient descent can work (step size 0.1) or fail spectacularly (step size 0.5). Today we ask the deeper questions: what can gradient descent guarantee at best? when does "best" mean "global optimum"? how to pick the step size more carefully? The answer to the first two leads to **convexity** — the most important concept in this course. The third leads to **exact line search** and 1D optimization.

I think I'm starting to see where this whole course is heading now.

## gradient descent recap (from Lec 3)

- update: $x^{k+1} = x^k - \alpha^k \nabla f(x^k)$
- backtracking: if $f$ doesn't decrease, halve $\alpha$ and retry
- this was our "crude" algorithm from last time

### gradient computation example: log-sum-exp

- new example: $f(x) = 4 \ln(e^{2x_1} + e^{4x_2}) - 6x_1 - 4x_2$
- gradient involves Softmax-like terms:
$$\nabla f = \left(4 \cdot \frac{2e^{2x_1}}{e^{2x_1} + e^{4x_2}} - 6, \quad 4 \cdot \frac{4e^{4x_2}}{e^{2x_1} + e^{4x_2}} - 4\right)$$
- I noticed that $\frac{e^{2x_1}}{e^{2x_1} + e^{4x_2}}$ is literally a Softmax probability! this is not a coincidence — the gradient of log-sum-exp IS Softmax: $\nabla \text{LSE}(z) = \text{softmax}(z)$
  - connects to Lec 2 where log-sum-exp appeared in the Softmax log-likelihood
- this function is convex: $\ln(e^{2x_1} + e^{4x_2})$ is convex (log-sum-exp), scale by 4 still convex, subtract linear still convex
<!-- log-sum-exp到处都是：逻辑回归、Softmax、attention... 掌握它的性质很重要 -->

## local minimum vs global minimum — the key distinction

- **global minimum** $x^*$: $f(x^*) \leq f(x)$ for ALL $x$
- **local minimum** $\hat{x}$: $f(\hat{x}) \leq f(x)$ for all $x$ in a small neighborhood of $\hat{x}$
- every global min is a local min, but not vice versa

- **gradient descent can only guarantee convergence to a local minimum**
  - gradient is a local quantity — it only sees the immediate neighborhood
  - $\nabla f = 0$ means "critical point" which could be local min, local max, or saddle point
  - if you start near a bad local min, you get stuck there

- in deep learning: the loss landscape is full of local minima and saddle points
  - but recent research suggests most local minima in large networks have loss values close to global minimum
  - saddle points are actually the bigger problem in high dimensions
  - SGD's randomness helps escape saddle points
  - this partially explains why DL "works" despite non-convexity

- this all connects back to Lec 1: the box problem with starting point (0,0) — that was a degenerate critical point where gradient was zero but it clearly wasn't the optimum

## convex functions — the "golden key"

This is THE concept of the course. I need to understand this well.

- **definition**: $f$ is convex if for any $x, y$ and $\lambda \in [0,1]$:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$
- geometric meaning: the line segment between any two points on the graph lies above the function curve. the region above the graph is "bowl-shaped"

### the fundamental theorem: local min = global min for convex functions

- **proof** (by contradiction): suppose $\hat{x}$ is local min but not global min. then exists $x^*$ with $f(x^*) < f(\hat{x})$. by convexity:
$$f(\lambda x^* + (1-\lambda)\hat{x}) \leq \lambda f(x^*) + (1-\lambda)f(\hat{x}) < f(\hat{x})$$
- take $\lambda$ very small → the point $\lambda x^* + (1-\lambda)\hat{x}$ is arbitrarily close to $\hat{x}$ but has strictly lower value → contradicts $\hat{x}$ being a local min

- this is actually a pretty elegant proof. the key insight: if there's a lower point anywhere, convexity forces the function to decrease along the line toward it, even for tiny steps. so there can be no "fake valleys."
<!-- 凸性的证明不难，但意义深远：没有假坑，所有局部最优都是全局最优 -->

### practical implications

| Non-convex | Convex |
|---|---|
| multiple local minima possible | every local min is global |
| GD result depends on starting point | any starting point works |
| need multiple random restarts | one run is enough |
| no theoretical success guarantee | strong convergence theory |

- looking back at our models:
  - Lec 1 box problem: non-convex → starting point matters
  - Lec 2 logistic regression: convex → always reliable
  - Lec 3 deep learning: non-convex → training is "mysterious"
- convexity is like a quality certificate for optimization problems

## exact line search

- instead of backtracking (halve until it works), find the OPTIMAL step size:
$$\alpha^* = \arg\min_{\alpha \geq 0} f(x^k - \alpha \nabla f(x^k))$$
- key observation: this is a **1D optimization problem** regardless of the dimension $n$ of the original problem
  - we fix the direction $d = -\nabla f(x^k)$ and only optimize how far to go along it
  - define $\phi(\alpha) = f(x^k + \alpha d)$, minimize $\phi$ over $\alpha \geq 0$

- a beautiful property: at the optimal $\alpha^*$, we have $\nabla f(x^{k+1})^\top \nabla f(x^k) = 0$
  - consecutive gradients are orthogonal! exact line search walks as far as possible in current direction, then turns 90 degrees
  - this gives a "zigzag" pattern on quadratic functions

### exact vs backtracking line search

| | Exact | Backtracking |
|---|---|---|
| Goal | find optimal $\alpha^*$ | find "good enough" $\alpha$ |
| Cost per step | higher (solve 1D problem) | lower (few halvings) |
| Per-step quality | optimal | suboptimal but sufficient |
| In practice | mostly theoretical | widely used |

- in DL practice nobody uses exact line search — fixed learning rate or Adam is more practical
- but exact line search is important for theory: it gives the "best possible" GD performance

## 1D optimization — bisection search

Since exact line search reduces to 1D optimization, we need tools for that.

### the framework: interval reduction

- start with interval $[a^1, b^1]$ containing the minimum (assumes **unimodal** function — convex functions are always unimodal)
- pick two test points $\lambda^k < \rho^k$ inside the interval
- compare $f(\lambda^k)$ vs $f(\rho^k)$:
  - if $f(\lambda^k) \leq f(\rho^k)$: minimum is in $[a^k, \rho^k]$ (discard right part)
  - if $f(\lambda^k) > f(\rho^k)$: minimum is in $[\lambda^k, b^k]$ (discard left part)
- repeat → interval shrinks each iteration

- **why two test points?** with one point you can't tell which side the minimum is on (unless you compute the derivative). two points let you compare and eliminate half the interval.

### bisection search specifically

- place test points symmetrically around the midpoint:
$$\lambda^k = \frac{a^k + b^k}{2} - \varepsilon, \quad \rho^k = \frac{a^k + b^k}{2} + \varepsilon$$
- each step: interval shrinks to roughly half (exactly $\frac{W_k}{2} + \varepsilon$)
- after $k$ iterations: width $\approx W_0 / 2^k$
- to reach precision $\delta$: need $k \approx \log_2(W_0 / \delta)$ iterations
  - example: initial width 10, target precision $10^{-6}$ → only ~23 iterations, each needing 2 function evaluations. very efficient!

- $\varepsilon$ tradeoff: too small → $f(\lambda^k) \approx f(\rho^k)$ so comparison is numerically unreliable. too large → slow convergence. practical choice: $\varepsilon \approx 10^{-8} \cdot W_0$.

- prof mentioned golden section search is better because it reuses one test point from previous iteration (each step only 1 new function eval instead of 2), but ratio is $0.618$ instead of $0.5$. overall fewer total evaluations.

## where I think this is all going

After 4 lectures I can see the structure of the course:
1. Lec 1-2: **why** — ML = optimization, build up to DL optimization problem
2. Lec 3-4: **how (basics)** — gradient descent, convexity, line search
3. Lec 5-9 (upcoming): **how (theory)** — when does GD converge? how fast? what step size?
4. Lec 10+: **how (scalable)** — SGD for big data, advanced methods

The Lipschitz condition mentioned a few times should tell us exactly what step size to use without trial and error. And "condition number" $\kappa = \lambda_{\max}/\lambda_{\min}$ from the Lec 3 quadratic example will apparently control how fast GD converges. Looking forward to seeing the full theory.
<!-- 前4讲算是铺垫完了，接下来应该是正式的收敛理论 -->

## things to review

- convexity definition — make sure I can state it precisely and use it in proofs
- the proof that local min = global min for convex functions (might appear on exam)
- bisection search convergence rate derivation
- relationship between Hessian eigenvalues and step size bounds (from Lec 3 example)
