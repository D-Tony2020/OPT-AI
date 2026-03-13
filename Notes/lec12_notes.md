# Lec 12 - Mar 9 - Constrained Optimization & Projected Gradient Descent

## Why constraints now?

- lec 09-11 were all about **unconstrained** optimization on $\mathbb{R}^n$
- but real problems have constraints: weights must be non-negative, probabilities sum to 1, norm bounded, etc.
- today we go from unconstrained to constrained — this is more practical
- convex sets from lec 02 finally come back! the whole "convex set" definition was building up to this moment
- this lecture combines set theory (lec 02-04) with the convergence theory (lec 10-11) into a complete framework
<!-- 凸集的定义从lec 02开始铺垫，现在终于派上用场了 -->

## Review: convex sets

- $C$ is convex if for all $x, y \in C$ and $\lambda \in [0,1]$: $\lambda x + (1-\lambda)y \in C$
- intuition: "line of sight" — any two points in the set can see each other in a straight line
- no "holes" or "dents" in the set
- examples: balls $\{x: \|x\| \leq r\}$, half-spaces $\{x: a^Tx \leq b\}$, polyhedra $\{x: Ax \leq b\}$, probability simplex $\{x: x_i \geq 0, \sum x_i = 1\}$
- non-examples: integer lattice $\mathbb{Z}^n$ (0.5 is between 0 and 1 but not integer), sphere surface $\{x: \|x\| = 1\}$ (midpoint of two surface points falls inside the ball)
- convexity of feasible set is what preserves the "local = global" property from lec 09

## Key property 1: intersection of convex sets is convex

- **Theorem**: if $C_1, C_2, \ldots, C_\ell$ are all convex, then $C_1 \cap C_2 \cap \cdots \cap C_\ell$ is convex
- proof: take $x, y$ in the intersection. they're in every $C_i$. Each $C_i$ is convex, so $\lambda x + (1-\lambda)y \in C_i$ for each $i$. Therefore it's in the intersection.
- this works for any number of sets, even infinitely many
- important: **union of convex sets is generally NOT convex**
  - example: $C_1 = \{x \leq 0\}$ and $C_2 = \{x \geq 1\}$ — both convex, but $C_1 \cup C_2$ is not (0.5 is between 0 and 1 but in neither set)
- this asymmetry matters: intersection preserves convexity, union does not
<!-- 交集保持凸性，但并集不行。这个不对称性要记住 -->

## Key property 2: sublevel sets of convex functions are convex

- **Theorem**: if $f$ is convex, then $C = \{x: f(x) \leq 0\}$ is a convex set
- more generally: $\{x: f(x) \leq \alpha\}$ is convex for any level $\alpha$
- proof: take $x, y \in C$, so $f(x) \leq 0$ and $f(y) \leq 0$
  - weighted average: $\lambda f(x) + (1-\lambda)f(y) \leq \lambda \cdot 0 + (1-\lambda) \cdot 0 = 0$
  - by convexity: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y) \leq 0$
  - so the convex combination is in $C$. done.
- intuition: "contour lines" of a convex function enclose convex regions — like a bowl, the region below any height is a disk (convex)
- a non-convex function with two valleys could have a disconnected sublevel set (not convex)
- note: some non-convex functions also have convex sublevel sets (called quasiconvex), but convex functions ALWAYS do

## Putting it together: convex constraints produce convex feasible region

- constrained optimization: $\min f(x)$ s.t. $g_1(x) \leq 0, \ldots, g_m(x) \leq 0$
- each constraint defines $C_i = \{x: g_i(x) \leq 0\}$
- if each $g_i$ is convex $\to$ each $C_i$ is convex (property 2) $\to$ $C = \cap C_i$ is convex (property 1)

$$\text{convex } g_i \xrightarrow{\text{sublevel set}} \text{convex } C_i \xrightarrow{\text{intersection}} \text{convex feasible region } C$$

- this is the "golden rule" of convex optimization: **convex objective + convex constraints = tractable**
- all the nice properties from lec 09 (local = global, stationary = optimal) still hold on convex feasible regions
- if constraints are non-convex (e.g., integer constraints), problem becomes NP-hard in general
<!-- 凸约束+凸目标=可解。这是凸优化的"黄金法则" -->

### ML examples of constrained convex optimization

- regularization: $\min L(w)$ s.t. $\|w\| \leq r$ — weight norm constraint, feasible set is a ball
- SVM: $\min \|w\|^2$ s.t. $y_i(w^Tx_i + b) \geq 1$ — linear constraints define a polyhedron
- probability distributions: $\min D_{KL}(p\|q)$ s.t. $p_i \geq 0, \sum p_i = 1$ — simplex constraint
- LASSO constraint form: $\min \|Xw - y\|^2$ s.t. $\|w\|_1 \leq t$ — $\ell_1$ ball is convex
- in all cases: constraints are convex, so feasible region is convex, so we can use PGD

## Strict convexity and uniqueness

- **Strict convexity**: $f(\lambda x + (1-\lambda)y) < \lambda f(x) + (1-\lambda)f(y)$ for $x \neq y$, $\lambda \in (0,1)$
- the difference from regular convexity: strict inequality $<$ instead of $\leq$
- geometrically: a "truly curved bowl" with no flat segments anywhere
- a convex but not strictly convex function can have flat parts (e.g., $f(x) = |x|$ near origin... wait, actually $|x|$ is strictly convex. better example: $f(x) = 0$ constant function is convex but not strictly convex)

| | Convex | Strictly convex |
|---|---|---|
| Inequality | $\leq$ | $<$ (strict) |
| Optimal solution | may be multiple | **unique** |
| Example | $f(x) = \max(0, x)$ on $[-1,0]$ | $f(x) = x^2$ |

### Uniqueness theorem

- **Theorem**: strictly convex function on a convex set has a **unique** minimizer
- proof by contradiction: assume two optima $x^*, y^*$ with same value $z^*$
  - their convex combination $c = \lambda x^* + (1-\lambda)y^*$ is feasible (convex set, both points in it)
  - strict convexity: $g(c) < \lambda g(x^*) + (1-\lambda)g(y^*) = \lambda z^* + (1-\lambda)z^* = z^*$
  - but $z^*$ was supposed to be optimal! $c$ achieves strictly lower value. contradiction!
- same proof technique as lec 09 (local=global) — "construct convex combination to get contradiction"
- this trick keeps coming back: whenever you want to prove something about convex optimization, try taking two candidate points and forming their convex combination
<!-- 反证法 + 凸组合构造矛盾，这个技巧反复出现 -->

## Why uniqueness matters: well-defined projection

- $\ell_2$ regularization: $f(x) + \frac{\lambda}{2}\|x\|^2$ is strictly convex even if $f$ is just convex
- ridge regression always has unique solution — bonus of $\ell_2$ beyond preventing overfitting
- more importantly: the **projection operator** needs uniqueness to be well-defined
- projection asks "what's the closest feasible point?" — that's $\min_{y \in \mathbb{X}} \|y-x\|^2$
- $\|y-x\|^2$ is strictly convex in $y$, so on convex set $\mathbb{X}$ the answer is unique

## Projected Gradient Descent (PGD)

### The problem

- in constrained optimization $\min_{x \in \mathbb{X}} f(x)$, standard GD update $x^{k+1} = x^k - \gamma\nabla f(x^k)$ might land outside $\mathbb{X}$
- even if current $x^k$ is feasible, the gradient step can violate constraints
- we can't just ignore the constraints — the optimal solution might be on the boundary

### The solution: two-step process

1. **Gradient step**: $y^{k+1} = x^k - \gamma\nabla f(x^k)$ (might be outside $\mathbb{X}$)
2. **Projection step**: $x^{k+1} = \Pi_\mathbb{X}(y^{k+1})$ (pull back to feasible region)

- **Projection operator**: $\Pi_\mathbb{X}(x) = \arg\min_{y \in \mathbb{X}} \|y - x\|^2$
- finds the closest point in $\mathbb{X}$ to $x$
- if $x \in \mathbb{X}$ already, $\Pi_\mathbb{X}(x) = x$ (no projection needed)
- when $\mathbb{X} = \mathbb{R}^n$ (unconstrained), projection is identity, PGD reduces to standard GD
<!-- 投影 = 找可行域中离你最近的点。PGD = GD + 投影 -->

### Projection cost for common constraint sets

| Feasible set $\mathbb{X}$ | Projection formula | Cost |
|---|---|---|
| Ball $\|x\| \leq r$ | $\Pi(x) = rx/\max(\|x\|, r)$ | $O(n)$ |
| Non-negative $x \geq 0$ | $\Pi(x) = \max(x, 0)$ elementwise | $O(n)$ |
| Hyperplane $a^Tx = b$ | $\Pi(x) = x + \frac{b-a^Tx}{\|a\|^2}a$ | $O(n)$ |
| Simplex | sort + threshold | $O(n\log n)$ |
| General convex set | iterative solver needed | expensive |

- good news: most ML constraints (norm, non-negativity, simplex) have cheap projections
- bad news: general convex sets might need solving another optimization just for the projection

### Key property: non-expansiveness

- $\|\Pi_\mathbb{X}(x) - \Pi_\mathbb{X}(y)\| \leq \|x - y\|$ — projection never amplifies distance
- two points close before projection stay close after projection
- this is crucial for convergence analysis: projection step doesn't destroy progress made by gradient step
- PGD on smooth convex functions achieves same $O(1/K)$ rate as standard GD (lec 11)

### ML applications of PGD

- weight norm constraint: after GD step, if $\|w\| > r$, rescale to $w \leftarrow rw/\|w\|$
- non-negative matrix factorization: after GD step, clip negatives to zero: $W \leftarrow \max(W, 0)$
- WGAN weight clipping: clip discriminator weights to $[-c, c]$ — projection onto a box

## Looking back at the course arc (lec 01-12)

now lec 03's convexity definition, lec 07's Taylor theorem, lec 09's first-order characterization, and lec 10-11's convergence theory all come together with today's constrained optimization:

| Lectures | Topic | Role |
|---|---|---|
| 01-04 | Convex sets, convex functions, GD intro | Foundation |
| 05-08 | Lipschitz, step sizes, theory | Tools |
| 09 | First-order characterization | Bridge theorem |
| 10 | $O(1/\sqrt{K})$ baseline convergence | First rate |
| 11 | $O(1/K)$ with smoothness | Improved rate |
| **12** | **Constraints + PGD** | **Practical extension** |

- we've built from definitions to a complete toolkit: unconstrained and constrained, with convergence guarantees
- the theory arc is complete: convex sets $\to$ convex functions $\to$ GD convergence $\to$ constrained GD
- next up: SGD, acceleration, and more advanced methods
<!-- 从定义到工具到收敛到约束，整个理论弧线完成了。接下来是SGD和加速方法 -->
