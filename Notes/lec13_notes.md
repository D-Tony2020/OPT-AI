# Lec 13 - Mar 11 - Projected Gradient Descent

## Big Picture: From Unconstrained to Constrained

- this lecture is where the course pivots from unconstrained to constrained optimization
- for 12 lectures we've been doing min f(x) over all of R^n, GD can move freely anywhere
- but real ML problems almost always have constraints: weights must be nonneg, norm bounded, probability distributions sum to 1, etc.
- today we introduce Projected Gradient Descent (PGD) and prove the key mathematical properties that make it work
- this is the beginning of the last block (lec 13-16) of the course, and honestly everything from lec 01-12 was building toward this
<!-- 终于从无约束进入约束优化了，这是课程后半段的核心 -->

## Projected Gradient Descent Algorithm

- the idea is very natural: "two-step" strategy
  - **gradient step**: ignore constraints, do normal GD: y^{k+1} = x^k - gamma * grad f(x^k)
  - **projection step**: if y^{k+1} landed outside the feasible set X, find the closest point in X
    - x^{k+1} = Pi_X(y^{k+1}) = argmin_{x in X} ||x - y||^2
- notation: y^{k+1} is the "raw" gradient step result (might be infeasible), x^{k+1} is the projected result (always feasible)
- so the actual iterates x^0, x^1, x^2, ... always stay inside X
- this is basically the same convergence proof framework from lec 10-11, just with an extra projection step added
- the two conditions we need: f is convex AND X is convex
  - convex X guarantees unique projection (strict convexity of ||x-y||^2 + convex X)
  - convex f guarantees local min = global min
<!-- 跟普通GD的区别就是多了一步投影，把跑出去的点拉回来 -->

## Projection Definition

- Pi_X(y) = argmin_{x in X} ||x - y||^2 — find the point in X closest to y
- since ||x-y||^2 is strictly convex (Hessian = 2I) and X is convex, the solution is unique
- that's why we can write Pi_X(y) as a function — one input, one output
- note that computing Pi_X is itself an optimization problem — if X is complicated, this could be expensive
- the practical question: is projection easy to compute? for common ML constraints (box, L2-ball, simplex, L1-ball), yes — lectures 14-16 will show this
- examples of constraints in ML:
  - ridge regression: ||w||_2 <= R (Euclidean ball)
  - LASSO: ||w||_1 <= R (L1-ball)
  - softmax output: sum p_i = 1, p_i >= 0 (probability simplex)
  - NMF: W >= 0, H >= 0 (box constraint)

## Geometric Intuition

- imagine standing inside a convex-shaped fenced area (X) at position x^k
- gradient tells you the steepest uphill direction, you walk the opposite way
- you step to y^{k+1} — but you might have stepped outside the fence
- projection = walk to the nearest point on the fence (or stay if you're still inside)
- the question: does getting "pulled back" by projection ruin the convergence? the next few pages prove it doesn't

## Obtuse Angle Property (Key Result 1)

- for convex X, any x in X, any y in R^n, let pi = Pi_X(y):
  - (x - pi)^T (y - pi) <= 0
- meaning: from the projection point pi, looking toward x (inside X) and toward y (outside X), the angle between these two directions is >= 90 degrees (obtuse)
- geometrically: the feasible set and the original point are on "opposite sides" from the projection point
- this relates to the supporting hyperplane concept from lec 03: at the projection point, there's a hyperplane with normal (y-pi) that separates X from y
- this is the foundational property — everything else builds on it

## Proof of Obtuse Angle Property

- proof uses the "convex combination perturbation" technique:
  1. for any lambda in [0,1), the point lambda*x + (1-lambda)*pi is in X (by convexity of X)
  2. since pi is the closest point to y in X: ||y - pi||^2 <= ||y - (lambda*x + (1-lambda)*pi)||^2
  3. rewrite RHS as ||y - pi + lambda*(pi - x)||^2
  4. apply Taylor expansion of g(v) = ||v||^2 at v = y-pi with perturbation delta = lambda*(pi-x)
  5. using grad(||v||^2) = 2v, we get: ||y-pi||^2 + 2*lambda*(y-pi)^T(pi-x) + o(lambda)
  6. cancel ||y-pi||^2 from both sides: 0 <= 2*lambda*(y-pi)^T(pi-x) + o(lambda)
  7. divide by lambda > 0: 0 <= 2*(y-pi)^T(pi-x) + o(lambda)/lambda
  8. take lambda -> 0, the o()/lambda term vanishes (careful analysis: o(lambda*||pi-x||)/(lambda*||pi-x||) -> 0, times ||pi-x|| constant)
  9. result: (y-pi)^T(pi-x) >= 0, equivalently (x-pi)^T(y-pi) <= 0
- this "perturb + Taylor + limit" technique is standard in convex optimization — same style as the first-order characterization proof from lec 03-04
- note: for g(x)=||x||^2 specifically, the higher order term is exactly lambda^2*||pi-x||^2, so we don't strictly need the limit. But the o() version generalizes to other convex functions
<!-- 这个证明技巧跟lec 03-04证凸函数一阶特征化的方法一样，扰动+Taylor+取极限 -->

## Pythagorean Inequality (Key Result 2)

- direct corollary of obtuse angle property:
  - ||x - y||^2 >= ||x - Pi_X(y)||^2 + ||y - Pi_X(y)||^2
- proof: decompose x-y = (x-pi) + (pi-y), expand the norm squared
  - ||x-y||^2 = ||x-pi||^2 + ||pi-y||^2 + 2(x-pi)^T(pi-y)
  - = ||x-pi||^2 + ||y-pi||^2 - 2(x-pi)^T(y-pi)
  - by obtuse angle property, (x-pi)^T(y-pi) <= 0, so -2(x-pi)^T(y-pi) >= 0
  - therefore ||x-y||^2 >= ||x-pi||^2 + ||y-pi||^2
- this is like Pythagorean theorem but with >= instead of = (because the angle at pi is obtuse, not right)
- **critical meaning**: projection does NOT increase distance to any feasible point
  - specifically: ||y^{k+1} - x*||^2 >= ||x^{k+1} - x*||^2 (since x* in X)
  - so projecting can only bring you closer to the optimum, never farther
- this is the mathematical guarantee that projection "doesn't break" GD convergence
- I can see this as a direct extension of the convergence framework from lec 10-11, just with this extra inequality to handle the projection step
- this property is called "firm non-expansiveness" in the literature

## Convergence Analysis Setup

- same strategy as unconstrained GD (lec 05-08):
  - step 1: convexity gives f(x^k) - f(x*) <= grad f(x^k)^T (x^k - x*)  (first-order characterization, lec 03-04)
  - step 2: substitute grad f(x^k) = (x^k - y^{k+1})/gamma  (from the GD update rule)
  - step 3: use polarization identity: 2a^T b = ||a||^2 + ||b||^2 - ||a-b||^2
- after expanding with a = x^k - y^{k+1}, b = x^k - x*:
  - grad f(x^k)^T(x^k - x*) = (1/2gamma)(||x^k - y^{k+1}||^2 + ||x^k - x*||^2 - ||y^{k+1} - x*||^2)
  - = (1/2gamma)(gamma^2 * ||grad f(x^k)||^2 + ||x^k - x*||^2 - ||y^{k+1} - x*||^2)
- the ||x^k - x*||^2 - ||y^{k+1} - x*||^2 terms would telescope if y^{k+1} = x^{k+1}
- but y^{k+1} != x^{k+1} (pre vs post projection)!
- **Pythagorean inequality saves us**: ||y^{k+1} - x*||^2 >= ||x^{k+1} - x*||^2, so -||y^{k+1} - x*||^2 <= -||x^{k+1} - x*||^2
- replacing y^{k+1} with x^{k+1} restores the telescoping structure (at the cost of a looser bound)
- this is exactly where the projection properties pay off
<!-- 投影的勾股不等式让伸缩求和结构成立，这是关键的一步 -->

## Understanding the Three Terms

- the expansion gives us: (1/2gamma)(gamma^2*||g^k||^2 + ||x^k - x*||^2 - ||y^{k+1} - x*||^2)
- first term: gamma^2*||g^k||^2 = ||x^k - y^{k+1}||^2 — how far the gradient step moved us, the "step size squared"
- second term: ||x^k - x*||^2 — current distance to optimum, "how far we still are"
- third term: -||y^{k+1} - x*||^2 — distance after gradient step
- the second and third terms together measure "did this step bring us closer to x*?"
- when we sum over k, the second/third terms telescope (after replacing y with x via Pythagorean)
- the first term accumulates into a "total gradient energy" term that scales with gamma

## Comparison with Unconstrained GD

- in unconstrained GD (lec 05-08): y^{k+1} = x^{k+1}, so no gap between gradient step and iterate
- in projected GD: y^{k+1} != x^{k+1}, the projection creates a gap
- the Pythagorean inequality bridges this gap: ||y^{k+1} - x*||^2 >= ||x^{k+1} - x*||^2
- this means projection can only help (bring us closer to x*), never hurt
- so PGD inherits the convergence guarantees of unconstrained GD — the proof goes through with minimal modification
- this is very satisfying: adding constraints doesn't fundamentally change the convergence story

## What's Coming Next

- lec 14 will complete the telescoping sum and give the final convergence rate O(RB/sqrt(K))
- then lec 14-16 will show how to actually compute projections for box, L2-ball, simplex, and L1-ball
- the hard mathematical infrastructure is done in this lecture — next lecture is "harvesting"
- the four projections form a nice progression: box (trivial) -> L2-ball (Lagrangian) -> simplex (Lagrangian + truncation) -> L1-ball (reduce to simplex)

## ML Applications for PGD

- ridge regression: min ||Xw - y||^2 s.t. ||w||_2 <= R -> PGD with L2-ball projection (lec 15)
- LASSO: min ||Xw - y||^2 s.t. ||w||_1 <= R -> PGD with L1-ball projection (lec 16)
- non-negative matrix factorization: min ||A - WH||^2 s.t. W >= 0, H >= 0 -> PGD with box projection (lec 14)
- probability estimation: min loss s.t. p in simplex -> PGD with simplex projection (lec 15)
- adversarial attacks (PGD attack): min -loss(x+delta) s.t. delta in L_inf ball -> PGD with box projection
- all these use the same algorithm framework, just different projections — that's the power of the modular design

## Key Takeaways

1. PGD = gradient step + projection step, the natural way to handle convex constrained optimization
2. obtuse angle property: (x-pi)^T(y-pi) <= 0 — projection separates feasible points from the original point
3. Pythagorean inequality: ||x-y||^2 >= ||x-pi||^2 + ||y-pi||^2 — projection only shrinks distances
4. the convergence analysis parallels unconstrained GD exactly, with projection properties filling the gap — same proof technique works!
5. convex combination perturbation proof technique = same tool from lec 03-04, very reusable
6. the two key properties (obtuse angle -> Pythagorean) form the mathematical foundation for everything in lec 14-16
