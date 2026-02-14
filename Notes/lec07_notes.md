# Lec 7 - Feb 11 - Barrier Method & Taylor's Theorem

## Two big topics today

- first half: barrier function method (the "dual" of penalty method from lec 6)
- second half: Taylor's theorem -- this is the math foundation for proving GD convergence
- these seem unrelated but they are connected: barrier is the last "crude" constrained method, then Taylor opens the door to convergence theory

<!-- 本讲分两半：约束优化的收尾 + 收敛分析的开始 -->

## Barrier function method (Interior Point Method)

- lec 6's penalty method works from the OUTSIDE -- iterates can be infeasible, penalty pushes them in
- barrier method works from the INSIDE -- iterates are always feasible, barrier prevents them from leaving
- this is like penalty but from the other direction. professor's analogy: penalty = electric fence (get shocked if you cross), barrier = cliff edge (you physically cannot go over)

### The math

- only works for **inequality** constraints: min f(x) s.t. g_i(x) <= 0
- (no equality constraints -- that's a limitation compared to penalty method)
- inside the feasible region: g_i(x) < 0, so -g_i(x) > 0 (small positive number near boundary)
- barrier term: -mu_i / g_i(x), which goes to +infinity as g_i(x) -> 0^-
- total problem: min f(x) - sum mu_i / g_i(x)

The intuition:
- far from boundary: g_i(x) << 0, barrier term is tiny, almost no effect
- near boundary: g_i(x) -> 0^-, barrier term -> +infinity, creates an "infinite wall"
- the wall prevents iterates from ever leaving the feasible region

### Comparison with penalty method

| | Penalty (lec 6) | Barrier (lec 7) |
|--|----------------|-----------------|
| Iterates | can be infeasible | always feasible |
| Direction | outside -> push in | inside -> pull toward boundary |
| Parameter | theta increases | mu decreases |
| Constraints | equality + inequality | inequality only |
| Initial point | any | must be strictly feasible |

- barrier is better for safety-critical applications (robotics, control) because every intermediate solution is feasible
- but finding an initial feasible point can itself be hard

<!-- 障碍法每步都可行，这对安全关键场景很重要 -->

### The algorithm

1. pick initial mu's, tolerance epsilon, shrink factor beta in (0,1), find a strictly feasible initial point x^0
2. solve: x^k = argmin f(x) - sum mu_i / g_i(x), using GD starting from x^{k-1} (warm start)
3. if barrier term <= epsilon, stop. otherwise mu_i *= beta, repeat

- mu starts large (thick wall far from boundary) and gradually shrinks (thin wall, solution approaches boundary)
- the path traced by x*(mu) as mu decreases is called the **central path** -- modern interior point methods follow this path efficiently

### Same example as lec 6

- min (x1-2)^4 + (x1-2x2)^2 s.t. x1^2 - x2 <= 0
- barrier version: min (x1-2)^4 + (x1-2x2)^2 - mu/(x1^2 - x2)
- initial point must satisfy x2 > x1^2, e.g., (0, 1)
- penalty method's intermediate solutions may have x1^2 > x2 (infeasible), barrier's never do
- in practice, log-barrier form is more common: -mu * sum ln(-g_i(x)), numerically more stable
- this is what CVXPY solvers (SCS, ECOS) use internally
- so when you call `prob.solve()` in CVXPY, it's probably running a log-barrier interior point method underneath

## Taylor's theorem -- the math gets heavier but I can follow

- from here the course shifts from "how to optimize" to "why optimization works"
- Taylor's theorem is THE tool for proving GD convergence. I think this is setup for lec 8

### Little-o notation

- o(t) means a function g(t) such that lim_{t->0} g(t)/t = 0
- in words: g(t) vanishes faster than t does as t -> 0

Three examples:
- t^2: is o(t)? yes, t^2/t = t -> 0. higher order, vanishes faster
- 2t: is o(t)? no, 2t/t = 2 (same order, not faster)
- sqrt(t): is o(t)? no, sqrt(t)/t = 1/sqrt(t) -> infinity (lower order, vanishes slower)

Order hierarchy as t -> 0: ... << t^3 << t^2 << t << sqrt(t) << 1

Why this matters: in Taylor's corollary, o(||y-x||) represents the error of linear approximation. Knowing it's o (not O) means the error is negligible compared to step size.

### Taylor's theorem statement

- for smooth f: R^n -> R:

$$f(y) = f(x) + \nabla f(\gamma x + (1-\gamma)y)^T (y-x) \quad \text{for some } \gamma \in (0,1)$$

- this is the **multidimensional mean value theorem**
- it's an EXACT equation (no approximation!), but gamma is unknown
- the exactness is nice but the unknown gamma limits usefulness -- that's why we need the corollary
- geometrically: the secant slope equals the tangent slope at some intermediate point
- reminds me of the 1D mean value theorem from calculus, this is just the vector version

### Taylor's theorem corollary (the useful version)

$$f(y) = f(x) + \nabla f(x)^T (y-x) + o(\|y-x\|)$$

- this replaces the unknown-gamma version with a known gradient at x, plus a small error
- the proof structure is similar to what we did before: split nabla f(c) = nabla f(x) + (nabla f(c) - nabla f(x))

**Proof sketch**:
1. start from Taylor: f(y) = f(x) + nabla f(c)^T (y-x)
2. add and subtract nabla f(x): = f(x) + nabla f(x)^T(y-x) + (nabla f(c) - nabla f(x))^T(y-x)
3. the remainder = ||nabla f(c) - nabla f(x)|| * ||y-x|| * cos(phi)
4. need to show this is o(||y-x||): divide by ||y-x||, get ||nabla f(c) - nabla f(x)|| * cos(phi)
5. as y -> x, c -> x, so nabla f(c) -> nabla f(x) by gradient continuity. cos(phi) is bounded. done.

<!-- 证明的核心：梯度连续性 + cos 有界 -->

The key assumption: gradient must be continuous (f is C^1). If gradient is discontinuous, the corollary fails.

### Why this corollary is so important

- it tells us: near x, f(y) ≈ f(x) + nabla f(x)^T(y-x), with negligible error
- this is the first-order linear approximation (tangent plane)
- lec 8 will plug y = x - alpha * nabla f(x) into this and prove GD decreases function value
- I think this Taylor stuff is setup for proving GD convergence -- the pieces are coming together

## Summary

- barrier method completes the "crude" constrained optimization methods (penalty from outside, barrier from inside)
- now we have two ways to handle constraints: penalty (lec 6) and barrier (lec 7). they are like two sides of the same coin
- Taylor's theorem + corollary = the mathematical foundation for convergence analysis
- the little-o notation is a precise way to say "negligible error"
- the corollary f(y) = f(x) + grad f(x)^T(y-x) + o(||y-x||) is THE formula I need to remember
- the proof technique of "add zero, subtract zero" (split grad f(c) into grad f(x) + difference) is a pattern I should get comfortable with
- the math is getting heavier but I can follow the logic. each step makes sense on its own
- next lecture should use these tools to finally prove why gradient descent works
- I think the key insight is: Taylor corollary says "locally, f behaves like its linear approximation". if the linear approximation goes down (which it does along negative gradient), so does f
