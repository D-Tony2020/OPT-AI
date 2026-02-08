# Lec 5 - Feb 4 - 1D Optimization: Bisection & Golden Section Search

## Why are we learning 1D optimization now?

- continuing from lec 4 where we introduced gradient descent
- in GD, we know the direction (negative gradient), but how far to step?
- choosing step size alpha is actually a 1D optimization problem: min over alpha of f(x^k - alpha * grad f(x^k))
- so lec 5 is a "tool lecture" -- we build the 1D search tools that lec 6 will plug into GD
- ah so THIS is why professor paused the multi-dimensional stuff, we need these 1D methods first

<!-- 这节课是工具课，为 lec 6 的 line search 做准备 -->

## The interval reduction framework

- core assumption: f is **unimodal** (one valley, like a bowl shape)
- unimodal means: unique minimum x*, f is decreasing on the left of x* and increasing on the right
- we maintain an "interval of uncertainty" [a^k, b^k] that contains the minimum
- pick two test points lambda^k < rho^k inside the interval
- compare f(lambda^k) vs f(rho^k):
  - if f(lambda) <= f(rho): minimum is not to the right of rho, so new interval = [a, rho]
  - if f(rho) < f(lambda): minimum is not to the left of lambda, so new interval = [lambda, b]
- each step we eliminate part of the interval -- it's an exclusion method
- the framework does not specify WHERE to place lambda and rho -- different placements give different algorithms
- the unimodal assumption is key. if function has multiple local minima this breaks
- but convex functions (from lec 3) are automatically unimodal, so we are good
- this connects back to why convexity matters -- convex = unimodal = interval reduction works

## Bisection search

- most natural approach: place test points near the midpoint
- lambda^k = (a^k + b^k)/2 - epsilon, rho^k = (a^k + b^k)/2 + epsilon
- each step the interval shrinks to roughly L/2 + epsilon (approximately halving)
- epsilon must be > 0 (if epsilon = 0 the two points coincide and we learn nothing)
- but epsilon should be small, like 10^{-6}, much smaller than initial interval
- stopping: when b^{k+1} - a^{k+1} <= theta (tolerance parameter)
- this is like the number guessing game -- always guess the middle, then eliminate half

**Convergence**: after k iterations, interval is about L_0 * (1/2)^k. Need about log_2(L_0/theta) iterations. But each iteration needs **two** function evaluations (f(lambda) and f(rho)). So total cost is ~2 * log_2(L_0/theta) evaluations.

### Bisection example

- minimize f(x) = x^2 + 2x on [-3, 5], minimum is at x* = -1 (easy to check by derivative)
- first iteration: midpoint = 1, lambda = 0.9, rho = 1.1
  - f(0.9) = 2.61, f(1.1) = 3.41, so new interval = [-3, 1.1]
- second iteration: midpoint = -0.95, lambda = -1.05, rho = -0.85
  - f(-1.05) = -0.9975, f(-0.85) = -0.9775, new interval = [-3, -0.85]
- notice: the lambda and rho from step 1 are completely thrown away in step 2
- we computed two function values, used them once, then discarded them
- this "waste" is what motivates golden section search -- can we recycle old evaluations?

## Golden section search -- the elegant solution

<!-- 黄金分割法很优雅，核心思想是复用试探点 -->

- the big question: can we place test points so that one of them carries over to the next iteration?
- we want TWO properties:
  1. **point reuse**: after shrinking, one old test point becomes a new test point
  2. **constant ratio**: interval shrinks by a fixed ratio beta each step

- these two constraints uniquely determine beta!

### Deriving beta

- the approach is very mathematical: state the properties you want, then solve for what satisfies them
- from the constant ratio condition: rho^k divides [a^k, b^k] at ratio beta:(1-beta), lambda^k at (1-beta):beta
- from the reuse condition in case 1: old rho becomes new lambda in the shrunken interval
- combining these gives the equation: 1 - beta = beta^2
- rearranging: beta^2 + beta - 1 = 0
- solving with quadratic formula: beta = (-1 + sqrt(5)) / 2 ≈ 0.618
- this is the golden ratio conjugate! (golden ratio phi ≈ 1.618, and beta = 1/phi = phi - 1)
- the self-similarity property of golden ratio is WHY point reuse works
- fun fact: 0.618^2 = 0.382 = 1 - 0.618. the ratio reproduces itself at every scale
- verified for both cases (case 1 and case 2) -- same equation, same beta. the symmetry is beautiful
- if the two cases gave different beta values, the algorithm wouldn't exist. luckily they agree

### How it works in practice

- example with [0, 10]:
  - lambda = 0.618*0 + 0.382*10 = 3.82, rho = 0.382*0 + 0.618*10 = 6.18
  - suppose f(3.82) > f(6.18), new interval = [3.82, 10], length ≈ 6.18
  - in new interval: lambda_new = 6.18 (= old rho!), only need to compute rho_new = 7.64
  - **one function evaluation saved per step**

- the algorithm is nearly identical to bisection, just different placement of test points
- IMPORTANT: must remember to reuse the old function value! if you forget, you lose the whole point

### Efficiency comparison

| Method | Evals/step | Shrink ratio | Total evals for precision theta |
|--------|-----------|-------------|-------------------------------|
| Bisection | 2 | ~0.5 | ~2 log_2(L_0/theta) |
| Golden section | 1 (first step 2) | 0.618 | ~2.08 log_2(L_0/theta) |

- per function evaluation, they are nearly the same efficiency
- but golden section is cleaner: no epsilon parameter, fixed shrink ratio, simpler implementation

## Derivative-based 1D search

- if we can compute f'(x), we can do even better
- evaluate f' at the midpoint:
  - f'(mid) < 0: function still decreasing, minimum is to the right -> keep right half
  - f'(mid) > 0: function increasing, minimum is to the left -> keep left half
  - f'(mid) = 0: we found the minimum!
- each step: ONE derivative evaluation, interval EXACTLY halved
- this is strictly better than bisection (which needs 2 evals for approximate halving)
- but requires differentiability -- not always available (black-box optimization, simulations)

**Big picture -- hierarchy of methods**:
- zero-order (no derivatives): bisection, golden section -- only need function values
- first-order (gradient): derivative-based search -- need f'
- this same hierarchy exists in multi-dimensional optimization: gradient-free vs GD vs Newton's method
- more information = more efficiency. this principle keeps coming back in this course

<!-- 零阶方法 vs 一阶方法，这个分类在多维优化里也一样 -->

## When to use which method

- bisection: simplest to implement, good when function evals are cheap
- golden section: best zero-order method, no epsilon parameter, clean implementation
- derivative-based: most efficient IF you have access to derivatives
- in ML context: function evaluation might mean running a full training loop (expensive!), so saving evaluations really matters
- for line search in GD, we can use any of these to find optimal step size alpha

## Looking forward

- all three methods here are "subroutines" for the line search in gradient descent
- lec 6 will directly use these to find optimal step size in GD
- I think the key insight from this lecture is: even though GD lives in R^n, step size selection is always a 1D problem -- dimension reduction!
- golden section is elegant. the fact that the golden ratio appears naturally from the reuse + constant-ratio constraints is really satisfying
- continuing the line search idea from lec 4 -- now we have concrete algorithms to do it
- the "information reuse" idea in golden section reminds me of how momentum methods reuse past gradients (I think we will see this later)
