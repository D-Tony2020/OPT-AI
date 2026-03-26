# Lec 16 - Mar 23 - L1-Ball Projection

## The Hardest Projection: L1-Ball

- X = {x : sum |x_i| <= R} — the L1-ball, diamond/rhombus shape in 2D
- this is the constraint behind LASSO from the very first lecture! min ||Xw - y||^2 s.t. ||w||_1 <= R
- so the course has come full circle: lec 01 introduced LASSO as a motivating example and I had no idea how to solve it. Now in lec 16 I actually have the full machinery
- the "sharp corners" of the L1-ball along coordinate axes is what causes sparsity — optimization solutions tend to land on corners where some coordinates are exactly zero
- this is the hardest projection we've seen so far
<!-- L1-ball投影连接回第一讲的LASSO！整个课程的闭环 -->

## Why L1-Ball Is Harder

| Set | Constraint | Difficulty |
|-----|-----------|------------|
| Box | 0 <= x_i <= U_i | fully separable, trivial |
| L2-ball | sum x_i^2 <= R^2 | coupled but smooth -> Lagrangian works directly |
| Simplex | sum x_i = 1, x_i >= 0 | coupled equality + non-negativity -> Lagrangian + truncation |
| L1-ball | sum \|x_i\| <= R | absolute value is **non-smooth** + coupled inequality |

- the absolute value |x_i| is not differentiable at x_i = 0 — can't directly take derivative and set to zero
- can't directly apply Lagrangian to sum|x_i| = R as an equality constraint (non-smooth)
- the strategy: don't fight the difficulty head-on, **reduce** it to a solved problem (simplex projection from lec 15)
- this is the art of mathematics — transform hard problems into easy ones through clever observations

## Observation 1: Can Assume y_hat_i >= 0

- L1-ball is symmetric about every coordinate axis: flipping x_j -> -x_j doesn't change sum|x_i|
- so if y_hat_j < 0, flip it to |y_hat_j|, do the projection in the non-negative "quadrant", then flip the result back
- formally: the two problems min sum(x_i - y_hat_i)^2 s.t. sum|x_i| <= R and min sum(x_i - |y_hat_i|)^2 s.t. sum|x_i| <= R have the same optimal value

### Proof of Observation 1

- suppose y_hat_j < 0. given optimal x* for the first problem, construct x_tilde: flip j-th coordinate (x_tilde_j = -x_j*), keep others
- feasibility: sum|x_tilde_i| = sum|x_i*| <= R (absolute values are flip-invariant)
- objective value: (-x_j* - (-y_hat_j))^2 = (-x_j* + y_hat_j)^2 = (x_j* - y_hat_j)^2, so same objective
- so x_tilde is feasible for second problem with same objective -> OPT(second) <= OPT(first)
- by symmetric argument (reverse direction): OPT(first) <= OPT(second)
- therefore OPT(first) = OPT(second). QED
- this "construct feasible solution + compare objectives" is a standard optimization proof pattern
<!-- 利用L1-ball的坐标轴对称性，先把所有分量取绝对值 -->

## Observation 2: Non-negative y_hat -> Non-negative Projection

- claim: if y_hat_i >= 0 for all i, then the projection Pi_X(y_hat) also has all non-negative components
- this is critical: combined with Obs 1, it means |x_i| = x_i in the constraint, removing the absolute values!

### Proof of Observation 2

- proof by contradiction: suppose x_j* < 0 but y_hat_j >= 0
- construct competitor x_tilde: set x_tilde_j = 0, keep others the same
- feasibility: sum|x_tilde_i| = sum_{i!=j}|x_i*| + 0 <= sum_{i!=j}|x_i*| + |x_j*| = sum|x_i*| <= R
  - actually uses strictly less L1 budget (since |x_j*| > 0)
- objective comparison for j-th coordinate:
  - x_tilde: (0 - y_hat_j)^2 = y_hat_j^2
  - x*: (x_j* - y_hat_j)^2 = (|x_j*| + y_hat_j)^2 (since x_j* < 0, y_hat_j >= 0)
  - clearly y_hat_j^2 < (|x_j*| + y_hat_j)^2 because |x_j*| > 0
- so x_tilde is feasible and has strictly better objective -> x* can't be optimal. Contradiction!
- intuition: if the target is non-negative, going negative is always wasteful — 0 is always strictly closer than any negative number to a non-negative target

## The Reduction Chain (The Beautiful Part)

- combining both observations step by step:
  1. **Obs 1**: replace y_hat_i with |y_hat_i| -> now all components non-negative
  2. **Obs 2**: since |y_hat_i| >= 0, optimal x_i* >= 0, so |x_i*| = x_i* -> constraint becomes sum x_i <= R with x_i >= 0
  3. **Boundary argument**: when ||y_hat||_1 > R (y_hat outside X), the inequality sum x_i <= R must be tight -> sum x_i = R
     - why? if sum x_i* < R, we have slack, and we could increase some x_i toward |y_hat_i| to reduce objective — contradiction with optimality
  4. **Reduced problem**: min sum(x_i - |y_hat_i|)^2 s.t. sum x_i = R, x_i >= 0
- this is exactly a **scaled probability simplex projection** (R instead of 1)!
- directly apply lec 15 formula: find lambda* from sum[|y_hat_i| - lambda*/2]+ = R, then x_i = [|y_hat_i| - lambda*/2]+
- the way everything reduces to simpler subproblems is really beautiful — L1-ball -> simplex -> Lagrangian -> separable 1D problems

## L1-Ball Projection: Full Formula

- find lambda* such that sum [|y_hat_i| - lambda*/2]+ = R
- then: **x_i* = sign(y_hat_i) * [|y_hat_i| - lambda*/2]+**

- this is a three-step operation:
  1. **take absolute value**: |y_hat_i| (handle signs, from Obs 1)
  2. **soft-threshold**: [|y_hat_i| - tau]+ where tau = lambda*/2 (simplex projection from lec 15)
  3. **restore sign**: multiply by sign(y_hat_i) (undo the absolute value step)

- this is exactly the **soft-thresholding operator** from LASSO / proximal gradient methods!
  - if |y_hat_i| > tau: x_i* = sign(y_hat_i) * (|y_hat_i| - tau) — shrink magnitude by tau, keep sign
  - if |y_hat_i| <= tau: x_i* = 0 — magnitude too small, crushed to zero
- the soft-thresholding is how L1 regularization induces sparsity: small components are killed, large ones are uniformly shrunk
- computation: O(n log n) — same as simplex projection (dominated by sorting to find lambda*)

## Complete Projection Comparison (All Four)

| Set | Formula | Key Trick | O(?) |
|-----|---------|-----------|------|
| Box | clip(y_i, 0, U_i) | separability | O(n) |
| L2-ball | R*y/\|\|y\|\| | Lagrangian + rotational symmetry | O(n) |
| Simplex | [y_i - lambda*/2]+ | Lagrangian + truncation | O(n log n) |
| L1-ball | sign(y_i)*[\|y_i\|-lambda*/2]+ | symmetry reduction -> simplex | O(n log n) |

- I need to memorize all four of these for the exam
- the difficulty increases left to right, but each builds on the previous
- PGD is just GD + projection, and the convergence proof uses the same framework from lec 10-11 — just add Pythagorean inequality
- all four projections are efficient enough that PGD has essentially the same per-iteration cost as unconstrained GD

## Why the Inequality Becomes Equality

- after the two observations, we have: min sum(x_i - |y_hat_i|)^2 s.t. sum x_i <= R, x_i >= 0
- but why does sum x_i <= R become sum x_i = R at the optimum?
- intuition: |y_hat_i| are all non-negative, and sum|y_hat_i| > R (y_hat is outside the ball)
- if sum x_i* < R, then there's slack — we could increase some x_i closer to |y_hat_i| without violating the constraint
- since |y_hat_i| >= x_i* >= 0 (we're projecting FROM |y_hat_i| which is further out), moving x_i toward |y_hat_i| reduces (x_i - |y_hat_i|)^2
- so strict inequality cannot hold at the optimum -> must have equality: sum x_i = R
- this is the same "active constraint" reasoning we saw for the L2-ball in lec 15

## Connections Across the Entire Course

- lec 01: LASSO introduced as motivation -> lec 16: L1-ball projection lets us solve it with PGD
- lec 02: convex sets defined -> lec 13: convexity of X guarantees unique projection
- lec 03-04: first-order characterization of convex functions -> lec 13-14: same tool drives convergence proof
- lec 06: penalty method -> lec 15: Lagrangian is the rigorous version
- lec 10-11: convergence proof framework for GD -> lec 13-14: same framework + projection properties = PGD convergence
- lec 15: simplex projection -> lec 16: L1-ball reduces to simplex

## Mid-Semester Reflection

Looking back at what we've covered so far, it's been quite a journey:

- **Lec 01-04**: foundations — what is optimization, convex sets, convex functions, GD basics. LASSO was introduced as a motivating example and I had no idea how to actually solve it efficiently.
- **Lec 05-08**: GD theory — convergence analysis, Lipschitz conditions, step size choices, strong convexity. I learned that convergence proofs all follow the same template: convexity bound -> algebraic expansion -> telescoping sum.
- **Lec 09-12**: convergence rates, SGD. Understood the O(1/K) vs O(1/sqrt(K)) distinction and when each applies.
- **Lec 13-16**: constrained optimization with PGD. The same convergence proof framework works, just with projection added. Then four projection formulas, increasing in complexity but each building on the previous one.

The arc from "what is optimization" in lec 01 to "here's how to efficiently solve constrained sparse optimization problems with convergence guarantees" by lec 16 is pretty satisfying. The most satisfying realization: L1-ball projection reduces to simplex projection (lec 15), which uses the Lagrangian method (lec 15), which plugs into PGD (lec 13-14), which has convergence guarantees from the same proof framework as vanilla GD (lec 05-08, 10-11). Everything is connected.

At this point I have a pretty solid toolbox: from the simplest unconstrained gradient descent to projected gradient descent that can handle LASSO, ridge regression, probability constraints, and more — all with provable convergence rates. Curious to see what comes next.

<!-- 从第一讲的"什么是优化"到第十六讲的约束稀疏优化，知识链很完整 -->

## Key Takeaways

1. L1-ball projection: sign(y_hat_i) * [|y_hat_i| - lambda*/2]+, where lambda* from sum[|y_hat_i| - lambda*/2]+ = R
2. two observations reduce L1-ball to simplex: (1) sign symmetry -> assume non-negative, (2) non-negative input -> non-negative output -> remove absolute values
3. soft-thresholding is the mechanism behind L1 sparsity: small components go to exact zero, large ones shrink uniformly
4. the "construct feasible solution + compare objective" proof pattern is used throughout (Obs 1 and Obs 2) — standard in optimization
5. so far: unconstrained GD -> convergence theory -> constrained PGD -> four projections (box -> L2-ball -> simplex -> L1-ball), complexity increases but concepts stay unified
6. L1-ball projection connects back to LASSO from lec 01 — things are coming full circle
