# Lec 15 - Mar 18 - Lagrangian Method + Euclidean Ball / Simplex Projections

## The Problem: Coupled Constraints

- box projection was easy because constraints are separable (each x_i independent)
- but ML has many coupled constraints: L2-ball sum x_i^2 <= R^2, simplex sum x_i = 1
- you can't decide each x_i independently when they are tied together by a shared constraint
- solution: the Lagrangian method — absorb the coupling equality constraint into the objective, making the problem separable again
- this feels like the penalty method from lec 06 but more principled — we find the exact lambda that makes the constraint satisfied, not just an approximation
- penalty method adds lambda*(violation)^2 and hopes it's close enough; Lagrangian adds lambda*(violation) and finds the exact lambda
<!-- Lagrangian方法比penalty method更精确，找到恰好满足约束的lambda -->

## Lagrangian Method: The General Framework

- original problem (dagger): min f(x) s.t. g(x) = b, x in X
- for fixed lambda in R, solve the relaxed problem (*): min f(x) + lambda*(g(x) - b) s.t. x in X
  - note: the equality constraint g(x) = b is removed, replaced by a penalty term
  - only the "simple" constraint x in X remains
- let x*(lambda) be the unique optimal solution to the relaxed problem
- find lambda* such that g(x*(lambda*)) = b — the "right" penalty strength that makes the constraint automatically satisfied
- then x*(lambda*) is the optimal solution to the original problem
- three-step process: (1) relax the hard constraint, (2) tune lambda to make constraint hold, (3) conclude optimality

## Lagrangian Correctness Proof

- the proof is a neat sandwich argument, very elegant:
  - let x_hat be the original problem's optimum, x*(lambda*) be the Lagrangian solution
  - direction 1 (original beats Lagrangian): x_hat is optimal for original, x*(lambda*) is feasible for original (since g(x*(lambda*))=b and x*(lambda*) in X) -> f(x_hat) <= f(x*(lambda*))
  - direction 2 (Lagrangian beats original): x*(lambda*) is optimal for relaxed problem (*), x_hat is also feasible for (*) (since x_hat in X) -> f(x*(lambda*)) + lambda*(g(x*(lambda*))-b) <= f(x_hat) + lambda*(g(x_hat)-b)
  - the key trick: BOTH points satisfy g = b, so the lambda terms on both sides equal zero!
  - after cancellation: f(x*(lambda*)) <= f(x_hat)
  - combining both directions: f(x*(lambda*)) = f(x_hat), and since x*(lambda*) is feasible, it must be optimal
- the proof is essentially the sufficiency part of KKT conditions — the Lagrangian multiplier lambda* is the dual variable
<!-- 两个点都满足g=b，所以lambda惩罚项在两个点上都是0，巧妙！ -->

## Projection onto Euclidean Ball

- X = {x : sum x_i^2 <= R^2}, the L2-ball centered at origin with radius R
- used in: ridge regression (min ||Xw-y||^2 s.t. ||w||<=R), weight decay, gradient clipping by norm, differential privacy (DP-SGD), trust region methods
- if y_hat is already inside the ball (||y_hat|| <= R), projection is y_hat itself — trivial case
- interesting case: y_hat outside (||y_hat|| > R), then projection must land on the boundary (sum x_i^2 = R^2)
  - why boundary? because the closest point in a ball to an exterior point is always on the surface
  - the inequality constraint becomes an equality at the optimum
- can't use box method because sum x_i^2 <= R^2 couples all coordinates together
- but the L2-ball has **rotational symmetry** — should preserve direction, only change magnitude

## Euclidean Ball: Lagrangian Derivation

- problem: min sum(x_i - y_hat_i)^2 s.t. sum x_i^2 = R^2 (equality because we know it's tight)
- Lagrangian relaxation: min_x { sum(x_i - y_hat_i)^2 + lambda*(sum x_i^2 - R^2) }
- key observation: after expanding, the problem is separable by coordinate!
  - each coordinate: min_{x_i} (x_i - y_hat_i)^2 + lambda*x_i^2
  - because both the original objective and the penalty are sums of per-coordinate terms
- per-coordinate first-order condition: 2(x_i - y_hat_i) + 2*lambda*x_i = 0
  - x_i*(1+lambda) = y_hat_i
  - x_i = y_hat_i / (1+lambda)
- amazingly: every coordinate gets the **same scaling factor** 1/(1+lambda) — direction preserved, only magnitude changes
- find lambda*: plug into constraint sum (y_hat_i/(1+lambda*))^2 = R^2
  - ||y_hat||^2 / (1+lambda*)^2 = R^2
  - 1/(1+lambda*) = R/||y_hat||
  - since ||y_hat|| > R, we get 1/(1+lambda*) < 1, so lambda* > 0 — the constraint is indeed active
- the Lagrangian turned an n-dimensional coupled problem into n trivial 1D problems + one 1D equation

## Euclidean Ball Projection Formula

- **pi = (R / ||y_hat||) * y_hat** — scale y_hat down to the ball surface along its direction
- or equivalently: pi = R * unit(y_hat) where unit(y) = y/||y|| is normalization
- complete formula with both cases:
  - if ||y_hat|| <= R: pi = y_hat (already inside)
  - if ||y_hat|| > R: pi = R * y_hat / ||y_hat|| (scale to boundary)
- O(n) computation: compute ||y_hat|| (one pass), then scalar multiply (one pass)
- this is exactly gradient clipping by norm in deep learning: `if norm(g) > C: g = C * g / norm(g)`
- also used in contrastive learning (SimCLR): normalize embeddings to unit sphere = L2 ball projection with R=1
- I've been doing this in every PyTorch training loop without realizing it's an L2-ball projection!

## Projection onto Probability Simplex

- X = {x : sum x_i = 1, x_i >= 0} — the set of all probability distributions over n outcomes
- used in: softmax output layers, mixture model weights, attention weights in Transformers, multi-armed bandit strategies
- two types of constraints:
  - equality (sum = 1): coupled, handles with Lagrangian
  - inequality (x_i >= 0): separable, keep in the subproblem
- more complex than ball because the non-negativity constraint creates a "clipping" effect that doesn't exist for the ball
- in 2D the simplex is just the line segment from (1,0) to (0,1); in 3D it's a triangle

## Simplex: Lagrangian Derivation

- problem: min sum(x_i - y_hat_i)^2 s.t. sum x_i = 1, x_i >= 0
- strategy: Lagrangian for the equality constraint sum x_i = 1, keep x_i >= 0 in the subproblem
- relaxed problem: min_{x_i >= 0} sum{ (x_i - y_hat_i)^2 + lambda*x_i } - lambda
- still separable! each coordinate: min_{x_i >= 0} (x_i - y_hat_i)^2 + lambda*x_i
- per-coordinate first-order condition (ignoring x_i >= 0): 2(x_i - y_hat_i) + lambda = 0 -> x_i = y_hat_i - lambda/2
- but we need x_i >= 0, so two cases:
  - if y_hat_i - lambda/2 >= 0: solution is y_hat_i - lambda/2 (unconstrained min is feasible)
  - if y_hat_i - lambda/2 < 0: solution is 0 (unconstrained min is infeasible, clip to boundary)
- unified: x_i*(lambda) = [y_hat_i - lambda/2]+ (positive part / ReLU operation)
- this [.]+ is the same soft-thresholding that appears in LASSO and proximal methods — it keeps showing up everywhere
<!-- [.]+ 操作在LASSO和proximal方法中也出现，数学结构是相同的 -->

## Simplex vs Ball: The Fundamental Difference

- ball projection: **multiplicative** scaling — every coordinate shrunk by same factor R/||y_hat||, no coordinate becomes zero (unless it was already zero)
- simplex projection: **additive** shift + clipping — every coordinate shifted down by lambda*/2, negatives clipped to 0
- the clipping creates **sparsity**: small coordinates become exactly zero after projection
- this is why simplex projection is more complex but also more interesting
- the sparsity connects to sparsemax (alternative to softmax) which is literally simplex projection

## Simplex Projection Formula and Lambda*

- find lambda* such that sum [y_hat_i - lambda*/2]+ = 1
- then pi_i = [y_hat_i - lambda*/2]+
- h(lambda) = sum [y_hat_i - lambda/2]+ is:
  - continuous (sum of continuous piecewise-linear functions)
  - strictly decreasing (in the range where h > 0)
  - goes from +infinity (lambda -> -inf) to 0 (lambda large enough)
  - so by IVT, unique lambda* exists with h(lambda*) = 1
- computing lambda* efficiently: sort y_hat values descending, then find the right "breakpoint" -> O(n log n)
  - there also exists an O(n) linear-time algorithm, but sorting is practical enough
- no closed-form for lambda* (unlike the ball), but still much cheaper than general constrained optimization

## Geometric Intuition (2D Simplex)

- 2D simplex is just the line segment from (1,0) to (0,1)
- case 1: y_hat above the line with both coordinates positive -> perpendicular projection onto line, both coordinates stay positive
  - this is the uniform shift case: each coordinate decreases by (y_hat_1 + y_hat_2 - 1)/2
- case 2: y_hat biased toward one axis (e.g. y_hat_1 large, y_hat_2 small or negative)
  - perpendicular projection onto the line might give negative second coordinate
  - non-negativity constraint kicks in: second coordinate clipped to 0, first gets the remaining budget
  - projection lands near a vertex of the simplex
- lambda* determines the shift amount: larger lambda -> lower threshold -> more coordinates zeroed out -> sparser result
<!-- simplex投影也会产生稀疏性——小的坐标被截到0，考试要记住这一点 -->

## Summary Comparison

| Set | Formula | Key Trick | Complexity |
|-----|---------|-----------|------------|
| Box | clip(y_i, 0, U_i) | separability | O(n) |
| L2-ball | R*y/\|\|y\|\| | Lagrangian + rotational symmetry | O(n) |
| Simplex | [y_i - lambda*/2]+ | Lagrangian + truncation | O(n log n) |

- clear difficulty progression: box (trivial) -> ball (Lagrangian, closed form lambda*) -> simplex (Lagrangian, need to solve for lambda*)
- all are practical tools that plug into the PGD framework from lec 13-14
- the Lagrangian method is the unifying technique for ball and simplex — it converts coupled constraints into separable subproblems

## Key Takeaways

1. Lagrangian method: absorb equality constraint into objective with multiplier lambda, find lambda* that satisfies the constraint — super powerful general technique
2. Euclidean ball projection: pi = R*y_hat/||y_hat||, direction preserved, magnitude clipped, O(n) — this is gradient clipping
3. simplex projection: pi_i = [y_hat_i - lambda*/2]+, additive shift + clip, O(n log n) — creates sparsity
4. ball = multiplicative scaling (no sparsity), simplex = additive shift + clip (sparsity from non-negativity)
5. the Lagrangian proof correctness relies on the fact that both comparison points satisfy g=b, so penalty terms cancel
6. I need to memorize all projection formulas for the exam: box (clip), ball (R*y/||y||), simplex ([y-lambda/2]+)
