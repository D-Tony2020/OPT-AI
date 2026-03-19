# Lec 14 - Mar 16 - PGD Convergence Theorem + Box Projection

## Overview

- this lecture finishes two things: (1) the convergence theorem for projected GD, and (2) the first concrete projection computation — box constraints
- the convergence theorem is the "payoff" for all the mathematical infrastructure from lec 13
- box projection is the simplest and most practical — it's literally np.clip
- both theoretical and practical in one lecture — I like the balance
<!-- 理论收敛定理+第一个实际投影计算，理论和实践都有了 -->

## Review: Two Projection Properties (from Lec 13)

- property 1 (obtuse angle): (x - Pi_X(y))^T (y - Pi_X(y)) <= 0
  - from projection point, feasible direction and original-point direction form obtuse angle
- property 2 (Pythagorean): ||x - y||^2 >= ||y - Pi_X(y)||^2 + ||x - Pi_X(y)||^2
  - projection only shrinks distances to any feasible point
- property 1 is the foundation, property 2 is the corollary
- property 2 is what we actually use in the convergence proof — it guarantees projection doesn't increase distance to x*
- these are pure geometric/algebraic properties of convex sets, independent of the objective f

## Convergence Proof: Key Step

- continuing from lec 13, we had:
  - f(x^k) - f(x*) <= grad f(x^k)^T(x^k - x*) (by convexity first-order characterization, lec 03-04)
  - grad f(x^k)^T(x^k - x*) = (1/2gamma)(gamma^2 ||g^k||^2 + ||x^k - x*||^2 - ||y^{k+1} - x*||^2)
  - this comes from the polarization identity: 2a^T b = ||a||^2 + ||b||^2 - ||a-b||^2
- the problem: we need ||x^k - x*||^2 - ||x^{k+1} - x*||^2 for telescoping, but we have ||y^{k+1} - x*||^2 instead
- **Pythagorean inequality to the rescue**: ||y^{k+1} - x*||^2 >= ||x^{k+1} - x*||^2 (since x* in X)
  - so -||y^{k+1} - x*||^2 <= -||x^{k+1} - x*||^2
  - this replaces y^{k+1} with x^{k+1}, restoring the telescoping structure
  - we pay a price: dropping ||y^{k+1} - x^{k+1}||^2 >= 0 makes the bound looser, but that's fine
- after substitution:
  - grad f(x^k)^T(x^k - x*) <= (gamma/2)||g^k||^2 + (1/2gamma)(||x^k - x*||^2 - ||x^{k+1} - x*||^2)
- in unconstrained GD (lec 05-08), y^{k+1} = x^{k+1} so this step is not needed

## Telescoping Sum

- sum from k=0 to K-1:
  - sum of (||x^k - x*||^2 - ||x^{k+1} - x*||^2):
    - k=0: ||x^0 - x*||^2 - ||x^1 - x*||^2
    - k=1: ||x^1 - x*||^2 - ||x^2 - x*||^2
    - ...
    - k=K-1: ||x^{K-1} - x*||^2 - ||x^K - x*||^2
  - everything in the middle cancels! only ||x^0 - x*||^2 - ||x^K - x*||^2 survives
  - drop the -||x^K - x*||^2 term (it's non-negative, so dropping it makes the bound looser)
- final bound: sum_{k=0}^{K-1} (f(x^k) - f(x*)) <= (gamma/2) * sum||g^k||^2 + (1/2gamma) * ||x^0 - x*||^2
- the two terms trade off:
  - gamma large -> gradient term (gamma/2)*K*B^2 dominates (too much "oscillation")
  - gamma small -> initial distance term (1/2gamma)*R^2 dominates (too slow to reach optimum)
  - optimal gamma balances the two
<!-- 伸缩求和太优雅了，几十项一加中间全消了，高中学级数就见过这个技巧 -->

## Convergence Theorem (Main Result)

- **assumptions**: f convex, X convex, ||x^0 - x*|| <= R (bounded initial distance), ||grad f(x)|| <= B for all x in X (bounded gradients)
- **step size**: gamma = R/(B*sqrt(K))
- **result**: (1/K) * sum_{k=0}^{K-1} (f(x^k) - f(x*)) <= RB/sqrt(K)
- proof:
  - sum(f(x^k) - f(x*)) <= (gamma/2)*K*B^2 + (1/2gamma)*R^2
  - divide by K: (1/K)*sum <= (gamma/2)*B^2 + R^2/(2*gamma*K)
  - plug in gamma = R/(B*sqrt(K)): first term = RB/(2*sqrt(K)), second term = RB/(2*sqrt(K))
  - total = RB/sqrt(K)
- the step size that makes both terms equal is optimal — classic AM-GM type argument
- convergence rate O(1/sqrt(K)) — to reach epsilon accuracy, need K >= (RB/epsilon)^2 iterations
- this is slower than O(1/K) under Lipschitz gradient (lec 10), because bounded gradient is a weaker assumption
  - weaker assumption -> weaker conclusion, makes sense
- the conclusion is about **average** error, not last iterate — can use x_bar = (1/K)*sum(x^k) by Jensen's inequality
- step size gamma = R/(B*sqrt(K)) requires knowing R, B, and K in advance — not always convenient in practice
<!-- 步长需要提前知道R, B, K，实际中不太方便但理论上是最优的 -->

## Bounded Gradient Assumption

- ||grad f(x)|| <= B for all x in X is not as strong as it sounds
- if X is bounded (compact) and f is continuously differentiable, then grad f is automatically bounded on X
  - this is just "continuous function on compact set achieves its maximum" from real analysis
- ML examples where X is bounded: L2-ball, simplex, box, L1-ball — all bounded, so assumption is automatic
- this connects back to convex sets from lec 02 — understanding what sets are bounded matters for convergence theory
- the bounded gradient assumption is weaker than the Lipschitz gradient assumption (||grad f(x) - grad f(y)|| <= L||x-y||) from lec 10
  - Lipschitz gradient -> O(1/K) convergence (faster)
  - bounded gradient -> O(1/sqrt(K)) convergence (slower, but weaker assumption)
  - weaker assumption = weaker conclusion, this tradeoff makes complete sense

## Computing Projections: The Practical Question

- Pi_X(y) = argmin_{x in X} ||x - y||^2 is itself a constrained optimization problem
- sounds circular: we use projection to solve constrained opt, but projection IS constrained opt!
- the key insight that breaks the circularity: ||x-y||^2 = sum(x_i - y_i)^2 is **separable** — no cross terms x_i*x_j
- this special structure of the projection objective makes it tractable for many common X
- if projection were as hard as the original problem, PGD would be useless — we'd be solving a constrained optimization subproblem at every step
- luckily, for ML-relevant constraint sets, projections are very efficient
- the story for lec 14-16: box (fully separable, easiest), ball (Lagrangian makes it separable), simplex (Lagrangian + truncation), L1-ball (reduce to simplex)
- this is one of those cases where theory and practice align beautifully

## Box Projection

- X = {x : 0 <= x_i <= U_i for all i} — a hyperrectangle/"box" in R^n
- projection problem: min sum(x_i - y_hat_i)^2 s.t. 0 <= x_i <= U_i
- both objective and constraints are **separable** by coordinate -> n independent 1D problems!
  - this is the key: no coordinate talks to any other coordinate
  - separability of objective + separability of constraints = complete decoupling
- each 1D problem: min_{0 <= x_i <= U_i} (x_i - y_hat_i)^2
- three cases (just a parabola on an interval):
  - case 1: 0 <= y_hat_i <= U_i -> x_i* = y_hat_i (minimum of parabola is inside interval, nothing to do)
  - case 2: y_hat_i < 0 -> x_i* = 0 (parabola minimum is to the left, clip to lower bound)
  - case 3: y_hat_i > U_i -> x_i* = U_i (parabola minimum is to the right, clip to upper bound)
- closed-form: pi_i = min{U_i, [y_hat_i]+} = clip(y_hat_i, 0, U_i)
- this is literally `np.clip(y, 0, U)` in numpy! I've been using this function without knowing it's a projection
<!-- np.clip就是box投影！一直在用但从不知道这是投影梯度下降的一部分 -->

## Why Box Projection Works: Separability

- the magic word here is **separability**: both objective and constraints decompose by coordinate
- objective: sum(x_i - y_hat_i)^2 = sum of independent per-coordinate terms (no cross terms x_i * x_j)
- constraints: 0 <= x_i <= U_i for each i independently (no coupling like sum x_i = 1)
- when both are separable: min sum f_i(x_i) s.t. x_i in C_i = sum of min f_i(x_i) s.t. x_i in C_i
- this is NOT possible when constraints couple coordinates: e.g., simplex has sum x_i = 1, so changing x_1 forces changes in other x_i
- separability = the ultimate computational gift, it's what makes box the easiest projection

## Box Projection: Properties and Applications

- O(n) computation — two comparisons per coordinate, same cost as one gradient evaluation, basically free
- most coordinates don't violate constraints most of the time, so projection is often a no-op for most entries
- applications everywhere in ML:
  - ReLU activation: max(0, x) = projection onto [0, inf) box (U_i = infinity)
  - clipped ReLU: min(U, max(0, x)) = full box projection
  - pixel value constraints in adversarial attacks (PGD attack): x in [0, 255]^n
  - per-coordinate gradient clipping: clip each g_i to [-C, C]
  - variable bounds in linear programming
- separability is why box projection is so easy — no coupling between coordinates
- when constraints couple coordinates (like sum x_i^2 <= R^2 or sum x_i = 1), separability breaks and we need the Lagrangian method -> lec 15

## Key Takeaways

1. convergence rate: O(RB/sqrt(K)) with step size gamma = R/(B*sqrt(K)), under bounded gradient assumption
2. telescoping sum is the core proof technique, same framework as lec 10-11 unconstrained analysis
3. Pythagorean inequality (from lec 13) is the bridge: it lets us swap y^{k+1} for x^{k+1} to enable telescoping
4. box projection = coordinate-wise clipping = np.clip, O(n), zero overhead compared to vanilla GD
5. separability is key: when both objective and constraints decompose by coordinate, the n-dim problem splits into n trivial 1D problems
6. non-separable constraints (ball, simplex) need the Lagrangian method to "artificially" restore separability -> next lecture
