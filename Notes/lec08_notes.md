# Lec 8 - Feb 18 - Taylor Applications, Convex Gradient Characterization & Jensen's Inequality

## This lecture is a milestone

- three big things happen today:
  1. we FINALLY prove that gradient descent actually decreases function value (using Taylor corollary from lec 7)
  2. Jensen's inequality -- the multi-point generalization of convexity
  3. gradient characterization of convex functions -- tangent plane is a global lower bound
- I have been waiting since lec 4 for the formal proof that GD works. today we get it

<!-- 里程碑：GD 下降的严格证明 + 凸函数的梯度刻画 -->

## Taylor theorem review (from lec 7)

Quick recap of the key formula:

**Taylor theorem**: f(y) = f(x) + nabla f(gamma*x + (1-gamma)*y)^T (y-x) for some gamma in (0,1)

**Corollary** (the useful one):

$$f(y) = f(x) + \nabla f(x)^T(y-x) + o(\|y-x\|)$$

Remember: ||x||^2 = x^T x. This identity will be used repeatedly below.
Professor spent a whole page re-deriving the corollary proof -- same as lec 7. I think he wants to make sure we really internalize it before applying it.

## The big application: proving GD decreases f

- this is the payoff of all the Taylor machinery
- we are at point x_hat, gradient is nonzero, we step to x_hat - alpha * nabla f(x_hat)

**Step 1 -- substitute into Taylor corollary**:

Set x = x_hat, y = x_hat - alpha * nabla f(x_hat):

$$f(\hat{x} - \alpha \nabla f(\hat{x})) = f(\hat{x}) + \nabla f(\hat{x})^T(-\alpha \nabla f(\hat{x})) + o(\alpha \|\nabla f(\hat{x})\|)$$

**Step 2 -- simplify the inner product**:

nabla f(x_hat)^T * (-alpha * nabla f(x_hat)) = -alpha * ||nabla f(x_hat)||^2

This is where ||x||^2 = x^T x comes in. The negative sign is crucial -- we walk in the NEGATIVE gradient direction, so the dot product is negative, meaning function value goes DOWN.

**Step 3 -- factor out**:

$$f(\hat{x} - \alpha \nabla f(\hat{x})) = f(\hat{x}) + \alpha \|\nabla f(\hat{x})\| \left(-\|\nabla f(\hat{x})\| + \frac{o(\alpha\|\nabla f\|)}{\alpha\|\nabla f\|}\right)$$

**Step 4 -- the key limit**:

As alpha -> 0, the o(...)/(...) term -> 0 (by definition of little-o). So the bracket approaches -||nabla f||, which is NEGATIVE (as long as gradient is nonzero).

**Step 5 -- make it rigorous**:

Choose alpha small enough so that the o-term ratio <= (1/2)||nabla f||. Then:

$$\boxed{f(\hat{x} - \alpha \nabla f(\hat{x})) \leq f(\hat{x}) - \frac{1}{2}\alpha \|\nabla f(\hat{x})\|^2}$$

This is the **descent guarantee**:
- each step decreases f by at least (1/2) * alpha * ||grad f||^2
- descent is proportional to gradient norm squared -- larger gradient = more descent
- gradient = 0 means no descent -- we are at a stationary point

<!-- 核心结论：每步至少下降 (1/2)*alpha*||grad f||^2 -->

### What this proof does NOT tell us

- it says "there exists a small enough alpha" but not HOW small
- alpha too small: descent amount ~0, convergence is glacially slow
- alpha too large: o-term dominates, function might actually increase
- to quantify the right alpha, we need Lipschitz continuous gradients (coming in lec 9+)
- the Descent Lemma will give us alpha <= 1/L where L is the Lipschitz constant
- so this proof is qualitative, the quantitative version comes later

## Convex functions review (from lec 3)

**Definition**: f is convex if f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y) for all x,y and lambda in [0,1]

- geometrically: chord is above the curve
- ah so THIS is why we spent so much time on convex functions back in lec 2-3
- the descent proof above works for ANY differentiable function
- but convexity is what upgrades "local descent" to "global convergence"
- for non-convex f: GD might get stuck at local minima or saddle points
- for convex f: every local min is global min, and gradient = 0 means global optimum

## Jensen's inequality

$$f\left(\sum_\ell \lambda^\ell x^\ell\right) \leq \sum_\ell \lambda^\ell f(x^\ell)$$

where lambda^l >= 0 and sum lambda^l = 1.

- this generalizes the two-point convexity definition to k points
- proof: induction. peel off the first point, apply convexity definition, recurse on the rest
- the "peel off" trick: write sum as lambda^1 * x^1 + (1-lambda^1) * [convex combo of remaining points]
  - the remaining weights are lambda^l / (1 - lambda^1), which still sum to 1

**Continuous version** (for random variables): f(E[X]) <= E[f(X)]
- this is used EVERYWHERE in ML theory:
  - KL divergence >= 0: follows from Jensen + (-log is convex)
  - EM algorithm: E-step lower bound from Jensen
  - Variational inference: ELBO derivation uses Jensen
- I think this will also be useful for convergence rate proofs later

<!-- Jensen 不等式在 ML 里无处不在：KL散度、EM、ELBO -->

## Gradient characterization of convex functions -- THE key theorem

**Theorem**: f is convex if and only if:

$$f(y) \geq f(x) + \nabla f(x)^T(y-x) \quad \text{for all } x, y$$

- tangent plane (first-order approximation) is a **global lower bound** on f
- this is MUCH stronger than the Taylor corollary

### Comparing Taylor corollary vs gradient characterization

| | Taylor corollary | Gradient characterization |
|--|-----------------|--------------------------|
| Formula | f(y) = f(x) + nabla f(x)^T(y-x) + o(||y-x||) | f(y) >= f(x) + nabla f(x)^T(y-x) |
| Type | equality (exact) | inequality (lower bound) |
| Applies to | any C^1 function | convex functions only |
| Range | local (y near x) | **global** (any y) |
| Error term | o(||y-x||), can be positive or negative | always >= 0 |

The Taylor corollary says "linear approximation has small error locally". The gradient characterization says "linear approximation is a global LOWER bound". The second is way more powerful.

### Why this matters so much

1. **Stationary point = global optimum**: if nabla f(x*) = 0, plug into the theorem:
   f(y) >= f(x*) + 0 = f(x*) for all y. So x* is global minimum!
   - for non-convex functions, nabla f = 0 could be saddle point or local max -- useless

2. **Convergence analysis foundation**: set x = x^k (current iterate), y = x* (optimum):
   f(x*) >= f(x^k) + nabla f(x^k)^T(x* - x^k)
   rearranging: f(x^k) - f(x*) <= nabla f(x^k)^T(x^k - x*)
   this bounds the "suboptimality gap" in terms of gradient -- starting point for convergence rate proofs

3. **Tangent planes form envelope**: all tangent lines together reconstruct the convex function from below (this connects to Fenchel conjugate / convex duality, more advanced topic)

<!-- 凸函数的梯度刻画 = 凸优化的灵魂 -->

## Connecting everything

- lec 3: defined convex functions
- lec 4: introduced GD (intuitively: follow negative gradient)
- lec 5: 1D optimization tools
- lec 6: line search completes GD + penalty for constraints
- lec 7: barrier method + Taylor theorem (the math foundation)
- **lec 8: Taylor proves GD works + convexity guarantees global optimum**
- this is building toward the convergence proof I think. we have "descent per step" from Taylor, we have "global structure" from convexity. lec 9 should add Lipschitz condition to quantify the step size and give us a convergence RATE

The big picture is becoming clear:
- Taylor corollary -> GD decreases f (qualitative)
- Lipschitz condition -> how much alpha to use (quantitative, coming next)
- Convexity -> local optimum = global optimum
- Combine all three -> convergence rate O(1/k) for convex, O(rho^k) for strongly convex

## Key formulas to remember

1. GD descent: f(x - alpha*grad f) <= f(x) - (1/2)*alpha*||grad f||^2
2. Jensen: f(sum lambda^l x^l) <= sum lambda^l f(x^l)
3. Gradient characterization: f convex <=> f(y) >= f(x) + grad f(x)^T(y-x) for all x,y
