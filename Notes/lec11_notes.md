# Lec 11 - Mar 4 - Smooth Convex GD: O(1/K) Convergence

## Connection to last lecture

- lec 10 gave us $O(1/\sqrt{K})$ for convex + bounded gradients — the baseline
- today: add **smoothness** assumption, get $O(1/K)$ — a full order of magnitude faster
- same preliminary analysis framework (first-order char + norm identity + telescoping), just a different "plugin" at the end
- the improvement from smoothness is huge. this comparison table will be on the exam for sure
<!-- O(1/√K) vs O(1/K) — smoothness的提升是质的飞跃 -->

## Quick review: where we left off

- preliminary analysis from lec 10:
$$\sum_{k=0}^{K-1}(f(x^k) - f(x^*)) \leq \frac{\gamma}{2}\sum\|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$
- last time we handled gradient term crudely: $\sum\|g^k\|^2 \leq KB^2$ (grows linearly with $K$)
- that gave $O(1/\sqrt{K})$ rate with $O(1/\varepsilon^2)$ iteration complexity
- at $\varepsilon = 0.001$, need $K = 10^6$ steps. expensive for large-scale ML. can we do better?
- yes — if the function has more structure. enter smoothness.

## What is smoothness?

- **Definition**: $f$ is $L$-smooth if:
$$f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2 \quad \forall x, y$$

- this is a **quadratic upper bound** — function value can't grow faster than a quadratic
- $L$ is the smoothness parameter — controls max "curvature" of the function
- for twice-differentiable functions, $L$ = max eigenvalue of Hessian ($\nabla^2 f(x) \preceq LI$)
- equivalent to: gradient is Lipschitz continuous, $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$
<!-- 光滑性 = 函数弯曲有上限 = 梯度是Lipschitz连续的 -->

### The sandwich structure

- combined with first-order characterization (from lec 09), we get a sandwich:

$$\underbrace{f(x) + \nabla f(x)^T(y-x)}_{\text{linear lower bound (lec 09)}} \leq f(y) \leq \underbrace{f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2}_{\text{quadratic upper bound (lec 11)}}$$

- at point $x$, all three curves meet. moving away from $x$, $f$ is squeezed between them
- the picture: tangent line below, quadratic bowl above, function in the middle
- now lec 09's first-order characterization and today's smoothness condition work together as lower and upper bounds
- small $L$ = gentle curvature (nearly linear), large $L$ = steep curvature (tight bowl)
- example: MSE loss $f(w) = \|Xw - y\|^2$ has $L = 2\lambda_{\max}(X^TX)$

## Result 1: monotone descent (the killer result from smoothness)

- plug GD update $x^{k+1} = x^k - \gamma g^k$ into the smoothness definition:

$$f(x^{k+1}) \leq f(x^k) + (g^k)^T(-\gamma g^k) + \frac{L}{2}\gamma^2\|g^k\|^2$$
$$= f(x^k) - \gamma\|g^k\|^2 + \frac{L\gamma^2}{2}\|g^k\|^2 = f(x^k) - \gamma(1 - \frac{L\gamma}{2})\|g^k\|^2$$

- the coefficient $h(\gamma) = \gamma(1 - L\gamma/2)$:
  - when $\gamma < 2/L$: positive, so $f(x^{k+1}) < f(x^k)$ — function value **decreases every step**!
  - when $\gamma = 2/L$: zero, no guarantee
  - when $\gamma > 2/L$: negative, might increase — step too big!
- maximized at $\gamma = 1/L$, giving $h = 1/(2L)$

- optimal choice $\gamma = 1/L$ gives maximum per-step descent:

$$f(x^{k+1}) \leq f(x^k) - \frac{1}{2L}\|g^k\|^2$$

- wow, this is fundamentally different from lec 10! no more bouncing!
- with $\gamma = 1/L$, every single step makes progress
- bigger gradient $\to$ bigger descent. far from optimum = big gradient = fast progress
- close to optimum = small gradient = slow but steady progress
- the step size $1/L$ is "safe" because smoothness limits how much the function can curve
<!-- 每一步都保证下降！不再弹跳！这就是光滑性的"杀手级"结论 -->

## Result 2: same preliminary analysis as lec 10

$$\sum_{k=0}^{K-1}(f(x^k) - f(x^*)) \leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

- exact same framework from lec 10, reused. the magic is in how we handle $\sum\|g^k\|^2$
- lec 10: crude bound $\sum\|g^k\|^2 \leq KB^2$ (grows with $K$ — bad)
- lec 11: from Result 1, rearrange to get $\|g^k\|^2 \leq 2L(f(x^k) - f(x^{k+1}))$
- sum over $k$: telescoping! $\sum\|g^k\|^2 \leq 2L(f(x^0) - f(x^K))$
- this bound does NOT grow with $K$ — it's just the total function decrease
- that's the core mathematical reason why smoothness gives a faster rate

## The convergence theorem

- **Theorem**: $f$ is $L$-smooth convex, $\|x^0 - x^*\| \leq R$, step size $\gamma = 1/L$:

$$f(x^K) - f(x^*) \leq \frac{LR^2}{2K}$$

- note: this bounds the **last iterate**, not the average! much stronger than lec 10
- step size $1/L$ is fixed — doesn't depend on $K$! much simpler than lec 10's $R/(B\sqrt{K})$

### Proof

1. From Result 1 (telescoping): $\frac{1}{2L}\sum_{k=0}^{K-1}\|g^k\|^2 \leq f(x^0) - f(x^K) \quad (\ast)$
2. From Result 2 with $\gamma = 1/L$:
$$\sum_{k=0}^{K-1}(f(x^k) - f(x^*)) \leq \frac{1}{2L}\sum\|g^k\|^2 + \frac{L}{2}R^2$$
3. Substitute $(\ast)$ into step 2:
$$\sum_{k=0}^{K-1}(f(x^k) - f(x^*)) \leq f(x^0) - f(x^K) + \frac{L}{2}R^2$$
4. Split the $k=0$ term from the left sum and rearrange:
$$\sum_{k=1}^{K}(f(x^k) - f(x^*)) \leq \frac{L}{2}R^2$$
5. Divide by $K$: $\frac{1}{K}\sum_{k=1}^{K}(f(x^k) - f(x^*)) \leq \frac{LR^2}{2K}$
6. By monotone descent ($f(x^K)$ is smallest), last iterate $\leq$ average:
$$f(x^K) - f(x^*) \leq \frac{1}{K}\sum_{k=1}^{K}(f(x^k) - f(x^*)) \leq \frac{LR^2}{2K}$$

- the two Results fit together perfectly — coefficients match because $\gamma = 1/L$
- same framework as lec 10, different plugin, rate jumps from $1/\sqrt{K}$ to $1/K$
<!-- 两个Result完美咬合，同一个框架不同插件，速率就从1/√K变成1/K -->

## Comparison: lec 10 vs lec 11 (this is the big picture)

OK let me organize the convergence results so far:

| | Lec 10 (bounded grad) | Lec 11 (smooth) |
|---|---|---|
| Assumption | $\|\nabla f\| \leq B$ | $L$-smooth |
| Step size | $R/(B\sqrt{K})$ (depends on $K$!) | $1/L$ (fixed!) |
| What's bounded | average error | **last iterate** error |
| Rate | $O(1/\sqrt{K})$ | $O(1/K)$ |
| Iteration complexity | $O(1/\varepsilon^2)$ | $O(1/\varepsilon)$ |
| Monotone descent? | No (bouncing) | **Yes** |
| Need to track best? | Yes | No (last is best) |

- smoothness gives improvement on EVERY dimension: faster rate, simpler step size, stronger conclusion
- this is why understanding function structure matters — it directly determines algorithm performance

## Iteration complexity comparison

| Target $\varepsilon$ | Lec 10: $O(1/\varepsilon^2)$ | Lec 11: $O(1/\varepsilon)$ | Speedup |
|---|---|---|---|
| 0.1 | 100 | 10 | $10\times$ |
| 0.01 | $10^4$ | 100 | $100\times$ |
| 0.001 | $10^6$ | $10^3$ | $1000\times$ |
| 0.0001 | $10^8$ | $10^4$ | $10^4\times$ |

- the gap grows with precision. at high accuracy, smoothness saves orders of magnitude of computation
- in ML: going from training loss 0.1 to 0.001 — lec 10 says 100x more work, lec 11 says only 10x

## Asymptotic convergence

- by squeeze theorem: $0 \leq f(x^K) - f(x^*) \leq LR^2/(2K) \to 0$ as $K \to \infty$
- so $\lim_{K\to\infty} f(x^K) = f(x^*)$ — GD converges to the optimal value
- in practice you don't run infinite steps, but the iteration complexity tells you when to stop

## The full convergence hierarchy so far

| Function class | Rate | Complexity | Lecture |
|---|---|---|---|
| Convex + bounded grad | $O(1/\sqrt{K})$ | $O(1/\varepsilon^2)$ | Lec 10 |
| Convex + smooth | $O(1/K)$ | $O(1/\varepsilon)$ | **Lec 11** |
| Strongly convex + smooth | $O(e^{-cK})$ | $O(\log(1/\varepsilon))$ | Later |

- stronger assumptions $\to$ faster convergence. that's the whole story of this part of the course
- each level gains roughly one "order" of improvement in complexity
- next: constrained optimization and projected gradient descent (lec 12) — making things more practical
<!-- 更强假设→更快收敛，这就是整个convergence理论的核心故事线 -->
