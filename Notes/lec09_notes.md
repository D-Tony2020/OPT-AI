# Lec 09 - Feb 23 - First-Order Characterization of Convexity

## Why this lecture matters

- today is the bridge lecture. we spent lec 03-04 defining convex functions and lec 05-08 building GD theory, but we never really answered: **why does GD actually find the global optimum on convex functions?**
- this lecture gives three theorems that answer that question completely
- now I see why we studied convex function definitions (lec 03) and Taylor's theorem (lec 07) — they both come together here
- the three results build on each other: first-order characterization $\to$ local=global $\to$ stationary point theorem
<!-- 这讲是桥梁，把前面的凸性定义和Taylor定理串起来了 -->

## Little-o notation and Taylor's theorem (review)

- $g(t) = o(t)$ means $g(t)/t \to 0$ as $t \to 0$ — basically "goes to zero faster than $t$"
- example: $t^2 = o(t)$ because $t^2/t = t \to 0$. but $2t \neq o(t)$ because $2t/t = 2$
- Taylor's theorem: $f(y) = f(x) + \nabla f(x)^T(y - x) + o(\|y - x\|)$
- the remainder $o(\|y-x\|)$ is the "negligible tail" that we can ignore when $y$ is close to $x$
- this is the tool that lets us go from "convexity definition" to "gradient-based characterization"
- I remember we saw Taylor expansion in lec 07, now it's being used as a proof tool

## The first-order characterization theorem

- **Theorem**: $f$ is convex $\iff$ $f(y) \geq f(x) + \nabla f(x)^T(y - x)$ for all $x, y$
- geometric meaning: the tangent line (or tangent plane) always lies below the function
- for non-convex functions, the tangent can cross through the curve — sometimes above, sometimes below
- this is THE key tool for all convergence analysis in lec 10-11
<!-- 一阶刻画：切线永远在函数下方，这是后面所有收敛分析的起手式 -->

### Why this is so useful for GD analysis

- whenever we need to bound $f(x^k) - f(x^*)$ in GD analysis, step 1 is always:
$$f(x^*) \geq f(x^k) + \nabla f(x^k)^T(x^* - x^k)$$
- rearrange: $f(x^k) - f(x^*) \leq (g^k)^T(x^k - x^*)$
- so function gap $\leq$ inner product of gradient and displacement — much easier to work with
- this converts an optimization question into a geometry question
- the power of convexity: local information (gradient at one point) gives GLOBAL guarantee (about function value everywhere)
- you will see this inequality at the start of EVERY convergence proof from now on

## Proof of the forward direction ($\Rightarrow$)

- assume $f$ is convex, want to show $f(y) \geq f(x) + \nabla f(x)^T(y-x)$
- core idea: start from convexity definition (discrete inequality), take limit to get gradient (continuous info)
- by convexity definition: $f(x + \lambda(y-x)) \leq (1-\lambda)f(x) + \lambda f(y)$
- rearrange and divide by $\lambda > 0$ (inequality direction preserved):

$$f(y) \geq f(x) + \frac{f(x + \lambda(y-x)) - f(x)}{\lambda}$$

- the right side is a difference quotient! as $\lambda \to 0$, it approaches $\nabla f(x)^T(y-x)$
- apply Taylor to the numerator: $f(x + \lambda(y-x)) - f(x) = \nabla f(x)^T \cdot \lambda(y-x) + o(\lambda\|y-x\|)$
- divide by $\lambda$, get:

$$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{o(\lambda\|y-x\|)}{\lambda}$$

- for the remainder: rewrite as $\frac{o(\lambda\|y-x\|)}{\lambda\|y-x\|} \cdot \|y-x\|$
- as $\lambda \to 0$ (with $x, y$ fixed), $\lambda\|y-x\| \to 0$, so the fraction $\to 0$ by definition of little-o
- $\|y-x\|$ is just a constant, so the whole remainder $\to 0$
- the trick: inequality holds for all $\lambda \in (0,1)$, so it holds in the limit too
- subtlety: we need to carefully track that $o$ is about its own argument going to zero, not about $\lambda$
<!-- 从离散的凸性不等式 → 取极限 → 得到连续的梯度信息，经典的"从离散到连续" -->

## Proof of the reverse direction ($\Leftarrow$)

- this direction has a beautiful algebraic trick — completely different style from the forward direction
- assume first-order condition holds for all $x, y$, want to prove convexity definition
- key idea: expand at the convex combination point $c = \lambda a + (1-\lambda)b$
- note: $a - c = (1-\lambda)(a-b)$ and $b - c = -\lambda(a-b)$ — opposite directions!
- apply first-order condition to both $a$ and $b$ from center $c$:

$$f(a) \geq f(c) + \nabla f(c)^T(a - c) = f(c) + (1-\lambda)\nabla f(c)^T(a-b)$$
$$f(b) \geq f(c) + \nabla f(c)^T(b - c) = f(c) - \lambda\nabla f(c)^T(a-b)$$

- multiply first by $\lambda$, second by $(1-\lambda)$, add them up
- the gradient terms: $\lambda(1-\lambda)\nabla f(c)^T(a-b) - \lambda(1-\lambda)\nabla f(c)^T(a-b) = 0$
- they cancel perfectly! because $a-c$ and $b-c$ point in opposite directions with the right ratio
- result: $\lambda f(a) + (1-\lambda)f(b) \geq [\lambda + (1-\lambda)]f(c) = f(c) = f(\lambda a + (1-\lambda)b)$
- that's the convexity definition. done!

- interesting: forward proof uses calculus (Taylor + limits), reverse proof is pure algebra
- "expand at the middle point, weighted sum kills the gradient" — remember this trick, it shows up again later
<!-- 正向用微积分，反向用纯代数。两种完全不同的证明风格 -->

## Local minimum = Global minimum

- **Theorem**: if $f$ is convex and $\hat{x}$ is a local minimum, then $\hat{x}$ is a global minimum
- this is HUGE for optimization — it means no "local traps" in convex world
- proof idea: for any far-away point $y$, construct convex combination $z = \lambda\hat{x} + (1-\lambda)y$
- choose $\lambda$ close to 1 so that $z$ falls inside the $\varepsilon$-neighborhood of $\hat{x}$
- specifically, need $(1-\lambda)\|y - \hat{x}\| \leq \varepsilon$, so pick $\lambda \geq 1 - \varepsilon/\|y-\hat{x}\|$
- then: local optimality gives $f(\hat{x}) \leq f(z)$
- convexity gives $f(z) \leq \lambda f(\hat{x}) + (1-\lambda)f(y)$
- combine: $f(\hat{x}) \leq \lambda f(\hat{x}) + (1-\lambda)f(y)$
- subtract $\lambda f(\hat{x})$ from both sides: $(1-\lambda)f(\hat{x}) \leq (1-\lambda)f(y)$
- divide by $(1-\lambda) > 0$: $f(\hat{x}) \leq f(y)$
<!-- 凸函数没有"局部陷阱"，这就是为什么凸优化是tractable的 -->

- this is why convex optimization is fundamentally easier than non-convex: no local traps
- in deep learning (non-convex), you can get stuck in bad local minima — not here
- for ML models like linear regression, logistic regression, SVM — all convex, so any local search finds global optimum

## Stationary point theorem

- **Theorem**: if $f$ is convex and $\nabla f(\hat{x}) = 0$, then $\hat{x}$ is a global minimum
- proof is literally two lines — probably shortest proof in this course:
$$f(y) \geq f(\hat{x}) + \underbrace{\nabla f(\hat{x})}_{=0} \cdot (y - \hat{x}) = f(\hat{x})$$
- so $f(y) \geq f(\hat{x})$ for all $y$. done.
- this is THE most direct application of first-order characterization

- in non-convex world, $\nabla f = 0$ could be saddle point or local max — you can't tell
- in convex world, $\nabla f = 0$ means you're at the global optimum. period.
- GD's goal is to make gradient small — on convex functions, this is equivalent to finding the global optimum

## Why these theorems matter for ML

- linear regression: $f(w) = \|Xw - y\|^2$ is convex $\to$ GD finds the global MSE minimizer, guaranteed
- logistic regression: convex loss $\to$ no worry about local minima
- SVM: convex formulation $\to$ unique solution (with $\ell_2$ regularization)
- deep learning: non-convex! that's why training is hard and you need careful initialization, learning rate scheduling, etc.
- the whole message: convexity is what makes optimization "easy" (polynomial time, global guarantees)
- in some sense, half the effort in ML is in formulating problems as convex optimization

## Summary table: convex vs general functions

| Property | Convex | General (non-convex) |
|----------|--------|---------|
| Local min $\Rightarrow$ Global min? | **Yes** | No |
| $\nabla f = 0 \Rightarrow$ Global min? | **Yes** | No (could be saddle/local max) |
| GD finds global opt? | **Yes (given enough steps)** | Usually no |
| Tangent always below curve? | **Yes** | No |

## Two styles of proof in this lecture

- forward direction ($\Rightarrow$): calculus style — Taylor expansion, limits, careful $o(\cdot)$ handling
- reverse direction ($\Leftarrow$): algebraic style — expand at middle point, weighted sum cancels gradient
- local=global proof: convex combination trick to "pull" far-away point into local neighborhood
- stationary point proof: direct one-line application of first-order characterization
- all these techniques will reappear in the convergence proofs of lec 10-11

## Looking ahead

- the first-order characterization $f(x^k) - f(x^*) \leq (g^k)^T(x^k - x^*)$ will be the starting point of ALL convergence proofs in lec 10 and 11
- lec 10: under bounded gradients, we get $O(1/\sqrt{K})$ convergence — the first real rate!
- lec 11: under smoothness, we get $O(1/K)$ — the improvement from extra structure is striking
- so the story is: stronger assumptions $\to$ faster convergence. this lecture gives us the foundation to prove all of that
- the proof techniques from today (Taylor expansion + limits, convex combination tricks) will be reused many times
<!-- 下一讲终于要证明GD真的收敛了！一阶刻画是所有证明的起手式 -->
