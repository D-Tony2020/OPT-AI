# Lec 10 - Mar 2 - Convergence of Gradient Descent

## Why constant step size is tricky

- ok so today we finally get to the punchline — does gradient descent actually converge?
- short answer: with constant step size, not exactly. it bounces around the minimum
- but the average error still goes down, which is the whole point of this lecture
- finally seeing why GD actually works, even when it doesn't perfectly land on the answer
- using the first-order characterization from lec 09 as the key starting tool
<!-- 终于要证明GD收敛了！用上了lec 09的一阶刻画 -->

## The bouncing example

- take $f(x) = (x-1)^2$, start at $x^0 = 2$, step size $\gamma = 1$
- $f'(x) = 2(x-1)$, so $f'(2) = 2$
- $x^1 = 2 - 1 \cdot 2 = 0$. jumped right past the minimum!
- $f'(0) = -2$, so $x^2 = 0 - 1 \cdot (-2) = 2$. back to the start!
- $x$ goes $2 \to 0 \to 2 \to 0 \to \ldots$ forever bouncing, never hits $x^* = 1$
- the function value at 0 and 2 is both 1, so the error never improves
- this is because step size too big for this problem — the Lipschitz constant $L = 2$ so we need $\gamma < 1/L = 0.5$
- if we picked $\gamma = 0.5$: $x^1 = 2 - 0.5 \cdot 2 = 1 = x^*$ — one step, done!
- the takeaway: step size too small = you barely move, step size too big = you overshoot like crazy
- but we usually don't know the "right" $\gamma$ in advance, so we need general theory

## Setting up the proof (preliminary analysis)

- the convergence proof uses everything from lec 08-09, especially the first-order characterization of convexity
- this is a three-step framework that gets reused in lec 11 too — worth memorizing
- **Step 1**: by first-order characterization (lec 09):
$$f(x^k) - f(x^*) \leq (g^k)^T(x^k - x^*) \quad \text{where } g^k = \nabla f(x^k)$$
- so bounding the function gap turns into bounding an inner product, which is easier to work with
- **Step 2**: from the GD update rule, $g^k = (1/\gamma)(x^k - x^{k+1})$, just rearranging
- this substitution converts gradients (abstract) into iteration differences (geometric)
<!-- 套路：一阶刻画把函数值差 → 内积 → 代换梯度为迭代点的差 -->

## The norm identity trick

- key identity: $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2a^Tb$
- equivalently: $a^Tb = \frac{1}{2}(\|a\|^2 + \|b\|^2 - \|a-b\|^2)$
- this looks basic but its the move that makes the whole proof work
- it converts the inner product into differences of squared norms
- and differences of norms telescope! thats the magic
- applying this with $a = x^k - x^{k+1}$ and $b = x^k - x^*$:

$$(g^k)^T(x^k - x^*) = \frac{1}{\gamma}(x^k - x^{k+1})^T(x^k - x^*)$$
$$= \frac{1}{2\gamma}(\|x^k - x^{k+1}\|^2 + \|x^k - x^*\|^2 - \|x^{k+1} - x^*\|^2)$$
$$= \frac{\gamma}{2}\|g^k\|^2 + \frac{1}{2\gamma}(\|x^k - x^*\|^2 - \|x^{k+1} - x^*\|^2)$$

- the last term has the "current distance - next distance" structure — this telescopes!

## Telescoping sum

- **Step 3**: sum over $k = 0$ to $K-1$
- the $\|x^k - x^*\|^2 - \|x^{k+1} - x^*\|^2$ terms cancel in pairs:
  - $(d_0^2 - d_1^2) + (d_1^2 - d_2^2) + \cdots + (d_{K-1}^2 - d_K^2) = d_0^2 - d_K^2$
- only $\|x^0 - x^*\|^2$ survives at the start, and $\|x^K - x^*\|^2$ at the end
- since $\|x^K - x^*\|^2 \geq 0$, we can drop it to get an upper bound
- final cumulative error bound (the "preliminary analysis"):

$$\sum_{k=0}^{K-1}(f(x^k) - f(x^*)) \leq \frac{\gamma}{2}\sum_{k=0}^{K-1}\|g^k\|^2 + \frac{1}{2\gamma}\|x^0 - x^*\|^2$$

- this is like the discrete version of Newton-Leibniz formula: sum of differences = endpoint difference

## The two-term tradeoff

- two terms fighting each other:
  - first term ($\gamma/2 \cdot$ gradient norms): big step size $\to$ this grows $\to$ overshoot penalty
  - second term ($1/(2\gamma) \cdot$ initial distance): small step size $\to$ this grows $\to$ slow progress
- $\gamma$ small: "snail mode" — barely move, initial distance dominates
- $\gamma$ large: "drunk mode" — bounce everywhere, gradient term dominates
- optimal $\gamma$ balances them — this is a bias-variance tradeoff in disguise!
<!-- 两项trade-off：步长大→过冲，步长小→走不动。和ML里的bias-variance一样 -->

## The convergence theorem

- assume $f$ is convex and gradients are bounded: $\|\nabla f(x)\| \leq B$ for all $x$
- let $R = \|x^0 - x^*\|$ (how far we start from the answer)
- pick step size $\gamma = R/(B\sqrt{K})$
- then the average error satisfies:

$$\frac{1}{K}\sum_{k=0}^{K-1}(f(x^k) - f(x^*)) \leq \frac{RB}{\sqrt{K}}$$

- this $O(1/\sqrt{K})$ rate is important — this is the baseline for convex GD
- it means to get error $\leq \varepsilon$, you need $K = O(R^2B^2/\varepsilon^2)$ iterations
- not super fast (sublinear) but its the best you can do with just convexity + bounded gradients
- $RB$ measures problem difficulty: $R$ = how far you start, $B$ = how steep the function can be

## Proof sketch

- plug bounded gradient into the cumulative bound: $\sum\|g^k\|^2 \leq KB^2$
- upper bound becomes $H(\gamma) = \frac{1}{2}(KB^2\gamma + R^2/\gamma)$
- take derivative, set to zero: $H'(\gamma) = \frac{1}{2}(KB^2 - R^2/\gamma^2) = 0$
- solve: $\gamma^2 = R^2/(KB^2)$, so $\gamma^* = R/(B\sqrt{K})$
- plug back in: both terms become $\frac{1}{2}RB\sqrt{K}$, total is $RB\sqrt{K}$
- divide by $K$: average error $\leq RB/\sqrt{K}$
- note: at the optimal $\gamma$, the two terms are **exactly equal** — this is AM-GM equality condition
- the whole proof only uses convexity + norm identity + basic calculus. pretty clean
<!-- 最优步长让两项恰好相等，AM-GM取等的条件 -->

## The step size depends on K — kind of annoying

- $\gamma = R/(B\sqrt{K})$ means you need to decide how many steps to run BEFORE you start
- also need to know $R$ and $B$ — might be hard to estimate in practice
- in practice people just use constant step size anyway (like in deep learning) and it works fine
- also: $\min_k f(x^k) \leq \frac{1}{K}\sum f(x^k)$, so the "best iterate" also has this bound
- but you need to track all iterates and pick the best one — not ideal

## Iteration complexity

| Target $\varepsilon$ | Steps needed $K$ | Growth |
|---|---|---|
| 0.1 | $100 \cdot (RB)^2$ | — |
| 0.01 | $10000 \cdot (RB)^2$ | $100\times$ |
| 0.001 | $10^6 \cdot (RB)^2$ | $100\times$ |

- precision improves by 10x $\to$ computation increases by 100x. that's $O(1/\varepsilon^2)$. pretty expensive.
- for high-precision problems this is a lot of iterations

## The three-step framework (summary)

| Step | Math | Intuition |
|---|---|---|
| 1. First-order char | $f(x^k) - f(x^*) \leq (g^k)^T(x^k - x^*)$ | function gap $\to$ inner product |
| 2. Norm identity | inner product $\to$ norm differences | geometry replaces algebra |
| 3. Telescoping sum | sum of differences $\to$ first - last | single-step $\to$ cumulative bound |

- this exact framework is reused in lec 11 — the only difference is what we plug in for $\sum\|g^k\|^2$

## What this result does NOT give us

- we only bound the **average** error, not the last iterate — you don't know which step was best without tracking
- the step size $\gamma = R/(B\sqrt{K})$ needs $K$ in advance — not fully "online"
- no guarantee of monotone descent — function values can oscillate wildly
- all of these limitations will be addressed in lec 11 with the smoothness assumption

## Practical relevance

- in SGD (stochastic GD) for deep learning, the situation is similar: constant learning rate causes oscillation around the optimum
- learning rate decay schedules ($\gamma_k = \gamma_0/\sqrt{k}$) are motivated by this kind of analysis
- the $RB$ factor explains why initialization matters: starting closer to the optimum ($R$ small) helps
- and why gradient clipping helps: keeping $B$ small reduces the $RB$ product

## Big picture

- this lecture ties together the whole theory arc from the last few weeks
- we went from "what is convexity" (lec 03) to "GD converges at rate $O(1/\sqrt{K})$" which feels like a real result
- the preliminary analysis framework (first-order char $\to$ norm identity $\to$ telescoping) is **reusable** — lec 11 uses the exact same framework with a different "plugin"
- next step: what happens with stronger assumptions like smoothness? spoiler: $O(1/K)$, which is way better
- $O(1/\sqrt{K})$ vs $O(1/K)$ — smoothness really helps. that's the key comparison for the exam
<!-- O(1/√K) vs O(1/K) — smoothness really helps. 下一讲见分晓 -->
