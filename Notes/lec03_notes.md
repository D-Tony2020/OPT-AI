# Lec 3 - Jan 28 - Deep Learning Optimization & Gradient Descent Basics

## overview

This lecture does two things. First half: combine everything from Lec 1-2 to write down the full deep learning MLE optimization problem. Second half: introduce gradient descent — the core algorithm for solving it. We go from "what problem are we solving" to "how do we start solving it." The math gets more intense this lecture.

## the complete deep learning optimization problem

- recall from Lec 2: neural network outputs learned features $z_j = z(x_j, Q^1, g^1, \ldots, Q^L, g^L)$
- then Softmax classifies using these features
- likelihood for entire dataset:
$$L = \prod_{j=1}^M \frac{e^{w_{y_j}^\top z_j}}{\sum_{\ell=1}^K e^{w_\ell^\top z_j}}$$
- taking log and rearranging, we get the MLE problem:
$$\max \left\{ -\sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top z_j}\right) + \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbb{1}(y_j = k) \, w_k^\top z_j \right\}$$

- key difference from Lec 2's multi-class logistic regression: features $z_j$ are NOT fixed, they depend on network parameters $(Q^\ell, g^\ell)$
  - this makes it **non-convex** (nested nonlinear activations destroy convexity)
  - in logistic regression, $x_j$ was constant so negative log-likelihood was convex
  - now $z_j$ itself is a complicated function of the parameters... much harder
<!-- 从凸变非凸，"一字之差"改变了一切 -->

### parameter count

- total parameters = network params + classifier params:
$$P = \sum_{\ell=1}^L d_\ell(d_{\ell-1} + 1) + K d_L$$
- for typical image classifier: ~100 million params
- for GPT-3: ~175 billion params
- we are optimizing over a space with 10^8 to 10^11 dimensions... that is insane

### connection back to ERM (full circle from Lec 1)

- flipping sign: $\min_\theta \frac{1}{M}\sum_j [\text{cross-entropy loss for point } j]$
- this is exactly the ERM framework from Lec 1 slide 3!
- the journey: general ERM (Lec 1) → Softmax + CE (Lec 2) → add deep network features (Lec 3)
- pretty satisfying to see how the framework from week 1 now encompasses deep learning
- now we have the "problem" fully defined. rest of course = how to solve it
- from here the course switches from "what are we solving" to "how do we solve it"

## gradient descent — the core algorithm

### what is a gradient

- gradient $\nabla f(x^0)$ is a vector of partial derivatives:
$$\nabla f(x^0) = \left(\frac{\partial f}{\partial x_1}\bigg|_{x^0}, \ldots, \frac{\partial f}{\partial x_n}\bigg|_{x^0}\right)$$
- each component tells you: if I slightly increase $x_i$, how much does $f$ change?
- gradient points in the direction of steepest ascent (can prove via Cauchy-Schwarz)
- so $-\nabla f$ is direction of steepest descent
- gradient descent update rule:
$$x^{k+1} = x^k - \alpha \nabla f(x^k)$$
  (move a small step in the negative gradient direction)

- the "blind hiker" analogy: you're blindfolded on a mountain, can only feel the slope under your feet. you step in the steepest downhill direction each time. that's gradient descent.

- **why small step matters**: by Taylor expansion,
$$f(x^0 - \alpha \nabla f(x^0)) \approx f(x^0) - \alpha \|\nabla f(x^0)\|^2$$
  the $-\alpha\|\nabla f\|^2$ term guarantees decrease... but only when $\alpha$ is small enough for the approximation to hold
<!-- 梯度就是"每个参数该怎么调"的处方，但这个处方只在局部有效 -->

### numerical example — step size 0.1 (success)

- $f(x) = x_1^2 + 4x_1 x_2 + 3x_2^2 - 5$
- gradient: $\nabla f = (2x_1 + 4x_2, \; 4x_1 + 6x_2)^\top$
- at $x^0 = (1, 1)^\top$: $\nabla f = (6, 10)^\top$
- with $\alpha = 0.1$: new point = $(1,1) - 0.1(6,10) = (0.4, 0)$
- $f(x^0) = 3$, $f(x^1) = -4.84$. It worked! function value dropped by 7.84

- I worked this out by hand and it checks out. the function is quadratic, can be written as $x^\top A x - 5$ where $A = \begin{pmatrix} 1 & 2 \\ 2 & 3 \end{pmatrix}$. Both eigenvalues positive (~4.24 and ~0.76) so $f$ is strictly convex. minimum at $(0,0)$, $f^* = -5$.

### numerical example — step size 0.5 (DISASTER)

- same starting point, same gradient, but $\alpha = 0.5$
- new point = $(1,1) - 0.5(6,10) = (-2, -4)$
- $f(-2, -4) = 79$. Function went from 3 to 79!!!
- step size increased only 5x but result went from great to catastrophic

- **why it failed**: the new point $(-2,-4)$ is distance $\sqrt{34} \approx 5.83$ from start, but the minimum is only $\sqrt{2} \approx 1.41$ away. we completely overshot the minimum and landed on the other side in a high-value region
- for this quadratic, the max eigenvalue of $A$ is ~4.24, so the step size upper bound is ~$1/4.24 \approx 0.236$. $\alpha = 0.1 < 0.236$ works, $\alpha = 0.5 > 0.236$ fails.
- this is exactly what happens when learning rate is too high in neural network training — loss explodes to NaN
<!-- 这个反例太有说服力了，步长选择真的是生死攸关 -->

### the overshoot diagram

- prof showed a 1D picture that is very clear:
  - if step size is small enough, the new point lands on the downhill side → function value decreases
  - if step size too large, new point jumps past the minimum to the other side → function value INCREASES
- this is the visual version of what happened in the numerical example
- in neural network training, this looks like: loss suddenly spikes to NaN

### step size selection — backtracking (trial and error)

- the basic algorithm:
  1. start at $x^k$, compute gradient $\nabla f(x^k)$
  2. pick a step size $\alpha$
  3. check: is $f(x^k - \alpha \nabla f(x^k)) < f(x^k)$?
  4. if yes, take the step. if no, halve $\alpha$ and try again

- this is called **Backtracking Line Search**
- guaranteed to terminate because Taylor expansion ensures small enough $\alpha$ always works
- each "halving" attempt requires evaluating $f$ once = one forward pass through all data
  - expensive when $M$ is large, which motivates SGD (coming in Lec 10-12)
- a more formal version is the **Armijo condition**: require $f(x^{k+1}) \leq f(x^k) - c\alpha\|\nabla f\|^2$ for some $c \in (0,1)$
  - prof's version is the $c = 0$ simplification

### how this connects to PyTorch

- loss.backward() = compute gradient
- optimizer.step() = take one gradient descent step
- learning rate = step size $\alpha$
- the whole training loop is just repeating: compute gradient → choose step → move → check

## what I'm not 100% sure about yet

- the proof that backtracking always terminates — I get the Taylor expansion argument intuitively but not sure I could write a rigorous proof
- the connection between eigenvalues of the Hessian and the step size bound — need to review this before midterm
- how does this work in practice for non-convex deep learning? prof mentioned that local minima in large networks tend to be "good enough" but didn't prove it

## takeaway

- the full DL training problem is now on paper: minimize cross-entropy over all network + classifier parameters. it's a massive non-convex optimization problem.
- gradient descent is the workhorse: compute gradient, take step, repeat
- step size is CRITICAL — too small = slow, too large = explosion. this is the #1 practical issue
- next lecture should go deeper into when gradient descent actually works (convexity?)
