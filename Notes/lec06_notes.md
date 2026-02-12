# Lec 6 - Feb 9 - Line Search in GD & Penalty Methods

## Exact line search for step size

- ok so this is where everything from lec 4-5 finally comes together
- in gradient descent we know the direction (negative gradient), but how far to go?
- the idea: treat step size alpha as a 1D optimization problem
- we want to solve min over alpha >= 0 of f(x^k - alpha * grad f(x^k))
- x^k is known, grad f is known, so this is literally just a function of alpha
- define phi(alpha) = f(x^k - alpha * grad f(x^k)), then minimize phi
- then we can use bisection search or golden section from lec 5 to find best alpha
- oh I see how this fits -- all those 1D methods were building up to exactly this moment

<!-- lec 4 的方向 + lec 5 的搜索 = 完整版梯度下降 -->

This is called **exact line search** -- we find the truly optimal alpha. In practice:
- exact line search is expensive: each alpha candidate needs a full function evaluation
- in deep learning, that means one full forward pass per candidate -- too costly
- say golden section needs 10 iterations to find alpha, that's 10 forward passes per GD step!
- practitioners use fixed learning rates + schedules, or adaptive methods (Adam, AdaGrad)
- but for theoretical analysis, exact line search gives the best-case convergence bound
- there's also **inexact line search** (Armijo condition, Wolfe condition) -- just needs alpha to be "good enough"
- now we can do "full" gradient descent: lec 3 gave us convexity, lec 4 gave us the direction, lec 5 gave us 1D tools, and now lec 6 connects them all

## Bisection search for step size -- example

- professor used a log-sum-exp function: f(x) = 4 ln(e^{2x1} + e^{4x2}) - 6x1 - 4x2
- gradient has those softmax-looking fractions, kind of messy but makes sense
  - grad_1 = 8e^{2x1}/(e^{2x1} + e^{4x2}) - 6
  - grad_2 = 16e^{4x2}/(e^{2x1} + e^{4x2}) - 4
- the gradient looks like softmax probabilities times constants minus constants -- this structure appears in cross-entropy loss gradients in ML
- you plot f(x^k - alpha * grad f(x^k)) as a curve over alpha, then do bisection on that curve
- pick initial interval [0, some large alpha_max], put test points near midpoint, compare, shrink
- the key insight is even though f lives in R^n, the step size problem is always 1D
- this is the same log-sum-exp that showed up as a convex function example in lec 3
- the level sets are like "rounded rectangles" because log-sum-exp approximates the max function
- the different curvatures in different directions is why GD might zigzag -- exact line search helps but doesn't fully fix this

## Constrained optimization -- the big picture

- now we move to constrained optimization, this feels like a new chapter
- general form: min f(x) subject to g_i(x) <= 0 (inequality) and h_j(x) = 0 (equality)
- professor said the "crude" methods all work by converting constrained problems back into unconstrained ones
- so we don't need brand new algorithms, we just reuse gradient descent!
- I like this approach, very practical -- transform the problem instead of inventing new tools
- he keeps saying "crude" which means there are better methods coming later (KKT, Lagrangian)

<!-- "crude" 方法 = 把约束塞进目标函数，继续用 GD -->

Why constrained optimization matters for ML:
- SVM: min ||w||^2 subject to margin constraints
- probability distributions: sum to 1, all non-negative
- regularization: L1/L2 constraints are equivalent to penalty terms (this connection is about to become very clear)

## Penalty function method

- core idea is actually really intuitive
- you take the constraints and shove them into the objective function as penalty terms
- if you violate a constraint, you pay a big cost
- for inequality g_i(x) <= 0: add theta_i * (max{0, g_i(x)})^2
  - when constraint is satisfied, max is 0, no penalty
  - when violated, penalty grows with the square of violation
- for equality h_i(x) = 0: add gamma_i * (h_i(x))^2
  - any deviation from zero gets penalized
- theta and gamma are large penalty parameters

The connection to ML regularization is very clear now:
- L2 regularization lambda * ||w||^2 is literally a quadratic penalty on parameter size
- the constraint "||w|| should be small" becomes a penalty term in the loss
- penalty parameter theta plays the same role as regularization coefficient lambda

## Why square the penalty?

- this is a detail I wouldn't have thought about but its actually important
- max{0, g(x)} by itself has a sharp corner where g(x) = 0 -- not differentiable there
- gradient descent needs derivatives so thats a problem
- but (max{0, g(x)})^2 smooths out the corner, becomes differentiable
- the math: from the left r'(b) = 0, from the right r'(x) = 2g(x)g'(x) which also goes to 0 at b
- left derivative = right derivative, so its smooth. kind of elegant actually

This "smoothing" trick appears everywhere in optimization:
- ReLU (sharp corner) vs smooth approximations
- Huber loss smooths L1 at zero
- softmax smooths max function
- log-barrier (next lecture) also achieves smoothness

<!-- 取平方 = 磨圆尖角，和 ReLU 的不可微问题是同一类 -->

## The full algorithm

- step 1: pick initial penalty params, tolerance epsilon, magnification beta > 1
- step 2: solve the unconstrained penalized problem (using GD), get x^k
- step 3: check if total penalty terms <= epsilon
  - if yes, constraints are basically satisfied, stop
  - if no, multiply all penalty params by beta, go back to step 2
- so you gradually increase the penalty instead of starting huge
- starting with huge params causes numerical issues (ill-conditioning), the gradual approach is better
- each round you use previous solution as starting point (warm start)
- beta should not be too large (like 100), otherwise penalty gradient overwhelms objective gradient
  - typical choice: beta = 2 or beta = 10

## Example problem

- min (x1 - 2)^4 + (x1 - 2x2)^2 subject to x1^2 - x2 <= 0
- the constraint means x2 >= x1^2, so feasible region is above the parabola
- unconstrained minimum is at (2, 1), but 2^2 = 4 > 1, so it violates the constraint!
- so the unconstrained optimum is NOT feasible -- we need the penalty to push us back
- penalized problem: min (x1-2)^4 + (x1-2x2)^2 + theta*(max{0, x1^2 - x2})^2
- as theta increases, solution gets pushed toward the constraint boundary x2 = x1^2
- one limitation: the solution is technically slightly infeasible, constraints only approximately satisfied
- professor hinted barrier methods (lec 7) will fix this -- they keep you inside the feasible region
- interesting observation: the optimal solution is on the constraint boundary. this is typical -- in constrained optimization, optima often sit on the boundary (this relates to KKT complementary slackness)

## Practical concerns with penalty method

- **ill-conditioning**: when theta is very large, the Hessian of the penalty term dominates, making the condition number bad. GD slows down significantly on the inner loop
- **accuracy vs difficulty tradeoff**: more accurate constraint satisfaction (larger theta) = harder subproblem. this is the fundamental curse of penalty methods
- **warm starting is essential**: without using x^{k-1} as initial point for the next round, you waste a lot of compute re-solving from scratch
- these issues motivate both barrier method (lec 7) and augmented Lagrangian (more advanced)

## Big takeaways

- exact line search completes the GD framework from lec 3-5, but is rarely used in practice
- penalty method is a beautifully simple idea: violate constraints -> pay a fine
- the squared penalty trick for differentiability is something I should remember
- this "convert constrained to unconstrained" philosophy is very powerful and practical
- looking forward to lec 7's barrier method, which sounds like penalty but from the inside
- now I can do exact line search for GD AND handle constraints -- my optimization toolbox is getting real
- the connection between penalty params and regularization coefficients is a nice insight from this lecture
- professor keeps calling these methods "crude" but they seem very useful for engineering practice
