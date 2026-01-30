# Lec 2 - Jan 26 - From Logistic Regression to Deep Learning

## overview

This lecture is a direct continuation of Lec 1. We finish the logistic regression derivation, then generalize in two directions: (1) binary → multi-class classification via Softmax, (2) fixed features → learned features via neural networks. By the end, we have the full deep learning optimization problem written down.

## completing the log-likelihood derivation

- recall from Lec 1: we have the likelihood $L(w) = \prod_j P(y_j | x_j, w)$ and we want to maximize it
- taking the log of each data point's likelihood:
  - if $y_j = +1$: $\ln P = -\ln(1 + e^{w^\top x_j})$
  - if $y_j = -1$: $\ln P = w^\top x_j - \ln(1 + e^{w^\top x_j})$
- both cases share the $-\ln(1 + e^{w^\top x_j})$ term, difference is the extra $w^\top x_j$ when $y_j = -1$
- using indicator function $\mathbb{1}(y_j = -1)$ to unify:
$$\ln L(w) = -\sum_{j=1}^{M} \ln(1 + e^{w^\top x_j}) + \sum_{j=1}^{M} \mathbb{1}(y_j = -1) \cdot w^\top x_j$$
- so the optimization problem is: $\max_{w} \ln L(w)$ — unconstrained, nonlinear

- important insight I didn't realize before: **negative log-likelihood for logistic regression is convex**
  - $\ln(1 + e^z)$ is convex in $z$ (second derivative $= \sigma(z)(1-\sigma(z)) > 0$)
  - sum of convex functions is convex
  - so gradient descent is guaranteed to find the global optimum here
  - this connects to the Lec 1 lesson about nonlinear optimization failing — convex problems don't have that issue!
<!-- 凸性保证了逻辑回归不会像盒子问题那样"初始点选错就完蛋" -->

### the log-odds perspective

- I found this interesting: logistic regression actually models the log-odds as a linear function of features
- odds = $P(+1)/P(-1) = e^{-w^\top x}$, so log-odds $= -w^\top x$
- this means logistic regression's fundamental assumption is "log-odds is linear in features"
- sigmoid is the ONLY function that satisfies this assumption — so it's not an arbitrary choice

## multi-class logistic regression (Softmax)

- now $y_j \in \{1, 2, \ldots, K\}$ instead of $\{+1, -1\}$
- prediction function uses **Softmax**:
$$\phi_k(x_j, w_1, \ldots, w_K) = \frac{e^{w_k^\top x_j}}{\sum_{\ell=1}^{K} e^{w_\ell^\top x_j}}$$
- this gives the probability that $x_j$ belongs to class $k$
- Softmax properties that I think are important:
  - all outputs positive and sum to 1 (valid probability distribution)
  - monotone: higher score $w_k^\top x_j$ → higher probability for class $k$
  - when $K=2$, Softmax reduces to sigmoid — so this is truly a generalization
- now we need $K$ parameter vectors $w_1, \ldots, w_K \in \mathbb{R}^n$, total $nK$ parameters
  - for binary case we only needed one $w$ (the "difference" vector $w_2 - w_1$)
- Softmax = smooth approximation of argmax. "Soft" because it gives probabilities instead of hard 0/1
<!-- Softmax这个名字原来是这个意思——soft版本的max -->

## multi-class log-likelihood

- same pattern as binary case: product of likelihoods → take log → sum
- likelihood for data point $j$ with label $y_j = k$:
$$P(y_j = k | x_j) = \frac{e^{w_k^\top x_j}}{\sum_\ell e^{w_\ell^\top x_j}}$$
- log-likelihood:
$$\ln L = \sum_{j=1}^{M} \sum_{k=1}^{K} \mathbb{1}(y_j = k) \, w_k^\top x_j - \sum_{j=1}^{M} \ln\left(\sum_{\ell=1}^{K} e^{w_\ell^\top x_j}\right)$$
- first term = reward correct class score, second term = penalize overall score inflation (log-sum-exp)
- prof's exam analogy was helpful: you want YOUR score high, but class average is also rising, so what matters is being significantly above average
- the negative log-likelihood = **Cross-Entropy Loss** — same thing different name
- still convex! (log-sum-exp is convex, subtract linear = still convex) so global optimum guaranteed

### the gradient has a nice form

- the gradient of log-likelihood w.r.t. $w_k$:
$$\nabla_{w_k} \ln L = \sum_{j=1}^M \left[\mathbb{1}(y_j = k) - \phi_k(x_j, w)\right] x_j$$
- intuition: it's (true label - predicted probability) times features. when prediction is accurate, residual is small, gradient is small → model doesn't change much. "learn a lot when wrong, little when right"
- this self-regulating behavior is very elegant

### numerical example

- prof showed a 2-class example with income and education features
- $w_A = (0.1, 0.5)$ emphasizes education, $w_B = (0.4, 0.1)$ emphasizes income
- computed by hand: for data point 1 (50K income, 11 years edu, true label A):
  - class A score: $e^{0.1 \times 50 + 0.5 \times 11} = e^{10.5}$
  - class B score: $e^{0.4 \times 50 + 0.1 \times 11} = e^{21.1}$
  - $P(A) = e^{10.5}/(e^{10.5} + e^{21.1}) \approx 0$ — model thinks it's class B with near certainty!
- these are clearly bad parameters. MLE would find better ones. training = adjusting params to match labels

## deep learning introduction

- key limitation of multi-class logistic regression: relies on hand-crafted features $x_j$
- parameter count = $nK$. for ImageNet ($n$ = 150528 pixels, $K$ = 1000 classes) that's 150 million params for a LINEAR model — way too many and not expressive enough
- if raw features are bad (e.g. raw pixels), model fails no matter how good the optimizer is
- solution: **learn the features automatically** using a neural network

### neural network structure

- each layer does: affine transform + nonlinear activation
$$a_j^\ell = \sigma_\ell(Q^\ell a_j^{\ell-1} + g^\ell)$$
- $Q^\ell$: weight matrix, $g^\ell$: bias vector, $\sigma_\ell$: activation function
- input layer: $a_j^0 = x_j$

- dimension chain: $x_j \in \mathbb{R}^n \to a^1 \in \mathbb{R}^{d_1} \to a^2 \in \mathbb{R}^{d_2} \to \cdots \to a^L \in \mathbb{R}^{d_L}$
<!-- 维度追踪是理解网络的基本功 -->

- **why activation functions are necessary**: without nonlinearity, $L$ layers collapse to single linear transform $Q^L Q^{L-1} \cdots Q^1 x = Wx$. Nonlinearity is the source of depth's expressive power.

### sigmoid vs ReLU

| | Sigmoid | ReLU $\max(t, 0)$ |
|---|---|---|
| Output range | $(0, 1)$ | $[0, \infty)$ |
| Gradient issue | vanishes (max 0.25) | no vanishing (gradient = 1 for positive) |
| Compute cost | expensive (exp) | cheap (comparison) |
| Modern usage | rare in hidden layers | standard choice |

## deep learning = feature extractor + classifier

- the whole network can be written as one function:
$$z(x_j, Q^1, g^1, \ldots, Q^L, g^L) = \sigma_L(Q^L \sigma_{L-1}(\ldots \sigma_1(Q^1 x_j + g^1) \ldots) + g^L)$$
- then do Softmax classification on the learned features $z_j$
- this two-stage structure is everywhere: BERT = Transformer + linear head, ResNet = CNN + FC head, etc.
- all parameters (network + classifier) optimized jointly — **end-to-end training**
- but now the optimization problem is **non-convex** because of the nested nonlinearities
  - this is a big deal: we lose the "local = global" guarantee from convex case
  - explains why DL training can be sensitive to initialization and hyperparameters

## takeaway

- the derivation chain is: probability model → likelihood (product) → log-likelihood (sum) → optimization problem
  - this chain works for any parametric model, not just logistic regression
- Softmax generalizes sigmoid to multi-class, Cross-Entropy Loss = negative log-likelihood
- deep learning replaces fixed features with learnable features, but makes optimization non-convex
- from Lec 1 to here, the "why" is answered — DL training is a massive nonlinear optimization. starting next lecture, we learn "how" to solve it
