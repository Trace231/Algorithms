# Formalized Algorithms

## Stochastic Mirror Descent

File: `StochasticMirrorDescent.lean`

Source target: Lan, FOML, Section 4.1, Theorem 4.1.

Problem:
```text
f^* = min_{x in X} f(x),        f(x) = E[F(x, xi)].
```

Mirror geometry:
```text
V(x, z) = v(z) - v(x) - <nabla v(x), z - x>,
v is 1-strongly convex on X.
```

Update and output:
```text
x_{t+1} = argmin_{x in X} { gamma_t <G(x_t, xi_t), x> + V(x_t, x) },

bar{x}_s^k =
  (sum_{t=s}^k gamma_t)^{-1} sum_{t=s}^k gamma_t x_t.
```

Main bound:
```text
E[f(bar{x}_s^k)] - f^*
  <= (sum_{t=s}^k gamma_t)^{-1}
       [ E[V(x_s, x^*)]
         + (M^2 + sigma^2) sum_{t=s}^k gamma_t^2 ].
```

Formalization highlight: the proof separates the deterministic Bregman
three-point descent from the stochastic martingale cancellation
`E[<delta_t, x_t - x^*>] = 0`, with oracle unbiasedness and variance represented
as source-level assumptions rather than hidden regularity shortcuts.

## Variance-Reduced Mirror Descent

File: `VarianceReducedMirrorDescent.lean`

Source target: Lan, FOML, Section 5.3, Algorithm 5.6 and Corollary 5.8.

Composite finite-sum problem:
```text
min_{x in X} Psi(x) = f(x) + h(x),
f(x) = (1/m) sum_{i=1}^m f_i(x).
```

Sampling constant:
```text
L_Q = (1/m) max_i L_i / q_i.
```

Variance-reduced estimator and prox step:
```text
G_t = [nabla f_{i_t}(x_t) - nabla f_{i_t}(tilde{x})] / (q_{i_t} m)
        + nabla f(tilde{x}),

x_{t+1} = argmin_{x in X}
  { gamma [<G_t, x> + h(x)] + V(x_t, x) }.
```

Epoch/output formulas:
```text
x^s = x_{T_s+1},
tilde{x}^s = (sum_{t=2}^{T_s} theta_t x_t) /
             (sum_{t=2}^{T_s} theta_t),

bar{x}^S = (sum_{s=1}^S w_s tilde{x}^s) / (sum_{s=1}^S w_s).
```

Corollary 5.8 parameter choice and bound:
```text
theta = 1,        gamma = 1 / (16 L_Q),        T_1 = 7,
T_s = 2 T_{s-1},

E[Psi(bar{x}^S) - Psi(x^*)]
  <= 8/(2^S - 1)
       [ (11/4)(Psi(x^0) - Psi(x^*)) + 16 L_Q V(x^0, x^*) ].
```

Formalization highlight: the Lean model keeps the component gradients, sampling
law `Q`, epoch recursion, snapshot averages, and composite prox objective
separate. The component convexity needed for the co-coercivity route is modeled
explicitly from the Section 5.3 prose setup.

## Nonconvex Stochastic Mirror Descent

File: `NonconvexStochasticMirrorDescent.lean`

Source target: Lan, FOML, Section 6.2.3.2, Theorem 6.7.

Composite nonconvex problem:
```text
Psi^* = min_{x in X} Psi(x),        Psi(x) = f(x) + h(x),
f is L-smooth and may be nonconvex,        h is simple convex.
```

Bregman prox mapping:
```text
x^+ = argmin_{u in X}
  { <g, u> + gamma^{-1} V(x, u) + h(u) },

P_X(x, g, gamma) = gamma^{-1}(x - x^+).
```

Mini-batch RSMD step:
```text
G_k = (1/m_k) sum_{i=1}^{m_k} G(x_k, xi_{k,i}),

x_{k+1} = argmin_{u in X}
  { <G_k, u> + gamma_k^{-1} V(x_k, u) + h(u) }.
```

Randomized stopping and post-optimization selection:
```text
Prob{R = k} =
  (gamma_k - L gamma_k^2) /
  sum_{j=1}^N (gamma_j - L gamma_j^2),

bar{G}_T(x) = (1/T) sum_{k=1}^T G(x, xi_k),
bar{g}_X(bar{x}_s) = P_X(bar{x}_s, bar{G}_T(bar{x}_s), gamma_{R_s}),

||bar{g}_X(bar{x}^*)|| = min_s ||bar{g}_X(bar{x}_s)||.
```

High-probability theorem shape:
```text
Prob{ ||g_X(bar{x}^*)||^2
      >= 2(4L B_{bar{N}} + 3 lambda sigma^2 / T) }
  <= S/lambda + 2^{-S}.
```

Formalization highlight: the file models the two-phase construction directly:
independent optimization runs, fresh validation samples, the empirical projected
gradient selector, and the final tail bound are represented as separate objects
instead of being compressed into one expectation statement.

## Stochastic Block Mirror Descent

File: `StochasticBlockMirrorDescent.lean`

Source target: Lan, FOML, Section 4.6, Algorithm 4.5 and Theorem 4.12.

Block problem:
```text
f^* = min_{x in X} f(x),        f(x) = E[F(x, xi)],

X = X_1 x X_2 x ... x X_b subset R^n,
X_i subset R^{n_i},        sum_{i=1}^b n_i = n.
```

Oracle and block moment assumptions:
```text
E[G(x, xi)] = g(x) in partial f(x),
G_i(x, xi) = U_i^T G(x, xi),

E[ ||G_i(x, xi)||_{i,*}^2 ] <= M_i^2.
```

Block Bregman geometry:
```text
V_i(z, x) =
  v_i(x) - v_i(z) - <nabla v_i(z), x - z>,

v_i : X_i -> R is continuously differentiable and
1-strongly convex with respect to ||.||_i.
```

Sampling and update:
```text
Prob{i_k = i} = p_i,        p_i in (0, 1],        sum_i p_i = 1,

x_{k+1}^{(i)} =
  argmin_{u in X_i} { <G_i(x_k, xi_k), u>
                      + gamma_k^{-1} V_i(x_k^{(i)}, u) },
  if i = i_k,

x_{k+1}^{(i)} = x_k^{(i)},        if i != i_k.
```

Weighted output:
```text
theta_k = gamma_k,

bar{x}_N =
  (sum_{k=1}^N theta_k)^{-1} sum_{k=1}^N theta_k x_k.
```

Theorem 4.12:
```text
E[f(bar{x}_N) - f(x_*)]
  <= (sum_{k=1}^N gamma_k)^{-1}
       [ sum_{i=1}^b p_i^{-1} V_i(x_1^{(i)}, x_*^{(i)})
         + (1/2) sum_{k=1}^N gamma_k^2 sum_{i=1}^b M_i^2 ].
```

Formalization highlight: the Lean model uses a dependent product
`PiLp 2 (fun i => R^{n_i})` for the block state, derives the finite block PMF
from the real probabilities `p_i`, derives the dual norm from each primal block
norm, keeps `v_i : X_i -> R` as the paper-facing restricted DGF, and uses an
ambient DGF only as a Mathlib differentiability realization on `X_i`. The proof
route exposes the weighted aggregate potential
`sum_i p_i^{-1} V_i`, the martingale cancellation for the sampled block oracle,
and the quadratic term expansion
`E[bar{delta}_k] <= sum_i M_i^2`.
