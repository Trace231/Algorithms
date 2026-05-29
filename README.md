# Formalized Algorithms

## Stochastic Mirror Descent

File: [`StochasticMirrorDescent.lean`](StochasticMirrorDescent.lean)

Source target: Lan, FOML, Section 4.1, Eq. (4.1.6), Theorem 4.1.

### Problem and Geometry

$$
f^*=\min_{x\in X} f(x),
\qquad
f(x)=\mathbb{E}[F(x,\xi)] .
$$

Lean:
- `MirrorDescentSetup` stores $X$, $F$, the SFO, the DGF, and the source assumptions.
- `MirrorDescentProcess.f` is the objective expectation.
- `MirrorDescentProcess.xStar` and `MirrorDescentProcess.fStar` realize the optimizer and optimum value.

The Bregman divergence is

$$
V(x,z)=v(z)-v(x)-\langle \nabla v(x),z-x\rangle .
$$

Lean:
- Top-level `V` wraps the SOptLib Bregman divergence.
- `MirrorDescentSetup.V`, `literalV`, and `literalVIntrinsic` distinguish the carrier-safe Bregman object from literal interior-gradient formulas.
- `V_lower_bound_of_intrinsicInterior`, `V_nonneg_on_carrier`, and related boundary lemmas supply the paper's $1$-strong convexity consequences on the feasible carrier.

### Update

$$
x_{t+1}
=\operatorname*{argmin}_{x\in X}
\left\{\gamma_t\langle G(x_t,\xi_t),x\rangle+V(x_t,x)\right\}.
$$

Lean:
- `paperMirrorObjective` is the displayed objective.
- `IsMirrorStep` states the argmin property.
- `proxStep`, `mirrorStep`, `proxStep_isMirrorStep`, and `mirrorStep_isMirrorStep` realize the update.
- `MirrorDescentProcess.x` and `MirrorDescentProcess.G` define the stochastic iterate and sampled oracle process.

### Output and Bound

$$
\bar{x}_s^k
=
\left(\sum_{t=s}^{k}\gamma_t\right)^{-1}
\sum_{t=s}^{k}\gamma_t x_t .
$$

Lean:
- `MirrorDescentSetup.OutputWindow` indexes the window $s,\ldots,k$.
- `MirrorDescentProcess.xBar` is the weighted output.
- `xBar_jensen_gap` is the Jensen step for convexity of $f$.

Theorem 4.1:

$$
\mathbb{E}[f(\bar{x}_s^k)]-f^*
\le
\left(\sum_{t=s}^{k}\gamma_t\right)^{-1}
\left[
\mathbb{E}[V(x_s,x^*)]
+(M^2+\sigma^2)\sum_{t=s}^{k}\gamma_t^2
\right].
$$

Lean:
- `lemma_3_4` is the mirror three-point descent lemma used in the update analysis.
- `stochasticMirrorDescent_oneStep_pathwise` proves the one-step pathwise bound.
- `stochasticMirrorDescent_martingale_term_integral_zero` formalizes the cancellation of $\mathbb{E}[\langle\delta_t,x_t-x^*\rangle]$.
- `summed_one_step_gap_bound` and `window_variance_sum_expectation_bound` assemble the telescope and variance terms.
- `stochasticMirrorDescent_convergence` is the final theorem.

## Variance-Reduced Mirror Descent

File: [`VarianceReducedMirrorDescent.lean`](VarianceReducedMirrorDescent.lean)

Source target: Lan, FOML, Section 5.3, Algorithm 5.6, Corollary 5.8.

### Composite Finite-Sum Model

$$
\min_{x\in X}\Psi(x),
\qquad
\Psi(x)=f(x)+h(x),
\qquad
f(x)=\frac{1}{m}\sum_{i=1}^{m} f_i(x).
$$

Lean:
- `VarianceReducedMirrorDescentSetup` stores $X$, $f_i$, $h$, $v$, $L_i$, $q_i$, and the sampled index stream.
- `componentOn`, `fOn`, `hOn`, `Psi`, and `PsiOn` are the carrier-facing functions.
- `componentGradOn`, `gradFOn`, `componentGrad`, and `gradF` implement the source gradients with boundary-safe carrier semantics.

The sampling constant is

$$
L_Q=\frac{1}{m}\max_{1\le i\le m}\frac{L_i}{q_i}.
$$

Lean:
- `vrmdLQCore` and `VarianceReducedMirrorDescentSetup.LQ` implement $L_Q$.
- `Q`, `Q_apply`, and `sampledIndexAt_law_Q` implement the finite law $Q=\{q_i\}$.

### Estimator and Prox Step

$$
G_t
=
\frac{\nabla f_{i_t}(x_t)-\nabla f_{i_t}(\tilde{x})}{q_{i_t}m}
+\nabla f(\tilde{x}).
$$

Lean:
- `estimator` is the displayed $G_t$.
- `delta` and `deltaProcessAt` represent $G_t-\nabla f(x_t)$.
- `delta_condexp_eq_zero_at_epoch` and `delta_sq_condexp_le_two_composite_gaps_at_epoch` formalize the two central stochastic estimator facts.

The prox update is

$$
x_{t+1}
=
\operatorname*{argmin}_{x\in X}
\left\{\gamma\,[\langle G_t,x\rangle+h(x)]+V(x_t,x)\right\}.
$$

Lean:
- `vrmdProxObjectiveCore` and `proxObjective` are the displayed objective.
- `proxPoint`, `proxStep`, `proxStep_is_argmin`, and `paperNextInnerIterAt_eq_proxStep` realize the update in the epoch process.

### Epochs, Output, and Corollary 5.8

$$
x^s=x_{T_s+1},
\qquad
\tilde{x}^{s}
=
\frac{\sum_{t=2}^{T_s}\theta_t x_t}{\sum_{t=2}^{T_s}\theta_t}.
$$

$$
\bar{x}^{S}
=
\frac{\sum_{s=1}^{S}w_s\tilde{x}^{s}}{\sum_{s=1}^{S}w_s}.
$$

Lean:
- `PaperEpochState`, `paperEpoch`, `paperInnerIterAt`, `xEpochIter`, and `snapshotIter` encode Algorithm 5.6 epochs.
- `T`, `θ`, `w`, `paperWeightFormula`, and `outputWeightSum` encode the Corollary 5.8 schedule.
- `barXCarrier` and `barX` encode the final weighted output.

With

$$
\theta=1,\qquad \gamma=\frac{1}{16L_Q},\qquad
T_1=7,\qquad T_s=2T_{s-1},
$$

Corollary 5.8 states

$$
\mathbb{E}[\Psi(\bar{x}^{S})-\Psi(x^*)]
\le
\frac{8}{2^S-1}
\left[
\frac{11}{4}\bigl(\Psi(x^0)-\Psi(x^*)\bigr)
+16L_QV(x^0,x^*)
\right].
$$

Lean:
- `lemma_5_12`, `lemma_5_13`, and `lemma_5_14` are the local variance-reduced descent chain.
- `theorem_5_6` is the epoch telescope.
- `corollary_5_8` is the final convergence bound.
- `corollary_5_8_sfo_complexity_rate` formalizes the displayed SFO complexity rate.

## Nonconvex Stochastic Mirror Descent

File: [`NonconvexStochasticMirrorDescent.lean`](NonconvexStochasticMirrorDescent.lean)

Source target: Lan, FOML, Section 6.2.3.2, Theorem 6.7.

Namespace: `SGD.NonconvexStochasticMirrorDescent`.

### Problem and Prox Mapping

$$
\Psi^*=\min_{x\in X}\Psi(x),
\qquad
\Psi(x)=f(x)+h(x),
$$

where $f$ is $L$-smooth and may be nonconvex, while $h$ is simple convex.

Lean:
- `Setup` stores $X$, $f$, $h$, $\nu$, the SFO kernel, and all paper parameters.
- `SimpleConvexTerm` models the simple convex term $h:X\to\mathbb{R}$.
- `Psi`, `PsiAmbient`, and `PsiStar` encode the composite objective and optimum.

The prox point and projected-gradient mapping are

$$
x^+
=
\operatorname*{argmin}_{u\in X}
\left\{
\langle g,u\rangle+\gamma^{-1}V(x,u)+h(u)
\right\},
\qquad
P_X(x,g,\gamma)=\gamma^{-1}(x-x^+).
$$

Lean:
- `paperProxObjective` is the displayed prox objective.
- `PaperProxSelector` stores the source argmin selector.
- `proxPoint`, `proxObjective`, and `projectedGradient` implement $x^+$ and $P_X$.
- `proposition_6_1_projectedGradient_lipschitz` is the projected-gradient stability theorem used in validation.

### RSMD Runs and Stopping Law

The mini-batch oracle is

$$
G_k
=
\frac{1}{m_k}\sum_{i=1}^{m_k}G(x_k,\xi_{k,i}),
$$

and the RSMD step is

$$
x_{k+1}
=
\operatorname*{argmin}_{u\in X}
\left\{
\langle G_k,u\rangle+\gamma_k^{-1}V(x_k,u)+h(u)
\right\}.
$$

Lean:
- `miniBatchOracle` is $G_k$.
- `step`, `process`, and `iterate` define each RSMD run.
- `StochasticRealization` stores the sample stream, while `OracleAssumptions` stores unbiasedness and variance assumptions.

The randomized stopping law is

$$
\mathbb{P}\{R=k\}
=
\frac{\gamma_k-L\gamma_k^2}
{\sum_{j=1}^{N}(\gamma_j-L\gamma_j^2)} .
$$

Lean:
- `outputMass` is the displayed mass.
- `stoppingPMF` is the finite PMF on output times.
- `randomizedOutput` is the selected iterate from one run.

### Two-Phase Selection and Theorem 6.7

Validation uses

$$
\bar{G}_T(x)=\frac{1}{T}\sum_{k=1}^{T}G(x,\xi_k),
\qquad
\bar{g}_X(\bar{x}_s)
=
P_X(\bar{x}_s,\bar{G}_T(\bar{x}_s),\gamma_{R_s}).
$$

The final candidate satisfies

$$
\|\bar{g}_X(\bar{x}^*)\|
=
\min_{1\le s\le S}\|\bar{g}_X(\bar{x}_s)\|.
$$

Lean:
- `empiricalOracleAverage`, `empiricalProjectedGradient`, and `runEmpiricalProjectedGradient` encode validation.
- `selectedRun`, `selectedOutput`, and `selectedRun_spec` encode the minimum empirical projected-gradient choice.
- `optimizationMinTailProbability`, `validationErrorProbability`, and `selectedTailProbability` keep the two probability terms separate.

Theorem 6.7(a):

$$
\mathbb{P}
\left\{
\|g_X(\bar{x}^*)\|^2
\ge
2\left(4L\mathscr{B}_{\bar{N}}+\frac{3\lambda\sigma^2}{T}\right)
\right\}
\le
\frac{S}{\lambda}+2^{-S}.
$$

Lean:
- `eq_6_2_61`, `eq_6_2_62`, `eq_6_2_63`, and `eq_6_2_64` formalize the decomposition, independent-run tail, and validation tail.
- `theorem_6_7a` is the high-probability bound.
- `runCountChoice`, `budgetChoice`, `validationCountChoice`, `theorem_6_7b`, and `theorem_6_7b_totalized` encode the $(\epsilon,\Lambda)$ parameter choices and SFO-call bound.

## Stochastic Block Mirror Descent

File: [`StochasticBlockMirrorDescent.lean`](StochasticBlockMirrorDescent.lean)

Source target: Lan, FOML, Section 4.6, Algorithm 4.5, Theorem 4.12.

### Block Model

$$
f^*=\min_{x\in X}f(x),
\qquad
f(x)=\mathbb{E}[F(x,\xi)] .
$$

$$
X=X_1\times X_2\times\cdots\times X_b\subseteq\mathbb{R}^{n},
\qquad
X_i\subseteq\mathbb{R}^{n_i},
\qquad
\sum_{i=1}^{b}n_i=n .
$$

Lean:
- `StochasticBlockMirrorDescentSetup` stores the source data.
- `StateSpace` is the dependent product $\prod_i\mathbb{R}^{n_i}$, implemented as `PiLp 2`.
- `X` is the product feasible set $\{x:\forall i,\ x_i\in X_i\}$.
- `blockCoord` implements $U_i^Tx=x^{(i)}$.

### Oracle, Norms, and Bregman Blocks

The oracle assumptions are

$$
\mathbb{E}[G(x,\xi)]=g(x)\in\partial f(x),
\qquad
G_i(x,\xi)=U_i^TG(x,\xi).
$$

$$
\mathbb{E}\left[\|G_i(x,\xi)\|_{i,*}^{2}\right]\le M_i^2 .
$$

Lean:
- `objective`, `objective_eq_integral`, `unbiased_oracleMean`, and `gBlock` implement $f$, $g$, and $G_i$.
- `block_second_moment_dualNorm`, `block_second_moment_expectation`, and `block_second_moment_stream` implement Eq. (4.6.4).
- `canonicalDualNormFromPrimal` derives $\|\cdot\|_{i,*}$ from the primal block seminorm.
- `blockDualNorm_support_bound` proves the paper's primal-dual support inequality.

The block Bregman divergence is

$$
V_i(z,x)
=
v_i(x)-v_i(z)-\langle\nabla v_i(z),x-z\rangle .
$$

Lean:
- `blockDGF` is the paper-facing restricted object $v_i:X_i\to\mathbb{R}$.
- `blockDGFAmbient` is only the Mathlib differentiability realization on $X_i$.
- `blockDivergence`, `blockDivergence_eq`, `blockDivergence_lower_bound`, and `blockDivergence_nonneg` encode Eq. (4.6.7).

### Sampling and One-Block Update

$$
\mathbb{P}\{i_k=i\}=p_i,
\qquad
p_i\in(0,1],
\qquad
\sum_{i=1}^{b}p_i=1 .
$$

Lean:
- `finiteBlockSamplingPMF`, `finiteBlockSamplingLaw`, and `finiteBlockSamplingLaw_singleton` build the finite law.
- `blockIndexLaw`, `blockIndexLaw_singleton`, `samplePairLaw`, and `P` build the generated same-time oracle/block stream.
- `currentSample_independent_past` and `currentSample_components_independent` encode the independence used in the proof.

Algorithm 4.5 updates only the sampled block:

$$
x_{k+1}^{(i)}
=
\begin{cases}
\operatorname*{argmin}_{u\in X_i}
\left\{
\langle G_i(x_k,\xi_k),u\rangle
+\gamma_k^{-1}V_i(x_k^{(i)},u)
\right\},
& i=i_k,\\[4pt]
x_k^{(i)}, & i\ne i_k .
\end{cases}
$$

Lean:
- `blockProxObjective` is the displayed one-block prox objective.
- `blockMirrorProx` performs the selected-block replacement.
- `blockMirrorProx_selected`, `blockMirrorProx_other`, and `blockMirrorProx_selected_isMinOn` state the two branches and the argmin certificate.
- `process`, `xIter_mem`, and `xIter_measurable_sampleFiltration` define the recursive stochastic process.

### Output, Telescope, and Theorem 4.12

The output uses $\theta_k=\gamma_k$:

$$
\bar{x}_N
=
\left(\sum_{k=1}^{N}\theta_k\right)^{-1}
\sum_{k=1}^{N}\theta_k x_k .
$$

Lean:
- `paperEta`, `paperTheta`, and `paperTheta_eq_paperEta` encode $\theta_k=\gamma_k$.
- `outputTimes`, `outputThetaSum`, `outputEtaSum`, and `weightedAverage` encode $\bar{x}_N$.

The aggregate potential in the proof is

$$
V(z,x)=\sum_{i=1}^{b}p_i^{-1}V_i(z^{(i)},x^{(i)}).
$$

Lean:
- `aggregatePotential` is this weighted block Bregman potential.
- `delta` and `deltaBar` are the stochastic linear and quadratic error terms.
- `step_4_6_18` is the aggregate one-step recursion.
- `step_4_6_20` is the finite-window telescope.
- `step_4_6_21` proves the martingale cancellation term.
- `step_4_6_22` proves the quadratic selected-block bound.

Theorem 4.12:

$$
\mathbb{E}[f(\bar{x}_N)-f(x_*)]
\le
\left(\sum_{k=1}^{N}\gamma_k\right)^{-1}
\left[
\sum_{i=1}^{b}p_i^{-1}V_i(x_1^{(i)},x_*^{(i)})
+\frac{1}{2}\sum_{k=1}^{N}\gamma_k^2
\sum_{i=1}^{b}M_i^2
\right].
$$

Lean:
- `weighted_average_output_mem_measurable` supplies output feasibility and measurability.
- `theorem_4_12_output_gap_integrable` proves the expectation is a genuine integral, not a totalized fallback.
- `theorem_4_12` is the final source theorem.
