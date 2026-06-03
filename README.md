# Formalized Algorithms

## Stochastic Mirror Descent

File: [`StochasticMirrorDescent.lean`](StochasticMirrorDescent.lean)

Source target: Lan, FOML, Section 4.1, Eq. (4.1.6), Theorem 4.1.

### Problem and Geometry

```math
f^{*} = \min_{x\in X} f(x)
```

```math
f(x) = \mathbb{E}[F(x,\xi)]
```

Lean objects:
- `MirrorDescentSetup` stores `X`, `F`, the stochastic first-order oracle, the distance-generating function, and the source assumptions.
- `MirrorDescentProcess.f` is the objective expectation.
- `MirrorDescentProcess.xStar` and `MirrorDescentProcess.fStar` realize the optimizer and optimum value.

The Bregman divergence is:

```math
V(x,z) = v(z)-v(x)-\langle \nabla v(x), z-x\rangle
```

Lean objects:
- Top-level `V` wraps the SOptLib Bregman divergence.
- `MirrorDescentSetup.V`, `literalV`, and `literalVIntrinsic` separate carrier-safe Bregman semantics from literal interior-gradient formulas.
- `V_lower_bound_of_intrinsicInterior` and `V_nonneg_on_carrier` supply the paper's strong-convexity consequences on the feasible carrier.

### Update

```math
x_{t+1} = \arg\min_{x\in X}\; \gamma_t\langle G(x_t,\xi_t),x\rangle + V(x_t,x)
```

Lean objects:
- `paperMirrorObjective` is the displayed objective.
- `IsMirrorStep` states the argmin property.
- `proxStep`, `mirrorStep`, `proxStep_isMirrorStep`, and `mirrorStep_isMirrorStep` realize the update.
- `MirrorDescentProcess.x` and `MirrorDescentProcess.G` define the stochastic iterate and sampled oracle process.

### Output and Bound

```math
\bar{x}_s^{k} = \frac{\sum_{t=s}^{k}\gamma_t x_t}{\sum_{t=s}^{k}\gamma_t}
```

Lean objects:
- `MirrorDescentSetup.OutputWindow` indexes the window.
- `MirrorDescentProcess.xBar` is the weighted output.
- `xBar_jensen_gap` is the Jensen step for convexity of `f`.

Theorem 4.1:

```math
\mathbb{E}[f(\bar{x}_s^{k})]-f^{*} \le \frac{\mathbb{E}[V(x_s,x^{*})] + (M^{2}+\sigma^{2})\sum_{t=s}^{k}\gamma_t^{2}}{\sum_{t=s}^{k}\gamma_t}
```

Lean objects:
- `lemma_3_4` is the mirror three-point descent lemma.
- `stochasticMirrorDescent_oneStep_pathwise` proves the one-step pathwise bound.
- `stochasticMirrorDescent_martingale_term_integral_zero` formalizes cancellation of the martingale inner product.
- `summed_one_step_gap_bound` and `window_variance_sum_expectation_bound` assemble the telescope and variance terms.
- `stochasticMirrorDescent_convergence` is the final theorem.

## Variance-Reduced Mirror Descent

File: [`VarianceReducedMirrorDescent.lean`](VarianceReducedMirrorDescent.lean)

Source target: Lan, FOML, Section 5.3, Algorithm 5.6, Corollary 5.8.

### Composite Finite-Sum Model

```math
\min_{x\in X}\Psi(x)
```

```math
\Psi(x)=f(x)+h(x)
```

```math
f(x)=\frac{1}{m}\sum_{i=1}^{m} f_i(x)
```

Lean objects:
- `VarianceReducedMirrorDescentSetup` stores `X`, component functions, `h`, `v`, `L_i`, `q_i`, and the sampled index stream.
- `componentOn`, `fOn`, `hOn`, `Psi`, and `PsiOn` are the carrier-facing functions.
- `componentGradOn`, `gradFOn`, `componentGrad`, and `gradF` implement source gradients with boundary-safe carrier semantics.

The sampling constant is:

```math
L_Q = \frac{1}{m}\max_{1\le i\le m}\frac{L_i}{q_i}
```

Lean objects:
- `vrmdLQCore` and `VarianceReducedMirrorDescentSetup.LQ` implement `L_Q`.
- `Q`, `Q_apply`, and `sampledIndexAt_law_Q` implement the finite sampling law.

### Estimator and Prox Step

```math
G_t = \frac{\nabla f_{i_t}(x_t)-\nabla f_{i_t}(\tilde{x})}{q_{i_t}m}+\nabla f(\tilde{x})
```

Lean objects:
- `estimator` is the displayed estimator.
- `delta` and `deltaProcessAt` represent `G_t - gradF x_t`.
- `delta_condexp_eq_zero_at_epoch` and `delta_sq_condexp_le_two_composite_gaps_at_epoch` formalize the key stochastic estimator facts.

The prox update is:

```math
x_{t+1} = \arg\min_{x\in X}\; \gamma(\langle G_t,x\rangle+h(x)) + V(x_t,x)
```

Lean objects:
- `vrmdProxObjectiveCore` and `proxObjective` are the displayed objective.
- `proxPoint`, `proxStep`, `proxStep_is_argmin`, and `paperNextInnerIterAt_eq_proxStep` realize the update in the epoch process.

### Epochs, Output, and Corollary 5.8

```math
x^{s}=x_{T_s+1}
```

```math
\tilde{x}^{s}=\frac{\sum_{t=2}^{T_s}\theta_t x_t}{\sum_{t=2}^{T_s}\theta_t}
```

```math
\bar{x}^{S}=\frac{\sum_{s=1}^{S}w_s\tilde{x}^{s}}{\sum_{s=1}^{S}w_s}
```

Lean objects:
- `PaperEpochState`, `paperEpoch`, `paperInnerIterAt`, `xEpochIter`, and `snapshotIter` encode Algorithm 5.6 epochs.
- `T`, `theta`, `w`, `paperWeightFormula`, and `outputWeightSum` encode the Corollary 5.8 schedule.
- `barXCarrier` and `barX` encode the final weighted output.

The Corollary 5.8 parameter choices are:

```math
\theta=1,\qquad \gamma=\frac{1}{16L_Q},\qquad T_1=7,\qquad T_s=2T_{s-1}
```

Corollary 5.8:

```math
\mathbb{E}[\Psi(\bar{x}^{S})-\Psi(x^{*})] \le \frac{8}{2^{S}-1}(\frac{11}{4}(\Psi(x^{0})-\Psi(x^{*}))+16L_QV(x^{0},x^{*}))
```

Lean objects:
- `lemma_5_12`, `lemma_5_13`, and `lemma_5_14` are the variance-reduced descent chain.
- `theorem_5_6` is the epoch telescope.
- `corollary_5_8` is the final convergence bound.
- `corollary_5_8_sfo_complexity_rate` formalizes the displayed SFO complexity rate.

## Nonconvex Stochastic Mirror Descent

File: [`NonconvexStochasticMirrorDescent.lean`](NonconvexStochasticMirrorDescent.lean)

Source target: Lan, FOML, Section 6.2.3.2, Theorem 6.7.

Namespace: `SGD.NonconvexStochasticMirrorDescent`.

### Problem and Prox Mapping

```math
\Psi^{*}=\min_{x\in X}\Psi(x)
```

```math
\Psi(x)=f(x)+h(x)
```

Lean objects:
- `Setup` stores `X`, `f`, `h`, `nu`, the stochastic oracle kernel, and the paper parameters.
- `SimpleConvexTerm` models the simple convex term.
- `Psi`, `PsiAmbient`, and `PsiStar` encode the composite objective and optimum.

The prox point and projected-gradient mapping are:

```math
x^{+}=\arg\min_{u\in X}\; \langle g,u\rangle+\gamma^{-1}V(x,u)+h(u)
```

```math
P_X(x,g,\gamma)=\gamma^{-1}(x-x^{+})
```

Lean objects:
- `paperProxObjective` is the displayed prox objective.
- `PaperProxSelector` stores the source argmin selector.
- `proxPoint`, `proxObjective`, and `projectedGradient` implement the prox point and projected-gradient map.
- `proposition_6_1_projectedGradient_lipschitz` is the projected-gradient stability theorem used in validation.

### RSMD Runs and Stopping Law

The mini-batch oracle is:

```math
G_k=\frac{1}{m_k}\sum_{i=1}^{m_k}G(x_k,\xi_{k,i})
```

The RSMD step is:

```math
x_{k+1}=\arg\min_{u\in X}\; \langle G_k,u\rangle+\gamma_k^{-1}V(x_k,u)+h(u)
```

Lean objects:
- `miniBatchOracle` is the mini-batch oracle.
- `step`, `process`, and `iterate` define each RSMD run.
- `StochasticRealization` stores the sample stream, while `OracleAssumptions` stores unbiasedness and variance assumptions.

The randomized stopping law is:

```math
\Pr(R=k)=\frac{\gamma_k-L\gamma_k^{2}}{\sum_{j=1}^{N}(\gamma_j-L\gamma_j^{2})}
```

Lean objects:
- `outputMass` is the displayed mass.
- `stoppingPMF` is the finite PMF on output times.
- `randomizedOutput` is the selected iterate from one run.

### Two-Phase Selection and Theorem 6.7

Validation uses:

```math
\bar{G}_T(x)=\frac{1}{T}\sum_{k=1}^{T}G(x,\xi_k)
```

```math
\bar{g}_X(\bar{x}_s)=P_X(\bar{x}_s,\bar{G}_T(\bar{x}_s),\gamma_{R_s})
```

The final candidate satisfies:

```math
\|\bar{g}_X(\bar{x}^{*})\|=\min_{1\le s\le S}\|\bar{g}_X(\bar{x}_s)\|
```

Lean objects:
- `empiricalOracleAverage`, `empiricalProjectedGradient`, and `runEmpiricalProjectedGradient` encode validation.
- `selectedRun`, `selectedOutput`, and `selectedRun_spec` encode the minimum empirical projected-gradient choice.
- `optimizationMinTailProbability`, `validationErrorProbability`, and `selectedTailProbability` keep the two probability terms separate.

Theorem 6.7(a):

```math
\Pr(\|g_X(\bar{x}^{*})\|^{2}\ge 2(4L\mathcal{B}_{\bar{N}}+3\lambda\sigma^{2}/T))\le S/\lambda+2^{-S}
```

Lean objects:
- `eq_6_2_61`, `eq_6_2_62`, `eq_6_2_63`, and `eq_6_2_64` formalize the decomposition, independent-run tail, and validation tail.
- `theorem_6_7a` is the high-probability bound.
- `runCountChoice`, `budgetChoice`, `validationCountChoice`, `theorem_6_7b`, and `theorem_6_7b_totalized` encode the parameter choices and SFO-call bound.

## Stochastic Block Mirror Descent

File: [`StochasticBlockMirrorDescent.lean`](StochasticBlockMirrorDescent.lean)

Source target: Lan, FOML, Section 4.6, Algorithm 4.5, Theorem 4.12.

### Block Model

```math
f^{*}=\min_{x\in X}f(x)
```

```math
f(x)=\mathbb{E}[F(x,\xi)]
```

```math
X=X_1\times X_2\times\cdots\times X_b\subseteq\mathbb{R}^{n}
```

```math
X_i\subseteq\mathbb{R}^{n_i},\qquad \sum_{i=1}^{b}n_i=n
```

Lean objects:
- `StochasticBlockMirrorDescentSetup` stores the source data.
- `StateSpace` is the dependent product, implemented as `PiLp 2`.
- `X` is the product feasible set.
- `blockCoord` implements the block coordinate map.

### Oracle, Norms, and Bregman Blocks

The oracle assumptions are:

```math
\mathbb{E}[G(x,\xi)]=g(x)\in\partial f(x)
```

```math
G_i(x,\xi)=U_i^{T}G(x,\xi)
```

```math
\mathbb{E}[\|G_i(x,\xi)\|_{i,*}^{2}]\le M_i^{2}
```

Lean objects:
- `objective`, `objective_eq_integral`, `unbiased_oracleMean`, and `gBlock` implement the objective, mean oracle, and block oracle.
- `block_second_moment_dualNorm`, `block_second_moment_expectation`, and `block_second_moment_stream` implement Eq. (4.6.4).
- `canonicalDualNormFromPrimal` derives the dual norm from the primal block seminorm.
- `blockDualNorm_support_bound` proves the paper's primal-dual support inequality.

The block Bregman divergence is:

```math
V_i(z,x)=v_i(x)-v_i(z)-\langle\nabla v_i(z),x-z\rangle
```

Lean objects:
- `blockDGF` is the paper-facing restricted object.
- `blockDGFAmbient` is only the Mathlib differentiability realization on `X_i`.
- `blockDivergence`, `blockDivergence_eq`, `blockDivergence_lower_bound`, and `blockDivergence_nonneg` encode Eq. (4.6.7).

### Sampling and One-Block Update

```math
\Pr(i_k=i)=p_i,\qquad p_i\in(0,1],\qquad \sum_{i=1}^{b}p_i=1
```

Lean objects:
- `finiteBlockSamplingPMF`, `finiteBlockSamplingLaw`, and `finiteBlockSamplingLaw_singleton` build the finite law.
- `blockIndexLaw`, `blockIndexLaw_singleton`, `samplePairLaw`, and `P` build the same-time oracle/block stream.
- `currentSample_independent_past` and `currentSample_components_independent` encode the independence used in the proof.

Algorithm 4.5 updates only the sampled block:

```math
x_{k+1}^{(i_k)}=\arg\min_{u\in X_{i_k}}\; \langle G_{i_k}(x_k,\xi_k),u\rangle+\gamma_k^{-1}V_{i_k}(x_k^{(i_k)},u)
```

```math
x_{k+1}^{(i)}=x_k^{(i)}\qquad\text{for }i\ne i_k
```

Lean objects:
- `blockProxObjective` is the one-block prox objective.
- `blockMirrorProx` performs the selected-block replacement.
- `blockMirrorProx_selected`, `blockMirrorProx_other`, and `blockMirrorProx_selected_isMinOn` state the two branches and the argmin certificate.
- `process`, `xIter_mem`, and `xIter_measurable_sampleFiltration` define the recursive stochastic process.

### Output, Telescope, and Theorem 4.12

The output uses `theta_k = gamma_k`:

```math
\bar{x}_N=\frac{\sum_{k=1}^{N}\theta_k x_k}{\sum_{k=1}^{N}\theta_k}
```

Lean objects:
- `paperEta`, `paperTheta`, and `paperTheta_eq_paperEta` encode `theta_k = gamma_k`.
- `outputTimes`, `outputThetaSum`, `outputEtaSum`, and `weightedAverage` encode the output.

The aggregate potential in the proof is:

```math
V(z,x)=\sum_{i=1}^{b}p_i^{-1}V_i(z^{(i)},x^{(i)})
```

Lean objects:
- `aggregatePotential` is this weighted block Bregman potential.
- `delta` and `deltaBar` are the stochastic linear and quadratic error terms.
- `step_4_6_18` is the aggregate one-step recursion.
- `step_4_6_20` is the finite-window telescope.
- `step_4_6_21` proves the martingale cancellation term.
- `step_4_6_22` proves the quadratic selected-block bound.

Theorem 4.12:

```math
\mathbb{E}[f(\bar{x}_N)-f(x_{*})]\le \frac{\sum_{i=1}^{b}p_i^{-1}V_i(x_1^{(i)},x_{*}^{(i)})+\frac{1}{2}\sum_{k=1}^{N}\gamma_k^{2}\sum_{i=1}^{b}M_i^{2}}{\sum_{k=1}^{N}\gamma_k}
```

Lean objects:
- `weighted_average_output_mem_measurable` supplies output feasibility and measurability.
- `theorem_4_12_output_gap_integrable` proves the expectation is a genuine integral.
- `theorem_4_12` is the final source theorem.

## Stochastic Nonconvex Conditional Gradient

File: [`StochasticNonconvexConditionalGradient.lean`](StochasticNonconvexConditionalGradient.lean)

Source target: Lan, FOML, Section 7.4, Algorithms 7.12 and 7.13, Theorems 7.16 and 7.17.

### Stochastic Model and Conditional-Gradient Geometry

```math
f(x)=\mathbb{E}[F(x,\xi)]
```

```math
X\subseteq\mathbb{R}^{n}\quad\text{closed, compact, and convex}
```

Lean objects:
- `StochasticNonconvexConditionalGradientSetup` stores the stochastic objective, oracle assumptions, feasible set, parameter schedules, and sample stream.
- `FiniteSumNonconvexConditionalGradientSetup` is the finite-sum setup used for Theorem 7.16.
- `linearMinimizer`, `wolfeGapMaximizer`, and `wolfeGap_spec` realize the conditional-gradient linear oracle and Wolfe-gap geometry.
- `barDX`, `diameterPair`, and `barDX_bound` encode the feasible-set diameter budget.

The Wolfe gap is:

```math
\operatorname{gap}(x)=\max_{y\in X}\langle\nabla f(x),x-y\rangle
```

Lean objects:
- `SOptLib.ConditionalGradient.wolfeGap` is the library Wolfe-gap object.
- `wolfeGap_eq_linearMinimizer` and `wolfeGap_surrogate_bound` connect the selected linear minimizer with the estimator-driven descent proof.

### Variance-Reduced Estimator and Process

Algorithm 7.13 uses epoch refreshes and recursive mini-batch updates:

```math
G_k=\frac{1}{m}\sum_{i=1}^{m}\nabla F(x_k,\xi_i)
```

```math
G_k=G_{k-1}+\frac{1}{b}\sum_{i=1}^{b}\left(\nabla F(x_k,\xi_i)-\nabla F(x_{k-1},\xi_i)\right)
```

Lean objects:
- `rawProcess`, `processOfWellDefined`, `rawIterProcess`, and `iterProcessOfWellDefined` define the stochastic recursion.
- `deltaProcessOfWellDefined` is the estimator error `G_k - gradf x_k`.
- `filtration`, `sampleBlockMeasurableSpace_mono`, and `sampleBlockMeasurableSpace_le` provide the adaptedness and sample-block measurability infrastructure.
- `eq_7_4_2_conditional`, `lemma_7_4`, and `lemma_7_5_conditional` formalize the one-step descent, recursive estimator variance, and expected Wolfe-gap telescope.

The randomized output law is:

```math
\Pr(R=k)=\frac{\alpha_k}{\sum_{j=1}^{N}\alpha_j}
```

Lean objects:
- `alphaSumOfWellDefined`, `outputMassOfWellDefined`, and `outputPMFOfWellDefined` encode the normalized output law.
- `randomOutputOfWellDefined`, `randomOutputWolfeGapOfWellDefined`, and `expectedWolfeGapOfWellDefined` encode the randomized output and its expected Wolfe gap.

### Finite-Sum Theorem 7.16

Algorithm 7.12 samples component indices using smoothness weights:

```math
q_i=\frac{L_i}{mL}
```

Lean objects:
- `componentQ`, `componentIndexPMF`, and `sample_identDistrib_componentQ` encode the importance-sampling law.
- `componentQ_weighted_gradDiff_sum_eq_gradf_sub` and `sampled_componentQ_gradDiff_residual_secondMoment_le` prove the finite-sum centering and second-moment bridge.
- `activeEpochSteps`, `epochDifferenceAlphaIndex`, and `theorem716EpochDifferencePenalty` implement the active epoch partition and corrected generated-process penalty.

Theorem 7.16 bounds the expected Wolfe gap:

```math
\begin{aligned}
\mathbb{E}[\operatorname{gap}(x_R)]
&\le \frac{f(x_1)-f^{*}}{\sum_{k=1}^{N}\alpha_k} \\
&\quad + \frac{L\bar{D}_X^2}{\sum_{k=1}^{N}\alpha_k}(\cdots) \\
&\quad + \frac{N\sigma^2}{Lm\sum_{k=1}^{N}\alpha_k}
\end{aligned}
```

Lean objects:
- `theorem_7_16_general_domainAware` is the domain-aware finite-sum theorem.
- `theorem_7_16_general_epochDifference_domainAware` is the generated-process coordinate-corrected theorem.
- `theorem_7_16_conditional` and `theorem_7_16_conditional_euclidean` expose the conditional and Euclidean-facing statements.

### Theorem 7.17 and Corollary 7.12

Theorem 7.17 uses the Algorithm 7.13 parameter choice:

```math
\alpha_k=\alpha
=\left(\frac{1/N+\sigma^2/(Lm)}{L\bar{D}_X^2}\right)^{1/2}
```

The corrected Theorem 7.17 statement keeps the explicit L1 floor from Lemma 7.5:

```math
\mathcal{A}_{N,S}
=
\frac{3}{2}\sum_{k=1}^{N}\alpha_k^2
+\sum_{s=0}^{S}\sum_{j=1}^{T}\alpha_{s,j}\max_{1\le r\le T}\alpha_{s,r}
```

```math
\mathbb{E}[\operatorname{gap}(x_R)]\le
\frac{f(x_1)-f^{*}}{\sum_{k=1}^{N}\alpha_k}
+ \frac{L\bar{D}_X^2\mathcal{A}_{N,S}}{\sum_{k=1}^{N}\alpha_k}
+ \frac{N\sigma^2}{2Lm\sum_{k=1}^{N}\alpha_k}
+ \frac{\bar{D}_X\sigma}{\sqrt{m}}
```

Equivalently, the aggregate term is:

```math
\left(
\frac{3}{2}\sum_{k=1}^{N}\alpha_k^2
+\sum_{s=0}^{S}\sum_{j=1}^{T}\alpha_{s,j}\max_{1\le r\le T}\alpha_{s,r}
\right)
```

Corollary 7.12 is recorded in the same corrected form: the formal corollary keeps the Theorem 7.17 right-hand side rather than compressing it to the printed coefficient-`4` display.

Lean objects:
- `paperAlphaOfWellDefined`, `αOfWellDefined`, and `alphaSumOfWellDefined` encode the displayed constant stepsize and normalizer.
- `theorem_7_17_general_with_wellDefined_l1_floor` and `theorem_7_17_general_domainAware_l1_floor` state the corrected rate with the unavoidable L1 floor.
- `theorem_7_17_conditional_euclidean_l1_floor` is the Euclidean conditional form.
- `corollary_7_12_general_with_wellDefined_l1_floor` and `corollary_7_12_conditional_euclidean_l1_floor` give the corollary-level specialization.
- `corollary_7_12_printedClaimViolatedWithWellDefined_concreteWitness` records the scalar obstruction to the compressed printed claim.
