import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.ProdL2
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Probability.Martingale.Basic
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.IdentDistrib
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Probability.ConditionalExpectation
import Mathlib.Probability.Process.Adapted
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Data.NNReal.Defs
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Convex.Strong
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.GroupWithZero.Finset
import Mathlib.Algebra.Order.Chebyshev
import SOptLib.Model.ConditionalGradient
import SOptLib.Model.Diameter
import SOptLib.Model.Filtration
import SOptLib.Model.Iterates
import SOptLib.Model.Objective
import SOptLib.Model.ParameterChoices
import SOptLib.Model.Selection
import SOptLib.Model.Stationarity
import SOptLib.Model.StochasticOracle
import SOptLib.Glue.Algebra
import SOptLib.Glue.Analysis
import SOptLib.Glue.Martingale
import SOptLib.Glue.Probability
import SOptLib.Layer0.ConditionalGradient
import SOptLib.Layer0.Oracle
import SOptLib.Layer0.Objective
import SOptLib.Layer0.ConvexFOC
import SOptLib.Layer1.Complexity
import SOptLib.Layer1.Descent
import SOptLib.Layer1.Telescope

/-!
# Stochastic Nonconvex Variance-Reduced Conditional Gradient

Statement-only formalization of Lan's stochastic nonconvex variance-reduced
conditional-gradient method for smooth nonconvex stochastic optimization from
Section 7.4 of *First-Order and Stochastic Optimization Methods for Machine
Learning*.

The file packages the compact feasible set, the stochastic oracle model, the
Wolfe-gap geometry, the recursive gradient estimator, the epoch indexing, and
the randomized output distribution into
`StochasticNonconvexConditionalGradientSetup`. It then defines the iteration
process, the estimator and noise term, the Wolfe gap and its surrogate, and
declares displayed-expression paper objects plus conditional/domain-aware
realizations of the lemma chain. Full theorem names that would require
non-source-derived realization facts are intentionally not exported as
contract-indexed paper theorem heads.
-/

open MeasureTheory ProbabilityTheory
open scoped InnerProductSpace
open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {Ω : Type*} [MeasurableSpace Ω]
variable {Ξ : Type*} [MeasurableSpace Ξ]

/-- Selector-first finite product integrals expand as weighted fiber integrals.

Candidate audit: considered SOptLib
`integral_selected_finite_index_prod_eq_sum_weights`, which has the sample
coordinate first. This bridge is the same finite-index expansion after
Mathlib's product-measure swap, matching the output law
`PMF.toMeasure.prod P` used by Algorithm 7.13/Theorem 7.16. -/
theorem integral_finite_index_first_prod_eq_sum_weights
    {S ι : Type*} [MeasurableSpace S] [MeasurableSpace ι]
    [Fintype ι] [MeasurableSingletonClass ι]
    (ν : Measure ι) (μ : Measure S) [SFinite μ] [IsFiniteMeasure ν]
    (p : ι → ℝ) (F : ι → S → ℝ)
    (hν_singleton : ∀ i, ν.real ({i} : Set ι) = p i)
    (hF_int : ∀ i, Integrable (F i) μ) :
    ∫ q : ι × S, F q.1 q.2 ∂(ν.prod μ) =
      Finset.sum Finset.univ (fun i => p i * ∫ s, F i s ∂μ) := by
  exact SOptLib.integral_finite_index_first_prod_eq_sum_weights
    (ν := ν) (μ := μ) (p := p) (F := F) hν_singleton hF_int

/-- A finite weighted centered second moment is bounded by the corresponding
uncentered second moment when the named center is the weighted mean.

Candidate audit: searched SOptLib/Mathlib for "weighted variance sum norm sub
mean less second moment", "Finset sum weights norm sub mean squared variance
identity", and "weighted residual second moment mean zero". The closest hits
were stochastic/integral variance transport lemmas such as
`miniBatchResidual_sum_secondMoment_le_card_mul_variance`; none expose this
deterministic finite Hilbert-space algebra needed for Lan Lemma 7.4's
component-law diagonal residual budget. -/
lemma finset_weighted_variance_le_second_moment
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (q : ι → ℝ) (a : ι → E) (μ : E)
    (hqsum : Finset.sum Finset.univ q = 1)
    (hmean : Finset.sum Finset.univ (fun i => q i • a i) = μ) :
    Finset.sum Finset.univ (fun i => q i * ‖a i - μ‖ ^ 2) ≤
      Finset.sum Finset.univ (fun i => q i * ‖a i‖ ^ 2) := by
  exact Finset.weighted_variance_le_second_moment
    (s := Finset.univ) q a μ hqsum hmean

/-- Paper variable space `ℝ^n` from Eqs. (7.4.1), (7.4.12).

No SOptLib match: searched "Euclidean variable space stochastic optimization"
and scanned `SOptLib/Model/Objective.lean` plus `SOptLib/Model/Iterates.lean`;
those files are paper-spec-neutral over an abstract normed space, while this
paper's source-facing setup states `X ⊆ ℝ^n`. -/
abbrev PaperVariableSpace (n : ℕ) : Type :=
  EuclideanSpace ℝ (Fin n)

/-- Paper sample space `Ξ ⊆ ℝ^d` from Eq. (7.4.12).

No SOptLib match: searched "Euclidean sample space stochastic oracle" and
scanned `SOptLib/Model/StochasticOracle.lean`; the library models abstract
sample types, while the paper-facing layer records the cited Euclidean sample
ambient space. -/
abbrev PaperSampleSpace (d : ℕ) : Type :=
  EuclideanSpace ℝ (Fin d)

noncomputable instance paperEuclideanMeasurableSpace (n : ℕ) :
    MeasurableSpace (EuclideanSpace ℝ (Fin n)) :=
  borel _

instance paperEuclideanBorelSpace (n : ℕ) :
    BorelSpace (EuclideanSpace ℝ (Fin n)) :=
  ⟨rfl⟩

/-- Local generic helper replacing the older unlisted `SOptLib.Glue.Measurable`
compatibility import: square-integrability of two vector processes implies
integrability of their inner product. -/
theorem integrable_inner_of_sq_integrable
    {Ω E : Type*} [MeasurableSpace Ω] [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {P : Measure Ω} {u v : Ω → E}
    (hu_meas : AEStronglyMeasurable u P) (hv_meas : AEStronglyMeasurable v P)
    (hu_sq : Integrable (fun ω => ‖u ω‖ ^ 2) P)
    (hv_sq : Integrable (fun ω => ‖v ω‖ ^ 2) P) :
    Integrable (fun ω => ⟪u ω, v ω⟫_ℝ) P := by
  apply Integrable.mono (hu_sq.add hv_sq) (hu_meas.inner hv_meas)
  refine Filter.Eventually.of_forall fun ω => ?_
  rw [Real.norm_eq_abs]
  calc
    |⟪u ω, v ω⟫_ℝ| ≤ ‖u ω‖ * ‖v ω‖ := abs_real_inner_le_norm (u ω) (v ω)
    _ ≤ ‖u ω‖ ^ 2 + ‖v ω‖ ^ 2 := by
      nlinarith [sq_nonneg (‖u ω‖ - ‖v ω‖)]
    _ = ‖(fun ω => ‖u ω‖ ^ 2 + ‖v ω‖ ^ 2) ω‖ := by
      simp [Real.norm_eq_abs,
        abs_of_nonneg (by positivity : 0 ≤ ‖u ω‖ ^ 2 + ‖v ω‖ ^ 2)]

/-- Local generic helper replacing the older unlisted compatibility import:
a pointwise norm bound gives integrability of the squared norm and the
corresponding second-moment bound. -/
theorem integrable_sq_norm_of_pointwise_bound
    {S β : Type*} [MeasurableSpace S] [NormedAddCommGroup β]
    {f : S → β} {G : ℝ} {ν : Measure S} [IsProbabilityMeasure ν]
    (hf_meas : AEStronglyMeasurable f ν)
    (hbounded : ∀ s, ‖f s‖ ≤ G) :
    Integrable (fun s => ‖f s‖ ^ 2) ν ∧ ∫ s, ‖f s‖ ^ 2 ∂ν ≤ G ^ 2 := by
  have hbound : ∀ s, ‖‖f s‖ ^ 2‖ ≤ ‖G ^ 2‖ := fun s => by
    rw [Real.norm_of_nonneg (sq_nonneg _), Real.norm_of_nonneg (sq_nonneg G)]
    exact pow_le_pow_left₀ (norm_nonneg _) (hbounded s) 2
  have hint : Integrable (fun s => ‖f s‖ ^ 2) ν :=
    Integrable.mono (integrable_const (G ^ 2)) (hf_meas.norm.pow 2)
      (Filter.Eventually.of_forall hbound)
  constructor
  · exact hint
  · calc
      ∫ s, ‖f s‖ ^ 2 ∂ν
          ≤ ∫ _s, G ^ 2 ∂ν :=
            integral_mono hint (integrable_const _)
              (fun s => pow_le_pow_left₀ (norm_nonneg _) (hbounded s) 2)
      _ = G ^ 2 := by simp [integral_const, probReal_univ]

/-- Local generic helper: the previous pointwise boundedness bridge also works
with an almost-everywhere norm bound, which is the form supplied by the paper's
single a.s. smoothness event. -/
theorem integrable_sq_norm_of_ae_bound
    {S β : Type*} [MeasurableSpace S] [NormedAddCommGroup β]
    {f : S → β} {G : ℝ} {ν : Measure S} [IsProbabilityMeasure ν]
    (hf_meas : AEStronglyMeasurable f ν)
    (hbounded : ∀ᵐ s ∂ν, ‖f s‖ ≤ G) :
    Integrable (fun s => ‖f s‖ ^ 2) ν ∧ ∫ s, ‖f s‖ ^ 2 ∂ν ≤ G ^ 2 := by
  exact SOptLib.integrable_sq_norm_of_ae_bound hf_meas hbounded

/-- Local generic helper replacing the older unlisted compatibility import:
on a finite measure space, square-integrability of the norm implies Bochner
integrability of the vector-valued function. -/
theorem integrable_of_sqnorm_integrable
    {Ω E : Type*} [MeasurableSpace Ω] [NormedAddCommGroup E]
    {P : Measure Ω} [IsFiniteMeasure P] {v : Ω → E}
    (hv_meas : AEStronglyMeasurable v P)
    (hv_sq : Integrable (fun ω => ‖v ω‖ ^ 2) P) :
    Integrable v P := by
  have hnorm_int : Integrable (fun ω => ‖v ω‖) P := by
    have hdom : Integrable (fun ω => ‖v ω‖ ^ 2 + 1) P := by
      simpa using hv_sq.add (integrable_const (1 : ℝ))
    refine Integrable.mono' hdom hv_meas.norm ?_
    refine Filter.Eventually.of_forall fun ω => ?_
    rw [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg (v ω))]
    nlinarith [sq_nonneg (‖v ω‖ - (1 / 2 : ℝ))]
  exact (MeasureTheory.integrable_norm_iff hv_meas).mp hnorm_int

/-- Local generic helper replacing the older unlisted compatibility import:
the squared norm of a sum is controlled by twice the squared norms of the
summands. -/
theorem norm_add_sq_le_two_mul_norm_sq_add_two_mul_norm_sq
    {E : Type*} [NormedAddCommGroup E] (a b : E) :
    ‖a + b‖ ^ 2 ≤ 2 * ‖a‖ ^ 2 + 2 * ‖b‖ ^ 2 := by
  exact SOptLib.norm_add_sq_le_two_mul_norm_sq_add_two_mul_norm_sq a b

/-!
## Falsity witness for the retired sample-measurability derivation

The stochastic setup below records `hξ_meas` as part of the paper's random
sample stream interface. The private construction in this namespace records why
the older theorem shape was not derivable from only `iIndepFun` and
`IdentDistrib`: a stream can be independent and identically distributed in law
while a chosen representative is not measurable.
-/
namespace OldIIDMeasurabilityCounterexample

inductive CtrΩ where
  | z
  | o
  | tw
  deriving DecidableEq

inductive CtrΞ where
  | ff
  | tt
  deriving DecidableEq

open CtrΩ CtrΞ

instance : MeasurableSpace CtrΩ :=
  MeasurableSpace.generateFrom ({({z} : Set CtrΩ)} : Set (Set CtrΩ))

instance : MeasurableSpace CtrΞ := ⊤

def badSample (ω : CtrΩ) : CtrΞ :=
  match ω with
  | z => ff
  | o => tt
  | tw => ff

def goodSample (_ : CtrΩ) : CtrΞ :=
  ff

private lemma measurableSet_z : MeasurableSet ({z} : Set CtrΩ) := by
  apply MeasurableSpace.measurableSet_generateFrom
  simp

private lemma badSample_z : badSample z = ff := rfl

private lemma dirac_z_eq_zero_of_notMem {s : Set CtrΩ} (hs : z ∉ s) :
    Measure.dirac z s = 0 := by
  exact dirac_eq_zero_of_not_mem measurableSet_z hs

private lemma badSample_not_measurable : ¬ Measurable badSample := by
  intro h
  have hpre : MeasurableSet (badSample ⁻¹' ({tt} : Set CtrΞ)) := h (by simp)
  have hsingleton : MeasurableSet ({o} : Set CtrΩ) := by
    convert hpre using 1
    ext x
    cases x <;> simp [badSample]
  rw [measurableSet_generateFrom_singleton_iff] at hsingleton
  rcases hsingleton with h | h | h | h
  · have : o ∈ ({o} : Set CtrΩ) := by simp
    rw [h] at this
    simp at this
  · have : z ∈ ({o} : Set CtrΩ) := by rw [h]; simp
    simp at this
  · have : tw ∈ ({o} : Set CtrΩ) := by rw [h]; simp
    simp at this
  · have : z ∈ ({o} : Set CtrΩ) := by rw [h]; simp
    simp at this

private lemma badSample_ae_eq_goodSample :
    badSample =ᵐ[Measure.dirac z] goodSample := by
  apply Filter.mem_of_superset (show ({z} : Set CtrΩ) ∈ ae (Measure.dirac z) from by
    exact (mem_ae_dirac_iff measurableSet_z).2 (by simp))
  intro x hx
  subst x
  rfl

private lemma badSample_aemeasurable :
    AEMeasurable badSample (Measure.dirac z) := by
  exact ⟨goodSample, measurable_const, badSample_ae_eq_goodSample⟩

/-- Formal counterexample to the retired proof-only claim that
`iIndepFun` plus `IdentDistrib` imply coordinate measurability. -/
private theorem old_iid_fields_do_not_imply_coordinate_measurability :
    ∃ (ξ : ℕ → CtrΩ → CtrΞ) (P : Measure CtrΩ),
      IsProbabilityMeasure P ∧
      iIndepFun (β := fun _ : ℕ => CtrΞ) ξ P ∧
      (∀ k : ℕ, IdentDistrib (ξ k) (ξ 0) P P) ∧
      ¬ (∀ k : ℕ, Measurable (ξ k)) := by
  simpa using
    (exists_iid_identDistrib_not_forall_measurable (Ω := CtrΩ) (Ξ := CtrΞ)
      z o tw ff tt (by decide) (by decide) (by decide) (by simp) (by decide))

end OldIIDMeasurabilityCounterexample

/-- State carried across iterations of the stochastic nonconvex
conditional-gradient algorithm. We track the current iterate `x`, the current
gradient estimator `G`, and the current epoch counter `s`. -/
abbrev StochasticNonconvexCGState (E : Type*) := SOptLib.ConditionalGradientState E E

/-- Finite-sum conditional-gradient setup for the paper's Theorem 7.16.

Book citation: source_json `key_lemmas[2].ref` quotes `Theorem 7.16`, and
source_json `key_lemmas[2].statement_math` quotes
`E[gap(x_R)] <= ... + (L Dbar_X^2 / sum alpha_k) [...]`.

Candidate audit: checked `Convex.carrier_smooth_quadratic_upper_bound` and
searched "finite sum nonconvex conditional gradient theorem bound"; SOptLib has
generic telescope and finite-PMF helpers, but no paper-local finite-sum
conditional-gradient process. This setup therefore exposes the finite component
functions, their gradients, the sampled component-index stream, and the paper
parameters from Theorem 7.16 rather than arbitrary bound-side real quantities. -/
abbrev FiniteSumNonconvexConditionalGradientSetup
    (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
      [CompleteSpace E] [MeasurableSpace E] [BorelSpace E]
    (Ω : Type*) [MeasurableSpace Ω] :=
  SOptLib.FiniteSumConditionalGradientSetup E Ω

/-- Complete setup for the stochastic nonconvex variance-reduced conditional-
gradient method (Algorithm 7.13 of Lan's FOML, Section 7.4).

Book citations: source_json `setup.variable_space.math` quotes
`X subseteq R^n closed compact convex` and
`f(x) = E[F(x, xi)]`; source_json `algorithm_spec.ref` quotes
`Algorithm 7.13`; source_json `assumptions[*].math` lists the stochastic
smoothness, paired-gradient, unbiasedness, and bounded-variance assumptions.

The structure stores the feasible-set data, the stochastic oracle, the Wolfe-gap
geometry, the epoch-length and mini-batch parameters, the stepsize schedule, and
all measurability and independence assumptions used throughout the lemma chain. -/
structure StochasticNonconvexConditionalGradientSetup
    (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
      [CompleteSpace E] [MeasurableSpace E] [BorelSpace E]
    (Ω : Type*) [MeasurableSpace Ω]
    (Ξ : Type*) [MeasurableSpace Ξ] where
  /-- Compact feasible set X.

  Book citation: source_json `setup.variable_space.math` quotes
  `X subseteq R^n closed compact convex`. -/
  X : Set E
  /-- Initial iterate x1.

  Book citation: source_json `algorithm_spec.initialization.math` quotes
  `x_1 in X`. -/
  x₁ : E
  /-- Stochastic function F(x, xi).

  Book citation: source_json `setup.variable_space.math` quotes
  `f(x) = E[F(x, xi)]`. -/
  F : E → Ξ → ℝ
  /-- Gradient grad f(x).

  Book citation: source_json `assumptions[2].math` quotes
  `E[grad F(x, xi)] = grad f(x)`. -/
  gradf : E → E
  /-- Stochastic gradient grad F(x, xi).

  Book citation: source_json `assumptions[1].math` quotes
  `compute grad F(x, xi) and grad F(y, xi)`. -/
  gradF : E → Ξ → E
  /-- Smoothness constant L.

  Book citation: source_json `assumptions[0].parameters[0]` quotes `L > 0`. -/
  L : ℝ
  /-- Stochastic-gradient standard-deviation parameter sigma.

  Book citation: source_json `assumptions[3].math` quotes
  `E[||grad F(x, xi) - grad f(x)||^2] <= sigma^2`. -/
  σ : ℝ
  /-- Total number of iterations N.

  Book citation: source_json `algorithm_spec.output.math` quotes
  `k = 1, ..., N`. -/
  N : ℕ
  /-- Batch size m for epoch-start gradient estimate.

  Book citation: source_json `algorithm_spec.steps[0].math` quotes
  `sample H^s = {xi_1^s, ..., xi_m^s}`. -/
  m : ℕ
  /-- Epoch length T from the paper parameter regime `b = T = sqrt(m)`.

  Book citation: source_json `algorithm_spec.parameters[1].math` quotes
  `b = T = sqrt(m)`. -/
  T_choice : ℕ
  /-- Recursive mini-batch size b from the paper parameter regime
  `b = T = sqrt(m)`.

  Book citation: source_json `algorithm_spec.parameters[1].math` quotes
  `b = T = sqrt(m)`. -/
  b_choice : ℕ
  /-- IID stochastic samples driving the oracle.

  Book citation: source_json `algorithm_spec.steps[0].math` and
  `algorithm_spec.steps[1].math` quote `Generate i.i.d. sample`. -/
  ξ : ℕ → Ω → Ξ
  /-- Coordinate measurability of the i.i.d. sample stream. Mathlib's
  `iIndepFun` and `IdentDistrib` do not imply this for arbitrary
  representatives, so the random-variable part of the paper phrase
  `Generate i.i.d. sample ... for the random variable xi` is recorded in the
  setup interface used by the generated filtration.

  Book citation: source_json `algorithm_spec.steps[0].math` and
  `algorithm_spec.steps[1].math` quote `Generate i.i.d. sample ... for the
  random variable xi`; PDF text around Algorithm 7.13 says the same and states
  `xi is a random vector supported on Xi`. -/
  hξ_meas : ∀ k : ℕ, Measurable (ξ k)
  /-- Linear minimization oracle from Algorithm 7.13.

  Book citation: source_json `algorithm_spec.steps[2].math` quotes
  `y_k = argmin_{x in X} <G_k, x>`. The bundled measurability component is the
  Lean realization of this algorithmic oracle needed to treat the generated
  process as adapted; it is not a uniqueness assumption on the argmin set. -/
  lmo : SOptLib.LinearMinimizationOracle E X
  /-- Probability measure on the sample space, realizing the expectations in
  the stochastic objective and oracle assumptions.

  Book citation: source_json `setup.variable_space.math` quotes
  `f(x) = E[F(x, xi)]`. -/
  P : Measure Ω
  hP : IsProbabilityMeasure P
  /-- X is closed.

  Book citation: source_json `setup.variable_space.math` quotes
  `X subseteq R^n closed compact convex`. -/
  hX_closed : IsClosed X
  /-- X is compact.

  Book citation: source_json `setup.variable_space.math` quotes
  `X subseteq R^n closed compact convex`. -/
  hX_compact : IsCompact X
  /-- X is convex.

  Book citation: source_json `setup.variable_space.math` quotes
  `X subseteq R^n closed compact convex`. -/
  hX_convex : Convex ℝ X
  /-- x1 is in X.

  Book citation: source_json `algorithm_spec.initialization.math` quotes
  `x_1 in X`. -/
  hx₁_mem : x₁ ∈ X
  /-- L > 0, as stated in the smoothness assumption.

  Book citation: source_json `assumptions[0].parameters[0]` quotes `L > 0`. -/
  hL_pos : 0 < L
  /-- sigma >= 0.

  Book citation: source_json `assumptions[3].parameters[0]` quotes
  `sigma^2 >= 0`. -/
  hσ_nonneg : 0 ≤ σ
  /-- N >= 1, the nonempty output-window realization for `k = 1, ..., N`.

  Book citation: source_json `algorithm_spec.output.math` quotes
  `k = 1, ..., N`. -/
  hN_pos : 1 ≤ N
  /-- m >= 1, the nonempty refresh-batch realization.

  Book citation: source_json `algorithm_spec.steps[0].math` quotes
  `{xi_1^s, ..., xi_m^s}`. -/
  hm_pos : 1 ≤ m
  /-- Paper integer realization of `b = T = sqrt(m)`: the full refresh batch
  size factors exactly as epoch length times recursive mini-batch size.

  Book citation: source_json `algorithm_spec.parameters[1].math` quotes
  `b = T = sqrt(m)`. -/
  hm_eq_T_mul_b_choice : m = T_choice * b_choice
  /-- Paper equality `b = T` from Eq. (7.4.7).

  Book citation: source_json `algorithm_spec.parameters[1].math` quotes
  `b = T = sqrt(m)`. -/
  hb_choice_eq_T : b_choice = T_choice
  /-- Objective expectation well-definedness on feasible points. This is the
  well-defined side of the paper setup equation `f(x) = E[F(x,xi)]`, not a
  later proof artifact.

  Book citation: source_json `setup.variable_space.math` quotes
  `f(x) = E[F(x, xi)]`. -/
  hF_objective_wellDefined :
    ∀ x : E, x ∈ X → SOptLib.objectiveWellDefined P F (ξ 0) x
  /-- Paired gradient access is tied to a single almost-sure realization event:
  for almost every sampled realization, `gradF` gives the gradient of
  `F(.,xi)` at every feasible query.

  Book citation: source_json `assumptions[1].math` quotes
  `compute grad F(x, xi) and grad F(y, xi)`. -/
  hF_hasGradientAt_ae :
    ∀ᵐ ω ∂P,
      ∀ x : E, x ∈ X →
        HasGradientAt (fun z : E => F z (ξ 0 ω)) (gradF x (ξ 0 ω)) x
  /-- `F(.,xi)` is `L`-smooth on feasible pairs on a single almost-sure event,
  matching the paper's a.s. smooth-function assumption rather than a separate
  a.e. event for every pair.

  Book citation: source_json `assumptions[0].math` quotes
  `F(., xi) is a smooth function with Lipschitz constant L ... almost surely`. -/
  hF_smooth_ae :
    ∀ᵐ ω ∂P,
      ∀ x y : E, x ∈ X → y ∈ X →
        ‖gradF x (ξ 0 ω) - gradF y (ξ 0 ω)‖ ≤ L * ‖x - y‖
  /-- Joint measurability of the stochastic-gradient kernel. This is the
  stochastic-oracle regularity needed to evaluate `gradF` at random adapted
  iterates and sampled realizations.

  Book citation: source_json `assumptions[1].math` quotes `compute grad F(x,
  xi) and grad F(y, xi)` for a generated realization, and
  `assumptions[2].math` / `assumptions[3].math` state expectations involving
  `grad F(x, xi)`. -/
  hgradF_meas : Measurable (fun z : E × Ξ => gradF z.1 z.2)
  /-- Well-defined unbiasedness: the stochastic-gradient expectation exists
  and equals `grad f(x)` for every feasible `x`.

  Book citation: source_json `assumptions[2].math` quotes
  `E[grad F(x, xi)] = grad f(x), forall x in X`. -/
  hgradF_unbiased :
    ∀ x : E, x ∈ X →
      SOptLib.oracleWellDefined P gradF (ξ 0) x ∧
        SOptLib.oracleMean P gradF (ξ 0) x = gradf x
  /-- Gradient interface for the paper objective expectation. The source setup
  states `f(x) = E[F(x, xi)]`, and the unbiased-gradient assumption states
  `E[grad F(x, xi)] = grad f(x)` for feasible `x`; this field records the
  corresponding differentiability-under-expectation interface needed to use
  `gradf` as the actual gradient of the paper objective.

  Book citation: source_json `setup.variable_space.math` quotes
  `f(x) = E[F(x, xi)]`, and `assumptions[2].math` quotes
  `E[grad F(x, xi)] = grad f(x), forall x in X`. -/
  hgradf_hasGradientAt :
    ∀ x : E, x ∈ X →
      HasGradientAt (SOptLib.objectiveExpectation P F (ξ 0)) (gradf x) x
  /-- Well-defined bounded variance (7.4.13): the squared stochastic-gradient
  residual is integrable and its expectation is at most `sigma^2` for every
  feasible `x`.

  Book citation: source_json `assumptions[3].math` quotes
  `E[||grad F(x, xi) - grad f(x)||^2] <= sigma^2`. -/
  hgradF_variance_bound :
    ∀ x : E, x ∈ X →
      Integrable (fun ω => ‖gradF x (ξ 0 ω) - gradf x‖ ^ 2) P ∧
        ∫ ω, ‖gradF x (ξ 0 ω) - gradf x‖ ^ 2 ∂P ≤ σ ^ 2
  /-- Independence of the sample stream.

  Book citation: source_json `algorithm_spec.steps[0].math` and
  `algorithm_spec.steps[1].math` quote `Generate i.i.d. sample`. -/
  hξ_indep : iIndepFun (β := fun _ => Ξ) ξ P
  /-- Identical distribution of all samples.

  Book citation: source_json `algorithm_spec.steps[0].math` and
  `algorithm_spec.steps[1].math` quote `Generate i.i.d. sample`. -/
  hξ_ident : ∀ k, IdentDistrib (ξ k) (ξ 0) P P

namespace StochasticNonconvexConditionalGradientSetup

variable (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)

/-- The feasible set is nonempty because Algorithm 7.13 initializes with
`x₁ ∈ X`; this is derived from the initialization datum, not a primitive setup
assumption. -/
theorem hX_nonempty : Set.Nonempty setup.X :=
  ⟨setup.x₁, setup.hx₁_mem⟩

/-- Source-backed well-definedness of the paper objective expectation
`f(x)=E[F(x,ξ)]` on feasible points. -/
theorem paperObjective_wellDefined (x : E) (hx : x ∈ setup.X) :
    SOptLib.objectiveWellDefined setup.P setup.F (setup.ξ 0) x :=
  setup.hF_objective_wellDefined x hx

/-- Source-backed integrability projection from the paper objective expectation
well-definedness predicate. -/
theorem paperObjective_integrable (x : E) (hx : x ∈ setup.X) :
    Integrable (fun ω => setup.F x (setup.ξ 0 ω)) setup.P := by
  simpa [SOptLib.objectiveWellDefined, SOptLib.objectiveKernel] using
    setup.paperObjective_wellDefined x hx

/-- Source-backed well-definedness of the stochastic-gradient expectation in
the unbiasedness assumption. -/
theorem paperGradientOracle_wellDefined (x : E) (hx : x ∈ setup.X) :
    SOptLib.oracleWellDefined setup.P setup.gradF (setup.ξ 0) x :=
  (setup.hgradF_unbiased x hx).1

/-- Source-backed integrability projection from the well-defined unbiasedness
assumption. -/
theorem paperGradientOracle_integrable (x : E) (hx : x ∈ setup.X) :
    Integrable (fun ω => setup.gradF x (setup.ξ 0 ω)) setup.P := by
  simpa [SOptLib.oracleWellDefined, SOptLib.oracleKernel] using
    setup.paperGradientOracle_wellDefined x hx

/-- Source-backed well-definedness side of Eq. (7.4.13). -/
theorem paperGradientVariance_wellDefined (x : E) (hx : x ∈ setup.X) :
    Integrable (fun ω => ‖setup.gradF x (setup.ξ 0 ω) - setup.gradf x‖ ^ 2)
      setup.P :=
  (setup.hgradF_variance_bound x hx).1

/-- Source-backed unbiasedness equation in the raw integral notation printed in
Section 7.4.2. -/
theorem paperGradient_unbiased_integral (x : E) (hx : x ∈ setup.X) :
    ∫ ω, setup.gradF x (setup.ξ 0 ω) ∂setup.P = setup.gradf x := by
  simpa [SOptLib.oracleMean, SOptLib.oracleKernel] using
    (setup.hgradF_unbiased x hx).2

/-- Paper epoch length `T` from Eq. (7.4.7), exposed through the setup's
integer square-root realization rather than Lean's floor `Nat.sqrt`.

No SOptLib match: searched "batch size epoch length square root parameter" and
scanned `SOptLib/Model/ParameterChoices.lean` and `SOptLib/Model/Iterates.lean`;
none align with Eq. (7.4.7) because the available selectors are RSMD batch
schedules, while this paper requires the exact integer regime `m = T*b` and
`b = T`. -/
noncomputable def T : ℕ :=
  setup.T_choice

/-- Paper recursive mini-batch size `b = T = sqrt(m)` from Eq. (7.4.7). -/
noncomputable def b : ℕ :=
  setup.b_choice

/-- The paper choice makes the recursive batch size equal to the epoch length. -/
theorem hb_eq_T : setup.b = setup.T := by
  simpa [b, T] using setup.hb_choice_eq_T

/-- The paper choice makes the recursive batch size equal to the epoch length. -/
theorem hb_ge_T : setup.T ≤ setup.b := by
  simpa [setup.hb_eq_T]

/-- The paper square-root regime is recorded as the exact integer bridge
`m = T*b`, avoiding the floor semantics of `Nat.sqrt`. -/
theorem hm_eq_T_mul_b : setup.m = setup.T * setup.b := by
  simpa [T, b] using setup.hm_eq_T_mul_b_choice

/-- The paper epoch length is positive in the integer square-root regime. -/
theorem hT_pos : 1 ≤ setup.T := by
  exact SOptLib.one_le_left_factor_of_one_le_nat_mul
    (by simpa [setup.hm_eq_T_mul_b] using setup.hm_pos)

/-- Single-stream index separation required by the local realization of Algorithm
7.13, derived from the paper's exact integer bridge `m = T*b`. -/
theorem hm_le_T_mul_b : setup.m ≤ setup.T * setup.b := by
  exact le_of_eq setup.hm_eq_T_mul_b

/-- The paper objective `f(x) = E[F(x, ξ)]`.

Candidate audit: chose SOptLib `objectiveExpectation` because it is exactly the
Bochner-expectation objective kernel listed for `setup.variable_space`; the
local wrapper fixes this paper's sample stream to the reference law of `ξ 0`,
matching Eqs. (7.4.1), (7.4.12). -/
noncomputable def f (x : E) : ℝ :=
  SOptLib.objectiveExpectation setup.P setup.F (setup.ξ 0) x

/-- The paper objective unfolds to the expected stochastic loss. -/
theorem f_eq_objectiveExpectation (x : E) :
    setup.f x = ∫ ω, setup.F x (setup.ξ 0 ω) ∂setup.P := by
  rfl

/-- Well-definedness of the paper objective on feasible points. -/
theorem f_wellDefined (x : E) (hx : x ∈ setup.X) :
    SOptLib.objectiveWellDefined setup.P setup.F (setup.ξ 0) x := by
  exact setup.paperObjective_wellDefined x hx

/-- The source-facing stochastic gradient `∇F(x, ξ)` is the gradient of the
sampled stochastic objective, as required by Section 7.4.2's paired-gradient
access assumption. -/
theorem stochasticGradient_hasGradientAt_ae (x : E) (hx : x ∈ setup.X) :
    ∀ᵐ ω ∂setup.P,
      HasGradientAt (fun z : E => setup.F z (setup.ξ 0 ω))
        (setup.gradF x (setup.ξ 0 ω)) x :=
  setup.hF_hasGradientAt_ae.mono (fun _ hω => hω x hx)

/-- Projection of the paper's single almost-sure smooth-function event to one
feasible pair. -/
theorem stochasticGradient_smooth_ae (x y : E) (hx : x ∈ setup.X) (hy : y ∈ setup.X) :
    ∀ᵐ ω ∂setup.P,
      ‖setup.gradF x (setup.ξ 0 ω) - setup.gradF y (setup.ξ 0 ω)‖ ≤
        setup.L * ‖x - y‖ :=
  setup.hF_smooth_ae.mono (fun _ hω => hω x y hx hy)

/-- Gradient interface for the paper objective `f(x) = E[F(x, ξ)]`.

Source-faithfulness note: the book writes `f(x)=E[F(x,ξ)]` and
`E[∇F(x,ξ)] = ∇f(x)`. The setup now records the corresponding
differentiability-under-expectation interface explicitly, rather than trying to
derive differentiability of an expectation from unbiasedness of a vector field
alone. -/
theorem gradf_hasGradientAt (x : E) (hx : x ∈ setup.X) :
    HasGradientAt setup.f (setup.gradf x) x := by
  simpa [StochasticNonconvexConditionalGradientSetup.f] using
    setup.hgradf_hasGradientAt x hx

/-- Feasible `L`-smoothness of `∇f` is derived from the almost-sure smooth
stochastic objective assumption. -/
theorem gradf_smooth (x y : E) (hx : x ∈ setup.X) (hy : y ∈ setup.X) :
    ‖setup.gradf x - setup.gradf y‖ ≤ setup.L * ‖x - y‖ := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  exact oracleMean_lipschitz_of_ae_lipschitz
    (μ := setup.P) (G := fun z ω => setup.gradF z (setup.ξ 0 ω))
    (g := setup.gradf) (L := setup.L) x y
    (setup.paperGradientOracle_integrable x hx)
    (setup.paperGradientOracle_integrable y hy)
    (by
      simpa [SOptLib.oracleMean, SOptLib.oracleKernel] using
        setup.paperGradient_unbiased_integral x hx)
    (by
      simpa [SOptLib.oracleMean, SOptLib.oracleKernel] using
        setup.paperGradient_unbiased_integral y hy)
    (setup.stochasticGradient_smooth_ae x y hx hy)

/-- The deterministic paper gradient is measurable on the feasible carrier.

Candidate audit: chose SOptLib `oracleMean_measurable_of_eq_integral`; it
matches the source form after restricting the query variable to `X` and
transporting the sample law through `ξ 0`, while full-space measurability would
not follow from the paper's feasible-only unbiasedness assumption. -/
theorem gradfOnX_measurable :
    Measurable (fun x : setup.X => setup.gradf (x : E)) := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  exact oracleMean_measurable_on_subtype_of_map_law_integral
    (P := setup.P) (A := setup.X) (Y := setup.ξ 0)
    (G := setup.gradF) (g := setup.gradf)
    (setup.hξ_meas 0)
    (by
      exact setup.hgradF_meas.comp
        ((measurable_subtype_coe.comp measurable_fst).prodMk measurable_snd))
    (fun x => setup.paperGradient_unbiased_integral (x : E) x.property)

/-- Well-definedness side of the paper variance assumption (7.4.13). -/
theorem gradF_variance_wellDefined (x : E) (hx : x ∈ setup.X) :
    Integrable (fun ω => ‖setup.gradF x (setup.ξ 0 ω) - setup.gradf x‖ ^ 2) setup.P :=
  oracle_variance_integrable_of_variance_bound
    (P := setup.P) (G := fun y ω => setup.gradF y (setup.ξ 0 ω))
    (g := setup.gradf) (feasible := setup.X) (σ2 := setup.σ ^ 2)
    (x := x) hx setup.hgradF_variance_bound

/-- Bound side of the paper variance assumption (7.4.13), paired with
`gradF_variance_wellDefined` so the expectation is not interpreted through
Lean's totalized integral semantics. -/
theorem gradF_variance_bound (x : E) (hx : x ∈ setup.X) :
    ∫ ω, ‖setup.gradF x (setup.ξ 0 ω) - setup.gradf x‖ ^ 2 ∂setup.P ≤
      setup.σ ^ 2 :=
  oracle_variance_bound_of_variance_bound
    (P := setup.P) (G := fun y ω => setup.gradF y (setup.ξ 0 ω))
    (g := setup.gradf) (feasible := setup.X) (σ2 := setup.σ ^ 2)
    (x := x) hx setup.hgradF_variance_bound

/-- Existence of an attained minimizer of the paper objective on compact `X`.

Candidate audit: SOptLib `optimizerValueOfMinimum` and
`optimizerValue_le_of_attained_minimum` were checked; they provide optimizer
value extraction once an attained minimum is available, but not this paper's
compact-existence bridge from `X` and `f`. The pre-searched
`recursive_process_measurable_of_measurable_update`, `blockMirrorUpdate`,
`BlockIterateState`, and `iIndepFun.indep_past_iSup_current` candidates are
process/adaptedness primitives rather than compact minimization, so this local
theorem uses `IsCompact.exists_isMinOn`. -/
theorem objectiveMinimum_exists :
    ∃ x : setup.X, ∀ y : setup.X, setup.f x ≤ setup.f y := by
  exact SOptLib.objectiveMinimum_exists_of_isCompact_continuousOn setup.f
    setup.hX_compact setup.hX_nonempty
    (by
      intro x hx
      exact (setup.gradf_hasGradientAt x hx).continuousAt.continuousWithinAt)

/-- Canonical selected minimizer for `f^* := min_{x ∈ X} f(x)`. -/
noncomputable def objectiveMinimum : {x : setup.X // ∀ y : setup.X, setup.f x ≤ setup.f y} :=
  ⟨Classical.choose setup.objectiveMinimum_exists,
    Classical.choose_spec setup.objectiveMinimum_exists⟩

/-- The paper optimum value `f^* = min_{x ∈ X} f(x)`.

Candidate audit: chose SOptLib `optimizerValueOfMinimum`; it exactly extracts
the objective value from an attained global minimum, while the local
`objectiveMinimum_exists` theorem supplies this paper's compact feasible-set
attainment obligation from Eq. (7.4.1). -/
noncomputable def fStar : ℝ :=
  SOptLib.optimizerValueOfMinimum (fun x : setup.X => setup.f x) setup.objectiveMinimum

/-- The selected optimizer realizes `f^*`. -/
theorem fStar_eq :
    setup.fStar = setup.f ((setup.objectiveMinimum : setup.X) : E) := by
  rfl

/-- The canonical optimum value lower-bounds every feasible objective value. -/
theorem fStar_lb (x : E) (hx : x ∈ setup.X) : setup.fStar ≤ setup.f x := by
  simpa [fStar, SOptLib.optimizerValueOfMinimum] using
    setup.objectiveMinimum.2 ⟨x, hx⟩

/-- The number of epochs `S = ⌈N / T⌉` used in Algorithm 7.13.

This aligns with `algorithm_spec.initialization`; no reusable SOptLib primitive
is needed because this is the paper's closed-form natural-number schedule. -/
noncomputable def S : ℕ := (setup.N + setup.T - 1) / setup.T

/-- Defining equation for the paper epoch count. -/
theorem S_def : setup.S = (setup.N + setup.T - 1) / setup.T := rfl

/-- Existence of a pair attaining the feasible-set diameter.

Candidate audit: SOptLib `bregmanDiameterSq`, `bregmanDiameterSq_eq_declared_max`,
and compact-bound lemmas were checked; they cover Bregman or absolute bounds,
not the literal Euclidean maximum `max_{x,y in X} ‖x-y‖` from Eq. (7.1.18). -/
theorem diameterPair_exists :
    ∃ p : setup.X × setup.X,
      ∀ q : setup.X × setup.X,
        ‖(q.1 : E) - (q.2 : E)‖ ≤ ‖(p.1 : E) - (p.2 : E)‖ := by
  simpa [dist_eq_norm] using
    SOptLib.diameterPair_exists_of_isCompact
      (show setup.X.Nonempty from ⟨setup.x₁, setup.hx₁_mem⟩) setup.hX_compact

/-- A canonical pair realizing `D̄_X`.

No SOptLib match: searched "diameter compact set max distance" and scanned
`SOptLib/Model/Bregman.lean` plus `SOptLib/Glue/Analysis.lean`; none align with
Eq. (7.1.18) because those candidates model Bregman diameters or non-attained
uniform bounds, while this paper uses the literal attained Euclidean diameter. -/
noncomputable def diameterPair : setup.X × setup.X :=
  Classical.choose setup.diameterPair_exists

/-- The paper diameter `D̄_X = max_{x,y ∈ X} ‖x-y‖`. -/
noncomputable def barDX : ℝ :=
  ‖(setup.diameterPair.1 : E) - (setup.diameterPair.2 : E)‖

/-- The canonical feasible-set diameter is nonnegative. -/
theorem barDX_nonneg : 0 ≤ setup.barDX := by
  exact norm_nonneg _

/-- Every feasible pair is bounded by the canonical paper diameter. -/
theorem barDX_bound (x y : E) (hx : x ∈ setup.X) (hy : y ∈ setup.X) :
    ‖x - y‖ ≤ setup.barDX := by
  simpa [barDX] using
    SOptLib.le_euclideanDiameterOfPair_of_mem (p := setup.diameterPair)
      (by simpa [diameterPair] using Classical.choose_spec setup.diameterPair_exists) hx hy

/-- Raw real expression printed for the constant paper stepsize in Eq. (7.4.15):
`α = sqrt(((1/N + σ²/(L m)) / (L D̄_X²)))`.

Candidate audit: SOptLib step-size and parameter-choice candidates were checked
(`stepSize`, `constantBatchSchedule`, and `SOptLib/Model/ParameterChoices.lean`);
none match Eq. (7.4.15), because those objects model RSMD schedules and budget
selectors rather than this SNCCG diameter/variance-balanced constant. -/
noncomputable def rawAlphaFormula : ℝ :=
  Real.sqrt
    ((1 / (setup.N : ℝ) + setup.σ ^ 2 / (setup.L * setup.m)) /
      (setup.L * setup.barDX ^ 2))

/-- The nondegeneracy condition under which Eq. (7.4.15) is a genuine paper
stepsize formula rather than Lean's totalized division by zero.

Book citation: `algorithm_spec.parameters[0]` states
`α_k = α := [((1/N + σ²/(Lm)) 1/(L D̄_X²))]^{1/2}`, and
`algorithm_spec.parameters[2]` defines `D̄_X := max_{x,y∈X} ‖x-y‖`.

No SOptLib match: searched "step size denominator nonzero normalized output"
and checked `SOptLib.outputWeightDenominator_pos`; that candidate applies after
positive weights are known, while Eq. (7.4.15) first needs the paper diameter
denominator `L * D̄_X^2` to be nonzero. -/
def paperAlphaFormulaWellDefined : Prop :=
  0 < setup.barDX

/-- Nondegenerate-regime well-definedness bridge for Eq. (7.4.15)'s diameter
denominator. This is deliberately proof-parameterized: the book JSON states the
formula with `D̄_X` in the denominator but does not list `0 < D̄_X` as a
primitive assumption, so the object layer must not assert it unconditionally.

Book citation: `algorithm_spec.parameters[0]` states
`α_k = α := [((1/N + σ²/(Lm)) 1/(L D̄_X²))]^{1/2}` and
`algorithm_spec.parameters[2]` defines
`D̄_X := max_{x,y∈X} ‖x-y‖`. -/
theorem paperAlphaFormula_wellDefined_of_nonzeroDiameter
    (hDX : 0 < setup.barDX) : setup.paperAlphaFormulaWellDefined := by
  exact hDX

/-- Constant paper stepsize from Eq. (7.4.15), available only after the paper
formula's denominator is known to be nondegenerate.

No SOptLib match: searched "real sqrt parameter choice step size positive
denominator" and scanned `SOptLib/Model/ParameterChoices.lean`; the available
selectors are positive totalizations for other Lan algorithms, while Eq.
(7.4.15) needs the literal SNCCG formula under the explicit
`paperAlphaFormulaWellDefined` boundary. -/
noncomputable def paperAlphaOfWellDefined
    (_hαwf : setup.paperAlphaFormulaWellDefined) : ℝ :=
  setup.rawAlphaFormula

/-- Paper stepsize schedule `α_k = α` from Eq. (7.4.15), proof-parameterized by
the nonzero-diameter condition needed for the displayed division. -/
noncomputable def αOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (_k : ℕ) : ℝ :=
  setup.paperAlphaOfWellDefined hαwf

/-- Raw compatibility stepsize schedule with the displayed Eq. (7.4.15)
expression.

Candidate audit: checked SOptLib step-size/parameter-choice primitives and the
local `αOfWellDefined`; the reusable candidates do not match Eq. (7.4.15).
The paper-semantic schedule is `αOfWellDefined`; this raw helper exposes no
paper well-definedness claim. -/
noncomputable def rawPaperAlpha (_k : ℕ) : ℝ :=
  setup.rawAlphaFormula

/-- Displayed real expression for the stepsize schedule `α_k = α` from
Eq. (7.4.15).

Candidate audit: checked SOptLib step-size and parameter-choice primitives in
`SOptLib/Model/Iterates.lean` and `SOptLib/Model/ParameterChoices.lean`; none
match the SNCCG constant `[((1/N + σ²/(Lm)))/(L D̄_X²)]^{1/2}`. This raw wrapper
has no paper semantics outside the explicit `paperAlphaFormulaWellDefined`
boundary. -/
noncomputable def rawAlpha (k : ℕ) : ℝ :=
  setup.rawAlphaFormula

/-- Internal compatibility schedule for old recursive-process scaffolding. It is
the literal displayed real expression from Eq. (7.4.15), definitionally aligned
with the raw compatibility schedule.

Candidate audit: checked SOptLib step-size and parameter-choice candidates in
`SOptLib/Model/Iterates.lean` and `SOptLib/Model/ParameterChoices.lean`; none
match the SNCCG formula because they model other algorithms' schedules rather
than Eq. (7.4.15)'s diameter/variance-balanced constant. -/
noncomputable def internalAlpha (_k : ℕ) : ℝ :=
  setup.rawAlpha _k

/-- Legacy compatibility alias for old internal proof scaffolding. -/
noncomputable def rawStepsize (_k : ℕ) : ℝ :=
  setup.internalAlpha _k

/-- Definitional bridge for the constant paper stepsize. -/
theorem alphaOfWellDefined_eq_paperAlphaOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) :
    setup.αOfWellDefined hαwf k = setup.paperAlphaOfWellDefined hαwf := by
  rfl

/-- The Lean stepsize unfolds to the literal paper expression in Eq. (7.4.15). -/
theorem paperAlphaOfWellDefined_eq_formula
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    setup.paperAlphaOfWellDefined hαwf = setup.rawAlphaFormula := by
  rfl

/-- Positivity of the paper stepsize schedule once Eq. (7.4.15)'s denominator
is known to be a genuine nonzero paper quantity. -/
theorem hα_pos_of_paperAlphaFormulaWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) :
    0 < setup.αOfWellDefined hαwf k := by
  simpa [αOfWellDefined, paperAlphaOfWellDefined, rawAlphaFormula] using
    (SOptLib.varianceBalancedStepSize_pos setup.N setup.m setup.L setup.barDX setup.σ
      setup.hN_pos setup.hm_pos setup.hL_pos
      (by simpa [paperAlphaFormulaWellDefined] using hαwf))

/-- The paper stepsize is at most one in the parameter regime used for the
convex-combination update; callers must supply this parameter-regime fact
instead of relying on Lean totalization of Eq. (7.4.15). -/
theorem hα_le_one
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ)
    (hunit : setup.paperAlphaOfWellDefined hαwf ≤ 1) :
    setup.αOfWellDefined hαwf k ≤ 1 := by
  simpa [αOfWellDefined] using hunit

/-- Existence of the compact linear-minimization oracle point.

Candidate audit: SOptLib `proxStepArgmin`, `continuous_argmin_of_compact_unique`,
and `finiteArgmin` were checked; they either minimize prox objectives, require a
pre-existing selector, or apply only to finite index types, not the conditional
gradient LMO `argmin_{x in X} ⟪g,x⟫` from Algorithm 7.13. -/
theorem linearMinimizer_exists (g : E) :
    ∃ y : E, y ∈ setup.X ∧ ∀ x : E, x ∈ setup.X → ⟪g, y⟫_ℝ ≤ ⟪g, x⟫_ℝ := by
  exact SOptLib.linearMinimizer_exists_of_isCompact setup.hX_compact setup.hX_nonempty g

/-- Canonical realized linear minimization oracle
`y(g) = argmin_{x ∈ X} ⟪g,x⟫`.

No SOptLib match: searched "linear minimization oracle argmin compact convex"
and scanned `SOptLib/Model/Prox.lean`, `SOptLib/Model/Iterates.lean`, and
`SOptLib/Glue/Analysis.lean`; none align with Algorithm 7.13 because the
available argmin tools are prox, finite, or selector-continuity helpers rather
than the paper's compact feasible-set linear minimization oracle. -/
def linearMinimizer (g : E) : E :=
  setup.lmo g

/-- The canonical linear minimizer is feasible. -/
theorem linearMinimizer_mem (g : E) : setup.linearMinimizer g ∈ setup.X :=
  setup.lmo.mem g

/-- The canonical linear minimizer satisfies the Algorithm 7.13 argmin property. -/
theorem linearMinimizer_spec (g x : E) (hx : x ∈ setup.X) :
    ⟪g, setup.linearMinimizer g⟫_ℝ ≤ ⟪g, x⟫_ℝ :=
  setup.lmo.is_argmin g x hx

/-- Measurability of the canonical LMO selector used in adaptedness proofs. -/
theorem linearMinimizer_measurable : Measurable setup.linearMinimizer := by
  simpa [linearMinimizer] using setup.lmo.measurable

/-- The epoch index of global iteration k: s = k / T (0-indexed). -/
noncomputable def epochOf (k : ℕ) : ℕ := k / setup.T

/-- The within-epoch step index t = k % T + 1 (1-indexed within [1, T]). -/
noncomputable def stepOf (k : ℕ) : ℕ := SOptLib.stepOf setup.T k

/-- The global iteration index from epoch s and within-epoch step t. -/
noncomputable def globalIndex (s t : ℕ) : ℕ :=
  s * setup.T + t

/-- The local global-index notation agrees with the staged fixed-epoch encoder. -/
theorem globalIndex_eq_soptlib (s t : ℕ) :
    setup.globalIndex s t = SOptLib.globalIndex setup.T s t := by
  rfl

/-- Prefix monotonicity for the source iteration window encoded by
`globalIndex s t ≤ N`.

Book/PDF citation: Algorithm 7.13 runs only for `k = 1, ..., N`; Lemma 7.5 is
stated for an iteration index `k` (equivalently `(s,t)`) generated by that
loop. This arithmetic bridge lets proofs pass the source-domain fact from a
within-epoch target step back to earlier steps in the same epoch. -/
lemma globalIndex_prefix_le_of_le
    {s i t : ℕ} (hi : i ≤ t)
    (hkt : setup.globalIndex s t ≤ setup.N) :
    setup.globalIndex s i ≤ setup.N := by
  unfold StochasticNonconvexConditionalGradientSetup.globalIndex at *
  omega

/-! ### Retired arbitrary-epoch refresh domain

The old, unguarded refresh-floor helper treated every epoch coordinate `s` as
if it came from Algorithm 7.13's loop `k = 1, ..., N`. The compiled arithmetic
witness below records the concrete schedule collision behind that retirement:
with `N = 1`, `T = b = 2`, `m = T*b = 4`, the out-of-range refresh block for
`s = 1` overlaps recursive samples already used before `globalIndex 1 1`.
The current replacement exposes the source-domain premise
`globalIndex s 1 ≤ N` instead of assuming such freshness for arbitrary `s`. -/
private theorem retired_arbitrary_epoch_refresh_sample_overlap :
    let N : ℕ := 1
    let T : ℕ := 2
    let b : ℕ := 2
    let m : ℕ := T * b
    let s : ℕ := 1
    N < s * T + 1 ∧
      ∃ i < m, ∃ r < s, ∃ j, 2 ≤ j ∧ j ≤ T ∧
        ∃ q < b, s * m + i = N * m + (r * T + (j - 1)) * b + q := by
  dsimp
  refine ⟨by norm_num, ?_⟩
  refine ⟨2, by norm_num, 0, by norm_num, 2, by norm_num, by norm_num, 0, by norm_num, ?_⟩
  norm_num

/-- Existence of an attained Wolfe-gap maximizer on compact `X`.

No SOptLib match: searched "compact maximum argmax continuous set"; checked
`finiteArgmax` and compact argmin/continuity candidates such as
`continuous_argmin_of_compact_unique` and `mirrorStep_exists_compact`. Those
tools are finite, uniqueness/measurability, or prox-specific, while Eq. (7.3.3)
uses the literal attained maximum of `⟪∇f(x), x-y⟫` over the compact feasible
set. -/
theorem wolfeGapMaximizer_exists (x : E) :
    ∃ y : setup.X,
      ∀ z : setup.X,
        ⟪setup.gradf x, x - (z : E)⟫_ℝ ≤ ⟪setup.gradf x, x - (y : E)⟫_ℝ := by
  classical
  let φ : E → ℝ := fun y => ⟪setup.gradf x, x - y⟫_ℝ
  have hcont : ContinuousOn φ setup.X := by
    exact (continuous_const.inner (continuous_const.sub continuous_id)).continuousOn
  obtain ⟨y, hyX, hymax⟩ := setup.hX_compact.exists_isMaxOn setup.hX_nonempty hcont
  refine ⟨⟨y, hyX⟩, ?_⟩
  intro z
  exact hymax z.property

/-- Canonical attained maximizer in the paper Wolfe gap
`max_{y ∈ X} ⟪∇f(x), x-y⟫`. -/
noncomputable def wolfeGapMaximizer (x : E) : setup.X :=
  Classical.choose (setup.wolfeGapMaximizer_exists x)


/-- The canonical Wolfe-gap maximizer attains the paper maximum. -/
theorem wolfeGap_spec (x : E) (z : setup.X) :
    ⟪setup.gradf x, x - (z : E)⟫_ℝ ≤ SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x := by
  simpa [wolfeGapMaximizer] using
    SOptLib.ConditionalGradient.le_wolfeGap
      (X := setup.X) setup.gradf setup.wolfeGapMaximizer_exists x z

/-- Existence of an attained estimator linear-model maximizer on compact `X`. -/
theorem maxLinModelMaximizer_exists (x G : E) :
    ∃ y : setup.X,
      ∀ z : setup.X,
        ⟪G, x - (z : E)⟫_ℝ ≤ ⟪G, x - (y : E)⟫_ℝ := by
  exact SOptLib.exists_linear_model_maximizer_on_compact setup.hX_compact setup.hX_nonempty x G

/-- Canonical attained maximizer for the estimator linear-model gap. -/
noncomputable def maxLinModelMaximizer (x G : E) : setup.X :=
  Classical.choose (setup.maxLinModelMaximizer_exists x G)

/-- The estimator-model Wolfe surrogate at x with estimator G:
`maxLinModel(x, G) = max_{y ∈ X} ⟪G, x-y⟫`, realized by the compact maximizer. -/
noncomputable def maxLinModel (x G : E) : ℝ :=
  ⟪G, x - (setup.maxLinModelMaximizer x G : E)⟫_ℝ

/-- The canonical estimator-model maximizer attains the paper maximum. -/
theorem maxLinModel_spec (x G : E) (z : setup.X) :
    ⟪G, x - (z : E)⟫_ℝ ≤ setup.maxLinModel x G := by
  simpa [maxLinModel, maxLinModelMaximizer] using
    (Classical.choose_spec (setup.maxLinModelMaximizer_exists x G) z)

/-- The estimator error δ_k = G_k - ∇f(x_k). -/
noncomputable def delta (G x : E) : E :=
  SOptLib.oracleEstimatorError setup.gradf G x

/-- The paper sum of stepsizes `Σ_{k=1}^{N} α_k`, available once the constant
stepsize formula is well-defined. -/
noncomputable def alphaSumOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) : ℝ :=
  Finset.sum (Finset.Icc 1 setup.N) (setup.αOfWellDefined hαwf)

/-- Raw displayed denominator `Σ_{k=1}^N α_k` from Algorithm 7.13. -/
noncomputable def rawPaperAlphaSum : ℝ :=
  Finset.sum (Finset.Icc 1 setup.N) setup.rawPaperAlpha

/-- Displayed sum `Σ_{k=1}^{N} α_k` formed from the literal Eq. (7.4.15)
expression. The normalized paper output law uses `alphaSumOfWellDefined`
instead, because Algorithm 7.13's probability masses are meaningful only after
the nonzero-diameter boundary has been supplied. -/
noncomputable def rawDisplayedAlphaSum : ℝ :=
  Finset.sum (Finset.Icc 1 setup.N) setup.rawAlpha

/-- Internal compatibility normalizer for old proof scaffolding. The canonical
paper normalizer is `alphaSumOfWellDefined`, which requires the
well-definedness proof for Eq. (7.4.15). -/
noncomputable def internalAlphaSum : ℝ :=
  Finset.sum (Finset.Icc 1 setup.N) setup.internalAlpha

/-- Legacy compatibility alias for old proof scaffolding. -/
noncomputable def rawAlphaSum : ℝ :=
  setup.internalAlphaSum

/-- Positivity of the canonical paper output normalizer. -/
theorem alphaSum_pos_of_wellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) :
    0 < setup.alphaSumOfWellDefined hαwf := by
  exact hR

/-- General output-law normalization bridge: positive weights on the paper
window imply the denominator in Algorithm 7.13's law is positive.

Candidate audit: checked SOptLib `outputWeightDenominator_pos`; this theorem is
the local specialization to the SNCCG window `{1, ..., N}` and does not assert
the paper parameter regime itself. -/
theorem paperOutputLawWellDefined_of_forall_alpha_pos
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_pos : ∀ k, k ∈ Finset.Icc 1 setup.N →
      0 < setup.αOfWellDefined hαwf k) :
    0 < setup.alphaSumOfWellDefined hαwf := by
  have hnonempty : (Finset.Icc 1 setup.N).Nonempty := ⟨1, by simp [setup.hN_pos]⟩
  simpa [alphaSumOfWellDefined, SOptLib.outputWeightDenominator]
    using SOptLib.outputWeightDenominator_pos (Finset.Icc 1 setup.N)
      (setup.αOfWellDefined hαwf) hα_pos hnonempty

/-- Any use of the internal explicit-normalizer output law exposes the positive
denominator it depends on.

Book citation: `algorithm_spec.output` states
`Prob{R = k} = α_k / Σ_{k=1}^{N} α_k`, `k = 1, ..., N`. Every normalized
output object now carries this proof explicitly, so the denominator condition
cannot be hidden behind a totalized PMF. -/
theorem paperOutputLaw_requires_positive_denominator
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) :
    0 < setup.alphaSumOfWellDefined hαwf := by
  exact hR

/-- In the nondegenerate-diameter branch, positivity of the Eq. (7.4.15)
stepsize gives the normalized Algorithm 7.13 output denominator.

Book citation: `algorithm_spec.output.math` states
`Prob{R = k} = α_k / Σ_{k=1}^{N} α_k`; the proof is conditional on
`0 < D̄_X`, so it does not promote denominator positivity to a primitive setup
assumption. -/
theorem alphaSum_pos_of_nonzeroDiameter
    (hDX : 0 < setup.barDX) :
    0 < setup.alphaSumOfWellDefined hDX := by
  simpa [alphaSumOfWellDefined, SOptLib.outputWeightDenominator] using
    SOptLib.outputWeightDenominator_pos_of_Icc_pos
      setup.N (setup.αOfWellDefined hDX) setup.hN_pos
      (fun k _hk => setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)

/-- The paper index window `{2, ..., T}` in the epoch maximum from Theorems
7.16/7.17.

Book citation: `key_lemmas[2].statement_math` contains
`max_{j=2,...,T} α_{s,j}`. -/
noncomputable def innerAlphaWindow : Finset ℕ :=
  Finset.Icc 2 setup.T

/-- Nonemptiness condition for the paper maximum over `{2, ..., T}`.

Book citation: `key_lemmas[2].statement_math` writes a maximum over
`j = 2, ..., T`, which is a genuine nonempty finite maximum only under
`2 ≤ T`. -/
noncomputable def InnerAlphaWindowWellDefined : Prop :=
  2 ≤ setup.T

/-- Conditional realization boundary for the source-facing Algorithm 7.13 path.

This is the single object-layer contract for facts needed to realize the
displayed paper expressions but not derivable from the current book JSON:
`0 < D̄_X` for Eq. (7.4.15)'s denominator, the parameter-regime fact
`α ≤ 1` needed for the displayed convex-combination update to be a genuine
convex combination, and `2 ≤ T` for the finite maximum `max_{j=2,...,T}`. It
is a realization boundary, not a paper assumption; zero diameter and
empty-window behavior remain only in declarations whose names include
`Extension` or `degenerate`.

Book citation: `algorithm_spec.parameters[0].math` gives the denominator
`L D̄_X^2`; `algorithm_spec.parameters[2].math` defines
`D̄_X := max_{x,y∈X} ‖x-y‖`; `algorithm_spec.steps[3].math` writes
`x_{k+1} = (1 - α_k)x_k + α_k y_k`; `key_lemmas[2].statement_math` contains
`max_{j=2,...,T} α_{s,j}`. The current JSON does not assert `0 < D̄_X`,
`α ≤ 1`, or `2 ≤ T`. -/
abbrev Algorithm713RealizationContract : Prop :=
  SOptLib.ConditionalGradientRealizationContract setup.barDX
    (fun hDX =>
      setup.paperAlphaOfWellDefined
        (setup.paperAlphaFormula_wellDefined_of_nonzeroDiameter hDX))
    setup.T

/-- The contract supplies the nonzero-diameter boundary for Eq. (7.4.15). -/
theorem paperAlphaFormulaWellDefined_of_algorithm713Contract
    (h : setup.Algorithm713RealizationContract) :
    setup.paperAlphaFormulaWellDefined :=
  h.nonzeroDiameter

/-- The contract supplies the parameter-regime boundary that makes the displayed
Algorithm 7.13 update a genuine convex combination. -/
theorem paperAlpha_le_one_of_algorithm713Contract
    (h : setup.Algorithm713RealizationContract) :
    setup.paperAlphaOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) ≤ 1 := by
  simpa [paperAlphaFormulaWellDefined_of_algorithm713Contract] using h.alpha_le_one

/-- The contract supplies the nonempty-window boundary for
`max_{j=2,...,T}`. -/
theorem innerAlphaWindowWellDefined_of_algorithm713Contract
    (h : setup.Algorithm713RealizationContract) :
    setup.InnerAlphaWindowWellDefined := by
  simpa [InnerAlphaWindowWellDefined] using h.two_le_T

/-- The Algorithm 7.13 realization contract derives the output-law normalizer
positivity required by `Prob{R=k}=α_k/Σα_k`; it is not stored as an independent
paper assumption. -/
theorem alphaSum_pos_of_algorithm713Contract
    (h : setup.Algorithm713RealizationContract) :
    0 < setup.alphaSumOfWellDefined
      (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) :=
by
  simpa [alphaSumOfWellDefined, paperAlphaFormulaWellDefined_of_algorithm713Contract]
    using SOptLib.outputWeightDenominator_pos_Icc_of_nonzeroDiameter
      h.nonzeroDiameter
      setup.N
      (fun hDX => setup.αOfWellDefined hDX)
      setup.hN_pos
      (fun k _hk => setup.hα_pos_of_paperAlphaFormulaWellDefined h.nonzeroDiameter k)

/-- Paper-side nonemptiness of `{2, ..., T}` is an explicit domain obligation.

Book citation: `key_lemmas[2].statement_math` uses
`max_{j=2,...,T} α_{s,j}`; the JSON gives `b = T = sqrt(m)` but does not state
`2 ≤ T`. The object layer therefore keeps the genuine finite maximum behind
`paperMaxInnerAlphaOfWellDefined` instead of asserting this condition for every
setup. -/
theorem innerAlphaWindow_requires_two_le_T
    (hT : setup.InnerAlphaWindowWellDefined) : 2 ≤ setup.T := by
  simpa [InnerAlphaWindowWellDefined] using hT

/-- Nonemptiness of the paper window `{2, ..., T}` once the real paper-side
domain condition `2 ≤ T` is available.

Candidate audit: checked SOptLib `finiteRunMaxValue` and Mathlib `Finset.sup'`;
the reusable maximum selectors require this exact nonemptiness proof but do not
derive `2 ≤ T` from the SNCCG parameters. -/
theorem innerAlphaWindow_nonempty_of_wellDefined
    (hT : setup.InnerAlphaWindowWellDefined) : setup.innerAlphaWindow.Nonempty := by
  exact ⟨2, by simpa [innerAlphaWindow, InnerAlphaWindowWellDefined] using hT⟩

/-- The within-epoch maximum `max_{j=2,...,T} α_{s,j}` appearing in Theorem
7.16/7.17.

Candidate audit: checked SOptLib `finiteRunMaxValue` and Mathlib `Finset.sup'`;
the definition uses the genuine finite maximum primitive under an explicit
nonempty-window proof. The book JSON does not state `2 ≤ T`, so this
nondegenerate paper object is proof-parameterized rather than exported through an
unconditional empty-window convention. -/
noncomputable def paperMaxInnerAlphaOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hT : setup.InnerAlphaWindowWellDefined) (s : ℕ) : ℝ :=
  Finset.sup' setup.innerAlphaWindow
    (setup.innerAlphaWindow_nonempty_of_wellDefined hT)
    (fun j => setup.αOfWellDefined hαwf (setup.globalIndex s j))

/-- Internal compatibility alias for proof scaffolding that previously used
`maxInnerAlpha`; it is now proof-parameterized like
`paperMaxInnerAlphaOfWellDefined`. -/
noncomputable def maxInnerAlphaOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hT : setup.InnerAlphaWindowWellDefined) (s : ℕ) : ℝ :=
  setup.paperMaxInnerAlphaOfWellDefined hαwf hT s

/-- Under the paper-domain proof `2 ≤ T`, `paperMaxInnerAlphaOfWellDefined` is
the genuine finite maximum over `{2, ..., T}`. -/
theorem paperMaxInnerAlphaOfWellDefined_eq_sup
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s : ℕ) (hT : setup.InnerAlphaWindowWellDefined) :
    setup.paperMaxInnerAlphaOfWellDefined hαwf hT s =
      Finset.sup' setup.innerAlphaWindow
        (setup.innerAlphaWindow_nonempty_of_wellDefined hT)
        (fun j => setup.αOfWellDefined hαwf (setup.globalIndex s j)) := by
  simp [paperMaxInnerAlphaOfWellDefined]

/-- In Algorithm 7.13, Eq. (7.4.15) makes the stochastic stepsize schedule
constant, so the displayed current-coordinate epoch maximum is just that
constant stepsize. This is the bridge that keeps the stochastic Theorem 7.17
literal maximum compatible with the corrected finite-sum epoch-difference
analysis. -/
theorem paperMaxInnerAlphaOfWellDefined_eq_paperAlphaOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hT : setup.InnerAlphaWindowWellDefined) (s : ℕ) :
    setup.paperMaxInnerAlphaOfWellDefined hαwf hT s =
      setup.paperAlphaOfWellDefined hαwf := by
  classical
  rw [setup.paperMaxInnerAlphaOfWellDefined_eq_sup hαwf s hT]
  simpa [αOfWellDefined] using
    (Finset.sup'_const (s := setup.innerAlphaWindow)
      (H := setup.innerAlphaWindow_nonempty_of_wellDefined hT)
      (a := setup.paperAlphaOfWellDefined hαwf))

/-- The internal compatibility alias agrees definitionally with the
domain-aware paper epoch maximum. -/
theorem maxInnerAlphaOfWellDefined_eq_paperMaxInnerAlphaOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hT : setup.InnerAlphaWindowWellDefined) (s : ℕ) :
    setup.maxInnerAlphaOfWellDefined hαwf hT s =
      setup.paperMaxInnerAlphaOfWellDefined hαwf hT s := by
  rfl

/-- Source-facing displayed epoch maximum built from the literal Eq. (7.4.15)
stepsize expression, before the nonzero-diameter realization boundary is
available. The only explicit domain proof is the nonempty paper window
`2 ≤ T`; zero-diameter semantics are not repaired here by an extension branch.

Candidate audit: checked SOptLib `finiteRunMaxValue` and Mathlib `Finset.sup'`;
`Finset.sup'` is the matching genuine finite maximum once the source-boundary
window proof is supplied, while SOptLib has no primitive for the SNCCG raw
Eq. (7.4.15) schedule. -/
noncomputable def rawPaperMaxInnerAlpha
    (hT : setup.InnerAlphaWindowWellDefined) (s : ℕ) : ℝ :=
  Finset.sup' setup.innerAlphaWindow
    (setup.innerAlphaWindow_nonempty_of_wellDefined hT)
    (fun j => setup.rawAlpha (setup.globalIndex s j))

/-- Raw literal displayed stepsize expression from Eq. (7.4.15). -/
noncomputable def rawDisplayedAlpha (_k : ℕ) : ℝ :=
  setup.rawAlphaFormula

/-- The displayed stepsize expression unfolds to Eq. (7.4.15). -/
theorem rawDisplayedAlpha_eq_rawAlphaFormula (k : ℕ) :
    setup.rawDisplayedAlpha k = setup.rawAlphaFormula := by
  rfl

/-- The epoch-start batch gradient estimator at x using mini-batch H^s of size m. -/
noncomputable def batchGrad (x : E) (samples : Fin setup.m → Ξ) : E :=
  (setup.m : ℝ)⁻¹ • Finset.sum Finset.univ (fun i => setup.gradF x (samples i))

/-- The recursive gradient-difference update inside an epoch. -/
noncomputable def recursiveGrad
    (G_prev x_prev x_curr : E) (samples : Fin setup.b → Ξ) : E :=
  SOptLib.recursiveGradientDifferenceAverage setup.gradF G_prev x_prev x_curr samples

/-- Raw compatibility convex-combination update using Lean's totalized displayed
stepsize expression. Source-facing Algorithm 7.13 objects use
`iterUpdateOfWellDefined` instead. -/
noncomputable def rawIterUpdate (x G : E) (k : ℕ) : E :=
  (1 - setup.rawPaperAlpha k) • x + setup.rawPaperAlpha k • setup.linearMinimizer G

/-- Domain-aware Algorithm 7.13 convex-combination update
`x_{k+1} = (1 - α_k) x_k + α_k y_k` using the genuine Eq. (7.4.15) stepsize.

Candidate audit: no SOptLib match for the conditional-gradient convex
combination update; searched `proxOracleStep`, iterate update wrappers, and
`SOptLib/Model/Iterates.lean`. Those candidates model prox/oracle state
updates, while Algorithm 7.13 requires the literal Frank-Wolfe combination with
the LMO point. -/
noncomputable def iterUpdateOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (x G : E) (k : ℕ) : E :=
  (1 - setup.αOfWellDefined hαwf k) • x +
    setup.αOfWellDefined hαwf k • setup.linearMinimizer G

/-- Raw compatibility recursive process generated with the totalized displayed
stepsize expression. The paper process is `processOfWellDefined`, which carries
the Eq. (7.4.15) nonzero-denominator boundary explicitly. -/
noncomputable def rawProcess : ℕ → Ω → StochasticNonconvexCGState E
  | 0 => fun _ =>
      { x := setup.x₁
        G := 0
        s := 0 }
  | 1 => fun ω =>
      { x := setup.x₁
        G :=
          (setup.m : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.m)
              (fun i => setup.gradF setup.x₁ (setup.ξ i ω))
        s := 0 }
  | k + 2 => fun ω =>
      let prev := rawProcess (k + 1) ω
      let xk := prev.x
      let Gk := prev.G
      let sk := prev.s
      let xk1 := setup.rawIterUpdate xk Gk (k + 1)
      -- If `k + 1` is the last step of its epoch, the next state starts a new
      -- epoch and refreshes the estimator at `x_{k+1}`.
      let isNextEpochStart := ((k + 1) % setup.T == 0)
      let sk1 := if isNextEpochStart then sk + 1 else sk
      let Gk1 : E :=
        if isNextEpochStart then
          (setup.m : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.m)
              (fun i => setup.gradF xk1 (setup.ξ (sk1 * setup.m + i) ω))
        else
          (setup.b : ℝ)⁻¹ •
              Finset.sum (Finset.range setup.b)
                (fun i =>
                  setup.gradF xk1 (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω) -
                  setup.gradF xk (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω)) +
            Gk
      { x := xk1
        G := Gk1
        s := sk1 }

/-- Domain-aware Algorithm 7.13 joint process driven by the paper stepsize
`αOfWellDefined`. It is parameterized by the nonzero-diameter boundary because
Eq. (7.4.15) contains `D̄_X` in the denominator and the source JSON does not
state `0 < D̄_X` as an unconditional assumption.

Candidate audit: checked SOptLib `iterateProcessView`, `proxOracleStep`, and
recursive-process helpers in `SOptLib/Model/Iterates.lean`; they provide process
views and prox-oracle updates, not Algorithm 7.13's epoch-refresh/recursive
gradient-estimator state. -/
noncomputable def processOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    ℕ → Ω → StochasticNonconvexCGState E :=
  SOptLib.varianceReducedConditionalGradientProcess
    (fun x G s => ({ x := x, G := G, s := s } : StochasticNonconvexCGState E))
    (fun state : StochasticNonconvexCGState E => state.x)
    (fun state : StochasticNonconvexCGState E => state.G)
    (fun state : StochasticNonconvexCGState E => state.s)
    setup.x₁ setup.m setup.b setup.T setup.N setup.ξ setup.gradF
    (setup.iterUpdateOfWellDefined hαwf)

@[simp]
theorem processOfWellDefined_zero
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    setup.processOfWellDefined hαwf 0 =
      fun _ =>
        { x := setup.x₁
          G := 0
          s := 0 } := by
  rfl

@[simp]
theorem processOfWellDefined_one
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    setup.processOfWellDefined hαwf 1 =
      fun ω =>
        { x := setup.x₁
          G :=
            (setup.m : ℝ)⁻¹ •
              Finset.sum (Finset.range setup.m)
                (fun i => setup.gradF setup.x₁ (setup.ξ i ω))
          s := 0 } := by
  rfl

@[simp]
theorem processOfWellDefined_succ_succ
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) :
    setup.processOfWellDefined hαwf (k + 2) =
      fun ω =>
        let prev := setup.processOfWellDefined hαwf (k + 1) ω
        let xk := prev.x
        let Gk := prev.G
        let sk := prev.s
        let xk1 := setup.iterUpdateOfWellDefined hαwf xk Gk (k + 1)
        let isNextEpochStart := ((k + 1) % setup.T == 0)
        let sk1 := if isNextEpochStart then sk + 1 else sk
        let Gk1 : E :=
          if isNextEpochStart then
            (setup.m : ℝ)⁻¹ •
              Finset.sum (Finset.range setup.m)
                (fun i => setup.gradF xk1 (setup.ξ (sk1 * setup.m + i) ω))
          else
            (setup.b : ℝ)⁻¹ •
                Finset.sum (Finset.range setup.b)
                  (fun i =>
                    setup.gradF xk1
                        (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω) -
                      setup.gradF xk
                        (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω)) +
              Gk
        { x := xk1
          G := Gk1
          s := sk1 } := by
  rfl

/-- Raw compatibility iterate at step k. -/
noncomputable def rawIterProcess (k : ℕ) : Ω → E :=
  fun ω => (setup.rawProcess k ω).x

/-- Domain-aware Algorithm 7.13 iterate `x_k` generated by
`processOfWellDefined`. -/
noncomputable def iterProcessOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) : Ω → E :=
  fun ω => (setup.processOfWellDefined hαwf k ω).x

/-- Raw compatibility estimator at step k. -/
noncomputable def rawEstimatorProcess (k : ℕ) : Ω → E :=
  fun ω => (setup.rawProcess k ω).G

/-- Domain-aware Algorithm 7.13 estimator `G_k` generated by
`processOfWellDefined`. -/
noncomputable def estimatorProcessOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) : Ω → E :=
  fun ω => (setup.processOfWellDefined hαwf k ω).G

/-- Raw compatibility realized noise δ_k = G_k - ∇f(x_k). -/
noncomputable def rawDeltaProcess (k : ℕ) : Ω → E :=
  fun ω => setup.delta (setup.rawEstimatorProcess k ω) (setup.rawIterProcess k ω)

/-- Domain-aware realized estimator error
`δ_k = G_k - ∇f(x_k)` for the process generated with the paper stepsize. -/
noncomputable def deltaProcessOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) : Ω → E :=
  SOptLib.estimatorResidualProcess
    (setup.estimatorProcessOfWellDefined hαwf)
    (setup.iterProcessOfWellDefined hαwf) setup.gradf k

/-- The natural filtration generated by the sample stream.

This reuses `SOptLib.filtration`, aligning Algorithm 7.13's i.i.d. sample stream
with the library's canonical sample-prefix filtration. -/
noncomputable def filtration : Filtration ℕ ‹MeasurableSpace Ω› :=
  SOptLib.filtration setup.ξ setup.hξ_meas

/-- Raw paper mass expression `α_k / Σ_j α_j` for the randomized output law.
Internal domain-aware helper; the PMF construction remains parameterized by the
positive-normalizer proof because Algorithm 7.13 does not state that proof as a
primitive assumption. -/
noncomputable def outputWeightsOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) : ℝ :=
  setup.αOfWellDefined hαwf k / setup.alphaSumOfWellDefined hαwf

/-- Positive output indices `R ∈ {1, ..., N}` for Algorithm 7.13.

Candidate audit: chose SOptLib `positiveTimeOutputWindowTimes` as the closest
library alignment for positive paper-time output windows; this local subtype
specializes that concept to the exact SNCCG window `{1, ..., N}` from Algorithm
7.13, while normalization is kept in `outputPMFOfWellDefined`. -/
abbrev OutputTime : Type :=
  {k : ℕ // k ∈ Finset.Icc 1 setup.N}

/-- Raw displayed Algorithm 7.13 stepsize expression. This is the literal
Eq. (7.4.15) real formula before the realization contract is supplied; it is
not the paper-canonical stepsize schedule.

No SOptLib match: searched "finite PMF normalized output weights" and
"step size denominator nonzero normalized output"; SOptLib provides normalized
weight helpers after positivity is known, but not this paper's
diameter/variance-balanced SNCCG constant from Eq. (7.4.15). -/
noncomputable def algorithmAlphaDisplayedExpression (k : ℕ) : ℝ :=
  setup.rawDisplayedAlpha k

/-- Conditional realization of the Algorithm 7.13 stepsize schedule under the
single non-source realization contract. -/
noncomputable def algorithmAlphaConditional
    (h : setup.Algorithm713RealizationContract) (k : ℕ) : ℝ :=
  setup.αOfWellDefined
    (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) k

/-- The raw displayed schedule unfolds to Eq. (7.4.15). -/
theorem algorithmAlphaDisplayedExpression_eq_rawDisplayedAlpha (k : ℕ) :
    setup.algorithmAlphaDisplayedExpression k = setup.rawDisplayedAlpha k := by
  rfl

/-- The conditional Algorithm 7.13 schedule is definitionally the paper schedule
inside the realization boundary. -/
theorem algorithmAlphaConditional_eq_of_wellDefined
    (h : setup.Algorithm713RealizationContract) (k : ℕ) :
    setup.algorithmAlphaConditional h k =
      setup.αOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) k := by
  rfl

/-- The raw displayed Algorithm 7.13 output normalizer formed from the literal
Eq. (7.4.15) expression. This is not the normalized paper output law because
the current JSON does not assert denominator positivity. -/
noncomputable def alphaSumDisplayedExpression : ℝ :=
  Finset.sum (Finset.Icc 1 setup.N) setup.algorithmAlphaDisplayedExpression

/-- Conditional realization of the Algorithm 7.13 output normalizer under the
single non-source realization contract. -/
noncomputable def alphaSumConditional
    (h : setup.Algorithm713RealizationContract) : ℝ :=
  Finset.sum (Finset.Icc 1 setup.N) (setup.algorithmAlphaConditional h)

/-- The conditional Algorithm 7.13 normalizer is the paper normalizer built from
Eq. (7.4.15) inside the realization boundary. -/
theorem alphaSumConditional_eq_of_wellDefined
    (h : setup.Algorithm713RealizationContract) :
    setup.alphaSumConditional h =
      setup.alphaSumOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) := by
  rfl

/-- Degenerate `D̄_X = 0` extension convention: all output is the initial
feasible point. This object is used only by declarations named `Extension` or
`degenerate`, not by the paper-facing Algorithm 7.13 process. -/
noncomputable def degenerateProcess : ℕ → Ω → StochasticNonconvexCGState E :=
  fun _ _ =>
    { x := setup.x₁
      G := 0
      s := 0 }

/-- Raw displayed Algorithm 7.13 recursive process generated by Lean's
totalized Eq. (7.4.15) real expression. This compatibility object is not the
paper-canonical process; the realized paper process is
`algorithmProcessConditional` under the single realization contract.

Book citation: `algorithm_spec.parameters[0]` gives the Eq. (7.4.15) denominator
`L D̄_X^2`; `algorithm_spec.output.math` gives the normalized output law
`Prob{R=k}=α_k/Σα_k`. -/
noncomputable def algorithmProcessDisplayedExpression :
    ℕ → Ω → StochasticNonconvexCGState E :=
  setup.rawProcess

/-- Conditional realization of the Algorithm 7.13 process under the single
non-source realization contract. -/
noncomputable def algorithmProcessConditional
    (h : setup.Algorithm713RealizationContract) :
    ℕ → Ω → StochasticNonconvexCGState E :=
  setup.processOfWellDefined
    (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h)

/-- Non-paper extension run boundary. It agrees with Algorithm 7.13 when
Eq. (7.4.15) is well-defined and otherwise uses the documented stationary
zero-diameter convention for auxiliary lemmas only. -/
noncomputable def algorithmProcessExtension : ℕ → Ω → StochasticNonconvexCGState E :=
  by
    classical
    exact
      if hDX : setup.paperAlphaFormulaWellDefined then
        setup.processOfWellDefined hDX
      else
        setup.degenerateProcess

/-- The raw displayed Algorithm 7.13 process unfolds to the compatibility
recursive process. -/
theorem algorithmProcessDisplayedExpression_eq_rawProcess :
    setup.algorithmProcessDisplayedExpression = setup.rawProcess := by
  rfl

/-- The conditional Algorithm 7.13 process is exactly the recursive process
generated inside the Eq. (7.4.15) boundary. -/
theorem algorithmProcessConditional_eq_of_wellDefined
    (h : setup.Algorithm713RealizationContract) :
    setup.algorithmProcessConditional h =
      setup.processOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) := by
  rfl

/-- Raw displayed Algorithm 7.13 iterate from the compatibility process. -/
noncomputable def algorithmIterProcessDisplayedExpression (k : ℕ) : Ω → E :=
  fun ω => (setup.algorithmProcessDisplayedExpression k ω).x

/-- Conditional Algorithm 7.13 iterate under the realization boundary. -/
noncomputable def algorithmIterProcessConditional
    (h : setup.Algorithm713RealizationContract) (k : ℕ) : Ω → E :=
  fun ω => (setup.algorithmProcessConditional h k ω).x

/-- Raw displayed Algorithm 7.13 estimator from the compatibility process. -/
noncomputable def algorithmEstimatorProcessDisplayedExpression (k : ℕ) : Ω → E :=
  fun ω => (setup.algorithmProcessDisplayedExpression k ω).G

/-- Conditional Algorithm 7.13 estimator under the realization boundary. -/
noncomputable def algorithmEstimatorProcessConditional
    (h : setup.Algorithm713RealizationContract) (k : ℕ) : Ω → E :=
  fun ω => (setup.algorithmProcessConditional h k ω).G

/-- Raw displayed Algorithm 7.13 estimator error
`δ_k = G_k - ∇f(x_k)` from the compatibility process. -/
noncomputable def algorithmDeltaProcessDisplayedExpression (k : ℕ) : Ω → E :=
  fun ω =>
    setup.delta (setup.algorithmEstimatorProcessDisplayedExpression k ω)
      (setup.algorithmIterProcessDisplayedExpression k ω)

/-- Conditional Algorithm 7.13 estimator error under the realization boundary. -/
noncomputable def algorithmDeltaProcessConditional
    (h : setup.Algorithm713RealizationContract) (k : ℕ) : Ω → E :=
  fun ω =>
    setup.delta (setup.algorithmEstimatorProcessConditional h k ω)
      (setup.algorithmIterProcessConditional h k ω)

/-- Non-paper extension iterate process for auxiliary zero-diameter lemmas. -/
noncomputable def algorithmIterProcessExtension (k : ℕ) : Ω → E :=
  fun ω => (setup.algorithmProcessExtension k ω).x

/-- Non-paper extension estimator process for auxiliary zero-diameter lemmas. -/
noncomputable def algorithmEstimatorProcessExtension (k : ℕ) : Ω → E :=
  fun ω => (setup.algorithmProcessExtension k ω).G

/-- Non-paper extension estimator error for auxiliary zero-diameter lemmas. -/
noncomputable def algorithmDeltaProcessExtension (k : ℕ) : Ω → E :=
  fun ω =>
    setup.delta (setup.algorithmEstimatorProcessExtension k ω)
      (setup.algorithmIterProcessExtension k ω)

/-- First output index, used by the explicit degenerate `D̄_X = 0` convention. -/
def firstOutputTime : setup.OutputTime :=
  ⟨1, Finset.mem_Icc.mpr ⟨le_rfl, setup.hN_pos⟩⟩

/-- Non-paper extension convention for `{2,...,T}`: use the genuine finite
maximum when nonempty and `0` otherwise. Conditional theorem statements use
`paperMaxInnerAlphaOfWellDefined`; this helper is retained only for auxiliary
compatibility lemmas about empty windows. -/
noncomputable def paperMaxInnerAlphaEmptyWindowExtension (s : ℕ) : ℝ :=
  if h : setup.innerAlphaWindow.Nonempty then
    Finset.sup' setup.innerAlphaWindow h
      (fun j => setup.rawAlpha (setup.globalIndex s j))
  else
    0

/-- Under `2 ≤ T`, the extension convention agrees with the genuine finite
maximum over `{2,...,T}` for the Eq. (7.4.15) schedule. -/
theorem paperMaxInnerAlphaEmptyWindowExtension_eq_sup_of_wellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s : ℕ) (hT : setup.InnerAlphaWindowWellDefined) :
    setup.paperMaxInnerAlphaEmptyWindowExtension s =
      Finset.sup' setup.innerAlphaWindow
        (setup.innerAlphaWindow_nonempty_of_wellDefined hT)
        (fun j =>
          setup.αOfWellDefined hαwf (setup.globalIndex s j)) := by
  classical
  have hne : setup.innerAlphaWindow.Nonempty :=
    setup.innerAlphaWindow_nonempty_of_wellDefined hT
  simp [paperMaxInnerAlphaEmptyWindowExtension, hne, rawAlpha, αOfWellDefined,
    paperAlphaOfWellDefined]

/-- Paper mass `Prob{R = k} = α_k / Σ_{j=1}^N α_j` on the finite output window. -/
noncomputable def outputMassOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (R : setup.OutputTime) : ℝ :=
  setup.outputWeightsOfWellDefined hαwf R.1

/-- Nonnegativity of the internal domain-aware output masses. -/
theorem outputMassOfWellDefined_nonneg (R : setup.OutputTime)
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    0 ≤ setup.outputMassOfWellDefined hαwf R := by
  exact div_nonneg
    (le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hαwf R.1))
    (le_of_lt (setup.alphaSum_pos_of_nonzeroDiameter hαwf))

/-- The Algorithm 7.13 output masses sum to one once the normalizer is known
positive. -/
theorem outputMass_sum_one_of_wellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) :
    (∑ R : setup.OutputTime, setup.outputMassOfWellDefined hαwf R) = 1 := by
  classical
  have hreindex :
      (∑ R : setup.OutputTime, setup.αOfWellDefined hαwf R.1) =
        setup.alphaSumOfWellDefined hαwf := by
    simpa [OutputTime, alphaSumOfWellDefined] using
      (Finset.sum_attach (s := Finset.Icc 1 setup.N)
        (f := setup.αOfWellDefined hαwf))
  unfold outputMassOfWellDefined outputWeightsOfWellDefined
  calc
    (∑ R : setup.OutputTime,
        setup.αOfWellDefined hαwf R.1 / setup.alphaSumOfWellDefined hαwf)
        = (∑ R : setup.OutputTime, setup.αOfWellDefined hαwf R.1) /
            setup.alphaSumOfWellDefined hαwf := by
          rw [← Finset.sum_div]
    _ = 1 := by
          rw [hreindex, div_self (ne_of_gt hR)]

/-- Internal randomized output index law for Algorithm 7.13 under an explicit
normalization proof.

Candidate audit: chose `PMF.ofFintypeOfReal` from `SOptLib/Model/Iterates.lean`;
it is exactly the finite-PMF construction from normalized real weights needed
for the paper output law `Prob{R=k}=α_k/Σα_j`; the required denominator
positivity is an explicit paper-domain proof parameter, avoiding any
Lean-totalized division-by-zero convention in the paper-facing law. -/
noncomputable def outputPMFOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) : PMF setup.OutputTime :=
  SOptLib.normalizedFiniteWindowPMF (Finset.Icc 1 setup.N) (setup.αOfWellDefined hαwf)
    (fun k _hk => le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hαwf k))
    (by simpa [alphaSumOfWellDefined] using hR)

/-- The PMF mass at an output time is the paper mass `α_k / Σα_j`. -/
theorem outputPMFOfWellDefined_apply
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) (R : setup.OutputTime) :
    setup.outputPMFOfWellDefined hαwf hR R =
      ENNReal.ofReal (setup.outputMassOfWellDefined hαwf R) := by
  simp [outputPMFOfWellDefined, outputMassOfWellDefined, outputWeightsOfWellDefined,
    alphaSumOfWellDefined]

/-- The stochastic output PMF has the paper mass `α_k / Σα_j` on singleton
output times. -/
theorem outputPMFOfWellDefined_singleton_real
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) (R : setup.OutputTime) :
    (setup.outputPMFOfWellDefined hαwf hR).toMeasure.real ({R} : Set setup.OutputTime) =
      setup.outputMassOfWellDefined hαwf R := by
  simpa [outputPMFOfWellDefined, SOptLib.normalizedFiniteWindowPMF,
    outputMassOfWellDefined, outputWeightsOfWellDefined, alphaSumOfWellDefined] using
    (SOptLib.ofFintypeOfReal_toMeasure_real_singleton (setup.outputMassOfWellDefined hαwf)
      (fun R => setup.outputMassOfWellDefined_nonneg R hαwf)
      (setup.outputMass_sum_one_of_wellDefined hαwf hR) R)

/-- Conditional randomized output index law from Algorithm 7.13:
`Prob{R=k}=α_k/Σ_{j=1}^N α_j`, under the realization boundary that makes
Eq. (7.4.15)'s denominator and the output normalizer genuine paper quantities.

Candidate audit: chose `PMF.ofFintypeOfReal` through the local
`outputPMFOfWellDefined`; SOptLib's finite-output helpers supply normalization
machinery but not the SNCCG Eq. (7.4.15) stepsize and `{1,...,N}` output window
as a single conditional realization object. -/
noncomputable def algorithmOutputPMFConditional
    (h : setup.Algorithm713RealizationContract) : PMF setup.OutputTime :=
  setup.outputPMFOfWellDefined
    (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h)
    (setup.alphaSum_pos_of_algorithm713Contract h)

/-- The conditional output PMF agrees with any proof-specific construction of
Algorithm 7.13's normalized output law inside the realization boundary. -/
theorem algorithmOutputPMFConditional_eq_of_wellDefined
    (h : setup.Algorithm713RealizationContract) :
    setup.algorithmOutputPMFConditional h =
      setup.outputPMFOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h)
        (setup.alphaSum_pos_of_algorithm713Contract h) := by
  rfl

/-- Internal joint law under an explicit output-normalization proof. -/
noncomputable def outputLawOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) :
    Measure (setup.OutputTime × Ω) :=
  (setup.outputPMFOfWellDefined hαwf hR).toMeasure.prod setup.P

/-- Conditional joint law of the Algorithm 7.13 output index and sample stream. -/
noncomputable def algorithmOutputLawConditional
    (h : setup.Algorithm713RealizationContract) : Measure (setup.OutputTime × Ω) :=
  (setup.algorithmOutputPMFConditional h).toMeasure.prod setup.P

/-- Randomized Algorithm 7.13 output `x_R` generated by the domain-aware paper
process. -/
noncomputable def randomOutputOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (q : setup.OutputTime × Ω) : E :=
  setup.iterProcessOfWellDefined hαwf q.1.1 q.2

/-- Conditional randomized Algorithm 7.13 output `x_R`. -/
noncomputable def algorithmRandomOutputConditional
    (h : setup.Algorithm713RealizationContract) (q : setup.OutputTime × Ω) : E :=
  setup.algorithmIterProcessConditional h q.1.1 q.2

/-- Wolfe gap evaluated at the domain-aware randomized output `x_R`. -/
noncomputable def randomOutputWolfeGapOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (q : setup.OutputTime × Ω) : ℝ :=
  SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.randomOutputOfWellDefined hαwf q)

/-- Wolfe gap evaluated at the conditional randomized output `x_R`. -/
noncomputable def algorithmRandomOutputWolfeGapConditional
    (h : setup.Algorithm713RealizationContract) (q : setup.OutputTime × Ω) : ℝ :=
  SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.algorithmRandomOutputConditional h q)

/-- Expected Wolfe gap under the normalized Algorithm 7.13 output law. This is
proof-parameterized because the paper mass formula is meaningful only once
`Σ_{k=1}^N α_k > 0` is supplied. -/
noncomputable def expectedWolfeGapOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) : ℝ :=
  ∫ q, setup.randomOutputWolfeGapOfWellDefined hαwf q ∂
    setup.outputLawOfWellDefined hαwf hR

/-- Conditional expected Wolfe gap under Algorithm 7.13's normalized output law. -/
noncomputable def algorithmExpectedWolfeGapConditional
    (h : setup.Algorithm713RealizationContract) : ℝ :=
  ∫ q, setup.algorithmRandomOutputWolfeGapConditional h q ∂
    setup.algorithmOutputLawConditional h

/-- Degenerate output law for `D̄_X = 0`: the randomized output index is Dirac
at the first paper output time, and the iterate process is constantly `x₁`. -/
noncomputable def degenerateOutputPMF : PMF setup.OutputTime :=
  PMF.pure setup.firstOutputTime

/-- Non-paper extension output PMF. It agrees with the normalized Algorithm
7.13 law when the paper stepsize formula is well-defined and otherwise uses the
documented Dirac convention for auxiliary zero-diameter lemmas. -/
noncomputable def algorithmOutputPMFExtension : PMF setup.OutputTime :=
  by
    classical
    exact
      if hDX : setup.paperAlphaFormulaWellDefined then
        setup.outputPMFOfWellDefined hDX
          (setup.alphaSum_pos_of_nonzeroDiameter hDX)
      else
        setup.degenerateOutputPMF

/-- The extension output PMF agrees with Algorithm 7.13's normalized law
`Prob{R=k}=α_k/Σ_j α_j` in the nonzero-diameter paper domain. -/
theorem algorithmOutputPMFExtension_eq_of_wellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    setup.algorithmOutputPMFExtension =
      setup.outputPMFOfWellDefined hαwf
        (setup.alphaSum_pos_of_nonzeroDiameter hαwf) := by
  simp [algorithmOutputPMFExtension, hαwf]

/-- Non-paper extension joint law of output index and sample stream. -/
noncomputable def algorithmOutputLawExtension : Measure (setup.OutputTime × Ω) :=
  setup.algorithmOutputPMFExtension.toMeasure.prod setup.P

/-- Non-paper extension randomized output paired with the extension run boundary. -/
noncomputable def randomOutputExtension (q : setup.OutputTime × Ω) : E :=
  setup.algorithmIterProcessExtension q.1.1 q.2

/-- Wolfe gap evaluated at the canonical randomized output. -/
noncomputable def randomOutputWolfeGapExtension (q : setup.OutputTime × Ω) : ℝ :=
  SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.randomOutputExtension q)

/-- Non-paper extension expected Wolfe gap: normalized Algorithm 7.13 output in
the nonzero-diameter branch, and the documented Dirac zero-diameter convention
otherwise. -/
noncomputable def expectedWolfeGapExtension : ℝ :=
  ∫ q, setup.randomOutputWolfeGapExtension q ∂setup.algorithmOutputLawExtension

/-- The extension expected gap agrees with the normalized paper-output
expectation in the nonzero-diameter paper domain. -/
theorem expectedWolfeGapExtension_eq_of_wellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    setup.expectedWolfeGapExtension =
      setup.expectedWolfeGapOfWellDefined hαwf
        (setup.alphaSum_pos_of_nonzeroDiameter hαwf) := by
  simpa [expectedWolfeGapExtension, expectedWolfeGapOfWellDefined,
    algorithmOutputLawExtension, outputLawOfWellDefined] using
    (SOptLib.expected_selected_output_certificate_eq_of_law_and_certificate_eq
      (μ := setup.P)
      (p_ext := setup.algorithmOutputPMFExtension)
      (p_ref :=
        setup.outputPMFOfWellDefined hαwf
          (setup.alphaSum_pos_of_nonzeroDiameter hαwf))
      (cert_ext := setup.randomOutputWolfeGapExtension)
      (cert_ref := setup.randomOutputWolfeGapOfWellDefined hαwf)
      (h_law := setup.algorithmOutputPMFExtension_eq_of_wellDefined hαwf)
      (h_cert := by
        intro q
        simp [randomOutputWolfeGapExtension, randomOutputExtension,
          algorithmIterProcessExtension, algorithmProcessExtension, hαwf,
          randomOutputWolfeGapOfWellDefined, randomOutputOfWellDefined,
          iterProcessOfWellDefined]))

/-- Raw displayed expectation corresponding to Algorithm 7.13's
randomized output law, written as the finite weighted sum
`(Σα_k)⁻¹ Σ α_k E[gap(x_k)]` using the literal displayed stepsize and process.
The conditional PMF realization is kept in `paperExpectedWolfeGapConditional`;
this declaration does not assert the non-source normalizer positivity fact.

Candidate audit: SOptLib `expectedOutput_eq_weighted_sum_div` was considered
for expanding finite randomized outputs; this local definition records the
paper's exact `{1,...,N}` weighted displayed expression because the source JSON
does not provide the positivity proof needed to construct an unconditional PMF. -/
noncomputable def paperExpectedWolfeGapDisplayedExpression : ℝ :=
  SOptLib.normalizedWeightedExpectedCertificate
    (Finset.Icc 1 setup.N) setup.alphaSumDisplayedExpression
    setup.algorithmAlphaDisplayedExpression setup.P
    (fun k ω =>
      SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
        (setup.algorithmIterProcessDisplayedExpression k ω))

/-- Conditional PMF realization of the paper expected Wolfe gap under the
single non-source Algorithm 7.13 realization contract. -/
noncomputable def paperExpectedWolfeGapConditional
    (h : setup.Algorithm713RealizationContract) : ℝ :=
  setup.algorithmExpectedWolfeGapConditional h

/-- In the nonzero-diameter branch, the conditional expected gap is the
normalized Algorithm 7.13 output expectation. -/
theorem paperExpectedWolfeGapConditional_eq_of_wellDefined
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h =
      setup.expectedWolfeGapOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h)
        (setup.alphaSum_pos_of_algorithm713Contract h) := by
  simp [paperExpectedWolfeGapConditional, algorithmExpectedWolfeGapConditional,
    expectedWolfeGapOfWellDefined, algorithmOutputLawConditional,
    algorithmOutputPMFConditional, outputLawOfWellDefined,
    algorithmRandomOutputWolfeGapConditional, algorithmRandomOutputConditional,
    randomOutputWolfeGapOfWellDefined, randomOutputOfWellDefined,
    algorithmIterProcessConditional, algorithmProcessConditional,
    iterProcessOfWellDefined]

/-- Extension joint law for the zero-diameter Dirac convention. -/
noncomputable def degenerateOutputLaw : Measure (setup.OutputTime × Ω) :=
  setup.degenerateOutputPMF.toMeasure.prod setup.P

/-- Extension randomized output for the zero-diameter Dirac convention. -/
noncomputable def degenerateRandomOutput (_q : setup.OutputTime × Ω) : E :=
  setup.x₁

/-- Extension Wolfe-gap expectation for the zero-diameter Dirac convention. -/
noncomputable def degenerateExpectedWolfeGap : ℝ :=
  ∫ q, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.degenerateRandomOutput q) ∂setup.degenerateOutputLaw

/-- In the degenerate branch convention, the feasible set has zero diameter, so
the Dirac output at `x₁` has zero Wolfe gap. -/
theorem degenerateExpectedWolfeGap_eq_zero_of_diameter_eq_zero
    (hDX : setup.barDX = 0) :
    setup.degenerateExpectedWolfeGap = 0 := by
  have hgap :
      SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer setup.x₁ = 0 :=
    SOptLib.ConditionalGradient.wolfeGap_eq_zero_of_diameter_eq_zero
      setup.gradf setup.wolfeGapMaximizer setup.x₁ setup.barDX setup.barDX_bound
      hDX setup.hx₁_mem
  rw [StochasticNonconvexConditionalGradientSetup.degenerateExpectedWolfeGap]
  simp only [StochasticNonconvexConditionalGradientSetup.degenerateRandomOutput, hgap]
  simp

/-- In the non-paper degenerate extension convention, the expected gap is zero:
`D̄_X = 0` forces every feasible point to coincide with `x₁`, so the Dirac
output has zero Wolfe gap. -/
theorem expectedWolfeGapExtension_eq_zero_of_diameter_eq_zero
    (hDX : setup.barDX = 0) :
    setup.expectedWolfeGapExtension = 0 := by
  classical
  have hnot : ¬ setup.paperAlphaFormulaWellDefined := by
    simpa [StochasticNonconvexConditionalGradientSetup.paperAlphaFormulaWellDefined, hDX]
  have hPMF : setup.algorithmOutputPMFExtension = setup.degenerateOutputPMF := by
    simp [StochasticNonconvexConditionalGradientSetup.algorithmOutputPMFExtension, hnot]
  have hProc : setup.algorithmProcessExtension = setup.degenerateProcess := by
    funext k ω
    simp [StochasticNonconvexConditionalGradientSetup.algorithmProcessExtension, hnot]
  have hdeg := setup.degenerateExpectedWolfeGap_eq_zero_of_diameter_eq_zero hDX
  simpa [StochasticNonconvexConditionalGradientSetup.expectedWolfeGapExtension,
    StochasticNonconvexConditionalGradientSetup.algorithmOutputLawExtension,
    StochasticNonconvexConditionalGradientSetup.randomOutputWolfeGapExtension,
    StochasticNonconvexConditionalGradientSetup.randomOutputExtension,
    StochasticNonconvexConditionalGradientSetup.algorithmIterProcessExtension,
    StochasticNonconvexConditionalGradientSetup.degenerateExpectedWolfeGap,
    StochasticNonconvexConditionalGradientSetup.degenerateOutputLaw,
    StochasticNonconvexConditionalGradientSetup.degenerateRandomOutput,
    StochasticNonconvexConditionalGradientSetup.degenerateProcess,
    hPMF, hProc] using hdeg

/-- Explicit-normalizer variant of the randomized-output expectation expansion,
kept as an internal proof bridge. -/
theorem expectedWolfeGapOfWellDefined_eq_weighted_sum
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf)
    (hgap_int :
      ∀ R : setup.OutputTime,
        Integrable
          (fun ω => SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hαwf R.1 ω))
          setup.P) :
    setup.expectedWolfeGapOfWellDefined hαwf hR =
      (setup.alphaSumOfWellDefined hαwf)⁻¹ *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k =>
            setup.αOfWellDefined hαwf k *
              ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hαwf k ω) ∂setup.P) := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  simpa [StochasticNonconvexConditionalGradientSetup.expectedWolfeGapOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.randomOutputWolfeGapOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.randomOutputOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.outputLawOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.outputPMFOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
    using
    SOptLib.finiteWindowSelectedOutputExpectation_eq_weighted_sum
        (times := Finset.Icc 1 setup.N) (α := setup.αOfWellDefined hαwf) (P := setup.P)
        (x := setup.iterProcessOfWellDefined hαwf)
        (gap := SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer)
        (hα_nonneg := fun k _hk =>
          le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hαwf k))
        (hden := by
          simpa [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined] using hR)
        (hgap_int := hgap_int)

/-- The domain-aware output law exposes the positive denominator used by the
paper output law.

Algorithm 7.13 normalizes by `Σ_{k=1}^N α_k`, while Eq. (7.4.15) contains the
denominator `L D̄_X^2`. The output-law objects therefore keep their denominator
proof visible rather than specializing it through an uncited bridge. -/
theorem outputLaw_keeps_domain_boundary
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hR : 0 < setup.alphaSumOfWellDefined hαwf) :
    0 < setup.alphaSumOfWellDefined hαwf :=
  hR

/-- Raw compatibility estimator second-moment accumulation from iterate
differences in epoch s and step t. The paper-facing stochastic bound uses
`epochDiffSumOfWellDefined`. -/
noncomputable def rawEpochDiffSum (s t : ℕ) : Ω → ℝ :=
  fun ω =>
    Finset.sum (Finset.Icc 2 t)
      (fun i =>
        ‖setup.rawIterProcess (setup.globalIndex s i) ω -
          setup.rawIterProcess (setup.globalIndex s (i - 1)) ω‖ ^ 2)

/-- Domain-aware estimator second-moment accumulation from iterate differences
in epoch s and step t, built from the Algorithm 7.13 process using the genuine
Eq. (7.4.15) stepsize. -/
noncomputable def epochDiffSumOfWellDefined
    (hαwf : setup.paperAlphaFormulaWellDefined) (s t : ℕ) : Ω → ℝ :=
  fun ω =>
    Finset.sum (Finset.Icc 2 t)
      (fun i =>
        ‖setup.iterProcessOfWellDefined hαwf (setup.globalIndex s i) ω -
          setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (i - 1)) ω‖ ^ 2)

/-- Estimator second-moment accumulation from iterate differences inside an
epoch, built from the conditional Algorithm 7.13 run boundary. -/
noncomputable def epochDiffSumConditional
    (h : setup.Algorithm713RealizationContract)
    (s t : ℕ) : Ω → ℝ :=
  fun ω =>
    Finset.sum (Finset.Icc 2 t)
      (fun i =>
        ‖setup.algorithmIterProcessConditional h (setup.globalIndex s i) ω -
          setup.algorithmIterProcessConditional h (setup.globalIndex s (i - 1)) ω‖ ^ 2)

/-- In the nondegenerate branch, the canonical epoch-difference accumulation
agrees with the Eq. (7.4.15) process-specific version. -/
theorem epochDiffSumConditional_eq_of_wellDefined
    (h : setup.Algorithm713RealizationContract) (s t : ℕ) :
    setup.epochDiffSumConditional h s t =
      setup.epochDiffSumOfWellDefined
        (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) s t := by
  simpa [epochDiffSumConditional, epochDiffSumOfWellDefined,
    SOptLib.epochSquaredDifferenceSum] using
    (SOptLib.epochSuccessiveDiffSqSum_eq_of_process_eq
      (H := setup.Algorithm713RealizationContract)
      (H' := setup.paperAlphaFormulaWellDefined)
      (x := fun h k ω => setup.algorithmIterProcessConditional h k ω)
      (y := fun hαwf k ω => setup.iterProcessOfWellDefined hαwf k ω)
      (index := setup.globalIndex)
      (h := h)
      (h' := setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h)
      (s := s) (t := t)
      (by
        intro k ω
        simp [algorithmIterProcessConditional, algorithmProcessConditional,
          iterProcessOfWellDefined]))

lemma measurable_xi_of_lt
    {n i : ℕ} (hi : i < n) :
    Measurable[setup.filtration n] (setup.ξ i) := by
  exact SOptLib.measurable_sample_of_lt_prefixFiltration setup.ξ setup.hξ_meas hi

lemma sample_indep_of_filtration
    {n i : ℕ} (hni : n ≤ i) :
    Indep (setup.filtration n)
      (MeasurableSpace.comap (setup.ξ i) ‹MeasurableSpace Ξ›) setup.P := by
  exact ProbabilityTheory.iIndepFun.indep_prefixFiltration_future
    setup.ξ setup.hξ_meas setup.hξ_indep hni

lemma indepFun_of_measurable_filtration
    {β : Type*} [MeasurableSpace β]
    {wt : Ω → β} {n i : ℕ}
    (hwt : Measurable[setup.filtration n] wt)
    (hni : n ≤ i) :
    IndepFun wt (setup.ξ i) setup.P := by
  exact ProbabilityTheory.iIndepFun.indepFun_prefixMeasurable_future
    setup.ξ setup.hξ_meas setup.hξ_indep hwt hni

lemma rawProcess_succ_s_eq_div
    (k : ℕ) :
    ∀ ω, (setup.rawProcess (k + 1) ω).s = k / setup.T := by
  exact
    SOptLib.epochCounter_succ_eq_div_of_divisibility_update
      (counter := fun state => state.s)
      (process := setup.rawProcess)
      (T := setup.T)
      (by
        intro ω
        simp [StochasticNonconvexConditionalGradientSetup.rawProcess])
      (by
        intro k ω hdiv
        have hmod : (k + 1) % setup.T = 0 := Nat.mod_eq_zero_of_dvd hdiv
        simp [StochasticNonconvexConditionalGradientSetup.rawProcess, hmod])
      (by
        intro k ω hdiv
        have hmod : (k + 1) % setup.T ≠ 0 := by
          intro hmod
          exact hdiv (Nat.dvd_of_mod_eq_zero hmod)
        simp [StochasticNonconvexConditionalGradientSetup.rawProcess, hmod])
      k

lemma rawProcess_s_eq_globalIndex
    (s t : ℕ) (ht : 1 ≤ t) (ht_le : t ≤ setup.T) :
    ∀ ω, (setup.rawProcess (setup.globalIndex s t) ω).s = s := by
  intro ω
  have hsucc :
      (setup.globalIndex s t - 1) + 1 = setup.globalIndex s t := by
    unfold StochasticNonconvexConditionalGradientSetup.globalIndex
    omega
  have hdiv :
      (setup.globalIndex s t - 1) / setup.T = s := by
    refine Nat.div_eq_of_lt_le ?_ ?_
    · unfold StochasticNonconvexConditionalGradientSetup.globalIndex
      omega
    · have hdecomp :
          setup.globalIndex s t - 1 = s * setup.T + (t - 1) := by
        unfold StochasticNonconvexConditionalGradientSetup.globalIndex
        omega
      have hlt : t - 1 < setup.T := by
        omega
      rw [hdecomp]
      rw [show (s + 1) * setup.T = s * setup.T + setup.T by ring]
      exact Nat.add_lt_add_left hlt (s * setup.T)
  simpa [hsucc, hdiv] using
    (setup.rawProcess_succ_s_eq_div (k := setup.globalIndex s t - 1) ω)

/-- The domain-aware Algorithm 7.13 process has the same epoch-counter
arithmetic as the raw displayed process.

Candidate audit: considered target-file `rawProcess_succ_s_eq_div`,
`rawProcess_s_eq_globalIndex`, and SOptLib process-view helpers from
`SOptLib/Model/Iterates.lean`; the raw lemmas prove the same arithmetic for a
different process, while SOptLib has no Algorithm 7.13 state counter. This
helper is the well-defined-process analogue needed for the refresh formula. -/
lemma processOfWellDefined_succ_s_eq_div
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (k : ℕ) :
    ∀ ω, (setup.processOfWellDefined hαwf (k + 1) ω).s = k / setup.T := by
  intro ω
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hdiv : setup.T ∣ k + 1
      · have hmod : (k + 1) % setup.T = 0 := Nat.mod_eq_zero_of_dvd hdiv
        simp [hmod, ih,
          Nat.succ_div_of_dvd hdiv]
      · have hmod : (k + 1) % setup.T ≠ 0 := by
          intro hmod
          exact hdiv (Nat.dvd_of_mod_eq_zero hmod)
        simp [hmod, ih,
          Nat.succ_div_of_not_dvd hdiv]

/-- At a paper global index `(s,t)`, the domain-aware process state carries
epoch counter `s`.

Candidate audit: considered target-file `rawProcess_s_eq_globalIndex` and
SOptLib `iterateProcessView_eq`; the former is for the raw process only, and
the latter has no state-counter arithmetic. This specializes Algorithm 7.13's
literal global-index convention. -/
lemma processOfWellDefined_s_eq_globalIndex
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s t : ℕ) (ht : 1 ≤ t) (ht_le : t ≤ setup.T) :
    ∀ ω, (setup.processOfWellDefined hαwf (setup.globalIndex s t) ω).s = s := by
  intro ω
  have hsucc :
      (setup.globalIndex s t - 1) + 1 = setup.globalIndex s t := by
    unfold StochasticNonconvexConditionalGradientSetup.globalIndex
    omega
  have hdiv :
      (setup.globalIndex s t - 1) / setup.T = s := by
    refine Nat.div_eq_of_lt_le ?_ ?_
    · unfold StochasticNonconvexConditionalGradientSetup.globalIndex
      omega
    · have hdecomp :
          setup.globalIndex s t - 1 = s * setup.T + (t - 1) := by
        unfold StochasticNonconvexConditionalGradientSetup.globalIndex
        omega
      have hlt : t - 1 < setup.T := by
        omega
      rw [hdecomp]
      rw [show (s + 1) * setup.T = s * setup.T + setup.T by ring]
      exact Nat.add_lt_add_left hlt (s * setup.T)
  simpa [hsucc, hdiv] using
    (setup.processOfWellDefined_succ_s_eq_div hαwf
      (k := setup.globalIndex s t - 1) ω)

/-- At the first step of epoch `s`, Algorithm 7.13's well-defined estimator is
the fresh epoch mini-batch average indexed by `s*m+i`.

Candidate audit: considered SOptLib mini-batch average/residual algebra
(`miniBatchAverage_sub_target_eq_average_residual`) and target-file raw process
counter lemmas. The SOptLib algebra starts after the estimator has already been
identified, and the raw lemmas are for the displayed raw process; this helper
supplies the paper-local Algorithm 7.13 index identification. -/
lemma estimatorProcessOfWellDefined_globalIndex_one_eq_refresh_average
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s : ℕ) :
    setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s 1) =
      fun ω =>
        ((setup.m : ℝ)⁻¹) •
          Finset.sum (Finset.range setup.m)
            (fun i =>
              setup.gradF
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω)
                (setup.ξ (s * setup.m + i) ω)) := by
  simpa [-SOptLib.varianceReducedConditionalGradientProcess_def,
    StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.globalIndex] using
    (SOptLib.estimator_at_epoch_start_eq_refresh_minibatch
      (fun x G s => ({ x := x, G := G, s := s } : StochasticNonconvexCGState E))
      (fun state : StochasticNonconvexCGState E => state.x)
      (fun state : StochasticNonconvexCGState E => state.G)
      (fun state : StochasticNonconvexCGState E => state.s)
      (by intro x G s; rfl)
      (by intro x G s; rfl)
      (by intro x G s; rfl)
      setup.x₁ setup.m setup.b setup.T setup.N setup.ξ setup.gradF
      (setup.iterUpdateOfWellDefined hαwf) setup.hT_pos s)

/-- Epoch-start estimator error is the centered refresh mini-batch average.

This aligns with Algorithm 7.13's refresh formula and Lan Lemma 7.5's
`σ²/m` base term. Candidate audit: `miniBatchAverage_sub_target_eq_average_residual`
is the matching SOptLib centering algebra after the paper-local estimator
identity is known; `oracleResidualAtRandomQuery` is too abstract because it
does not encode the Algorithm 7.13 refresh indices `s*m+i`. -/
lemma refresh_delta_eq_centered_minibatch_average
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s : ℕ) :
    setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s 1) =
      fun ω =>
        ((setup.m : ℝ)⁻¹) •
          Finset.sum (Finset.range setup.m)
            (fun i =>
              setup.gradF
                  (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω)
                  (setup.ξ (s * setup.m + i) ω) -
                setup.gradf
                  (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω)) := by
  funext ω
  have hG :=
    congrFun
      (setup.estimatorProcessOfWellDefined_globalIndex_one_eq_refresh_average
        hαwf s) ω
  have hcenter :=
    miniBatchAverage_sub_target_eq_average_residual
      (I := Finset.range setup.m) (m := setup.m) (by simp)
      (lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos)
      (a := fun i =>
        setup.gradF
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω)
          (setup.ξ (s * setup.m + i) ω))
      (target :=
        setup.gradf
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω))
  simp [StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.delta, hG, hcenter]

/-- Distinct samples inside the epoch-refresh block are independent.

This is the coordinate-level off-diagonal independence fact used before
pairing a peer sample with the epoch-start query. Candidate audit: checked
SOptLib `iIndepFun.indepFun_finset_subtype_blocks`,
`indep_sampleBlock_singleton_of_disjoint_indices`, and
`indep_strictPast_sup_coordinate_of_disjoint_current`; the finite-block
`iIndepFun.indepFun_finset` specialization is the exact coordinate proof for
the two singleton refresh indices. -/
lemma refresh_samples_indep_distinct
    (s i j : ℕ) (hij : i ≠ j) :
    IndepFun (setup.ξ (s * setup.m + j)) (setup.ξ (s * setup.m + i)) setup.P := by
  classical
  let idxJ : ℕ := s * setup.m + j
  let idxI : ℕ := s * setup.m + i
  have hidx_ne : idxJ ≠ idxI := by
    intro h
    dsimp [idxJ, idxI] at h
    omega
  let Jset : Finset ℕ := {idxJ}
  let Iset : Finset ℕ := {idxI}
  have hdisj : Disjoint Jset Iset := by
    rw [Finset.disjoint_left]
    intro x hx hy
    have hx' : x = idxJ := by simpa [Jset] using hx
    have hy' : x = idxI := by simpa [Iset] using hy
    exact hidx_ne (hx'.symm.trans hy')
  have hvec :
      IndepFun
        (fun ω => fun q : {q // q ∈ Jset} => setup.ξ q.1 ω)
        (fun ω => fun q : {q // q ∈ Iset} => setup.ξ q.1 ω)
        setup.P :=
    setup.hξ_indep.indepFun_finset Jset Iset hdisj setup.hξ_meas
  let projJ : ({q // q ∈ Jset} → Ξ) → Ξ :=
    fun y => y ⟨idxJ, by simp [Jset]⟩
  let projI : ({q // q ∈ Iset} → Ξ) → Ξ :=
    fun y => y ⟨idxI, by simp [Iset]⟩
  have hprojJ : Measurable projJ := by
    exact measurable_pi_apply (⟨idxJ, by simp [Jset]⟩ : {q // q ∈ Jset})
  have hprojI : Measurable projI := by
    exact measurable_pi_apply (⟨idxI, by simp [Iset]⟩ : {q // q ∈ Iset})
  simpa [projJ, projI, idxJ, idxI, Jset, Iset] using hvec.comp hprojJ hprojI

/-- A finite block of sample coordinates is independent of any fresh sample
coordinate outside the block.

Candidate audit: `indep_sampleBlock_singleton_of_disjoint_indices` is the
matching SOptLib bridge from vector-valued block independence to generated
sigma-algebras; this helper specializes it to the Algorithm 7.13 sample stream
and the setup's i.i.d. hypothesis. -/
lemma sampleBlock_indep_singleton_of_not_mem
    (block : Finset ℕ) {j : ℕ} (hj : j ∉ block) :
    Indep (SOptLib.sampleBlockMeasurableSpace setup.ξ block)
      (MeasurableSpace.comap (setup.ξ j) (by infer_instance : MeasurableSpace Ξ))
      setup.P := by
  exact iIndepFun.indep_sampleBlock_singleton_of_not_mem
    setup.ξ setup.P block setup.hξ_meas setup.hξ_indep hj

/-- The current epoch-refresh coordinates are outside the finite sample
footprint of the epoch-start query.

Candidate audit: searched SOptLib for finite sample-block freshness and checked
`sampleBlockMeasurableSpace`, `indep_sampleBlock_singleton_of_disjoint_indices`,
and target-file cutoff lemmas. Those give sigma-algebra independence once a
block is known disjoint from the coordinate; this helper supplies the
Algorithm 7.13 arithmetic disjointness using `globalIndex s 1 ≤ N` and
`m = T*b`. -/
lemma epoch_refresh_index_not_mem_epoch_start_footprint
    (s i : ℕ) (hi : i ∈ Finset.range setup.m)
    (hsN : setup.globalIndex s 1 ≤ setup.N) :
    s * setup.m + i ∉
      (Finset.range (s * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex s 1 * setup.b)) := by
  simpa [StochasticNonconvexConditionalGradientSetup.globalIndex, SOptLib.global_index] using
    refresh_index_not_mem_epoch_start_footprint_of_global_index_le
      (T := setup.T) (N := setup.N) (m := setup.m) (b := setup.b)
      (s := s) (i := i) setup.hT_pos hi hsN

/-- The finite sample footprint of the epoch-start query is independent of any
current refresh sample in the same epoch block.

Candidate audit: this specializes the proved local
`sampleBlock_indep_singleton_of_not_mem`, which in turn specializes SOptLib's
finite-block independence bridge. The paper-local content is the Algorithm 7.13
footprint arithmetic in `epoch_refresh_index_not_mem_epoch_start_footprint`. -/
lemma epoch_start_footprint_indep_refresh_sample
    (s i : ℕ) (hi : i ∈ Finset.range setup.m)
    (hsN : setup.globalIndex s 1 ≤ setup.N) :
    Indep
      (SOptLib.sampleBlockMeasurableSpace setup.ξ
        (Finset.range (s * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex s 1 * setup.b)))
      (MeasurableSpace.comap (setup.ξ (s * setup.m + i))
        (by infer_instance : MeasurableSpace Ξ))
      setup.P := by
  exact setup.sampleBlock_indep_singleton_of_not_mem
    (Finset.range (s * setup.m) ∪
      Finset.Ico (setup.N * setup.m)
        (setup.N * setup.m + setup.globalIndex s 1 * setup.b))
    (setup.epoch_refresh_index_not_mem_epoch_start_footprint s i hi hsN)

/-- The epoch-start finite footprint enlarged by a distinct peer refresh sample
is still independent of the current refresh sample.

Candidate audit: this uses the same SOptLib finite-block independence bridge as
`epoch_start_footprint_indep_refresh_sample`; the extra paper-local obligation
is only the singleton peer/current disjointness inside the refresh block. -/
lemma epoch_start_footprint_with_peer_indep_refresh_sample
    (s i j : ℕ) (hi : i ∈ Finset.range setup.m) (_hj : j ∈ Finset.range setup.m)
    (hij : i ≠ j) (hsN : setup.globalIndex s 1 ≤ setup.N) :
    Indep
      (SOptLib.sampleBlockMeasurableSpace setup.ξ
        (insert (s * setup.m + j)
          (Finset.range (s * setup.m) ∪
            Finset.Ico (setup.N * setup.m)
              (setup.N * setup.m + setup.globalIndex s 1 * setup.b))))
      (MeasurableSpace.comap (setup.ξ (s * setup.m + i))
        (by infer_instance : MeasurableSpace Ξ))
      setup.P := by
  have hnot_footprint :
      s * setup.m + i ∉
        (Finset.range (s * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex s 1 * setup.b)) :=
    setup.epoch_refresh_index_not_mem_epoch_start_footprint s i hi hsN
  have hidx_ne : s * setup.m + i ≠ s * setup.m + j := by
    omega
  exact setup.sampleBlock_indep_singleton_of_not_mem
    (insert (s * setup.m + j)
      (Finset.range (s * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex s 1 * setup.b)))
    (by
      rw [Finset.mem_insert]
      intro hmem
      exact hmem.elim (fun h => hidx_ne h) (fun h => hnot_footprint h))

/-- Fixed feasible oracle residual variance transports from the base sample
`ξ 0` to any i.i.d. stream coordinate.

Candidate audit: `randomIterate_variance_bound_of_fixed_variance` and
`integral_sq_oracleResidual_le_of_indep_fixed_variance` handle random-query
transfers once independence is available; this fixed-query helper is the
smaller law-transport step needed to feed those SOptLib lemmas from the setup's
`hξ_ident` and `hgradF_variance_bound`. -/
lemma fixed_sample_residual_variance_bound
    (idx : ℕ) (x : E) (hx : x ∈ setup.X) :
    Integrable
        (fun ω => ‖setup.gradF x (setup.ξ idx ω) - setup.gradf x‖ ^ 2) setup.P ∧
      ∫ ω, ‖setup.gradF x (setup.ξ idx ω) - setup.gradf x‖ ^ 2 ∂setup.P ≤
        setup.σ ^ 2 := by
  exact
    fixed_oracle_residual_variance_bound_of_identDistrib
      (P := setup.P) (G := setup.gradF) (g := setup.gradf) (xi := setup.ξ)
      (sigma2 := setup.σ ^ 2) (idx := idx) (x := x)
      (hres_meas := by
        exact
          ((setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ξ => x).prodMk measurable_id)).sub
            (measurable_const : Measurable fun _ : Ξ => setup.gradf x)).norm.pow_const 2)
      (hxi_ident := setup.hξ_ident idx)
      (hbase := setup.hgradF_variance_bound x hx)

/-- Finite sample-block sigma-algebras are monotone in the coordinate block.

Candidate audit: `SOptLib.sampleBlockMeasurableSpace` is the matching finite
generated sigma-algebra primitive; searched `sample block measurable monotone`
and found no exported monotonicity theorem, so this local helper unfolds the
definition and includes each old generator in the larger block. -/
lemma sampleBlockMeasurableSpace_mono
    {A B : Finset ℕ} (hAB : A ⊆ B) :
    SOptLib.sampleBlockMeasurableSpace setup.ξ A ≤
      SOptLib.sampleBlockMeasurableSpace setup.ξ B := by
  exact SOptLib.sampleBlockMeasurableSpace_mono setup.ξ hAB

/-- A finite sample-block sigma-algebra generated by the measurable sample
stream is a sub-sigma-algebra of the ambient measurable space.

Candidate audit: searched SOptLib for sample-block ambient monotonicity; the
available primitive is the definition `sampleBlockMeasurableSpace`, and this
helper unfolds it and applies the setup's coordinate measurability
`hξ_meas`. -/
lemma sampleBlockMeasurableSpace_le
    {block : Finset ℕ} :
    SOptLib.sampleBlockMeasurableSpace setup.ξ block ≤
      (by infer_instance : MeasurableSpace Ω) := by
  exact SOptLib.sampleBlockMeasurableSpace_le setup.ξ block (fun i _ => setup.hξ_meas i)

/-- A sample coordinate listed in a finite block is measurable for that block.

Candidate audit: this is exactly SOptLib
`sampleBlock_coordinate_measurable`, specialized to the identity index map used
by Algorithm 7.13's flattened sample stream. -/
lemma xi_measurable_sampleBlock
    {block : Finset ℕ} {q : ℕ} (hq : q ∈ block) :
    Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block] (setup.ξ q) := by
  simpa [SOptLib.sampleBlockMeasurableSpace] using
    (SOptLib.sampleBlock_coordinate_measurable
      (ξ := setup.ξ) (idx := fun n : ℕ => n) (block := block) hq)

/-- The well-defined Algorithm 7.13 LMO update preserves measurability over any
fixed source sigma-algebra.

Candidate audit: target-file `rawIterUpdate_measurable` proves the same
measurability for the raw compatibility update, while SOptLib process-update
lemmas are abstract; this specializes the same LMO-measurability argument to
the paper-domain `iterUpdateOfWellDefined`. -/
lemma iterUpdateOfWellDefined_measurable_sampleBlock
    (hαwf : setup.paperAlphaFormulaWellDefined)
    {block : Finset ℕ} {k : ℕ} {x G : Ω → E}
    (hx : Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block] x)
    (hG : Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block] G) :
    Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block]
      (fun ω => setup.iterUpdateOfWellDefined hαwf (x ω) (G ω) k) := by
  simpa [iterUpdateOfWellDefined] using
    SOptLib.conditionalGradientUpdate_measurable_of_lmo
      (mΩ := SOptLib.sampleBlockMeasurableSpace setup.ξ block)
      (alpha := setup.αOfWellDefined hαwf)
      (linearMinimizer := setup.linearMinimizer)
      setup.linearMinimizer_measurable hx hG

/-- A finite sum of oracle gradients is measurable over a finite sample block
when the query and every used sample coordinate are block-measurable.

Candidate audit: SOptLib `oracleValue_measurable_of_query_sample_measurable`
and `miniBatchOracle_measurable_of_coordinate_measurable` cover the generic
composition/sum pattern; this local helper specializes them to the concrete
`gradF` kernel and Algorithm 7.13's flattened coordinate membership proof. -/
lemma grad_sum_measurable_sampleBlock
    {block : Finset ℕ} {n : ℕ} {x : Ω → E}
    (hx : Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block] x)
    {idx : ℕ → ℕ}
    (hidx : ∀ i, i < n → idx i ∈ block) :
    Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block]
      (fun ω =>
        Finset.sum (Finset.range n)
          (fun i => setup.gradF (x ω) (setup.ξ (idx i) ω))) := by
  refine Finset.measurable_sum _ ?_
  intro i hi
  have hxi :
      Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block]
        (setup.ξ (idx i)) :=
    setup.xi_measurable_sampleBlock (hidx i (Finset.mem_range.mp hi))
  exact setup.hgradF_meas.comp (hx.prodMk hxi)

/-- A finite sum of paired oracle-gradient differences is measurable over a
finite sample block when both queries and all used sample coordinates are
block-measurable.

Candidate audit: this is the sample-block analogue of the existing
`grad_diff_sum_measurable_of_lt`; SOptLib generic oracle measurability lemmas
handle the kernel composition but not this Algorithm 7.13 difference schedule
directly. -/
lemma grad_diff_sum_measurable_sampleBlock
    {block : Finset ℕ} {n : ℕ} {x y : Ω → E}
    (hx : Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block] x)
    (hy : Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block] y)
    {idx : ℕ → ℕ}
    (hidx : ∀ i, i < n → idx i ∈ block) :
    Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block]
      (fun ω =>
        Finset.sum (Finset.range n)
          (fun i =>
            setup.gradF (x ω) (setup.ξ (idx i) ω) -
              setup.gradF (y ω) (setup.ξ (idx i) ω))) := by
  exact SOptLib.oracle_value_sub_sum_measurable_of_coordinate_measurable
    (G := setup.gradF) (hG := setup.hgradF_meas) (ξ := setup.ξ)
    (block := block) (terms := Finset.range n) (x := x) (y := y) (idx := idx)
    hx hy (fun i hi => hidx i (Finset.mem_range.mp hi))

/-- The epoch-start footprint grows from epoch `s` to epoch `s+1`.

Candidate audit: SOptLib supplies finite-block monotonicity once a set inclusion
is known, but this Algorithm 7.13 helper proves the concrete index arithmetic
for the refresh-prefix plus recursive-sample footprint. -/
lemma epoch_start_footprint_subset_next
    (s : ℕ) :
    (Finset.range (s * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex s 1 * setup.b)) ⊆
      (Finset.range ((s + 1) * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex (s + 1) 1 * setup.b)) := by
  exact epoch_start_sample_footprint_subset_succ (N := setup.N) (m := setup.m)
    (b := setup.b) (globalIndex := setup.globalIndex) s
    (by
      simpa [StochasticNonconvexConditionalGradientSetup.globalIndex] using
        Nat.add_le_add_right
          (Nat.mul_le_mul_right setup.T (Nat.le_succ s)) 1)

/-- A completed epoch's refresh mini-batch belongs to the next epoch-start
finite footprint.

Candidate audit: no SOptLib match is expected for this Algorithm 7.13 flattened
index arithmetic; it is the refresh-prefix membership needed by the
epoch-start sample-block measurability induction. -/
lemma refresh_index_mem_next_epoch_start_footprint
    {s i : ℕ} (hi : i < setup.m) :
    s * setup.m + i ∈
      (Finset.range ((s + 1) * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex (s + 1) 1 * setup.b)) := by
  rw [Finset.mem_union]
  left
  exact Finset.mem_range.mpr (by
    have hlt : s * setup.m + i < s * setup.m + setup.m := by omega
    have hsucc : s * setup.m + setup.m = (s + 1) * setup.m := by ring
    simpa [hsucc] using hlt)

/-- Recursive mini-batch coordinates strictly before the next epoch start
belong to that next epoch-start finite footprint.

Candidate audit: this is a paper-local flattened-index bridge; SOptLib provides
the generated sigma-algebra machinery, but not Algorithm 7.13's `N*m+k*b+i`
coordinate arithmetic. -/
lemma recursive_index_mem_next_epoch_start_footprint
    {s k i : ℕ} (hk : k < setup.globalIndex (s + 1) 1) (hi : i < setup.b) :
    setup.N * setup.m + k * setup.b + i ∈
      (Finset.range ((s + 1) * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex (s + 1) 1 * setup.b)) := by
  rw [Finset.mem_union]
  right
  exact recursive_index_mem_Ico_of_lt_cutoff
    (N := setup.N) (m := setup.m) (b := setup.b)
    (cutoff := setup.globalIndex (s + 1) 1) (k := k) (i := i) hk hi

/-- Any earlier epoch-refresh mini-batch coordinate belongs to a later
epoch-start finite footprint.

Candidate audit: SOptLib has the finite sample-block primitive
`sampleBlock_coordinate_measurable`, while the existing local
`refresh_index_mem_next_epoch_start_footprint` only covers the immediately
next epoch. This helper is the Algorithm 7.13 flattened-index arithmetic needed
for strict-prefix adaptedness. -/
lemma refresh_index_mem_epoch_start_footprint_of_lt
    {s r i : ℕ} (hr : r < s) (hi : i < setup.m) :
    r * setup.m + i ∈
      (Finset.range (s * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex s 1 * setup.b)) := by
  exact refresh_index_mem_later_epoch_start_sample_footprint_of_epoch_lt
    (tail := Finset.Ico (setup.N * setup.m)
      (setup.N * setup.m + setup.globalIndex s 1 * setup.b)) hr hi

/-- Recursive mini-batch coordinates strictly before an epoch start belong to
that epoch-start finite footprint.

Candidate audit: this specializes the already-proved local
`recursive_index_mem_next_epoch_start_footprint`; SOptLib supplies only the
sample-block measurability machinery, not this Algorithm 7.13 index schedule. -/
lemma recursive_index_mem_epoch_start_footprint_of_lt
    {s k i : ℕ} (hk : k < setup.globalIndex s 1) (hi : i < setup.b) :
    setup.N * setup.m + k * setup.b + i ∈
      (Finset.range (s * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex s 1 * setup.b)) := by
  exact Finset.mem_union.mpr (Or.inr
    (recursive_index_mem_Ico_of_lt_cutoff
      (N := setup.N) (m := setup.m) (b := setup.b)
      (cutoff := setup.globalIndex s 1) (k := k) (i := i) hk hi))

/-- The initial epoch-start iterate `x_(0,1)` is constant, hence measurable for
the finite footprint that excludes the first refresh mini-batch.

Candidate audit: this is the base case of the requested
`iterProcess_epoch_start_measurable_sampleBlock` bridge; SOptLib recursive
process measurability lemmas are unnecessary for the constant initial slice. -/
lemma iterProcess_epoch_start_zero_measurable_sampleBlock
    (hαwf : setup.paperAlphaFormulaWellDefined) :
    Measurable[
      SOptLib.sampleBlockMeasurableSpace setup.ξ
        (Finset.range (0 * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex 0 1 * setup.b))]
      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex 0 1)) := by
  convert
    (measurable_const : Measurable[
      SOptLib.sampleBlockMeasurableSpace setup.ξ
        (Finset.range (0 * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex 0 1 * setup.b))]
      (fun _ : Ω => setup.x₁)) using 1
  funext ω
  simp [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.globalIndex]

/-- Before an epoch-start boundary, the full well-defined state is measurable
with respect to that boundary's finite sample footprint.

Candidate audit: checked SOptLib `process_prefix_measurable_wrt_sampleBlock`
and `recursive_process_measurable_of_measurable_update`; they provide the
generic induction scheme but do not encode Algorithm 7.13's two-branch sample
schedule. This helper supplies the paper-local refresh/recursive coordinate
membership while reusing the local sample-block measurability lemmas. -/
lemma processOfWellDefined_pair_measurable_before_epoch_start
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s k : ℕ) (hk : k < setup.globalIndex s 1) :
    Measurable[
      SOptLib.sampleBlockMeasurableSpace setup.ξ
        (Finset.range (s * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex s 1 * setup.b))]
      (fun ω =>
        (setup.iterProcessOfWellDefined hαwf k ω,
          setup.estimatorProcessOfWellDefined hαwf k ω)) := by
  classical
  simpa [-SOptLib.varianceReducedConditionalGradientProcess_def,
    StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined] using
    (SOptLib.recursive_refresh_process_pair_measurable_before_epoch_start
      (mkState := fun x G s => ({ x := x, G := G, s := s } : StochasticNonconvexCGState E))
      (xOf := fun state : StochasticNonconvexCGState E => state.x)
      (estimatorOf := fun state : StochasticNonconvexCGState E => state.G)
      (epochOf := fun state : StochasticNonconvexCGState E => state.s)
      (x0 := setup.x₁) (m := setup.m) (b := setup.b) (T := setup.T) (N := setup.N)
      (ξ := setup.ξ) (gradF := setup.gradF)
      (update := setup.iterUpdateOfWellDefined hαwf)
      (boundary := fun s => setup.globalIndex s 1)
      (footprint := fun s =>
        Finset.range (s * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex s 1 * setup.b))
      setup.hgradF_meas
      (by
        intro s n x G hx hG
        exact setup.iterUpdateOfWellDefined_measurable_sampleBlock hαwf
          (block :=
            Finset.range (s * setup.m) ∪
              Finset.Ico (setup.N * setup.m)
                (setup.N * setup.m + setup.globalIndex s 1 * setup.b))
          (k := n) hx hG)
      (by intros; rfl)
      (by intros; rfl)
      (by
        intro s n ω _ hstart
        have hdiv : setup.T ∣ n + 1 := Nat.dvd_of_mod_eq_zero hstart
        have hs := setup.processOfWellDefined_succ_s_eq_div hαwf n ω
        simpa [-SOptLib.varianceReducedConditionalGradientProcess_def,
          StochasticNonconvexConditionalGradientSetup.processOfWellDefined,
          Nat.succ_div_of_dvd hdiv] using congrArg Nat.succ hs)
      (by
        intro s i hs_boundary hi
        have hspos : 0 < s := by
          cases s with
          | zero =>
              simp [StochasticNonconvexConditionalGradientSetup.globalIndex] at hs_boundary
          | succ r => omega
        simpa using
          setup.refresh_index_mem_epoch_start_footprint_of_lt
            (s := s) (r := 0) (i := i) hspos hi)
      (by
        intro s n i hk_boundary _ hi
        have hidx_epoch : (n + 1) / setup.T < s := by
          have hlt : n + 1 < s * setup.T := by
            simp [StochasticNonconvexConditionalGradientSetup.globalIndex] at hk_boundary
            omega
          exact Nat.div_lt_of_lt_mul (by
            simpa [Nat.mul_comm] using hlt)
        simpa using
          setup.refresh_index_mem_epoch_start_footprint_of_lt
            (s := s) (r := (n + 1) / setup.T) (i := i) hidx_epoch hi)
      (by
        intro s n i hprev_lt hi
        simpa using
          setup.recursive_index_mem_epoch_start_footprint_of_lt
            (s := s) (k := n + 1) (i := i) hprev_lt hi)
      s k hk)

/-- The epoch-start iterate is measurable with respect to the finite sample
footprint that contains all earlier refresh and recursive mini-batches but
excludes the current refresh mini-batch.

Candidate audit: the existing local base case
`iterProcess_epoch_start_zero_measurable_sampleBlock` handles `s = 0`, and
SOptLib `process_prefix_measurable_wrt_sampleBlock` gives only an abstract
prefix induction. The proved local
`processOfWellDefined_pair_measurable_before_epoch_start` supplies the
Algorithm 7.13 strict-prefix state measurability needed for the final update. -/
lemma iterProcess_epoch_start_measurable_sampleBlock
    (hαwf : setup.paperAlphaFormulaWellDefined) (s : ℕ) :
    Measurable[
      SOptLib.sampleBlockMeasurableSpace setup.ξ
        (Finset.range (s * setup.m) ∪
          Finset.Ico (setup.N * setup.m)
            (setup.N * setup.m + setup.globalIndex s 1 * setup.b))]
      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1)) := by
  classical
  exact SOptLib.epoch_start_iterate_measurable_strict_sample_footprint
    (ξ := setup.ξ)
    (x := setup.iterProcessOfWellDefined hαwf)
    (G := setup.estimatorProcessOfWellDefined hαwf)
    (update := setup.iterUpdateOfWellDefined hαwf)
    (footprint := fun s =>
      Finset.range (s * setup.m) ∪
        Finset.Ico (setup.N * setup.m)
          (setup.N * setup.m + setup.globalIndex s 1 * setup.b))
    (boundary := fun s => setup.globalIndex s 1)
    (prevBoundary := fun s => s * setup.T)
    (hbase := setup.iterProcess_epoch_start_zero_measurable_sampleBlock hαwf)
    (hprev_pair := by
      intro r
      have hprev_lt : (r + 1) * setup.T < setup.globalIndex (r + 1) 1 := by
        unfold StochasticNonconvexConditionalGradientSetup.globalIndex
        omega
      simpa using
        setup.processOfWellDefined_pair_measurable_before_epoch_start
          hαwf (r + 1) ((r + 1) * setup.T) hprev_lt)
    (hupdate := by
      intro r hx_prev hG_prev
      exact setup.iterUpdateOfWellDefined_measurable_sampleBlock hαwf
        (block :=
          Finset.range ((r + 1) * setup.m) ∪
            Finset.Ico (setup.N * setup.m)
              (setup.N * setup.m + setup.globalIndex (r + 1) 1 * setup.b))
        hx_prev hG_prev)
    (hboundary_update := by
      intro r ω
      have hTpos : 0 < setup.T := lt_of_lt_of_le Nat.zero_lt_one setup.hT_pos
      have hprev_pos : 1 ≤ (r + 1) * setup.T := by
        exact Nat.succ_le_of_lt (Nat.mul_pos (Nat.succ_pos r) hTpos)
      have hprev :
          ((r + 1) * setup.T - 1) + 1 = (r + 1) * setup.T :=
        Nat.sub_add_cancel hprev_pos
      have hidx :
          setup.globalIndex (r + 1) 1 = ((r + 1) * setup.T - 1) + 2 := by
        unfold StochasticNonconvexConditionalGradientSetup.globalIndex
        omega
      have hmod : (((r + 1) * setup.T - 1) + 1) % setup.T = 0 := by
        rw [hprev]
        exact Nat.mod_eq_zero_of_dvd (Nat.dvd_mul_left setup.T (r + 1))
      change
        (setup.processOfWellDefined hαwf (setup.globalIndex (r + 1) 1) ω).x =
          setup.iterUpdateOfWellDefined hαwf
            (setup.iterProcessOfWellDefined hαwf ((r + 1) * setup.T) ω)
            (setup.estimatorProcessOfWellDefined hαwf ((r + 1) * setup.T) ω)
            ((r + 1) * setup.T)
      simp [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
        StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
        hidx, hprev, hmod])
    s

/-- Block-measurability of the epoch-start ambient iterate gives independence
of the feasible query from each fresh refresh sample.

Candidate audit: this is exactly SOptLib
`indepFun_of_block_measurable_and_fresh_sample` applied to the Algorithm 7.13
epoch-start footprint; the remaining paper-local input is the separate
measurability bridge for `x_(s,1)`. -/
lemma epoch_start_query_indep_refresh_sample_of_measurable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s i : ℕ) (hi : i ∈ Finset.range setup.m)
    (hsN : setup.globalIndex s 1 ≤ setup.N)
    (hx :
      ∀ ω, setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω ∈ setup.X)
    (hXblock :
      Measurable[
        SOptLib.sampleBlockMeasurableSpace setup.ξ
          (Finset.range (s * setup.m) ∪
            Finset.Ico (setup.N * setup.m)
              (setup.N * setup.m + setup.globalIndex s 1 * setup.b))]
        (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1))) :
    IndepFun
      (fun ω => (⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
        hx ω⟩ : setup.X))
      (setup.ξ (s * setup.m + i)) setup.P := by
  let block : Finset ℕ :=
    Finset.range (s * setup.m) ∪
      Finset.Ico (setup.N * setup.m)
        (setup.N * setup.m + setup.globalIndex s 1 * setup.b)
  have hquery :
      Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ block]
        (fun ω => (⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
          hx ω⟩ : setup.X)) := by
    exact hXblock.subtype_mk
  exact indepFun_of_block_measurable_and_fresh_sample
    setup.P (SOptLib.sampleBlockMeasurableSpace setup.ξ block)
    (fun ω => (⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
      hx ω⟩ : setup.X))
    (setup.ξ (s * setup.m + i)) hquery
    (by
      simpa [block] using
        setup.epoch_start_footprint_indep_refresh_sample s i hi hsN)

/-- Block-measurability of the epoch-start iterate also gives independence of
the query paired with a distinct peer refresh sample from the current refresh
sample.

Candidate audit: this combines SOptLib
`indepFun_of_block_measurable_and_fresh_sample`,
`sampleBlock_coordinate_measurable`, and the local finite-block monotonicity
helper. It is the exact off-diagonal independence bridge requested for Lemma
7.5's covariance cancellation. -/
lemma epoch_start_query_peer_indep_refresh_sample_of_measurable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s i j : ℕ) (hi : i ∈ Finset.range setup.m) (hj : j ∈ Finset.range setup.m)
    (hij : i ≠ j) (hsN : setup.globalIndex s 1 ≤ setup.N)
    (hx :
      ∀ ω, setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω ∈ setup.X)
    (hXblock :
      Measurable[
        SOptLib.sampleBlockMeasurableSpace setup.ξ
          (Finset.range (s * setup.m) ∪
            Finset.Ico (setup.N * setup.m)
              (setup.N * setup.m + setup.globalIndex s 1 * setup.b))]
        (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1))) :
    IndepFun
      (fun ω =>
        ((⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
          hx ω⟩ : setup.X), setup.ξ (s * setup.m + j) ω))
      (setup.ξ (s * setup.m + i)) setup.P := by
  let block : Finset ℕ :=
    Finset.range (s * setup.m) ∪
      Finset.Ico (setup.N * setup.m)
        (setup.N * setup.m + setup.globalIndex s 1 * setup.b)
  let peerBlock : Finset ℕ := insert (s * setup.m + j) block
  exact indepFun_prod_of_measurable_le_of_indep_comap
    (X := fun ω => (⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
      hx ω⟩ : setup.X))
    (Y := setup.ξ (s * setup.m + j)) (Z := setup.ξ (s * setup.m + i))
    (m := SOptLib.sampleBlockMeasurableSpace setup.ξ block)
    (m' := SOptLib.sampleBlockMeasurableSpace setup.ξ peerBlock)
    (by exact hXblock.subtype_mk)
    (by
      exact setup.sampleBlockMeasurableSpace_mono (by
        intro q hq
        exact Finset.mem_insert_of_mem hq))
    (by
      exact setup.xi_measurable_sampleBlock (by
        dsimp [peerBlock]
        simp))
    (by
      simpa [block, peerBlock] using
        setup.epoch_start_footprint_with_peer_indep_refresh_sample
          s i j hi hj hij hsN)

lemma recursive_batch_index_lt_next_cutoff
    {k i : ℕ} (hi : i < setup.b) :
    setup.N * setup.m + k * setup.b + i <
      setup.N * setup.m + (k + 1) * setup.b := by
  nlinarith

lemma refresh_index_lt_recursive_cutoff
    {r s t i : ℕ}
    (hr : r ≤ s)
    (ht : 2 ≤ t)
    (hi : i < setup.m) :
    r * setup.m + i <
      setup.N * setup.m +
        setup.globalIndex s (t - 1) * setup.b := by
  simpa [StochasticNonconvexConditionalGradientSetup.globalIndex, SOptLib.global_index]
    using refresh_index_lt_epoch_recursive_cutoff
      (N := setup.N) (m := setup.m) (T := setup.T) (b := setup.b)
      (r := r) (s := s) (t := t) (i := i)
      hr hi setup.hN_pos setup.hm_le_T_mul_b

lemma grad_sum_measurable_of_lt
    {cutoff n : ℕ} {x : Ω → E}
    (hx : Measurable[setup.filtration cutoff] x)
    {idx : ℕ → ℕ}
    (hidx : ∀ i, i < n → idx i < cutoff) :
    Measurable[setup.filtration cutoff]
      (fun ω =>
        Finset.sum (Finset.range n)
          (fun i => setup.gradF (x ω) (setup.ξ (idx i) ω))) := by
  exact SOptLib.measurable_finset_sum_kernel_comp_of_coordinate_measurable
    (mΩ := setup.filtration cutoff) (I := Finset.range n)
    (K := fun p : E × Ξ => setup.gradF p.1 p.2) setup.hgradF_meas hx
    setup.ξ idx (by
      intro i hi
      exact setup.measurable_xi_of_lt (hidx i (Finset.mem_range.mp hi)))

lemma grad_diff_sum_measurable_of_lt
    {cutoff n : ℕ} {x y : Ω → E}
    (hx : Measurable[setup.filtration cutoff] x)
    (hy : Measurable[setup.filtration cutoff] y)
    {idx : ℕ → ℕ}
    (hidx : ∀ i, i < n → idx i < cutoff) :
    Measurable[setup.filtration cutoff]
      (fun ω =>
        Finset.sum (Finset.range n)
          (fun i =>
            setup.gradF (x ω) (setup.ξ (idx i) ω) -
              setup.gradF (y ω) (setup.ξ (idx i) ω))) := by
  exact SOptLib.measurable_finset_sum_kernel_diff_comp_of_coordinate_measurable
    (mΩ := setup.filtration cutoff) (I := Finset.range n)
    (K := fun p : E × Ξ => setup.gradF p.1 p.2) setup.hgradF_meas hx hy
    setup.ξ idx (by
      intro i hi
      exact setup.measurable_xi_of_lt (hidx i (Finset.mem_range.mp hi)))

lemma rawIterUpdate_measurable
    {cutoff k : ℕ} {x G : Ω → E}
    (hx : Measurable[setup.filtration cutoff] x)
    (hG : Measurable[setup.filtration cutoff] G) :
    Measurable[setup.filtration cutoff]
      (fun ω => setup.rawIterUpdate (x ω) (G ω) k) := by
  have hy : Measurable[setup.filtration cutoff]
      (fun ω => setup.linearMinimizer (G ω)) :=
    setup.linearMinimizer_measurable.comp hG
  simpa [rawIterUpdate] using
    ((hx.const_smul (1 - setup.rawPaperAlpha k)).add
      (hy.const_smul (setup.rawPaperAlpha k)))

/-- The well-defined Algorithm 7.13 LMO update is measurable over a filtration
cutoff whenever its iterate and estimator inputs are measurable there.

Candidate audit: considered SOptLib recursive-process measurability helpers
`recursive_process_measurable_of_measurable_update` and
`recursive_process_measurable_of_measurable_update_oracle`, but their abstract
state-update framework does not expose Algorithm 7.13's concrete sample cutoff.
This is the filtration analogue of the existing local sample-block helper
`iterUpdateOfWellDefined_measurable_sampleBlock`. -/
lemma iterUpdateOfWellDefined_measurable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    {cutoff k : ℕ} {x G : Ω → E}
    (hx : Measurable[setup.filtration cutoff] x)
    (hG : Measurable[setup.filtration cutoff] G) :
    Measurable[setup.filtration cutoff]
      (fun ω => setup.iterUpdateOfWellDefined hαwf (x ω) (G ω) k) := by
  have hy : Measurable[setup.filtration cutoff]
      (fun ω => setup.linearMinimizer (G ω)) :=
    setup.linearMinimizer_measurable.comp hG
  simpa [iterUpdateOfWellDefined] using
    ((hx.const_smul (1 - setup.αOfWellDefined hαwf k)).add
      (hy.const_smul (setup.αOfWellDefined hαwf k)))

/-- The state available before the `k`-th recursive mini-batch is measurable
with respect to the filtration cutoff that stops just before that batch begins.
This is the adaptedness fact unlocked by `hm_le_T_mul_b`: refresh indices
`r * m + i` stay below the recursive cutoff `N * m + k * b`, and all earlier
recursive indices are below it by construction. -/
lemma rawProcess_pair_measurable_recursive_cutoff
    (k : ℕ) :
    Measurable[setup.filtration (setup.N * setup.m + k * setup.b)]
      (fun ω => (setup.rawIterProcess k ω, setup.rawEstimatorProcess k ω)) := by
  induction k with
  | zero =>
      simpa [StochasticNonconvexConditionalGradientSetup.rawIterProcess,
        StochasticNonconvexConditionalGradientSetup.rawEstimatorProcess,
        StochasticNonconvexConditionalGradientSetup.rawProcess] using
        (measurable_const : Measurable fun _ : Ω => (setup.x₁, (0 : E)))
  | succ k ih =>
      cases k with
      | zero =>
          have hx : Measurable[setup.filtration (setup.N * setup.m + (0 + 1) * setup.b)]
              (fun _ : Ω => setup.x₁) :=
            measurable_const
          have hGsum :
              Measurable[setup.filtration (setup.N * setup.m + (0 + 1) * setup.b)]
                (fun ω =>
                  Finset.sum (Finset.range setup.m)
                    (fun i => setup.gradF setup.x₁ (setup.ξ i ω))) :=
            setup.grad_sum_measurable_of_lt
              (cutoff := setup.N * setup.m + (0 + 1) * setup.b) (n := setup.m) hx
              (idx := fun i => i) (by
                intro i hi
                have hNm : setup.m ≤ setup.N * setup.m := by
                  nlinarith [setup.hN_pos, setup.hm_pos]
                have hi' : i < setup.N * setup.m + setup.b := by
                  exact lt_of_lt_of_le
                    hi (le_trans hNm (Nat.le_add_right (setup.N * setup.m) setup.b))
                simpa using hi')
          have hG : Measurable[setup.filtration (setup.N * setup.m + (0 + 1) * setup.b)]
              (fun ω =>
                (setup.m : ℝ)⁻¹ •
                  Finset.sum (Finset.range setup.m)
                    (fun i => setup.gradF setup.x₁ (setup.ξ i ω))) :=
            hGsum.const_smul ((setup.m : ℝ)⁻¹)
          simpa [StochasticNonconvexConditionalGradientSetup.rawIterProcess,
            StochasticNonconvexConditionalGradientSetup.rawEstimatorProcess,
            StochasticNonconvexConditionalGradientSetup.rawProcess] using
            hx.prodMk hG
      | succ n =>
          let cutoffPrev := setup.N * setup.m + (n + 1) * setup.b
          let cutoff := setup.N * setup.m + (n + 2) * setup.b
          have hcutoff : cutoffPrev ≤ cutoff := by
            dsimp [cutoffPrev, cutoff]
            have hmul : (n + 1) * setup.b ≤ (n + 2) * setup.b := by
              exact Nat.mul_le_mul_right setup.b (Nat.le_succ (n + 1))
            exact Nat.add_le_add_left hmul (setup.N * setup.m)
          have hpair_prev :
              Measurable[setup.filtration cutoff]
                (fun ω => (setup.rawIterProcess (n + 1) ω, setup.rawEstimatorProcess (n + 1) ω)) :=
            ih.mono (setup.filtration.mono hcutoff) le_rfl
          have hx_prev : Measurable[setup.filtration cutoff] (setup.rawIterProcess (n + 1)) :=
            measurable_fst.comp hpair_prev
          have hG_prev :
              Measurable[setup.filtration cutoff] (setup.rawEstimatorProcess (n + 1)) :=
            measurable_snd.comp hpair_prev
          have hx_next :
              Measurable[setup.filtration cutoff]
                (fun ω =>
                  setup.rawIterUpdate
                    (setup.rawIterProcess (n + 1) ω)
                    (setup.rawEstimatorProcess (n + 1) ω) (n + 1)) :=
            setup.rawIterUpdate_measurable (cutoff := cutoff) hx_prev hG_prev
          by_cases hstart : (n + 1) % setup.T = 0
          · have hdiv : setup.T ∣ n + 1 := Nat.dvd_of_mod_eq_zero hstart
            have hs_div : n / setup.T + 1 = (n + 1) / setup.T := by
              symm
              exact Nat.succ_div_of_dvd hdiv
            have hGsum :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    Finset.sum (Finset.range setup.m)
                      (fun i =>
                        setup.gradF
                          (setup.rawIterUpdate
                            (setup.rawIterProcess (n + 1) ω)
                            (setup.rawEstimatorProcess (n + 1) ω) (n + 1))
                          (setup.ξ (((n + 1) / setup.T) * setup.m + i) ω))) :=
              setup.grad_sum_measurable_of_lt (cutoff := cutoff) (n := setup.m) hx_next
                (idx := fun i => ((n + 1) / setup.T) * setup.m + i) (by
                  intro i hi
                  have hidx_le :
                      ((n + 1) / setup.T) * setup.m ≤ ((n + 1) / setup.T) * (setup.T * setup.b) := by
                    exact Nat.mul_le_mul_left ((n + 1) / setup.T) setup.hm_le_T_mul_b
                  have hdiv_le : ((n + 1) / setup.T) * setup.T ≤ n + 1 := Nat.div_mul_le_self _ _
                  have hqTb :
                      ((n + 1) / setup.T) * (setup.T * setup.b) ≤ (n + 1) * setup.b := by
                    simpa [Nat.mul_assoc] using Nat.mul_le_mul_right setup.b hdiv_le
                  have hqM : ((n + 1) / setup.T) * setup.m ≤ (n + 1) * setup.b := by
                    exact le_trans hidx_le hqTb
                  have hNm : setup.m ≤ setup.N * setup.m := by
                    nlinarith [setup.hN_pos, setup.hm_pos]
                  have hiNm : i < setup.N * setup.m := lt_of_lt_of_le hi hNm
                  have hsum :
                      ((n + 1) / setup.T) * setup.m + i <
                        (n + 1) * setup.b + setup.N * setup.m :=
                    Nat.add_lt_add_of_le_of_lt hqM hiNm
                  have hmul : (n + 1) * setup.b ≤ (n + 2) * setup.b := by
                    exact Nat.mul_le_mul_right setup.b (Nat.le_succ (n + 1))
                  have hfinal :
                      (n + 1) * setup.b + setup.N * setup.m ≤
                        setup.N * setup.m + (n + 2) * setup.b := by
                    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
                      Nat.add_le_add_right hmul (setup.N * setup.m)
                  dsimp [cutoff]
                  exact lt_of_lt_of_le hsum hfinal)
            have hG :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    (setup.m : ℝ)⁻¹ •
                      Finset.sum (Finset.range setup.m)
                        (fun i =>
                          setup.gradF
                            (setup.rawIterUpdate
                              (setup.rawIterProcess (n + 1) ω)
                              (setup.rawEstimatorProcess (n + 1) ω) (n + 1))
                            (setup.ξ (((n + 1) / setup.T) * setup.m + i) ω))) :=
              hGsum.const_smul ((setup.m : ℝ)⁻¹)
            simpa [cutoff, StochasticNonconvexConditionalGradientSetup.rawIterProcess,
              StochasticNonconvexConditionalGradientSetup.rawEstimatorProcess,
              StochasticNonconvexConditionalGradientSetup.rawProcess, hstart,
              setup.rawProcess_succ_s_eq_div, hs_div] using
              hx_next.prodMk hG
          · have hGdiff :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    Finset.sum (Finset.range setup.b)
                      (fun i =>
                        setup.gradF
                            (setup.rawIterUpdate
                              (setup.rawIterProcess (n + 1) ω)
                              (setup.rawEstimatorProcess (n + 1) ω) (n + 1))
                            (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω) -
                          setup.gradF (setup.rawIterProcess (n + 1) ω)
                            (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω))) :=
              setup.grad_diff_sum_measurable_of_lt (cutoff := cutoff) (n := setup.b)
                hx_next hx_prev
                (idx := fun i => setup.N * setup.m + (n + 1) * setup.b + i) (by
                  intro i hi
                  simpa [cutoff] using
                    (setup.recursive_batch_index_lt_next_cutoff (k := n + 1) hi))
            have hG :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    (setup.b : ℝ)⁻¹ •
                        Finset.sum (Finset.range setup.b)
                          (fun i =>
                            setup.gradF
                                (setup.rawIterUpdate
                                  (setup.rawIterProcess (n + 1) ω)
                                  (setup.rawEstimatorProcess (n + 1) ω) (n + 1))
                                (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω) -
                              setup.gradF (setup.rawIterProcess (n + 1) ω)
                                (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω)) +
                      setup.rawEstimatorProcess (n + 1) ω) := by
              simpa using (hGdiff.const_smul ((setup.b : ℝ)⁻¹)).add hG_prev
            simpa [cutoff, StochasticNonconvexConditionalGradientSetup.rawIterProcess,
              StochasticNonconvexConditionalGradientSetup.rawEstimatorProcess,
              StochasticNonconvexConditionalGradientSetup.rawProcess, hstart] using
              hx_next.prodMk hG

/-- The well-defined Algorithm 7.13 state available before the `k`-th recursive
mini-batch is measurable with respect to the filtration cutoff stopping just
before that batch begins.

Candidate audit: considered SOptLib `recursive_process_measurable_of_measurable_update`
and `recursive_process_measurable_of_measurable_update_oracle`, plus target-file
`rawProcess_pair_measurable_recursive_cutoff`. The SOptLib helpers do not encode
the two-branch refresh/recursive sample schedule, and the raw helper uses the
totalized displayed stepsize. This is the domain-aware adaptedness bridge needed
for Lan Lemma 7.5's fresh recursive mini-batch estimates. -/
lemma processOfWellDefined_pair_measurable_recursive_cutoff
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (k : ℕ) :
    Measurable[setup.filtration (setup.N * setup.m + k * setup.b)]
      (fun ω =>
        (setup.iterProcessOfWellDefined hαwf k ω,
          setup.estimatorProcessOfWellDefined hαwf k ω)) := by
  induction k with
  | zero =>
      simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
        StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined] using
        (measurable_const : Measurable fun _ : Ω => (setup.x₁, (0 : E)))
  | succ k ih =>
      cases k with
      | zero =>
          have hx : Measurable[setup.filtration (setup.N * setup.m + (0 + 1) * setup.b)]
              (fun _ : Ω => setup.x₁) :=
            measurable_const
          have hGsum :
              Measurable[setup.filtration (setup.N * setup.m + (0 + 1) * setup.b)]
                (fun ω =>
                  Finset.sum (Finset.range setup.m)
                    (fun i => setup.gradF setup.x₁ (setup.ξ i ω))) :=
            setup.grad_sum_measurable_of_lt
              (cutoff := setup.N * setup.m + (0 + 1) * setup.b) (n := setup.m) hx
              (idx := fun i => i) (by
                intro i hi
                have hNm : setup.m ≤ setup.N * setup.m := by
                  nlinarith [setup.hN_pos, setup.hm_pos]
                have hi' : i < setup.N * setup.m + setup.b := by
                  exact lt_of_lt_of_le
                    hi (le_trans hNm (Nat.le_add_right (setup.N * setup.m) setup.b))
                simpa using hi')
          have hG : Measurable[setup.filtration (setup.N * setup.m + (0 + 1) * setup.b)]
              (fun ω =>
                (setup.m : ℝ)⁻¹ •
                  Finset.sum (Finset.range setup.m)
                    (fun i => setup.gradF setup.x₁ (setup.ξ i ω))) :=
            hGsum.const_smul ((setup.m : ℝ)⁻¹)
          simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
            StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined] using
            hx.prodMk hG
      | succ n =>
          let cutoffPrev := setup.N * setup.m + (n + 1) * setup.b
          let cutoff := setup.N * setup.m + (n + 2) * setup.b
          have hcutoff : cutoffPrev ≤ cutoff := by
            dsimp [cutoffPrev, cutoff]
            have hmul : (n + 1) * setup.b ≤ (n + 2) * setup.b := by
              exact Nat.mul_le_mul_right setup.b (Nat.le_succ (n + 1))
            exact Nat.add_le_add_left hmul (setup.N * setup.m)
          have hpair_prev :
              Measurable[setup.filtration cutoff]
                (fun ω =>
                  (setup.iterProcessOfWellDefined hαwf (n + 1) ω,
                    setup.estimatorProcessOfWellDefined hαwf (n + 1) ω)) :=
            ih.mono (setup.filtration.mono hcutoff) le_rfl
          have hx_prev :
              Measurable[setup.filtration cutoff]
                (setup.iterProcessOfWellDefined hαwf (n + 1)) :=
            measurable_fst.comp hpair_prev
          have hG_prev :
              Measurable[setup.filtration cutoff]
                (setup.estimatorProcessOfWellDefined hαwf (n + 1)) :=
            measurable_snd.comp hpair_prev
          have hx_next :
              Measurable[setup.filtration cutoff]
                (fun ω =>
                  setup.iterUpdateOfWellDefined hαwf
                    (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                    (setup.estimatorProcessOfWellDefined hαwf (n + 1) ω) (n + 1)) :=
            setup.iterUpdateOfWellDefined_measurable hαwf
              (cutoff := cutoff) hx_prev hG_prev
          by_cases hstart : (n + 1) % setup.T = 0
          · have hdiv : setup.T ∣ n + 1 := Nat.dvd_of_mod_eq_zero hstart
            have hs_div : n / setup.T + 1 = (n + 1) / setup.T := by
              symm
              exact Nat.succ_div_of_dvd hdiv
            have hGsum :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    Finset.sum (Finset.range setup.m)
                      (fun i =>
                        setup.gradF
                          (setup.iterUpdateOfWellDefined hαwf
                            (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                            (setup.estimatorProcessOfWellDefined hαwf (n + 1) ω) (n + 1))
                          (setup.ξ (((n + 1) / setup.T) * setup.m + i) ω))) :=
              setup.grad_sum_measurable_of_lt (cutoff := cutoff) (n := setup.m) hx_next
                (idx := fun i => ((n + 1) / setup.T) * setup.m + i) (by
                  intro i hi
                  have hidx_le :
                      ((n + 1) / setup.T) * setup.m ≤
                        ((n + 1) / setup.T) * (setup.T * setup.b) := by
                    exact Nat.mul_le_mul_left ((n + 1) / setup.T) setup.hm_le_T_mul_b
                  have hdiv_le : ((n + 1) / setup.T) * setup.T ≤ n + 1 :=
                    Nat.div_mul_le_self _ _
                  have hqTb :
                      ((n + 1) / setup.T) * (setup.T * setup.b) ≤
                        (n + 1) * setup.b := by
                    simpa [Nat.mul_assoc] using Nat.mul_le_mul_right setup.b hdiv_le
                  have hqM : ((n + 1) / setup.T) * setup.m ≤ (n + 1) * setup.b := by
                    exact le_trans hidx_le hqTb
                  have hNm : setup.m ≤ setup.N * setup.m := by
                    nlinarith [setup.hN_pos, setup.hm_pos]
                  have hiNm : i < setup.N * setup.m := lt_of_lt_of_le hi hNm
                  have hsum :
                      ((n + 1) / setup.T) * setup.m + i <
                        (n + 1) * setup.b + setup.N * setup.m :=
                    Nat.add_lt_add_of_le_of_lt hqM hiNm
                  have hmul : (n + 1) * setup.b ≤ (n + 2) * setup.b := by
                    exact Nat.mul_le_mul_right setup.b (Nat.le_succ (n + 1))
                  have hfinal :
                      (n + 1) * setup.b + setup.N * setup.m ≤
                        setup.N * setup.m + (n + 2) * setup.b := by
                    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
                      Nat.add_le_add_right hmul (setup.N * setup.m)
                  dsimp [cutoff]
                  exact lt_of_lt_of_le hsum hfinal)
            have hG :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    (setup.m : ℝ)⁻¹ •
                      Finset.sum (Finset.range setup.m)
                        (fun i =>
                          setup.gradF
                            (setup.iterUpdateOfWellDefined hαwf
                              (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                              (setup.estimatorProcessOfWellDefined hαwf (n + 1) ω) (n + 1))
                            (setup.ξ (((n + 1) / setup.T) * setup.m + i) ω))) :=
              hGsum.const_smul ((setup.m : ℝ)⁻¹)
            simpa [cutoff, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
              StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
              hstart,
              setup.processOfWellDefined_succ_s_eq_div hαwf, hs_div] using
              hx_next.prodMk hG
          · have hGdiff :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    Finset.sum (Finset.range setup.b)
                      (fun i =>
                        setup.gradF
                            (setup.iterUpdateOfWellDefined hαwf
                              (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                              (setup.estimatorProcessOfWellDefined hαwf (n + 1) ω) (n + 1))
                            (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω) -
                          setup.gradF (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                            (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω))) :=
              setup.grad_diff_sum_measurable_of_lt (cutoff := cutoff) (n := setup.b)
                hx_next hx_prev
                (idx := fun i => setup.N * setup.m + (n + 1) * setup.b + i) (by
                  intro i hi
                  simpa [cutoff] using
                    (setup.recursive_batch_index_lt_next_cutoff (k := n + 1) hi))
            have hG :
                Measurable[setup.filtration cutoff]
                  (fun ω =>
                    (setup.b : ℝ)⁻¹ •
                        Finset.sum (Finset.range setup.b)
                          (fun i =>
                            setup.gradF
                                (setup.iterUpdateOfWellDefined hαwf
                                  (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                                  (setup.estimatorProcessOfWellDefined hαwf (n + 1) ω) (n + 1))
                                (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω) -
                              setup.gradF (setup.iterProcessOfWellDefined hαwf (n + 1) ω)
                                (setup.ξ (setup.N * setup.m + (n + 1) * setup.b + i) ω)) +
                      setup.estimatorProcessOfWellDefined hαwf (n + 1) ω) := by
              simpa using (hGdiff.const_smul ((setup.b : ℝ)⁻¹)).add hG_prev
            simpa [cutoff, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
              StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
              hstart] using
              hx_next.prodMk hG

/-- The current within-epoch iterate `x_(s,j)` is already measurable before the
fresh recursive mini-batch that updates `G_(s,j)`.

Candidate audit: considered SOptLib recursive adaptedness helpers and the local
`processOfWellDefined_pair_measurable_recursive_cutoff`. The SOptLib lemmas do
not encode Algorithm 7.13's update-before-sampling order, while the pair-cutoff
helper places the full state after the current batch. This lemma extracts the
paper-specific fact needed for Lemma 7.5's martingale increment at steps
`j ≥ 2`. -/
lemma iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T) :
    Measurable[
      setup.filtration
        (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b)]
      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)) := by
  simpa [StochasticNonconvexConditionalGradientSetup.globalIndex, SOptLib.global_index] using
    current_iterate_measurable_before_recursive_batch
      (filt := setup.filtration)
      (T := setup.T) (b := setup.b) (offset := setup.N * setup.m)
      (iter := setup.iterProcessOfWellDefined hαwf)
      (estimator := setup.estimatorProcessOfWellDefined hαwf)
      (update := fun x G k => setup.iterUpdateOfWellDefined hαwf x G k)
      (h_pair := by
        intro k
        exact setup.processOfWellDefined_pair_measurable_recursive_cutoff hαwf k)
      (h_update_meas := by
        intro cutoff n hx hG
        exact setup.iterUpdateOfWellDefined_measurable hαwf (cutoff := cutoff) hx hG)
      (h_iter_update := by
        intro n hmod
        funext ω
        simp [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
          StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
          hmod])
      (s := s) (j := j) hj2 hjT

/-- Every well-defined Algorithm 7.13 iterate is Borel-measurable.

Candidate audit: SOptLib recursive-process measurability helpers are abstract and
do not encode Algorithm 7.13's estimator schedule; the proved local
`processOfWellDefined_pair_measurable_recursive_cutoff` gives this as an
immediate corollary by forgetting the cutoff filtration. -/
lemma iterProcessOfWellDefined_measurable
    (hαwf : setup.paperAlphaFormulaWellDefined) (k : ℕ) :
    Measurable (setup.iterProcessOfWellDefined hαwf k) := by
  have hpair := setup.processOfWellDefined_pair_measurable_recursive_cutoff hαwf k
  exact (measurable_fst.comp hpair).mono
    (setup.filtration.le (setup.N * setup.m + k * setup.b)) le_rfl

/-- Formal falsity witness for the retired unconditional raw feasibility
invariant: if the displayed affine update is used with a coefficient `α > 1`,
then even the interval `X = [0,1]`, current point `1`, and linear-oracle point
`0` produce the next point `1 - α < 0`, outside `X`.

This is not a replacement for Algorithm 7.13. It records why raw feasibility
must consume a visible `α ≤ 1` boundary before the convexity proof below can
apply. -/
private theorem affine_update_Icc_counterexample_of_one_lt_alpha
    {α : ℝ} (hα : 1 < α) :
    (1 - α) • (1 : ℝ) + α • (0 : ℝ) ∉ Set.Icc (0 : ℝ) 1 := by
  intro hmem
  have hleft : 0 ≤ (1 - α) • (1 : ℝ) + α • (0 : ℝ) := hmem.1
  have hle : 0 ≤ 1 - α := by
    simpa using hleft
  linarith

lemma rawIterUpdate_mem_of_alpha_le_one
    {x G : E} {k : ℕ} (hx : x ∈ setup.X)
    (hα_le_one : setup.rawPaperAlpha k ≤ 1) :
    setup.rawIterUpdate x G k ∈ setup.X := by
  let y := setup.linearMinimizer G
  have hy : y ∈ setup.X := setup.linearMinimizer_mem G
  have hα_nonneg : 0 ≤ setup.rawPaperAlpha k := by
    simpa [rawPaperAlpha, rawAlphaFormula] using
      (Real.sqrt_nonneg
        ((1 / (setup.N : ℝ) + setup.σ ^ 2 / (setup.L * setup.m)) /
          (setup.L * setup.barDX ^ 2)))
  have hseg :
      x + setup.rawPaperAlpha k • (y - x) ∈ setup.X :=
    setup.hX_convex.add_smul_sub_mem hx hy
      ⟨hα_nonneg, hα_le_one⟩
  convert hseg using 1
  simp only [rawIterUpdate, y]
  module

lemma rawProcess_mem
    (hα_le_one : ∀ k, setup.rawPaperAlpha k ≤ 1) :
    ∀ k ω, (setup.rawProcess k ω).x ∈ setup.X := by
  intro k
  induction k with
  | zero =>
      intro ω
      simpa [rawProcess] using setup.hx₁_mem
  | succ k ih =>
      cases k with
      | zero =>
          intro ω
          simpa [rawProcess] using setup.hx₁_mem
      | succ n =>
          intro ω
          simpa [rawProcess] using
            setup.rawIterUpdate_mem_of_alpha_le_one
              (x := (setup.rawProcess (n + 1) ω).x)
              (G := (setup.rawProcess (n + 1) ω).G)
              (k := n + 1) (ih ω) (hα_le_one (n + 1))

lemma rawIterProcess_mem
    (hα_le_one : ∀ k, setup.rawPaperAlpha k ≤ 1) :
    ∀ k ω, setup.rawIterProcess k ω ∈ setup.X := by
  intro k ω
  simpa [StochasticNonconvexConditionalGradientSetup.rawIterProcess] using
    setup.rawProcess_mem hα_le_one k ω

lemma iterUpdateOfWellDefined_mem_of_alpha_le_one
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    {x G : E} {k : ℕ} (hx : x ∈ setup.X) :
    setup.iterUpdateOfWellDefined hαwf x G k ∈ setup.X := by
  let y := setup.linearMinimizer G
  have hy : y ∈ setup.X := setup.linearMinimizer_mem G
  have hα_nonneg : 0 ≤ setup.αOfWellDefined hαwf k :=
    le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hαwf k)
  have hα_le : setup.αOfWellDefined hαwf k ≤ 1 :=
    setup.hα_le_one hαwf k hα_le_one
  have hseg :
      x + setup.αOfWellDefined hαwf k • (y - x) ∈ setup.X :=
    setup.hX_convex.add_smul_sub_mem hx hy ⟨hα_nonneg, hα_le⟩
  convert hseg using 1
  simp only [iterUpdateOfWellDefined, y]
  module

lemma processOfWellDefined_mem_of_alpha_le_one
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1) :
    ∀ k ω, (setup.processOfWellDefined hαwf k ω).x ∈ setup.X := by
  intro k
  induction k with
  | zero =>
      intro ω
      simpa using setup.hx₁_mem
  | succ k ih =>
      cases k with
      | zero =>
          intro ω
          simpa using setup.hx₁_mem
      | succ n =>
          intro ω
          simpa using
            setup.iterUpdateOfWellDefined_mem_of_alpha_le_one
              (hαwf := hαwf) (hα_le_one := hα_le_one)
              (x := (setup.processOfWellDefined hαwf (n + 1) ω).x)
              (G := (setup.processOfWellDefined hαwf (n + 1) ω).G)
              (k := n + 1) (ih ω)

/-- Squared differences of well-defined Algorithm 7.13 iterates are integrable,
using compact feasible diameter as the dominating bound.

Candidate audit: considered the target-file raw analogue `rawIterDiff_sq_integrable`
and SOptLib bounded-measurable integrability helpers such as
`integrable_sq_norm_of_pointwise_bound`. The raw lemma uses the wrong process,
while the generic SOptLib/local boundedness helper still needs this
paper-specific feasibility and measurability assembly. -/
lemma iterProcessOfWellDefined_diff_sq_integrable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1) (k l : ℕ) :
    Integrable
      (fun ω =>
        ‖setup.iterProcessOfWellDefined hαwf k ω -
          setup.iterProcessOfWellDefined hαwf l ω‖ ^ 2) setup.P := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  exact
    integrable_sq_norm_sub_of_measurable_mem_diameter_bound
      (μ := setup.P) (X := setup.X) (D := setup.barDX)
      (x := setup.iterProcessOfWellDefined hαwf k)
      (y := setup.iterProcessOfWellDefined hαwf l)
      (setup.iterProcessOfWellDefined_measurable hαwf k).aestronglyMeasurable
      (setup.iterProcessOfWellDefined_measurable hαwf l).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
          setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le_one k ω)
      (Filter.Eventually.of_forall fun ω => by
        simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
          setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le_one l ω)
      (fun {a b} ha hb => setup.barDX_bound a b ha hb)

/-- The epoch-start refresh estimator error is square-integrable.

Candidate audit: SOptLib `integrable_sq_norm_centeredMiniBatchAverage` is the
matching finite-average L2 closure; `integrable_sq_oracleResidual_of_indep_fixed_variance_bound`
and the target-file epoch-start independence helpers provide the
random-query coordinate integrability. `epochRefresh_variance_floor` was
considered but only exposes the numerical inequality, while the strengthened
Lan Lemma 7.5 induction also needs the L2 invariant itself. -/
lemma epochRefresh_delta_sq_integrable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s : ℕ) (hsN : setup.globalIndex s 1 ≤ setup.N) :
    Integrable
      (fun ω =>
        ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s 1) ω‖ ^ 2)
      setup.P := by
  have hx :
      ∀ ω, setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω ∈ setup.X := by
    intro ω
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le_one
        (setup.globalIndex s 1) ω
  have hdelta_eq :=
    setup.refresh_delta_eq_centered_minibatch_average hαwf s
  let epochFootprint : Finset ℕ :=
    Finset.range (s * setup.m) ∪
      Finset.Ico (setup.N * setup.m)
        (setup.N * setup.m + setup.globalIndex s 1 * setup.b)
  have hXblock :
      Measurable[
        SOptLib.sampleBlockMeasurableSpace setup.ξ epochFootprint]
        (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1)) := by
    simpa [epochFootprint] using
      setup.iterProcess_epoch_start_measurable_sampleBlock hαwf s
  have hquery_indep_refresh :
      ∀ i ∈ Finset.range setup.m,
        IndepFun
          (fun ω => (⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
            hx ω⟩ : setup.X))
          (setup.ξ (s * setup.m + i)) setup.P := by
    intro i hi
    exact setup.epoch_start_query_indep_refresh_sample_of_measurable
      hαwf s i hi hsN hx (by simpa [epochFootprint] using hXblock)
  have hfixed_refresh_variance :
      ∀ i ∈ Finset.range setup.m, ∀ x : E, x ∈ setup.X →
        Integrable
            (fun ω => ‖setup.gradF x (setup.ξ (s * setup.m + i) ω) -
              setup.gradf x‖ ^ 2) setup.P ∧
          ∫ ω, ‖setup.gradF x (setup.ξ (s * setup.m + i) ω) -
              setup.gradf x‖ ^ 2 ∂setup.P ≤ setup.σ ^ 2 := by
    intro i _hi x hx
    exact setup.fixed_sample_residual_variance_bound (s * setup.m + i) x hx
  let xq : Ω → setup.X :=
    fun ω =>
      ⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω, hx ω⟩
  have hxq_block :
      Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ epochFootprint] xq := by
    exact hXblock.subtype_mk
  have hxq_meas : Measurable xq :=
    hxq_block.mono setup.sampleBlockMeasurableSpace_le le_rfl
  have hres_meas :
      Measurable
        (fun p : setup.X × Ξ =>
          setup.gradF (p.1 : E) p.2 - setup.gradf (p.1 : E)) := by
    exact
      (setup.hgradF_meas.comp
        ((measurable_subtype_coe.comp measurable_fst).prodMk measurable_snd)).sub
        (setup.gradfOnX_measurable.comp measurable_fst)
  haveI : IsProbabilityMeasure setup.P := setup.hP
  rw [hdelta_eq]
  simpa [xq] using
    (integrable_sq_norm_randomQuery_centeredMiniBatchAverage_of_fixed_variance
      (P := setup.P) (I := Finset.range setup.m) (m := setup.m)
      (G := fun z : setup.X => fun ξ : Ξ => setup.gradF (z : E) ξ)
      (target := fun z : setup.X => setup.gradf (z : E))
      (xq := xq) (Y := fun i => setup.ξ (s * setup.m + i))
      (σ2 := setup.σ ^ 2)
      hres_meas hxq_meas
      (by
        intro i _hi
        exact setup.hξ_meas (s * setup.m + i))
      (by
        intro i hi
        exact hquery_indep_refresh i hi)
      (sq_nonneg setup.σ)
      (by
        intro i hi z
        exact (hfixed_refresh_variance i hi (z : E) z.property).1)
      (by
        intro i hi z
        exact (hfixed_refresh_variance i hi (z : E) z.property).2))

/-- Epoch-start refreshes remain stochastic mini-batch averages in Algorithm
7.13; their error is controlled by the variance floor `σ² / m`, not by exact
equality to `∇f`.

Candidate audit: `randomIterate_variance_bound_of_fixed_variance` and
`oracleRandomIterateVarianceBound_of_fixedVariance` were checked; they provide
the fixed-variance transfer used in the proof, but the epoch-start mini-batch
average and index schedule are paper-local to Algorithm 7.13. -/
lemma epochRefresh_variance_floor
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s : ℕ) (hsN : setup.globalIndex s 1 ≤ setup.N) :
    ∫ ω, ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s 1) ω‖ ^ 2 ∂setup.P ≤
      setup.σ ^ 2 / setup.m := by
  have hx :
      ∀ ω, setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω ∈ setup.X := by
    intro ω
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le_one
        (setup.globalIndex s 1) ω
  have hdelta_eq :=
    setup.refresh_delta_eq_centered_minibatch_average hαwf s
  let epochFootprint : Finset ℕ :=
    Finset.range (s * setup.m) ∪
      Finset.Ico (setup.N * setup.m)
        (setup.N * setup.m + setup.globalIndex s 1 * setup.b)
  have hXblock :
      Measurable[
        SOptLib.sampleBlockMeasurableSpace setup.ξ epochFootprint]
        (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1)) := by
    simpa [epochFootprint] using
      setup.iterProcess_epoch_start_measurable_sampleBlock hαwf s
  have hquery_indep_refresh :
      ∀ i ∈ Finset.range setup.m,
        IndepFun
          (fun ω => (⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
            hx ω⟩ : setup.X))
          (setup.ξ (s * setup.m + i)) setup.P := by
    intro i hi
    exact setup.epoch_start_query_indep_refresh_sample_of_measurable
      hαwf s i hi hsN hx (by simpa [epochFootprint] using hXblock)
  have hquery_peer_indep_refresh :
      ∀ i ∈ Finset.range setup.m, ∀ j ∈ Finset.range setup.m, i ≠ j →
        IndepFun
          (fun ω =>
            ((⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω,
              hx ω⟩ : setup.X), setup.ξ (s * setup.m + j) ω))
          (setup.ξ (s * setup.m + i)) setup.P := by
    intro i hi j hj hij
    exact setup.epoch_start_query_peer_indep_refresh_sample_of_measurable
      hαwf s i j hi hj hij hsN hx (by simpa [epochFootprint] using hXblock)
  have hfixed_refresh_variance :
      ∀ i ∈ Finset.range setup.m, ∀ x : E, x ∈ setup.X →
        Integrable
            (fun ω => ‖setup.gradF x (setup.ξ (s * setup.m + i) ω) -
              setup.gradf x‖ ^ 2) setup.P ∧
          ∫ ω, ‖setup.gradF x (setup.ξ (s * setup.m + i) ω) -
              setup.gradf x‖ ^ 2 ∂setup.P ≤ setup.σ ^ 2 := by
    intro i _hi x hx
    exact setup.fixed_sample_residual_variance_bound (s * setup.m + i) x hx
  let xq : Ω → setup.X :=
    fun ω =>
      ⟨setup.iterProcessOfWellDefined hαwf (setup.globalIndex s 1) ω, hx ω⟩
  have hxq_block :
      Measurable[SOptLib.sampleBlockMeasurableSpace setup.ξ epochFootprint] xq := by
    exact hXblock.subtype_mk
  have hxq_meas : Measurable xq :=
    hxq_block.mono setup.sampleBlockMeasurableSpace_le le_rfl
  have hres_meas :
      Measurable
        (fun p : setup.X × Ξ =>
          setup.gradF (p.1 : E) p.2 - setup.gradf (p.1 : E)) := by
    exact
      (setup.hgradF_meas.comp
        ((measurable_subtype_coe.comp measurable_fst).prodMk measurable_snd)).sub
        (setup.gradfOnX_measurable.comp measurable_fst)
  rw [hdelta_eq]
  haveI : IsProbabilityMeasure setup.P := setup.hP
  simpa [xq] using
    (randomQuery_centeredMiniBatchAverage_secondMoment_le_variance_div_card
      (P := setup.P) (I := Finset.range setup.m) (m := setup.m)
      (G := fun z : setup.X => fun ξ : Ξ => setup.gradF (z : E) ξ)
      (target := fun z : setup.X => setup.gradf (z : E))
      (xq := xq) (Y := fun i => setup.ξ (s * setup.m + i)) (σ2 := setup.σ ^ 2)
      (lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos)
      (by simp)
      hres_meas hxq_meas
      (by
        intro i _hi
        exact setup.hξ_meas (s * setup.m + i))
      (by
        intro i hi
        exact hquery_indep_refresh i hi)
      (by
        intro i hi j hj hij
        exact hquery_peer_indep_refresh i hi j hj hij)
      (sq_nonneg setup.σ)
      (by
        intro i hi z
        exact (hfixed_refresh_variance i hi (z : E) z.property).1)
      (by
        intro i hi z
        exact (hfixed_refresh_variance i hi (z : E) z.property).2)
      (by
        intro i _hi z
        let Y : Ω → Ξ := setup.ξ (s * setup.m + i)
        have hY_meas : Measurable Y := setup.hξ_meas (s * setup.m + i)
        have hGz_meas : Measurable (fun yi : Ξ => setup.gradF (z : E) yi) :=
          setup.hgradF_meas.comp
            ((measurable_const : Measurable fun _ : Ξ => (z : E)).prodMk measurable_id)
        simpa [Y] using
          (fixedOracleDeviation_integral_law_eq_zero
            (μ := setup.P)
            (G := fun z : setup.X => fun yi : Ξ => setup.gradF (z : E) yi)
            (g := fun z : setup.X => setup.gradf (z : E))
            (x := z) (ξ := setup.ξ 0) (Y := Y)
            hY_meas hGz_meas (setup.hξ_ident (s * setup.m + i))
            (setup.paperGradientOracle_integrable (z : E) z.property)
            (setup.paperGradient_unbiased_integral (z : E) z.property))))

lemma rawDeltaProcess_recursive_of_not_epochStart
    (k : ℕ) (hk : (k + 1) % setup.T ≠ 0) :
    ∀ ω,
      setup.rawDeltaProcess (k + 2) ω =
        setup.rawDeltaProcess (k + 1) ω +
          (setup.b : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.b)
              (fun i =>
                setup.gradF (setup.rawIterProcess (k + 2) ω)
                    (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω) -
                  setup.gradF (setup.rawIterProcess (k + 1) ω)
                    (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω)) -
          (setup.gradf (setup.rawIterProcess (k + 2) ω) -
            setup.gradf (setup.rawIterProcess (k + 1) ω)) := by
  intro ω
  simp [StochasticNonconvexConditionalGradientSetup.rawDeltaProcess,
    StochasticNonconvexConditionalGradientSetup.rawEstimatorProcess,
    StochasticNonconvexConditionalGradientSetup.rawIterProcess,
    StochasticNonconvexConditionalGradientSetup.delta,
    StochasticNonconvexConditionalGradientSetup.rawProcess, hk]
  abel

/-- The well-defined Algorithm 7.13 estimator error obeys the recursive
within-epoch centered gradient-difference identity away from epoch starts.

Candidate audit: considered SOptLib mirror-descent/update candidates
`mirrorStep_minimizes_of_update`, `literalMirrorStep_of_update_of_interior`,
`mirrorDescent_three_point_of_update`, `adapted_iterate_of_recursive_adapted_update`,
and `BlockIterateState`; they concern generic iterate updates or state containers,
not Algorithm 7.13's estimator-error algebra. The target-file
`rawDeltaProcess_recursive_of_not_epochStart` has the right shape only for the
raw displayed process, so this is the paper-process analogue for Lan Lemma 7.5
step 1. -/
lemma deltaProcessOfWellDefined_recursive_of_not_epochStart
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (k : ℕ) (hk : (k + 1) % setup.T ≠ 0) :
    ∀ ω,
      setup.deltaProcessOfWellDefined hαwf (k + 2) ω =
        setup.deltaProcessOfWellDefined hαwf (k + 1) ω +
          (setup.b : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.b)
              (fun i =>
                setup.gradF (setup.iterProcessOfWellDefined hαwf (k + 2) ω)
                    (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω) -
                  setup.gradF (setup.iterProcessOfWellDefined hαwf (k + 1) ω)
                    (setup.ξ (setup.N * setup.m + (k + 1) * setup.b + i) ω)) -
          (setup.gradf (setup.iterProcessOfWellDefined hαwf (k + 2) ω) -
            setup.gradf (setup.iterProcessOfWellDefined hαwf (k + 1) ω)) := by
  simpa [StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined] using
    (SOptLib.recursive_estimator_residual_eq_prev_add_centered_batch_diff_of_not_epoch_start
      (m := setup.b) (batch := fun _ => Finset.range setup.b)
      (k := k) (hbatch_card := by simp)
      (hmpos := lt_of_lt_of_le Nat.zero_lt_one
        (le_trans setup.hT_pos setup.hb_ge_T))
      (estimator := setup.estimatorProcessOfWellDefined hαwf)
      (iterate := setup.iterProcessOfWellDefined hαwf)
      (oracle := setup.gradF)
      (target := setup.gradf)
      (sample := fun t i ω => setup.ξ (setup.N * setup.m + t * setup.b + i) ω)
      (hupdate := by
        intro ω
        simp [StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
          StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
          hk]
        abel)
      ).1

/-- At a paper global index inside an epoch, the well-defined estimator error
recurses by adding the centered fresh recursive mini-batch gradient difference.

Candidate audit: checked the pre-searched SOptLib candidates
`mirrorStep_minimizes_of_update`, `literalMirrorStep_of_update_of_interior`,
`mirrorDescent_three_point_of_update`, `adapted_iterate_of_recursive_adapted_update`,
`BlockIterateState`, and `iIndepFun.indep_past_iSup_current`; they do not encode
Algorithm 7.13's flattened recursive sample schedule. The target-file
`rawDeltaProcess_globalIndex_recursive` supplies only the raw-process analogue,
so this helper transports Lan Lemma 7.5 / Eq. (7.4.14) step 1 to the
domain-aware process. -/
lemma deltaProcessOfWellDefined_globalIndex_recursive
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T) :
    ∀ ω,
      setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s j) ω =
        setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω +
          (setup.b : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.b)
              (fun i =>
                setup.gradF
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω) -
                  setup.gradF
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω)) -
          (setup.gradf (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
            setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
  simpa [StochasticNonconvexConditionalGradientSetup.globalIndex, SOptLib.global_index_def,
    Nat.add_assoc] using
    (SOptLib.recursive_estimator_residual_global_index_eq_prev_add_centered_batch_diff
      (T := setup.T) (I := Finset.range setup.b) (m := setup.b)
      (hmcard := by simp)
      (hmpos := lt_of_lt_of_le Nat.zero_lt_one
        (le_trans setup.hT_pos setup.hb_ge_T))
      (δ := setup.deltaProcessOfWellDefined hαwf)
      (batch_diff := fun k k' i ω =>
        setup.gradF (setup.iterProcessOfWellDefined hαwf k' ω)
            (setup.ξ (setup.N * setup.m + k * setup.b + i) ω) -
          setup.gradF (setup.iterProcessOfWellDefined hαwf k ω)
            (setup.ξ (setup.N * setup.m + k * setup.b + i) ω))
      (target_diff := fun k k' ω =>
        setup.gradf (setup.iterProcessOfWellDefined hαwf k' ω) -
          setup.gradf (setup.iterProcessOfWellDefined hαwf k ω))
      (hrec := fun k hk =>
        setup.deltaProcessOfWellDefined_recursive_of_not_epochStart hαwf k hk)
      (s := s) (j := j) hj2 hjT).1

lemma maxLinModel_le_inner_sub_lmo
    {x G : E} :
    setup.maxLinModel x G ≤ ⟪G, x - setup.linearMinimizer G⟫_ℝ := by
  simpa [StochasticNonconvexConditionalGradientSetup.maxLinModel,
    SOptLib.ConditionalGradient.maxLinearModel] using
    (max_linear_model_le_inner_sub_linearMinimizer
      (maximizer := setup.maxLinModelMaximizer)
      (linearMinimizer := setup.linearMinimizer)
      (linearMinimizer_is_argmin := setup.linearMinimizer_spec) x G)

lemma inner_lmo_sub_le_neg_maxLinModel
    {x G : E} :
    ⟪G, setup.linearMinimizer G - x⟫_ℝ ≤ -setup.maxLinModel x G := by
  exact inner_sub_le_neg_of_le_inner_sub G x (setup.linearMinimizer G) (setup.maxLinModel x G)
    (setup.maxLinModel_le_inner_sub_lmo (x := x) (G := G))

/-- The stochastic Wolfe gap can be evaluated with the measurable linear
minimization selector.

Candidate audit: considered the finite target-file
`SOptLib.FiniteSumConditionalGradientSetup.wolfeGap_eq_linearMinimizer` and
compact maximum helpers from SOptLib/Mathlib. The finite lemma has the wrong
setup type, while compact maximum existence does not provide measurability of
the nonunique `wolfeGapMaximizer`; the Algorithm 7.13 LMO selector realizes the
same source maximum and is measurable. -/
theorem wolfeGap_eq_linearMinimizer (x : E) (hx : x ∈ setup.X) :
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x =
      ⟪setup.gradf x, x - setup.linearMinimizer (setup.gradf x)⟫_ℝ := by
  simpa [SOptLib.ConditionalGradient.wolfeGap,
    StochasticNonconvexConditionalGradientSetup.linearMinimizer,
    SOptLib.ConditionalGradient.wolfeGap] using
    (SOptLib.ConditionalGradient.wolfeGap_eq_linearMinimizer
      (grad := setup.gradf) (maximizer := setup.wolfeGapMaximizer)
      (linearMinimizer := setup.linearMinimizer)
      (linearMinimizer_mem := setup.linearMinimizer_mem)
      (linearMinimizer_is_argmin := setup.linearMinimizer_spec)
      (hmax := fun x z => by
        simpa [SOptLib.ConditionalGradient.wolfeGap] using
          setup.wolfeGap_spec x z)
      x)

/-- The stochastic Wolfe-gap value is measurable on the feasible carrier.

Candidate audit: searched `Wolfe gap integrable iterProcessOfWellDefined` and
checked the finite analogue `wolfeGapOnX_measurable`; no SOptLib lemma packages
Algorithm 7.13's measurable LMO realization of the nonunique Wolfe maximizer. -/
theorem wolfeGapOnX_measurable :
    Measurable (fun x : setup.X => SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (x : E)) := by
  exact SOptLib.ConditionalGradient.wolfeGap_measurable_of_lmo
    (eval := fun x : setup.X => (x : E))
    (grad := setup.gradf) (maximizer := setup.wolfeGapMaximizer)
    (linearMinimizer := setup.linearMinimizer)
    (heval := measurable_subtype_coe)
    (hgrad := setup.gradfOnX_measurable)
    (hlmo := setup.linearMinimizer_measurable)
    (linearMinimizer_mem := setup.linearMinimizer_mem)
    (linearMinimizer_is_argmin := setup.linearMinimizer_spec)
    (hmax := fun x z => by
      simpa [SOptLib.ConditionalGradient.wolfeGap] using setup.wolfeGap_spec x z)

/-- The stochastic objective gradient is uniformly bounded on compact `X` by
the start-gradient plus the smoothness/diameter budget.

Candidate audit: checked SOptLib compact-bound helpers and the finite target
analogue `gradf_norm_bound_on_X`; the explicit source-shaped estimate needed
here follows directly from `setup.gradf_smooth` and `setup.barDX_bound`. -/
theorem gradf_norm_bound_on_X (x : E) (hx : x ∈ setup.X) :
    ‖setup.gradf x‖ ≤ ‖setup.gradf setup.x₁‖ + setup.L * setup.barDX := by
  exact norm_le_ref_norm_add_lipschitz_mul_diam setup.gradf x setup.x₁ setup.L setup.barDX
    (setup.gradf_smooth x setup.x₁ hx setup.hx₁_mem)
    (setup.barDX_bound x setup.x₁ hx setup.hx₁_mem)
    (le_of_lt setup.hL_pos)

/-- Wolfe-gap fibers along the well-defined stochastic Algorithm 7.13 process
are integrable.

Candidate audit: searched `Wolfe gap integrable iterProcessOfWellDefined` and
checked target-file finite `finite_wolfeGap_iterProcess_integrable_of_le`.
The finite lemma has the wrong setup type; this stochastic version reuses the
same measurable-LMO and compact-diameter argument with
`iterProcessOfWellDefined`. -/
theorem wolfeGap_iterProcessOfWellDefined_integrable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1) (k : ℕ) :
    Integrable
      (fun ω => SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hαwf k ω))
      setup.P := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  have hmem : ∀ ω, setup.iterProcessOfWellDefined hαwf k ω ∈ setup.X := by
    intro ω
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le_one k ω
  exact
    SOptLib.ConditionalGradient.integrable_stationarity_of_measurable_bounded_on_feasible_process
      (eval := fun x : setup.X => (x : E))
      (grad := setup.gradf)
      (maximizer := setup.wolfeGapMaximizer)
      (linearMinimizer := setup.linearMinimizer)
      (process := fun ω => (⟨setup.iterProcessOfWellDefined hαwf k ω, hmem ω⟩ : setup.X))
      (G := ‖setup.gradf setup.x₁‖ + setup.L * setup.barDX)
      (D := setup.barDX)
      (heval := measurable_subtype_coe)
      (hgrad := setup.gradfOnX_measurable)
      (hlmo := setup.linearMinimizer_measurable)
      (hprocess := (setup.iterProcessOfWellDefined_measurable hαwf k).subtype_mk)
      (linearMinimizer_mem := setup.linearMinimizer_mem)
      (linearMinimizer_is_argmin := setup.linearMinimizer_spec)
      (hmax := fun x z => by
        simpa [SOptLib.ConditionalGradient.wolfeGap] using setup.wolfeGap_spec x z)
      (hgrad_bound := fun ω =>
        setup.gradf_norm_bound_on_X (setup.iterProcessOfWellDefined hαwf k ω) (hmem ω))
      (hdiam := fun ω =>
        setup.barDX_bound (setup.iterProcessOfWellDefined hαwf k ω)
          (setup.linearMinimizer
            (setup.gradf (setup.iterProcessOfWellDefined hαwf k ω)))
          (hmem ω)
          (setup.linearMinimizer_mem
            (setup.gradf (setup.iterProcessOfWellDefined hαwf k ω))))
      (hG_nonneg := by
        have hL_nonneg : 0 ≤ setup.L := le_of_lt setup.hL_pos
        have hDX_nonneg : 0 ≤ setup.barDX := norm_nonneg _
        positivity)

/-- Objective values along the well-defined stochastic Algorithm 7.13 iterates
are integrable.

Candidate audit: searched `iterProcessOfWellDefined f integrable objective`;
target-file hits only contained the finite helper
`finite_f_iterProcess_integrable_of_le`, while SOptLib supplied the generic
compact bounded-measurable infrastructure. This stochastic bridge is the direct
analogue using `iterProcessOfWellDefined_measurable` and
`processOfWellDefined_mem_of_alpha_le_one`, matching the objective-drop
telescope required in Lan Theorem 7.17's Theorem 7.16 template. -/
theorem f_iterProcessOfWellDefined_integrable
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1) (k : ℕ) :
    Integrable
      (fun ω => setup.f (setup.iterProcessOfWellDefined hαwf k ω)) setup.P := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  have hiter : Measurable (setup.iterProcessOfWellDefined hαwf k) :=
    setup.iterProcessOfWellDefined_measurable hαwf k
  have hmem : ∀ ω, setup.iterProcessOfWellDefined hαwf k ω ∈ setup.X := by
    intro ω
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le_one k ω
  have hiterX : Measurable
      (fun ω => (⟨setup.iterProcessOfWellDefined hαwf k ω, hmem ω⟩ : setup.X)) :=
    hiter.subtype_mk
  have hcontOn : ContinuousOn setup.f setup.X := by
    intro x hx
    exact (setup.gradf_hasGradientAt x hx).continuousAt.continuousWithinAt
  have hfX_meas : Measurable (fun x : setup.X => setup.f (x : E)) :=
    (continuous_subtype_of_continuousOn_ambient
      (fun x : setup.X => setup.f (x : E)) setup.f hcontOn (fun _ => rfl)).measurable
  have hmeas :
      Measurable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hαwf k ω)) :=
    hfX_meas.comp hiterX
  obtain ⟨C, _hC_nonneg, hC⟩ :=
    exists_nonneg_norm_bound_of_isCompact_of_continuousOn
      setup.f setup.hX_compact hcontOn
  exact integrable_of_measurable_bounded_real hmeas
    (C := C)
    (fun ω => hC (setup.iterProcessOfWellDefined hαwf k ω) (hmem ω))

end StochasticNonconvexConditionalGradientSetup

namespace SOptLib.FiniteSumConditionalGradientSetup

variable (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)

/-- The finite-sum feasible set is nonempty because the theorem setup includes
the initial feasible point `x₁ ∈ X`; this removes the former primitive
nonemptiness witness. -/
theorem hX_nonempty : Set.Nonempty fs.X :=
  ⟨fs.x₁, fs.hx₁_mem⟩

/-- Algorithm 7.12 component sampling probabilities `q_i = L_i/(m L)`. -/
noncomputable def componentQ (i : Fin fs.componentCount) : ℝ :=
  fs.componentL i / ((fs.componentCount : ℝ) * fs.L)

/-- Algorithm 7.12 component sampling probabilities are nonnegative. -/
theorem componentQ_nonneg (i : Fin fs.componentCount) : 0 ≤ fs.componentQ i := by
  simpa [componentQ, SOptLib.smoothnessImportanceWeight] using
    SOptLib.smoothnessImportanceWeight_nonneg
      (Lcomp := fs.componentL) (n := (fs.componentCount : ℝ)) (L := fs.L)
      (fun j => le_of_lt (fs.hcomponentL_pos j))
      (Nat.cast_nonneg fs.componentCount)
      (le_of_lt fs.hL_pos) i

/-- Algorithm 7.12 component sampling probabilities are strictly positive under
the finite-sum smoothness setup. -/
theorem componentQ_pos (i : Fin fs.componentCount) : 0 < fs.componentQ i := by
  unfold componentQ
  have hn_pos_nat : 0 < fs.componentCount :=
    Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hcomponentCount_pos
  have hn_pos : 0 < (fs.componentCount : ℝ) := by exact_mod_cast hn_pos_nat
  exact div_pos (fs.hcomponentL_pos i) (mul_pos hn_pos fs.hL_pos)

/-- Algorithm 7.12 component sampling probabilities sum to one. -/
theorem componentQ_sum_one : (∑ i : Fin fs.componentCount, fs.componentQ i) = 1 := by
  have hden_ne : (fs.componentCount : ℝ) * fs.L ≠ 0 := by
    have hn_pos_nat : 0 < fs.componentCount :=
      Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hcomponentCount_pos
    have hn_pos : 0 < (fs.componentCount : ℝ) := by exact_mod_cast hn_pos_nat
    exact mul_ne_zero (ne_of_gt hn_pos) (ne_of_gt fs.hL_pos)
  calc
    (∑ i : Fin fs.componentCount, fs.componentQ i)
        = (∑ i : Fin fs.componentCount, fs.componentL i) /
            ((fs.componentCount : ℝ) * fs.L) := by
          simp [componentQ, Finset.sum_div]
    _ = 1 := by
          rw [fs.hcomponentL_sum_eq, div_self hden_ne]

/-- Canonical Algorithm 7.12 law for the finite-sum component indices sampled in
Theorem 7.16. -/
noncomputable def componentIndexPMF : PMF (Fin fs.componentCount) :=
  PMF.ofFintypeOfReal fs.componentQ fs.componentQ_nonneg fs.componentQ_sum_one

/-- The component-index PMF assigns the paper mass `q_i = L_i/(m L)`. -/
theorem componentIndexPMF_apply (i : Fin fs.componentCount) :
    fs.componentIndexPMF i = ENNReal.ofReal (fs.componentQ i) := by
  classical
  simp [componentIndexPMF]

/-- Component-index sample measurability is a Lean realization obligation for
Algorithm 7.12's finite component sampling law, not a primitive theorem-side
paper field. -/
theorem sample_measurable (k : ℕ) : Measurable (fs.sample k) :=
  fs.hsample_meas k

/-- The finite-sum component-index samples are i.i.d. with Algorithm 7.12's
importance sampling law. -/
theorem sample_identDistrib_componentQ (k : ℕ) :
    IdentDistrib (fs.sample k) (fun i : Fin fs.componentCount => i) fs.P
      fs.componentIndexPMF.toMeasure := by
  exact SOptLib.sample_identDistrib_finiteImportancePMF fs.sample fs.P fs.componentIndexPMF k
    (fs.hsample_meas k).aemeasurable
    (by simpa [componentIndexPMF, componentQ] using fs.hsample_componentQ k)

/-- The finite-sum component-index stream is independent across sampling calls. -/
theorem sample_iIndep :
    iIndepFun (β := fun _ => Fin fs.componentCount) fs.sample fs.P :=
  fs.hsample_iIndep

/-- Finite-sum objective `f(x) = n⁻¹ ∑ᵢ Fᵢ(x)` from the finite-sum theorem
template used in Theorem 7.16.

Candidate audit: SOptLib `objectiveExpectation` models stochastic expectations,
and `objectiveKernel`/`objectiveWellDefined` model sampled objective kernels;
none are the literal finite average from Theorem 7.16, so the local finite-sum
objective is defined by the component average. -/
noncomputable def f (x : E) : ℝ :=
  (fs.componentCount : ℝ)⁻¹ *
    Finset.sum Finset.univ (fun i : Fin fs.componentCount => fs.Fcomp i x)

/-- Finite-sum gradient `∇f(x) = n⁻¹ ∑ᵢ ∇Fᵢ(x)`, derived from component
gradients rather than accepted as an arbitrary theorem-side quantity. -/
noncomputable def gradf (x : E) : E :=
  (fs.componentCount : ℝ)⁻¹ •
    Finset.sum Finset.univ (fun i : Fin fs.componentCount => fs.gradFcomp i x)

/-- Importance sampling with Algorithm 7.12 probabilities is centered at the
finite-sum full-gradient difference.

Candidate audit: checked `sample_identDistrib_componentQ`,
`componentIndexPMF_apply`, `integral_selected_finite_index_prod_eq_sum_weights`,
and searched `importance sampling componentQ finite sum gradient difference mean
zero`. Those results handle laws or product integrals; this helper isolates the
deterministic finite-sum algebra needed before applying those probabilistic
transport lemmas in Lan Lemma 7.4. -/
lemma componentQ_weighted_gradDiff_sum_eq_gradf_sub
    (x y : E) :
    Finset.sum Finset.univ
        (fun i : Fin fs.componentCount =>
          fs.componentQ i •
            (((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp i x - fs.gradFcomp i y))) =
      fs.gradf x - fs.gradf y := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.gradf] using
    (finset_importance_weighted_diff_mean_eq_uniform_mean
      (s := Finset.univ) (q := fs.componentQ) (n := (fs.componentCount : ℝ))
      (a := fun i : Fin fs.componentCount => fs.gradFcomp i x)
      (b := fun i : Fin fs.componentCount => fs.gradFcomp i y)
      (hq_ne := fun i _hi => ne_of_gt (fs.componentQ_pos i))
      (hn_ne := by
        have hn_pos_nat : 0 < fs.componentCount :=
          Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hcomponentCount_pos
        have hn_pos : 0 < (fs.componentCount : ℝ) := by exact_mod_cast hn_pos_nat
        exact ne_of_gt hn_pos))

/-- The Algorithm 7.12 inverse-probability component gradient difference has
the paper's `L`-smooth pointwise bound.

Candidate audit: checked SOptLib `lipschitzWith_of_norm_sub_le_mul` and
`Convex.carrier_smooth_quadratic_upper_bound`; those package generic Lipschitz
or quadratic-smoothness consequences, while Lan Lemma 7.4 needs the literal
importance-weighted component-gradient difference with `q_i = L_i/(mL)`.
This helper is the deterministic diagonal estimate used before integration. -/
lemma componentQ_weighted_gradDiff_norm_le
    (i : Fin fs.componentCount) (x y : E)
    (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    ‖((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
        (fs.gradFcomp i x - fs.gradFcomp i y)‖ ≤
      fs.L * ‖x - y‖ := by
  simpa [componentQ, SOptLib.normalizedOutputMass] using
    norm_inv_smoothness_importance_smul_le
      (Lcomp := fs.componentL) (n := (fs.componentCount : ℝ)) (L := fs.L)
      (r := ‖x - y‖) i (fs.gradFcomp i x - fs.gradFcomp i y)
      (hn_pos := by
        have hn_pos_nat : 0 < fs.componentCount :=
          Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hcomponentCount_pos
        exact_mod_cast hn_pos_nat)
      (hL_pos := fs.hL_pos)
      (hLi_pos := fs.hcomponentL_pos i)
      (hu := fs.hFcomp_smooth i x y hx hy)

/-- The diagonal second moment of one importance-sampled component difference
is bounded by the paper `L²` budget after summing over Algorithm 7.12's
component law.

Candidate audit: checked SOptLib `selected_block_second_moment_integral_le_sum_bounds`
and `integral_selected_finite_index_prod_eq_sum_weights`; those are integration
transport lemmas for selected finite indices. This helper supplies the
deterministic finite-index budget they need for Lan Lemma 7.4's recursive
mini-batch variance bound. -/
lemma componentQ_weighted_gradDiff_secondMoment_sum_le
    (x y : E) (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    Finset.sum Finset.univ
        (fun i : Fin fs.componentCount =>
          fs.componentQ i *
            ‖((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp i x - fs.gradFcomp i y)‖ ^ 2) ≤
      fs.L ^ 2 * ‖x - y‖ ^ 2 := by
  classical
  simpa [mul_pow] using
    finset_weighted_second_moment_le_of_pointwise_norm_bound
      (s := (Finset.univ : Finset (Fin fs.componentCount)))
      (q := fs.componentQ)
      (A := fun i : Fin fs.componentCount =>
        ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
          (fs.gradFcomp i x - fs.gradFcomp i y))
      (C := fs.L * ‖x - y‖)
      (by intro i _hi; exact fs.componentQ_nonneg i)
      (by simpa using fs.componentQ_sum_one)
      (by
        intro i _hi
        exact fs.componentQ_weighted_gradDiff_norm_le i x y hx hy)

/-- The finite component residual in Algorithm 7.12 has zero mean under the
importance-sampling probabilities.

Candidate audit: this specializes the already proved
`componentQ_weighted_gradDiff_sum_eq_gradf_sub`; probabilistic hits such as
`sample_identDistrib_componentQ` and `IdentDistrib.integral_comp_eq_of_measurable`
transport this algebra to sampled coordinates, but do not themselves prove the
finite-sum centering identity. -/
lemma componentQ_weighted_gradDiff_residual_sum_eq_zero
    (x y : E) :
    Finset.sum Finset.univ
        (fun i : Fin fs.componentCount =>
          fs.componentQ i •
            ((((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp i x - fs.gradFcomp i y)) -
              (fs.gradf x - fs.gradf y))) = 0 := by
  simpa using
    (finset_weighted_residual_sum_eq_zero
      (s := Finset.univ) (q := fs.componentQ)
      (a := fun i : Fin fs.componentCount =>
        ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
          (fs.gradFcomp i x - fs.gradFcomp i y))
      (μ := fs.gradf x - fs.gradf y)
      fs.componentQ_sum_one
      (fs.componentQ_weighted_gradDiff_sum_eq_gradf_sub x y))

/-- The centered finite component residual has no larger deterministic
second-moment budget than the uncentered importance-sampled difference, hence
inherits the `L²‖x-y‖²` diagonal bound used in Lan Lemma 7.4.

Candidate audit: checked the existing deterministic candidates
`componentQ_weighted_gradDiff_residual_sum_eq_zero` and
`componentQ_weighted_gradDiff_secondMoment_sum_le`, plus SOptLib
`miniBatchResidual_sum_secondMoment_le_card_mul_variance` and
`miniBatchAverage_secondMoment_le_variance_div_card`. The SOptLib lemmas start
from diagonal residual bounds; this helper supplies that fixed component-law
bound from the paper's importance sampling algebra. -/
lemma componentQ_weighted_gradDiff_residual_secondMoment_sum_le
    (x y : E) (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    Finset.sum Finset.univ
        (fun i : Fin fs.componentCount =>
          fs.componentQ i *
            ‖(((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp i x - fs.gradFcomp i y)) -
              (fs.gradf x - fs.gradf y)‖ ^ 2) ≤
      fs.L ^ 2 * ‖x - y‖ ^ 2 := by
  classical
  exact
    finset_weighted_residual_second_moment_le_of_second_moment
      (s := (Finset.univ : Finset (Fin fs.componentCount)))
      (q := fs.componentQ)
      (a := fun i : Fin fs.componentCount =>
        ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
          (fs.gradFcomp i x - fs.gradFcomp i y))
      (μ := fs.gradf x - fs.gradf y)
      (C := fs.L ^ 2 * ‖x - y‖ ^ 2)
      fs.componentQ_sum_one
      (by simpa using fs.componentQ_weighted_gradDiff_sum_eq_gradf_sub x y)
      (by simpa using fs.componentQ_weighted_gradDiff_secondMoment_sum_le x y hx hy)

/-- A fixed feasible pair transported through one finite component sample has
the Lan Lemma 7.4 diagonal residual second-moment budget.

Candidate audit: this consumes `sample_identDistrib_componentQ` and
`IdentDistrib.integral_comp_eq_of_measurable` after the deterministic
component-law bound
`componentQ_weighted_gradDiff_residual_secondMoment_sum_le`. SOptLib's
mini-batch lemmas require this as an input rather than deriving the
finite-index law transport themselves. -/
lemma sampled_componentQ_gradDiff_residual_secondMoment_le
    (r : ℕ) (x y : E) (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    Integrable
        (fun ω =>
          ‖(((fs.componentQ (fs.sample r ω) * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample r ω) x -
                fs.gradFcomp (fs.sample r ω) y)) -
              (fs.gradf x - fs.gradf y)‖ ^ 2)
        fs.P ∧
      ∫ ω,
          ‖(((fs.componentQ (fs.sample r ω) * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample r ω) x -
                fs.gradFcomp (fs.sample r ω) y)) -
              (fs.gradf x - fs.gradf y)‖ ^ 2 ∂fs.P ≤
        fs.L ^ 2 * ‖x - y‖ ^ 2 := by
  classical
  exact
    identDistrib_finite_pmf_integrable_integral_le_weighted_sum_bound
      fs.P (fs.sample r) fs.componentIndexPMF fs.componentQ
      (fun i =>
        ‖(((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
            (fs.gradFcomp i x - fs.gradFcomp i y)) -
            (fs.gradf x - fs.gradf y)‖ ^ 2)
      (fs.L ^ 2 * ‖x - y‖ ^ 2)
      (fs.sample_identDistrib_componentQ r)
      (fun i => by
        rw [fs.componentIndexPMF_apply i,
          ENNReal.toReal_ofReal (fs.componentQ_nonneg i)])
      (by
        simpa using
          fs.componentQ_weighted_gradDiff_residual_secondMoment_sum_le x y hx hy)

/-- The averaged finite-sum gradient is the gradient of the finite-sum
objective on feasible points. -/
theorem gradf_hasGradientAt (x : E) (hx : x ∈ fs.X) :
    HasGradientAt fs.f (fs.gradf x) x := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.f,
    SOptLib.FiniteSumConditionalGradientSetup.gradf] using
    SOptLib.finiteAverageObjective_hasGradientAt
      (F := fs.Fcomp) (gradF := fs.gradFcomp) x
      (fun i : Fin fs.componentCount => fs.hFcomp_hasGradientAt i x hx)

/-- Finite-sum smoothness of the averaged objective, derived from component
smoothness. -/
theorem gradf_smooth (x y : E) (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    ‖fs.gradf x - fs.gradf y‖ ≤ fs.L * ‖x - y‖ := by
  classical
  unfold SOptLib.FiniteSumConditionalGradientSetup.gradf
  have hn_pos_nat : 0 < fs.componentCount := by
    exact Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hcomponentCount_pos
  have hn_pos : 0 < (fs.componentCount : ℝ) := by
    exact_mod_cast hn_pos_nat
  have hc_nonneg : 0 ≤ (fs.componentCount : ℝ)⁻¹ := by
    exact inv_nonneg.mpr (le_of_lt hn_pos)
  have hdiff :
      (fs.componentCount : ℝ)⁻¹ • Finset.sum Finset.univ
          (fun i : Fin fs.componentCount => fs.gradFcomp i x) -
        (fs.componentCount : ℝ)⁻¹ • Finset.sum Finset.univ
          (fun i : Fin fs.componentCount => fs.gradFcomp i y) =
      (fs.componentCount : ℝ)⁻¹ •
        Finset.sum Finset.univ
          (fun i : Fin fs.componentCount => fs.gradFcomp i x - fs.gradFcomp i y) := by
    simp [Finset.sum_sub_distrib, smul_sub]
  calc
    ‖(fs.componentCount : ℝ)⁻¹ • Finset.sum Finset.univ
        (fun i : Fin fs.componentCount => fs.gradFcomp i x) -
        (fs.componentCount : ℝ)⁻¹ • Finset.sum Finset.univ
          (fun i : Fin fs.componentCount => fs.gradFcomp i y)‖
        = ‖(fs.componentCount : ℝ)⁻¹ •
            Finset.sum Finset.univ
              (fun i : Fin fs.componentCount => fs.gradFcomp i x - fs.gradFcomp i y)‖ := by
          rw [hdiff]
    _ = (fs.componentCount : ℝ)⁻¹ *
          ‖Finset.sum Finset.univ
            (fun i : Fin fs.componentCount => fs.gradFcomp i x - fs.gradFcomp i y)‖ := by
          rw [norm_smul, Real.norm_of_nonneg hc_nonneg]
    _ ≤ (fs.componentCount : ℝ)⁻¹ *
          Finset.sum Finset.univ
            (fun i : Fin fs.componentCount => ‖fs.gradFcomp i x - fs.gradFcomp i y‖) := by
          exact mul_le_mul_of_nonneg_left
            (norm_sum_le (s := Finset.univ)
              (f := fun i : Fin fs.componentCount => fs.gradFcomp i x - fs.gradFcomp i y))
            hc_nonneg
    _ ≤ (fs.componentCount : ℝ)⁻¹ *
          Finset.sum Finset.univ (fun _i : Fin fs.componentCount => fs.L * ‖x - y‖) := by
          have hsum_smooth :
              Finset.sum Finset.univ
                  (fun i : Fin fs.componentCount => ‖fs.gradFcomp i x - fs.gradFcomp i y‖) ≤
                Finset.sum Finset.univ
                  (fun i : Fin fs.componentCount => fs.componentL i * ‖x - y‖) :=
            Finset.sum_le_sum (fun i hi => fs.hFcomp_smooth i x y hx hy)
          have hfactor :
              Finset.sum Finset.univ
                  (fun i : Fin fs.componentCount => fs.componentL i * ‖x - y‖) =
                Finset.sum Finset.univ (fun _i : Fin fs.componentCount => fs.L * ‖x - y‖) := by
            calc
              Finset.sum Finset.univ
                  (fun i : Fin fs.componentCount => fs.componentL i * ‖x - y‖)
                  = (Finset.sum Finset.univ fs.componentL) * ‖x - y‖ := by
                    rw [Finset.sum_mul]
              _ = ((fs.componentCount : ℝ) * fs.L) * ‖x - y‖ := by
                    rw [fs.hcomponentL_sum_eq]
              _ = Finset.sum Finset.univ
                    (fun _i : Fin fs.componentCount => fs.L * ‖x - y‖) := by
                    simp [Finset.sum_const, mul_assoc]
          exact mul_le_mul_of_nonneg_left (hsum_smooth.trans_eq hfactor) hc_nonneg
    _ = fs.L * ‖x - y‖ := by
          have hn_ne : (fs.componentCount : ℝ) ≠ 0 := ne_of_gt hn_pos
          simp [Finset.sum_const, hn_ne]

/-- Each finite component gradient is measurable on the feasible carrier.

Candidate audit: checked SOptLib `lipschitzWith_of_norm_sub_le_mul` and
`LipschitzWith.measurable`. They match the proof shape: component smoothness is
a feasible-subtype Lipschitz bound, and Lipschitz maps on the carrier are
measurable. This is the finite-sum analogue of the stochastic
`gradfOnX_measurable` bridge used in Lemma 7.5. -/
theorem gradFcompOnX_measurable (i : Fin fs.componentCount) :
    Measurable (fun x : fs.X => fs.gradFcomp i (x : E)) := by
  exact measurable_of_norm_sub_le_mul_on_subtype
    (fun x : fs.X => fs.gradFcomp i (x : E)) (fs.componentL i)
    (fun x y => by
      simpa using fs.hFcomp_smooth i (x : E) (y : E) x.property y.property)

/-- The averaged finite-sum gradient is measurable on the feasible carrier.

Candidate audit: reused the newly proved component-carrier measurability and
the local finite-sum definition of `gradf`; SOptLib objective/oracle mean
measurability lemmas model expectation kernels, while Theorem 7.16's finite
gradient is a literal finite average. -/
theorem gradfOnX_measurable :
    Measurable (fun x : fs.X => fs.gradf (x : E)) := by
  unfold SOptLib.FiniteSumConditionalGradientSetup.gradf
  exact (Finset.measurable_sum Finset.univ
    (fun i _hi => fs.gradFcompOnX_measurable i)).const_smul
      ((fs.componentCount : ℝ)⁻¹)

/-- Existence of an attained finite-sum objective minimizer on compact `X`.

Candidate audit: SOptLib `objectiveGapRadius`/`objectiveGapRadius_eq` are budget
radius facts, while `objectiveKernel`, `objectiveWellDefined`,
`compositeObjective`, and `compositeObjectiveAmbient` model objective packaging
rather than compact attainment; this local theorem aligns with Lan Theorem
7.16's `f^* = min_{x ∈ X} f(x)` by applying `IsCompact.exists_isMinOn`. -/
theorem objectiveMinimum_exists :
    ∃ x : fs.X, ∀ y : fs.X, fs.f x ≤ fs.f y := by
  exact SOptLib.objectiveMinimum_exists_of_isCompact_continuousOn fs.f
    fs.hX_compact fs.hX_nonempty
    (by
      intro x hx
      exact (fs.gradf_hasGradientAt x hx).continuousAt.continuousWithinAt)

/-- Canonical selected finite-sum minimizer realizing `f^*`. -/
noncomputable def objectiveMinimum : {x : fs.X // ∀ y : fs.X, fs.f x ≤ fs.f y} :=
  ⟨Classical.choose fs.objectiveMinimum_exists,
    Classical.choose_spec fs.objectiveMinimum_exists⟩

/-- Finite-sum optimum value `f^* = min_{x ∈ X} f(x)`. -/
noncomputable def fStar : ℝ :=
  SOptLib.optimizerValueOfMinimum (fun x : fs.X => fs.f x) fs.objectiveMinimum

/-- Finite-sum optimum lower-bounds all feasible objective values. -/
theorem fStar_lb (x : E) (hx : x ∈ fs.X) : fs.fStar ≤ fs.f x := by
  simpa [fStar, SOptLib.optimizerValueOfMinimum] using
    fs.objectiveMinimum.2 ⟨x, hx⟩

/-- Number of epochs used by the finite-sum process in Theorem 7.16. -/
noncomputable def S : ℕ := (fs.N + fs.T - 1) / fs.T

/-- The global iteration index from epoch s and within-epoch step t. -/
noncomputable def globalIndex (s t : ℕ) : ℕ := s * fs.T + t

/-- The actual within-epoch steps generated by Algorithm 7.12 in epoch `s`.

No SOptLib match: searched `Finset filter sum nonnegative subset le` and checked
Mathlib `Finset.filter`; SOptLib has generic finite-sum/telescope helpers but no
paper-local epoch object for Algorithm 7.12's last partial epoch. The source text
after Algorithm 7.12 says the last epoch consists of only the remaining iterations,
so the canonical Lean realization is the paper epoch window filtered by
`globalIndex s j ≤ N`. -/
noncomputable def activeEpochSteps (s : ℕ) : Finset ℕ :=
  SOptLib.activeEpochSteps fs.T fs.N fs.globalIndex s

/-- Membership in the generated within-epoch window. -/
theorem mem_activeEpochSteps {s j : ℕ} :
    j ∈ fs.activeEpochSteps s ↔
      j ∈ Finset.Icc 1 fs.T ∧ fs.globalIndex s j ≤ fs.N := by
  simp [activeEpochSteps]

/-- Active epoch steps are genuine within-epoch coordinates. -/
theorem activeEpochSteps_mem_epoch {s j : ℕ}
    (hj : j ∈ fs.activeEpochSteps s) :
    1 ≤ j ∧ j ≤ fs.T := by
  exact (Finset.mem_Icc.mp ((fs.mem_activeEpochSteps).mp hj).1)

/-- Active epoch steps carry the run-window fact required by `lemma_7_4`. -/
theorem activeEpochSteps_globalIndex_le {s j : ℕ}
    (hj : j ∈ fs.activeEpochSteps s) :
    fs.globalIndex s j ≤ fs.N :=
  ((fs.mem_activeEpochSteps).mp hj).2

/-- The global index of an active epoch step lies in the output/update window
`{1, ..., N}`. -/
theorem globalIndex_mem_output_of_active {s j : ℕ}
    (hj : j ∈ fs.activeEpochSteps s) :
    fs.globalIndex s j ∈ Finset.Icc 1 fs.N := by
  exact global_index_mem_output_window_of_active_epoch_step
    (T := fs.T) (N := fs.N) (globalIndex := fs.globalIndex)
    (hglobalIndex_pos := by
      have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
      unfold SOptLib.FiniteSumConditionalGradientSetup.globalIndex
      omega)
    hj

/-- Epoch number of a generated global index `k`. -/
noncomputable def epochOfIndex (k : ℕ) : ℕ :=
  SOptLib.epochOfIndex fs.T k

/-- Within-epoch step number of a generated global index `k`. -/
noncomputable def stepOfIndex (k : ℕ) : ℕ :=
  SOptLib.stepOfIndex fs.T k

/-- The step coordinate decoded from a global index lies in `{1, ..., T}`. -/
theorem stepOfIndex_mem_epoch (k : ℕ) :
    1 ≤ fs.stepOfIndex k ∧ fs.stepOfIndex k ≤ fs.T := by
  have hTpos : 0 < fs.T := Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hT_pos
  simpa [SOptLib.FiniteSumConditionalGradientSetup.stepOfIndex] using
    SOptLib.stepOfIndex_mem_epoch (T := fs.T) (k := k) hTpos

/-- Decoding and re-encoding a positive global index returns the same index. -/
theorem globalIndex_epochOfIndex_stepOfIndex
    {k : ℕ} (hk : 1 ≤ k) :
    fs.globalIndex (fs.epochOfIndex k) (fs.stepOfIndex k) = k := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.globalIndex,
    SOptLib.FiniteSumConditionalGradientSetup.epochOfIndex,
    SOptLib.FiniteSumConditionalGradientSetup.stepOfIndex] using
    (SOptLib.global_index_epochOfIndex_stepOfIndex (T := fs.T) (k := k) hk)

/-- Every generated output index belongs to its decoded active epoch window. -/
theorem stepOfIndex_mem_activeEpochSteps
    {k : ℕ} (hk : k ∈ Finset.Icc 1 fs.N) :
    fs.stepOfIndex k ∈ fs.activeEpochSteps (fs.epochOfIndex k) := by
  exact SOptLib.stepOfIndex_mem_activeEpochSteps
    (T := fs.T) (N := fs.N)
    (globalIndex := fs.globalIndex)
    (epochOfIndex := fs.epochOfIndex)
    (stepOfIndex := fs.stepOfIndex)
    (fun {k} _hk => Finset.mem_Icc.mpr (fs.stepOfIndex_mem_epoch k))
    (fun {_k} hk => fs.globalIndex_epochOfIndex_stepOfIndex (Finset.mem_Icc.mp hk).1)
    hk

/-- Prefix monotonicity for finite-sum epoch coordinates inside the source run
window `k = 1, ..., N`.

Book/PDF citation: Algorithm 7.12 runs for `k = 1, 2, ..., N`, and Lemma 7.4
states its bound when `k`, equivalently `(s,t)`, represents the generated
iteration. This arithmetic bridge transports that run-domain fact to earlier
coordinates in the same epoch so component smoothness can be applied only to
generated feasible iterates. -/
lemma globalIndex_prefix_le_of_le
    {s i t : ℕ} (hi : i ≤ t)
    (hkt : fs.globalIndex s t ≤ fs.N) :
    fs.globalIndex s i ≤ fs.N := by
  unfold SOptLib.FiniteSumConditionalGradientSetup.globalIndex at *
  omega

/-- Existence of a pair attaining the feasible-set diameter in Theorem 7.16.

No SOptLib match: searched "compact maximum attained diameter product norm" and
found only bound/declared-max packaging lemmas such as
`exists_abs_bound_on_compact_product_of_continuousOn`; the paper object
requires constructing the actual compact maximum pair for
`max_{x,y ∈ X} ‖x-y‖`. -/
theorem diameterPair_exists :
    ∃ p : fs.X × fs.X,
      ∀ q : fs.X × fs.X,
        ‖(q.1 : E) - (q.2 : E)‖ ≤ ‖(p.1 : E) - (p.2 : E)‖ := by
  classical
  let S : Set (E × E) := fs.X.prod fs.X
  have hSne : S.Nonempty := by
    refine ⟨(fs.x₁, fs.x₁), ?_⟩
    exact ⟨fs.hx₁_mem, fs.hx₁_mem⟩
  have hScompact : IsCompact S := fs.hX_compact.prod fs.hX_compact
  have hcont : ContinuousOn (fun p : E × E => ‖p.1 - p.2‖) S := by
    exact ((continuous_fst.sub continuous_snd).norm).continuousOn
  obtain ⟨p, hpS, hpmax⟩ := hScompact.exists_isMaxOn hSne hcont
  refine ⟨(⟨p.1, hpS.1⟩, ⟨p.2, hpS.2⟩), ?_⟩
  intro q
  exact hpmax (show ((q.1 : E), (q.2 : E)) ∈ S from ⟨q.1.property, q.2.property⟩)

/-- Canonical finite-sum feasible-set diameter pair. -/
noncomputable def diameterPair : fs.X × fs.X :=
  Classical.choose fs.diameterPair_exists

/-- Finite-sum theorem diameter `D̄_X = max_{x,y ∈ X} ‖x-y‖`. -/
noncomputable def barDX : ℝ :=
  ‖(fs.diameterPair.1 : E) - (fs.diameterPair.2 : E)‖

/-- Every finite-sum feasible pair is bounded by `D̄_X`. -/
theorem barDX_bound (x y : E) (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    ‖x - y‖ ≤ fs.barDX := by
  simpa [barDX] using
    SOptLib.le_euclideanDiameterOfPair_of_mem (p := fs.diameterPair)
      (by simpa [diameterPair] using Classical.choose_spec fs.diameterPair_exists) hx hy

/-- Existence of the finite-sum compact linear-minimization oracle point.

Candidate audit: checked SOptLib candidates including `Measurable.proxStep_comp`,
`continuous_argmin_of_compact_unique`, and prox/F.O.C. argmin lemmas; they are
measurability, uniqueness, or prox-specific facts, while Algorithm 7.16 needs
only the attained compact minimum of the linear model `⟪g,x⟫`. -/
theorem linearMinimizer_exists (g : E) :
    ∃ y : E, y ∈ fs.X ∧ ∀ x : E, x ∈ fs.X → ⟪g, y⟫_ℝ ≤ ⟪g, x⟫_ℝ := by
  classical
  have hcont : ContinuousOn (fun y : E => ⟪g, y⟫_ℝ) fs.X := by
    exact (continuous_const.inner continuous_id).continuousOn
  obtain ⟨y, hyX, hymin⟩ :=
    fs.hX_compact.exists_isMinOn (show fs.X.Nonempty from ⟨fs.x₁, fs.hx₁_mem⟩) hcont
  exact ⟨y, hyX, fun x hx => hymin hx⟩

/-- Finite-sum Algorithm 7.12 linear minimization oracle
`argmin_{x ∈ X} ⟪G,x⟫`. -/
def linearMinimizer (g : E) : E :=
  fs.lmo g

/-- The finite-sum LMO output is feasible. -/
theorem linearMinimizer_mem (g : E) : fs.linearMinimizer g ∈ fs.X :=
  fs.lmo.mem g

/-- The finite-sum LMO output minimizes the linear model over `X`. -/
theorem linearMinimizer_spec (g x : E) (hx : x ∈ fs.X) :
    ⟪g, fs.linearMinimizer g⟫_ℝ ≤ ⟪g, x⟫_ℝ :=
  fs.lmo.is_argmin g x hx

/-- Measurability of the finite-sum LMO selector used in adaptedness proofs. -/
theorem linearMinimizer_measurable : Measurable fs.linearMinimizer := by
  simpa [linearMinimizer] using fs.lmo.measurable

/-- Finite-sum component mini-batch estimator at `x`.

This remains available for auxiliary sampled finite-sum calculations. It is not
the Algorithm 7.12 epoch-refresh object: Algorithm 7.12 refreshes with the exact
full gradient `∇f(x)`. -/
noncomputable def batchGrad (x : E) (start count : ℕ) (ω : Ω) : E :=
  (count : ℝ)⁻¹ •
    Finset.sum (Finset.range count)
      (fun i => fs.gradFcomp (fs.sample (start + i) ω) x)

/-- Finite-sum recursive gradient-difference estimator inside an epoch.

PDF source: Algorithm 7.12 sets
`G_k = 1/b sum_{i in I_b} (grad f_i(x_k) - grad f_i(x_{k-1}))/(q_i m)
+ G_{k-1}` with `q_i = L_i/(m L)`. Here `componentCount` is the paper's
number of components `m`. -/
noncomputable def recursiveGrad
    (G_prev x_prev x_curr : E) (start count : ℕ) (ω : Ω) : E :=
  (count : ℝ)⁻¹ •
      Finset.sum (Finset.range count)
        (fun i =>
          let idx := fs.sample (start + i) ω
          ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
            (fs.gradFcomp idx x_curr - fs.gradFcomp idx x_prev)) +
    G_prev

/-- Finite-sum iterate update from Theorem 7.16:
`x_{k+1} = (1 - α_k)x_k + α_k y_k`. -/
noncomputable def iterUpdate (x G : E) (k : ℕ) : E :=
  SOptLib.conditionalGradientIterUpdate fs.α fs.linearMinimizer x G k

/-- The finite-sum affine update is feasible when its stepsize is in
`[0,1]` on the paper output window. -/
lemma iterUpdate_mem_of_alpha_le_one
    {x G : E} {k : ℕ} (hx : x ∈ fs.X)
    (hk : k ∈ Finset.Icc 1 fs.N)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1) :
    fs.iterUpdate x G k ∈ fs.X := by
  let y := fs.linearMinimizer G
  have hy : y ∈ fs.X := fs.linearMinimizer_mem G
  have hα_nonneg : 0 ≤ fs.α k := fs.hα_nonneg k
  have hα_le : fs.α k ≤ 1 := hα_le_one k hk
  have hseg : x + fs.α k • (y - x) ∈ fs.X :=
    fs.hX_convex.add_smul_sub_mem hx hy ⟨hα_nonneg, hα_le⟩
  convert hseg using 1
  simp only [iterUpdate, SOptLib.conditionalGradientIterUpdate, y]
  module

/-- Canonical finite-sum Algorithm 7.16 process, with estimator refresh and
recursive component-gradient updates built from the component-index stream.

PDF source: Algorithm 7.12 states that at an epoch refresh, `G_k = ∇f(x_k)`.
The sampled component-index stream is used only in the recursive update branch;
the epoch-start estimator is the exact finite-sum gradient `fs.gradf`. -/
noncomputable def process : ℕ → Ω → StochasticNonconvexCGState E :=
    SOptLib.finiteSumConditionalGradientProcess
      (fun x G s => ({ x := x, G := G, s := s } : StochasticNonconvexCGState E))
      (fun state : StochasticNonconvexCGState E => state.x)
      (fun state : StochasticNonconvexCGState E => state.G)
      (fun state : StochasticNonconvexCGState E => state.s)
      fs.x₁ fs.gradf fs.recursiveGrad fs.iterUpdate fs.T fs.b

/-- Finite-sum iterate at step k. -/
noncomputable def iterProcess (k : ℕ) : Ω → E :=
  fun ω => (fs.process k ω).x

/-- The finite-sum process stays feasible through the paper output window when
the displayed update coefficients are genuine convex-combination weights. -/
lemma process_mem_of_alpha_le_one
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1) :
    ∀ k ω, k ≤ fs.N → (fs.process k ω).x ∈ fs.X := by
  exact
    SOptLib.process_mem_of_update_mem fs.process
      (fun state : StochasticNonconvexCGState E => state.x) fs.N
      (fun ω => by simpa [process] using fs.hx₁_mem)
      (fun ω => by simpa [process] using fs.hx₁_mem)
      (fun {n : ℕ} {ω : Ω} (hk : n + 1 ∈ Finset.Icc 1 fs.N)
          (hprev : (fs.process (n + 1) ω).x ∈ fs.X) => by
        simpa [process] using
          fs.iterUpdate_mem_of_alpha_le_one
            (x := (fs.process (n + 1) ω).x)
            (G := (fs.process (n + 1) ω).G)
            (k := n + 1) hprev hk hα_le_one)

/-- The finite-sum iterate component of the process is feasible through the
paper output window under the explicit `α_k ≤ 1` boundary. -/
lemma iterProcess_mem_of_alpha_le_one
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1) :
    ∀ k ω, k ≤ fs.N → fs.iterProcess k ω ∈ fs.X := by
  exact
    SOptLib.iterateProcess_mem_of_process_mem fs.process fs.iterProcess
      (fun state : StochasticNonconvexCGState E => state.x)
      (fun k ω => by simp [iterProcess])
      (fun k ω hk => fs.process_mem_of_alpha_le_one hα_le_one k ω hk)

/-- Finite-sum estimator at step k. -/
noncomputable def estimatorProcess (k : ℕ) : Ω → E :=
  fun ω => (fs.process k ω).G

/-- The generated finite-sum iterate advances by the Algorithm 7.12 affine LMO
update after the initialized first step.

Candidate audit: searched target-file `iterProcess succ update iterUpdate` and
checked SOptLib recursive-process update helpers; those abstract helpers do not
encode Algorithm 7.12's two-initial-state indexing, while this identity is the
literal process-unfolding bridge needed before applying the smooth one-step
descent. -/
lemma iterProcess_succ_eq_iterUpdate
    {k : ℕ} (hk : 1 ≤ k) (ω : Ω) :
    fs.iterProcess (k + 1) ω =
      fs.iterUpdate (fs.iterProcess k ω) (fs.estimatorProcess k ω) k := by
  cases k with
  | zero =>
      omega
  | succ n =>
      simp [iterProcess, estimatorProcess,
        SOptLib.FiniteSumConditionalGradientSetup.process]

/-- Finite-sum estimator error `δ_k = G_k - ∇f(x_k)`. -/
noncomputable def deltaProcess (k : ℕ) : Ω → E :=
  fun ω => fs.estimatorProcess k ω - fs.gradf (fs.iterProcess k ω)

/-- Algorithm 7.12 exact epoch refresh: at the first step of any epoch, the
finite-sum estimator is the exact full gradient. -/
theorem estimatorProcess_globalIndex_one_eq_gradf (s : ℕ) :
    fs.estimatorProcess (fs.globalIndex s 1) =
      fun ω => fs.gradf (fs.iterProcess (fs.globalIndex s 1) ω) := by
  simpa [estimatorProcess, iterProcess, process, globalIndex] using
    (SOptLib.estimatorProcess_epochStart_eq_target
      (fun x G s => ({ x := x, G := G, s := s } : StochasticNonconvexCGState E))
      (fun state : StochasticNonconvexCGState E => state.x)
      (fun state : StochasticNonconvexCGState E => state.G)
      (fun state : StochasticNonconvexCGState E => state.s)
      (by intro x G s; rfl)
      (by intro x G s; rfl)
      fs.x₁ fs.gradf fs.recursiveGrad fs.iterUpdate fs.T fs.b fs.hT_pos s)

/-- Algorithm 7.12 exact epoch refresh makes the finite-sum estimator error
zero at `t = 1`. This is the base case required by Lemma 7.4, and it is false
for a sampled refresh. -/
theorem deltaProcess_globalIndex_one_eq_zero (s : ℕ) :
    fs.deltaProcess (fs.globalIndex s 1) = fun _ => 0 := by
  simpa [deltaProcess, SOptLib.estimatorResidualProcess] using
    (SOptLib.estimatorResidualProcess_epochStart_eq_zero
      fs.estimatorProcess fs.iterProcess fs.gradf (fs.globalIndex s 1)
      (fs.estimatorProcess_globalIndex_one_eq_gradf s))

/-- Away from an epoch refresh, the finite-sum estimator error obeys the
centered recursive update from Algorithm 7.12.

Candidate audit: checked the pre-searched SOptLib mini-batch centering lemma
`miniBatchAverage_sub_target_eq_average_residual`, the target-file stochastic
helpers `deltaProcessOfWellDefined_recursive_of_not_epochStart` and
`deltaProcessOfWellDefined_globalIndex_recursive`, and searched
`finite sum deltaProcess recursive globalIndex centered residual recursiveGrad`.
The SOptLib lemma starts after the process branch has been identified, and the
stochastic helpers use the stochastic sample stream rather than Algorithm
7.12's importance-weighted component-index estimator. This helper supplies the
literal finite-sum process identity used in Lan Lemma 7.4. -/
lemma deltaProcess_recursive_of_not_epochStart
    (k : ℕ) (hk : (k + 1) % fs.T ≠ 0) :
    ∀ ω,
      fs.deltaProcess (k + 2) ω =
        fs.deltaProcess (k + 1) ω +
          (fs.b : ℝ)⁻¹ •
            Finset.sum (Finset.range fs.b)
              (fun i =>
                let idx := fs.sample ((k + 1) * fs.b + i) ω
                ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp idx (fs.iterProcess (k + 2) ω) -
                    fs.gradFcomp idx (fs.iterProcess (k + 1) ω))) -
          (fs.gradf (fs.iterProcess (k + 2) ω) -
            fs.gradf (fs.iterProcess (k + 1) ω)) := by
  intro ω
  simp [deltaProcess, SOptLib.estimatorResidualProcess, estimatorProcess,
    iterProcess, process, recursiveGrad, hk, Nat.add_assoc]
  abel

/-- At a paper global index inside a finite-sum epoch, the estimator error
recurses by adding the centered fresh importance-sampled mini-batch increment.

Candidate audit: checked `deltaProcessOfWellDefined_globalIndex_recursive`,
`rawDeltaProcess_globalIndex_recursive`, `miniBatchAverage_sub_target_eq_average_residual`,
and searched `finite sum deltaProcess recursive globalIndex centered residual
recursiveGrad`. Existing helpers either apply to the stochastic process or to
generic centering after the branch is known; this is the finite-sum Algorithm
7.12 branch identification required by Lemma 7.4. -/
lemma deltaProcess_globalIndex_recursive
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T) :
    ∀ ω,
      fs.deltaProcess (fs.globalIndex s j) ω =
        fs.deltaProcess (fs.globalIndex s (j - 1)) ω +
          (fs.b : ℝ)⁻¹ •
            Finset.sum (Finset.range fs.b)
              (fun i =>
                let idx := fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω
                ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp idx (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.gradFcomp idx
                      (fs.iterProcess (fs.globalIndex s (j - 1)) ω))) -
          (fs.gradf (fs.iterProcess (fs.globalIndex s j) ω) -
            fs.gradf (fs.iterProcess (fs.globalIndex s (j - 1)) ω)) := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.globalIndex,
    SOptLib.global_index_def] using
    (SOptLib.global_index_recursive_of_succ
      (T := fs.T)
      (process := fs.deltaProcess)
      (increment := fun k k' ω =>
        (fs.b : ℝ)⁻¹ •
          Finset.sum (Finset.range fs.b)
            (fun i =>
              let idx := fs.sample (k * fs.b + i) ω
              ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
                (fs.gradFcomp idx (fs.iterProcess k' ω) -
                  fs.gradFcomp idx (fs.iterProcess k ω))))
      (correction := fun k k' ω =>
        fs.gradf (fs.iterProcess k' ω) - fs.gradf (fs.iterProcess k ω))
      (hrec := fs.deltaProcess_recursive_of_not_epochStart)
      (s := s) (j := j) hj2 hjT)

/-- Finite-sum global-index recursion in residual-average form.

This is the same Algorithm 7.12 recursive identity as
`deltaProcess_globalIndex_recursive`, but centered with
`miniBatchAverage_sub_target_eq_average_residual`, aligning with Lan Lemma 7.4's
variance proof and the SOptLib mini-batch second-moment API. -/
lemma deltaProcess_globalIndex_recursive_residual
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T) :
    ∀ ω,
      fs.deltaProcess (fs.globalIndex s j) ω =
        fs.deltaProcess (fs.globalIndex s (j - 1)) ω +
          (fs.b : ℝ)⁻¹ •
            Finset.sum (Finset.range fs.b)
              (fun i =>
                (let idx := fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω
                 ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp idx (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.gradFcomp idx
                      (fs.iterProcess (fs.globalIndex s (j - 1)) ω))) -
                (fs.gradf (fs.iterProcess (fs.globalIndex s j) ω) -
                  fs.gradf (fs.iterProcess (fs.globalIndex s (j - 1)) ω))) := by
  refine
    SOptLib.process_recursive_average_sub_target_eq_average_residual
      (I := Finset.range fs.b) (m := fs.b) (hmcard := by simp)
      (hmpos := Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hb_pos)
      (process := fs.deltaProcess)
      (prev := fs.globalIndex s (j - 1)) (curr := fs.globalIndex s j)
      (a := fun i ω =>
        let idx := fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω
        ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
          (fs.gradFcomp idx (fs.iterProcess (fs.globalIndex s j) ω) -
            fs.gradFcomp idx (fs.iterProcess (fs.globalIndex s (j - 1)) ω)))
      (target := fun ω =>
        fs.gradf (fs.iterProcess (fs.globalIndex s j) ω) -
          fs.gradf (fs.iterProcess (fs.globalIndex s (j - 1)) ω))
      ?_
  intro ω
  simpa using fs.deltaProcess_globalIndex_recursive s j hj2 hjT ω

/-- Existence of an attained finite-sum Wolfe-gap maximizer on compact `X`.

No SOptLib match: searched "wolfe gap maximizer compact maximum inner"; the
usable precedent is the target-file stochastic compact maximum proof, while
SOptLib hits are prox, finite-selector, or declared-max wrappers rather than
the literal Wolfe-gap maximum over compact `X`. -/
theorem wolfeGapMaximizer_exists (x : E) :
    ∃ y : fs.X,
      ∀ z : fs.X,
        ⟪fs.gradf x, x - (z : E)⟫_ℝ ≤ ⟪fs.gradf x, x - (y : E)⟫_ℝ := by
  classical
  have hcont : ContinuousOn (fun y : E => ⟪fs.gradf x, x - y⟫_ℝ) fs.X := by
    exact (continuous_const.inner (continuous_const.sub continuous_id)).continuousOn
  obtain ⟨y, hyX, hymax⟩ :=
    fs.hX_compact.exists_isMaxOn (show fs.X.Nonempty from ⟨fs.x₁, fs.hx₁_mem⟩) hcont
  refine ⟨⟨y, hyX⟩, ?_⟩
  intro z
  exact hymax z.property

/-- Canonical finite-sum Wolfe-gap maximizer. -/
noncomputable def wolfeGapMaximizer (x : E) : fs.X :=
  Classical.choose (fs.wolfeGapMaximizer_exists x)


local notation "setup.X" => fs.X
local notation "setup.gradf" => fs.gradf

/-- The canonical finite-sum Wolfe-gap maximizer attains the paper maximum. -/
theorem wolfeGap_spec (x : E) (z : setup.X) :
    ⟪setup.gradf x, x - (z : E)⟫_ℝ ≤ SOptLib.ConditionalGradient.wolfeGap setup.gradf fs.wolfeGapMaximizer x := by
  simpa [SOptLib.ConditionalGradient.wolfeGap, wolfeGapMaximizer] using
    (Classical.choose_spec (wolfeGapMaximizer_exists (fs := fs) x) z)

/-- The finite Wolfe gap can be evaluated with the measurable linear-minimizer
selector.

Candidate audit: considered SOptLib compact-maximum measurability/continuity
routes such as `exists_nonneg_norm_bound_of_isCompact_of_continuousOn` and
target-file `wolfeGapMaximizer`; those control compact extrema but do not avoid
the nonmeasurable `Classical.choose` maximizer. This helper aligns with Eq.
(7.3.3)/(Theorem 7.16 by using Algorithm 7.12's already-measurable LMO selector
to realize the same maximum value. -/
theorem wolfeGap_eq_linearMinimizer (x : E) (hx : x ∈ fs.X) :
    SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x =
      ⟪fs.gradf x, x - fs.linearMinimizer (fs.gradf x)⟫_ℝ := by
  classical
  let y : fs.X := ⟨fs.linearMinimizer (fs.gradf x),
    fs.linearMinimizer_mem (fs.gradf x)⟩
  let z : fs.X := fs.wolfeGapMaximizer x
  have hle_lmo : ⟪fs.gradf x, fs.linearMinimizer (fs.gradf x)⟫_ℝ ≤
      ⟪fs.gradf x, (z : E)⟫_ℝ :=
    fs.linearMinimizer_spec (fs.gradf x) (z : E) z.property
  have hgap_le :
      SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x ≤
        ⟪fs.gradf x, x - fs.linearMinimizer (fs.gradf x)⟫_ℝ := by
    have hvalue : SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x = ⟪fs.gradf x, x - (z : E)⟫_ℝ := by
      simp [SOptLib.ConditionalGradient.wolfeGap, z]
    rw [hvalue]
    rw [inner_sub_right, inner_sub_right]
    linarith
  have hlmo_le :
      ⟪fs.gradf x, x - fs.linearMinimizer (fs.gradf x)⟫_ℝ ≤
        SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x := by
    simpa [y] using wolfeGap_spec (fs := fs) x y
  exact le_antisymm hgap_le hlmo_le

/-- Finite-sum Wolfe-gap surrogate behind Eq. (7.4.2), with the compact
linear oracle realizing the model maximum.

Candidate audit: checked the target-file stochastic
`wolfeGap_surrogate_bound`, finite `wolfeGap_eq_linearMinimizer`, and finite
LMO lemmas. The stochastic theorem has the wrong setup type and finite
Theorem 7.16 has no separate `maxLinModel` object, so this is the finite
Algorithm 7.12 specialization in LMO form. -/
lemma finite_wolfeGap_surrogate_bound
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (x G : E) (hx : x ∈ fs.X) :
    SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x ≤
      ⟪G, x - fs.linearMinimizer G⟫_ℝ + ‖G - fs.gradf x‖ * fs.barDX := by
  exact
    SOptLib.ConditionalGradient.wolfeGap_le_linearMinimizer_model_plus_gradient_error_mul_diameter
      fs.gradf fs.wolfeGapMaximizer fs.linearMinimizer fs.linearMinimizer_spec
      fs.barDX x G (fun y hy => fs.barDX_bound x y hx hy)

/-- The finite Wolfe-gap value is measurable on the feasible carrier.

Candidate audit: searched `continuous compact maximum value`, `measurable inner
product`, and the SOptLib compact-bound candidates. The chosen route is not a
new primitive: it rewrites the compact maximum value through the existing
measurable LMO selector, matching Theorem 7.16's linear-oracle model without
requiring measurability of the nonunique Wolfe maximizer. -/
theorem wolfeGapOnX_measurable :
    Measurable (fun x : fs.X => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (x : E)) := by
  exact SOptLib.ConditionalGradient.wolfeGap_measurable_of_lmo
    (eval := fun x : fs.X => (x : E))
    (grad := fs.gradf) (maximizer := fs.wolfeGapMaximizer)
    (linearMinimizer := fs.linearMinimizer)
    (heval := measurable_subtype_coe)
    (hgrad := fs.gradfOnX_measurable)
    (hlmo := fs.linearMinimizer_measurable)
    (linearMinimizer_mem := fs.linearMinimizer_mem)
    (linearMinimizer_is_argmin := fs.linearMinimizer_spec)
    (hmax := fun x z => by
      simpa [SOptLib.ConditionalGradient.wolfeGap] using fs.wolfeGap_spec x z)

/-- The finite-sum gradient is uniformly bounded on the compact feasible set by
the start gradient plus the smoothness/diameter budget.

Candidate audit: searched SOptLib for compact continuous norm bounds and
Lipschitz packaging. A compact-existence bound is available, but Theorem 7.16
needs the source-shaped explicit estimate from `fs.gradf_smooth` and
`fs.barDX_bound`, so this route-local helper keeps the paper constants visible. -/
theorem gradf_norm_bound_on_X (x : E) (hx : x ∈ fs.X) :
    ‖fs.gradf x‖ ≤ ‖fs.gradf fs.x₁‖ + fs.L * fs.barDX := by
  exact norm_le_ref_norm_add_lipschitz_mul_diam fs.gradf x fs.x₁ fs.L fs.barDX
    (fs.gradf_smooth x fs.x₁ hx fs.hx₁_mem)
    (fs.barDX_bound x fs.x₁ hx fs.hx₁_mem)
    (le_of_lt fs.hL_pos)

/-- Finite-sum stepsize denominator `Σ_{k=1}^N α_k`. -/
noncomputable def alphaSum : ℝ :=
  Finset.sum (Finset.Icc 1 fs.N) fs.α

/-- Positive output indices for Theorem 7.16. -/
abbrev OutputTime : Type :=
  {k : ℕ // k ∈ Finset.Icc 1 fs.N}

/-- Theorem 7.16 output mass `α_k / Σ_j α_j`. -/
noncomputable def outputMass (R : fs.OutputTime) : ℝ :=
  fs.α R.1 / fs.alphaSum

/-- Nonnegativity of the finite-sum output masses once the displayed
normalizer is known positive. -/
theorem outputMass_nonneg_of_wellDefined
    (hR : 0 < fs.alphaSum) (R : fs.OutputTime) : 0 ≤ fs.outputMass R := by
  exact div_nonneg (fs.hα_nonneg R.1) (le_of_lt hR)

/-- Finite-sum output masses sum to one once the displayed normalizer is known
positive. -/
theorem outputMass_sum_one_of_wellDefined
    (hR : 0 < fs.alphaSum) : (∑ R : fs.OutputTime, fs.outputMass R) = 1 := by
  classical
  have hreindex : (∑ R : fs.OutputTime, fs.α R.1) = fs.alphaSum := by
    simpa [OutputTime, alphaSum] using
      (Finset.sum_attach (s := Finset.Icc 1 fs.N) (f := fs.α))
  unfold outputMass
  calc
    (∑ R : fs.OutputTime, fs.α R.1 / fs.alphaSum)
        = (∑ R : fs.OutputTime, fs.α R.1) / fs.alphaSum := by
          rw [← Finset.sum_div]
    _ = 1 := by
          rw [hreindex, div_self (ne_of_gt hR)]

/-- Finite-sum randomized output index law from Theorem 7.16 under the explicit
realization boundary that `Σ_{k=1}^N α_k > 0`.

Book citation: `algorithm_spec.output.math` states
`Prob{R=k}=α_k/Σα_k`; the current JSON does not separately assert positivity
of the denominator, so this PMF constructor is domain-aware rather than a
primitive setup field. -/
noncomputable def outputPMFOfWellDefined
    (hR : 0 < fs.alphaSum) : PMF fs.OutputTime :=
  PMF.ofFintypeOfReal fs.outputMass
    (fs.outputMass_nonneg_of_wellDefined hR)
    (fs.outputMass_sum_one_of_wellDefined hR)

/-- The finite-sum output PMF has the paper mass `α_k / Σα_j` on singleton
output times. -/
theorem outputPMFOfWellDefined_singleton_real
    (hR : 0 < fs.alphaSum) (R : fs.OutputTime) :
    (fs.outputPMFOfWellDefined hR).toMeasure.real ({R} : Set fs.OutputTime) =
      fs.outputMass R := by
  have hmass_nonneg : 0 ≤ fs.outputMass R :=
    fs.outputMass_nonneg_of_wellDefined hR R
  simp [Measure.real_def, outputPMFOfWellDefined, hmass_nonneg]

/-- Joint law of the finite-sum randomized output index and sample stream under
an explicit positive-normalizer proof. -/
noncomputable def outputLawOfWellDefined
    (hR : 0 < fs.alphaSum) : Measure (fs.OutputTime × Ω) :=
  (fs.outputPMFOfWellDefined hR).toMeasure.prod fs.P

/-- Finite-sum randomized output `x_R`. -/
noncomputable def randomOutput (q : fs.OutputTime × Ω) : E :=
  fs.iterProcess q.1.1 q.2

/-- Wolfe gap evaluated at the finite-sum randomized output. -/
noncomputable def randomOutputWolfeGap (q : fs.OutputTime × Ω) : ℝ :=
  SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.randomOutput q)

/-- Expected finite-sum Wolfe gap under the normalized output law, available
only after the output normalizer is proved positive. -/
noncomputable def expectedWolfeGapOfWellDefined
    (hR : 0 < fs.alphaSum) : ℝ :=
  ∫ q, fs.randomOutputWolfeGap q ∂fs.outputLawOfWellDefined hR

/-- Raw displayed finite-sum expectation from Theorem 7.16, written
as `(Σα_k)⁻¹ Σ α_k E[gap(x_k)]` without asserting the non-source PMF
normalizer positivity fact.

Candidate audit: SOptLib `expectedOutput_eq_weighted_sum_div` was considered
for expanding finite randomized outputs; this local expression records the
paper's exact `{1,...,N}` finite weighted display while keeping PMF
well-definedness in `outputPMFOfWellDefined`. -/
noncomputable def paperExpectedWolfeGapDisplayedExpression : ℝ :=
  SOptLib.normalizedWeightedExpectedCertificate
    (Finset.Icc 1 fs.N) fs.alphaSum fs.α fs.P
    (fun k ω =>
      SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω))

/-- Finite output-law expansion for Theorem 7.16, assuming the Wolfe-gap
fibers are integrable.

Candidate audit: chose the local selector-first specialization of SOptLib
`integral_selected_finite_index_prod_eq_sum_weights`; the SOptLib theorem is
sample-first, while `outputLawOfWellDefined` is `PMF.toMeasure.prod P`.
This lemma aligns with Algorithm 7.13's displayed randomized output law
`Prob{R=k}=α_k/Σα_j`. -/
theorem expectedWolfeGapOfWellDefined_eq_weighted_sum_of_integrable
    (hR : 0 < fs.alphaSum)
    (hgap_int :
      ∀ R : fs.OutputTime,
        Integrable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess R.1 ω)) fs.P) :
    fs.expectedWolfeGapOfWellDefined hR =
      (fs.alphaSum)⁻¹ *
        Finset.sum (Finset.Icc 1 fs.N)
          (fun k => fs.α k * ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P) := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  simpa [SOptLib.FiniteSumConditionalGradientSetup.expectedWolfeGapOfWellDefined,
    SOptLib.FiniteSumConditionalGradientSetup.outputLawOfWellDefined,
    SOptLib.FiniteSumConditionalGradientSetup.randomOutputWolfeGap,
    SOptLib.FiniteSumConditionalGradientSetup.randomOutput] using
    SOptLib.expectedSelectedOutput_eq_inv_mul_Icc_weighted_sum
      (N := fs.N) (weight := fs.α) (denom := fs.alphaSum) (P := fs.P)
      (ν := (fs.outputPMFOfWellDefined hR).toMeasure)
      (score := fun k ω =>
        SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
          (fs.iterProcess k ω))
      (by
        intro R
        simpa [SOptLib.FiniteSumConditionalGradientSetup.outputMass] using
          fs.outputPMFOfWellDefined_singleton_real hR R)
      (ne_of_gt hR)
      hgap_int

/-- The finite-sum paper index window `{2, ..., T}` in Theorem 7.16. -/
def innerAlphaWindow : Finset ℕ :=
  Finset.Icc 2 fs.T

/-- Nonemptiness condition for the finite-sum paper maximum over `{2, ..., T}`. -/
def InnerAlphaWindowWellDefined : Prop :=
  2 ≤ fs.T

/-- Conditional realization boundary for the finite-sum Theorem 7.16 output and
window.

The current JSON for Theorem 7.16 does not assert `2 ≤ T`, while the displayed
expression uses the nonempty maximum `max_{j=2,...,T}`. The finite-sum template
also uses a normalized randomized output, whose positive denominator is not
separately asserted by the current JSON. These are realization-only boundary
facts, not paper assumptions, and the non-paper empty-window convention remains
only in declarations named `Extension`. -/
structure Theorem716DomainBoundary : Prop extends
    WeightedEpochOutputBoundary fs.alphaSum fs.α fs.N fs.T fs.b

/-- The finite-sum realization contract exposes the positive normalizer needed
for the randomized output law in Theorem 7.16. -/
theorem alphaSum_pos_of_theorem716Boundary
    (h : fs.Theorem716DomainBoundary) :
    0 < fs.alphaSum :=
  h.outputNormalizerPositive

/-- The finite-sum realization contract supplies the upper stepsize bound
needed to keep the displayed affine update inside the convex feasible set. -/
theorem alpha_le_one_of_theorem716Boundary
    (h : fs.Theorem716DomainBoundary) :
    ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1 :=
  h.weight_le_one

/-- The finite-sum realization contract supplies the nonempty-window boundary
for Theorem 7.16's `max_{j=2,...,T}`. -/
theorem innerAlphaWindowWellDefined_of_theorem716Boundary
    (h : fs.Theorem716DomainBoundary) :
    fs.InnerAlphaWindowWellDefined := by
  simpa [InnerAlphaWindowWellDefined] using h.two_le_T

/-- The finite-sum realization contract supplies the source-stated Theorem 7.16
batch-size boundary `b ≥ T`. -/
theorem batch_ge_epoch_of_theorem716Boundary
    (h : fs.Theorem716DomainBoundary) :
    fs.T ≤ fs.b :=
  h.epoch_le_batch

/-- Paper-side nonemptiness of the finite-sum window `{2, ..., T}` is an
explicit domain obligation.

Book citation: `key_lemmas[2].statement_math` uses
`max_{j=2,...,T} α_{s,j}`; the JSON gives `b = T = sqrt(m)` but does not state
`2 ≤ T`. The object layer therefore keeps the finite maximum behind
`paperMaxInnerAlphaOfWellDefined` instead of asserting this condition for every
finite-sum setup. -/
theorem innerAlphaWindow_requires_two_le_T
    (hT : fs.InnerAlphaWindowWellDefined) : 2 ≤ fs.T := by
  simpa [InnerAlphaWindowWellDefined] using hT

/-- Nonemptiness of the finite-sum paper window `{2, ..., T}` in Theorem 7.16
once the paper-side domain condition `2 ≤ T` is available.

Candidate audit: checked SOptLib `finiteRunMaxValue` and Mathlib `Finset.sup'`;
the maximum primitive is available once the nonempty index-window bridge is
proved, but the bridge is paper-specific to Theorem 7.16's parameter regime. -/
theorem innerAlphaWindow_nonempty_of_wellDefined
    (hT : fs.InnerAlphaWindowWellDefined) : fs.innerAlphaWindow.Nonempty := by
  exact ⟨2, by simpa [innerAlphaWindow, InnerAlphaWindowWellDefined] using hT⟩

/-- Every displayed within-epoch stepsize used by the paper maximum
`max_{j=2,...,T} α_{s,j}` is nonnegative.

This is the finite-sum interface needed for the square-root/max absorption after
Eq. (7.4.6): unlike active generated steps, the displayed maximum can read
off-output entries in the final partial epoch. -/
theorem alpha_globalIndex_nonneg (s j : ℕ) :
    0 ≤ fs.α (fs.globalIndex s j) :=
  fs.hα_nonneg (fs.globalIndex s j)

/-- Source epoch-difference stepsize index for the increment
`x_{s,i} - x_{s,i-1}` in the proof of Theorem 7.16.

The generated Lean process stores iterates by the global output index `k` and
updates `x_k` to `x_{k+1}` with `α_k`. Therefore the displayed source
difference `x_{s,i} - x_{s,i-1}` is produced by the global update
`globalIndex s (i - 1)`. This is not a new algorithmic object; it is the
source epoch coordinate needed to read Eq. (7.4.6)'s
`∑_{i=2}^t α_{s,i}^2` in the generated process model. -/
noncomputable def epochDifferenceAlphaIndex (s i : ℕ) : ℕ :=
  fs.globalIndex s (i - 1)

/-- Nonnegativity of the source epoch-difference stepsize. -/
theorem epochDifferenceAlpha_nonneg (s i : ℕ) :
    0 ≤ fs.α (fs.epochDifferenceAlphaIndex s i) := by
  simpa [epochDifferenceAlphaIndex] using
    (SOptLib.epochDifferenceWeight_nonneg fs.α fs.hα_nonneg fs.globalIndex s i)

/-- The inner epoch maximum `max_{j=2,...,T} α_{s,j}` from Theorem 7.16.

Candidate audit: checked SOptLib `finiteRunMaxValue` and Mathlib `Finset.sup'`;
the definition uses the genuine finite maximum primitive under
an explicit nonempty-window proof. The book JSON does not state `2 ≤ T`, so
this finite-sum paper object is proof-parameterized rather than exported through an
unconditional empty-window convention. -/
noncomputable def paperMaxInnerAlphaOfWellDefined
    (hT : fs.InnerAlphaWindowWellDefined) (s : ℕ) : ℝ :=
  Finset.sup' fs.innerAlphaWindow
    (fs.innerAlphaWindow_nonempty_of_wellDefined hT)
    (fun j => fs.α (fs.globalIndex s j))

/-- Internal compatibility alias for proof scaffolding that previously used
`maxInnerAlpha`; it is now proof-parameterized like
`paperMaxInnerAlphaOfWellDefined`. -/
noncomputable def maxInnerAlphaOfWellDefined
    (hT : fs.InnerAlphaWindowWellDefined) (s : ℕ) : ℝ :=
  fs.paperMaxInnerAlphaOfWellDefined hT s

/-- Under the paper-domain proof `2 ≤ T`, `paperMaxInnerAlphaOfWellDefined` is
the genuine finite maximum over `{2, ..., T}`. -/
theorem paperMaxInnerAlphaOfWellDefined_eq_sup
    (s : ℕ) (hT : fs.InnerAlphaWindowWellDefined) :
    fs.paperMaxInnerAlphaOfWellDefined hT s =
      Finset.sup' fs.innerAlphaWindow
        (fs.innerAlphaWindow_nonempty_of_wellDefined hT)
        (fun j => fs.α (fs.globalIndex s j)) := by
  simp [paperMaxInnerAlphaOfWellDefined]

/-- The displayed epoch maximum is nonnegative because every displayed
stepsize in the schedule is nonnegative. -/
theorem paperMaxInnerAlphaOfWellDefined_nonneg
    (s : ℕ) (hT : fs.InnerAlphaWindowWellDefined) :
    0 ≤ fs.paperMaxInnerAlphaOfWellDefined hT s := by
  classical
  have h2 : 2 ∈ fs.innerAlphaWindow := by
    simpa [innerAlphaWindow, InnerAlphaWindowWellDefined] using hT
  calc
    0 ≤ fs.α (fs.globalIndex s 2) := fs.alpha_globalIndex_nonneg s 2
    _ ≤ fs.paperMaxInnerAlphaOfWellDefined hT s := by
      rw [fs.paperMaxInnerAlphaOfWellDefined_eq_sup s hT]
      exact Finset.le_sup'
        (s := fs.innerAlphaWindow)
        (f := fun j => fs.α (fs.globalIndex s j)) h2

/-- The Eq. (7.4.6) maximum over the source epoch-difference stepsizes.

The paper writes the variance substitution with
`∑_{i=2}^t α_{s,i}^2`; in the generated global-index process this displayed
`α_{s,i}` is the update coefficient that produced
`x_{s,i} - x_{s,i-1}`, namely `α (globalIndex s (i - 1))`. This maximum is the
one that controls the triangular variance and square-root absorption terms
without requiring any monotonicity assumption on the schedule. -/
noncomputable def paperMaxEpochDifferenceAlphaOfWellDefined
    (hT : fs.InnerAlphaWindowWellDefined) (s : ℕ) : ℝ :=
  Finset.sup' fs.innerAlphaWindow
    (fs.innerAlphaWindow_nonempty_of_wellDefined hT)
    (fun i => fs.α (fs.epochDifferenceAlphaIndex s i))

/-- The source epoch-difference maximum is the genuine finite maximum over the
paper window `{2, ..., T}`. -/
theorem paperMaxEpochDifferenceAlphaOfWellDefined_eq_sup
    (s : ℕ) (hT : fs.InnerAlphaWindowWellDefined) :
    fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s =
      Finset.sup' fs.innerAlphaWindow
        (fs.innerAlphaWindow_nonempty_of_wellDefined hT)
        (fun i => fs.α (fs.epochDifferenceAlphaIndex s i)) := by
  simp [paperMaxEpochDifferenceAlphaOfWellDefined]

/-- The source epoch-difference maximum is nonnegative. -/
theorem paperMaxEpochDifferenceAlphaOfWellDefined_nonneg
    (s : ℕ) (hT : fs.InnerAlphaWindowWellDefined) :
    0 ≤ fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s := by
  classical
  have h2 : 2 ∈ fs.innerAlphaWindow := by
    simpa [innerAlphaWindow, InnerAlphaWindowWellDefined] using hT
  calc
    0 ≤ fs.α (fs.epochDifferenceAlphaIndex s 2) :=
      fs.epochDifferenceAlpha_nonneg s 2
    _ ≤ fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s := by
      rw [fs.paperMaxEpochDifferenceAlphaOfWellDefined_eq_sup s hT]
      exact Finset.le_sup'
        (s := fs.innerAlphaWindow)
        (f := fun i => fs.α (fs.epochDifferenceAlphaIndex s i)) h2

/-- Any displayed source epoch-difference stepsize in `{2, ..., T}` is bounded
by the source epoch-difference maximum. -/
theorem epochDifferenceAlpha_le_paperMaxEpochDifferenceAlpha
    (s i : ℕ) (hT : fs.InnerAlphaWindowWellDefined)
    (hi : i ∈ fs.innerAlphaWindow) :
    fs.α (fs.epochDifferenceAlphaIndex s i) ≤
      fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s := by
  classical
  rw [fs.paperMaxEpochDifferenceAlphaOfWellDefined_eq_sup s hT]
  exact Finset.le_sup'
    (s := fs.innerAlphaWindow)
    (f := fun i => fs.α (fs.epochDifferenceAlphaIndex s i)) hi

/-- The internal compatibility alias agrees definitionally with the
domain-aware paper epoch maximum. -/
theorem maxInnerAlphaOfWellDefined_eq_paperMaxInnerAlphaOfWellDefined
    (hT : fs.InnerAlphaWindowWellDefined) (s : ℕ) :
    fs.maxInnerAlphaOfWellDefined hT s =
      fs.paperMaxInnerAlphaOfWellDefined hT s := by
  rfl

/-- Non-paper extension convention for the finite-sum `{2,...,T}` maximum: use
the genuine finite maximum when nonempty and `0` otherwise. Paper-facing
Theorem 7.16 uses the domain-aware `paperMaxInnerAlphaOfWellDefined`; this
helper is retained only for auxiliary compatibility lemmas about empty
windows. -/
noncomputable def paperMaxInnerAlphaEmptyWindowExtension (s : ℕ) : ℝ :=
  if h : fs.innerAlphaWindow.Nonempty then
    Finset.sup' fs.innerAlphaWindow h (fun j => fs.α (fs.globalIndex s j))
  else
    0

/-- Under `2 ≤ T`, the finite-sum extension convention agrees with the
genuine finite maximum over `{2,...,T}`. -/
theorem paperMaxInnerAlphaEmptyWindowExtension_eq_sup_of_wellDefined
    (s : ℕ) (hT : fs.InnerAlphaWindowWellDefined) :
    fs.paperMaxInnerAlphaEmptyWindowExtension s =
      Finset.sup' fs.innerAlphaWindow
        (fs.innerAlphaWindow_nonempty_of_wellDefined hT)
        (fun j => fs.α (fs.globalIndex s j)) := by
  classical
  have hne : fs.innerAlphaWindow.Nonempty :=
    fs.innerAlphaWindow_nonempty_of_wellDefined hT
  simp [paperMaxInnerAlphaEmptyWindowExtension, hne]

/-- The finite-sum empty-window extension keeps its domain bridge available for
proof code.

Theorem 7.16 writes `max_{j=2,...,T} α_{s,j}`, but the source JSON gives only
`b = T = sqrt(m)` and does not state `2 ≤ T`. The extension convention is kept
outside the paper-facing theorem path. -/
theorem paperMaxInnerAlphaEmptyWindowExtension_keeps_domain_boundary
    (hT : fs.InnerAlphaWindowWellDefined) : 2 ≤ fs.T :=
  fs.innerAlphaWindow_requires_two_le_T hT

/-- Non-paper extension convention for the finite-sum epoch-difference maximum:
use the genuine finite maximum when nonempty and `0` otherwise. -/
noncomputable def paperMaxEpochDifferenceAlphaEmptyWindowExtension (s : ℕ) : ℝ :=
  if h : fs.innerAlphaWindow.Nonempty then
    Finset.sup' fs.innerAlphaWindow h
      (fun i => fs.α (fs.epochDifferenceAlphaIndex s i))
  else
    0

/-- Under `2 ≤ T`, the epoch-difference extension convention agrees with the
genuine finite maximum over the source difference window `{2,...,T}`. -/
theorem paperMaxEpochDifferenceAlphaEmptyWindowExtension_eq_sup_of_wellDefined
    (s : ℕ) (hT : fs.InnerAlphaWindowWellDefined) :
    fs.paperMaxEpochDifferenceAlphaEmptyWindowExtension s =
      Finset.sup' fs.innerAlphaWindow
        (fs.innerAlphaWindow_nonempty_of_wellDefined hT)
        (fun i => fs.α (fs.epochDifferenceAlphaIndex s i)) := by
  classical
  have hne : fs.innerAlphaWindow.Nonempty :=
    fs.innerAlphaWindow_nonempty_of_wellDefined hT
  simp [paperMaxEpochDifferenceAlphaEmptyWindowExtension, hne]

/-- The generated-iteration epoch penalty in Theorem 7.16 using an arbitrary
epoch maximum. The last epoch is filtered to the remaining generated iterations.

No SOptLib match: searched `Finset filter sum nonnegative subset le` and scanned
SOptLib telescope/output-window helpers; those cover generic finite sums but not
Algorithm 7.12's epoch notation with a partial final epoch. This definition is the
paper's `∑_s ∑_j α_{s,j} max_j α_{s,j}` over the generated epoch coordinates. -/
noncomputable def theorem716EpochPenaltyWithMax (M : ℕ → ℝ) : ℝ :=
  SOptLib.epochPenaltyWithMax fs.S fs.activeEpochSteps fs.α fs.globalIndex M

/-- Literal displayed epoch penalty from the printed Theorem 7.16 statement,
using `max_{j=2,...,T} α_{s,j}` with the current epoch coordinate. -/
noncomputable def theorem716DisplayedEpochPenalty
    (hT : fs.InnerAlphaWindowWellDefined) : ℝ :=
  fs.theorem716EpochPenaltyWithMax
    (fun s => fs.paperMaxInnerAlphaOfWellDefined hT s)

/-- Paper-facing epoch penalty from the printed Theorem 7.16 statement.

The generated-process correction is intentionally kept under
`theorem716EpochDifferencePenalty`, because the proof after Eq. (7.4.6) uses the
shifted identity `x_{s,i} - x_{s,i-1} = α_{s,i}(y_{s,i}-x_{s,i})`, while
Algorithm 7.12's global process updates `x_k` to `x_{k+1}` with `α_k`. -/
noncomputable def theorem716EpochPenalty
    (hT : fs.InnerAlphaWindowWellDefined) : ℝ :=
  fs.theorem716DisplayedEpochPenalty hT

/-- Corrected generated-process epoch penalty using the epoch-difference
coordinate for the increment `x_{s,i} - x_{s,i-1}`.

This is the statement-corrected realization of the Theorem 7.16 scalar penalty:
the update producing `x_{s,i} - x_{s,i-1}` uses
`α (globalIndex s (i - 1))`, so this maximum is the one actually controlled by
Lemma 7.4 and the Algorithm 7.12 affine update. The literal printed maximum is
retained separately as `theorem716DisplayedEpochPenalty`. -/
noncomputable def theorem716EpochDifferencePenalty
    (hT : fs.InnerAlphaWindowWellDefined) : ℝ :=
  fs.theorem716EpochPenaltyWithMax
    (fun s => fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s)

/-- The literal displayed penalty is exactly the paper-facing penalty name. -/
theorem theorem716DisplayedEpochPenalty_eq_theorem716EpochPenalty
    (hT : fs.InnerAlphaWindowWellDefined) :
    fs.theorem716DisplayedEpochPenalty hT = fs.theorem716EpochPenalty hT := by
  rfl

/-- Guard required to recover the literal printed Theorem 7.16 penalty from the
generated-process epoch-difference proof.

This is not a setup assumption. It is the exact missing coordinate bridge:
the generated update proof controls the epoch-difference maximum, while the
printed theorem displays the current-coordinate maximum. The counterexamples
below show this proposition is not derivable from nonnegativity of the schedule
alone. -/
def Theorem716LiteralCoordinateBridge
    (hT : fs.InnerAlphaWindowWellDefined) : Prop :=
  fs.theorem716EpochDifferencePenalty hT ≤ fs.theorem716EpochPenalty hT

/-- Scalar counterexample showing why the literal current-coordinate maximum
does not control the generated-process epoch-difference maximum.

For `T = 2`, the displayed maximum sees only `α_2`, while the update producing
`x_{s,2} - x_{s,1}` uses the predecessor coefficient `α_1`. Thus the missing
comparison cannot be proved from nonnegativity alone. -/
theorem theorem716_literal_max_not_general_epoch_difference_bound :
    ∃ α : ℕ → ℝ, (∀ k, 0 ≤ α k) ∧
      Finset.sup' (Finset.Icc 2 2)
          (show (Finset.Icc 2 2).Nonempty from ⟨2, by simp⟩)
          (fun j => α j) <
        Finset.sup' (Finset.Icc 2 2)
          (show (Finset.Icc 2 2).Nonempty from ⟨2, by simp⟩)
          (fun i => α (i - 1)) := by
  refine ⟨fun k => if k = 1 then (1 : ℝ) else 0, ?_, ?_⟩
  · intro k
    by_cases hk : k = 1 <;> simp [hk]
  · norm_num

/-- Stronger scalar counterexample for the statement-correction protocol:
the whole epoch-penalty factor with the literal displayed current-coordinate
maximum need not dominate the generated-process epoch-difference penalty.

For `T = 2`, the displayed maximum sees only `α₂`, while the difference
coordinate for `x_{s,2} - x_{s,1}` sees `α₁`. -/
theorem theorem716_literal_penalty_not_general_epoch_difference_penalty :
    ∃ α : ℕ → ℝ, (∀ k, 0 ≤ α k) ∧
      (Finset.sum (Finset.Icc 1 2) (fun j => α j)) *
          Finset.sup' (Finset.Icc 2 2)
            (show (Finset.Icc 2 2).Nonempty from ⟨2, by simp⟩)
            (fun j => α j) <
        (Finset.sum (Finset.Icc 1 2) (fun j => α j)) *
          Finset.sup' (Finset.Icc 2 2)
            (show (Finset.Icc 2 2).Nonempty from ⟨2, by simp⟩)
            (fun i => α (i - 1)) := by
  refine ⟨fun k => if k = 1 then (1 : ℝ) else 0, ?_, ?_⟩
  · intro k
    by_cases hk : k = 1 <;> simp [hk]
  · norm_num

/-- The generated-iteration epoch penalty using the auxiliary empty-window
maximum convention, in the literal current coordinate. -/
noncomputable def theorem716EpochPenaltyEmptyWindowExtension : ℝ :=
  fs.theorem716EpochPenaltyWithMax
    (fun s => fs.paperMaxInnerAlphaEmptyWindowExtension s)

/-- Auxiliary empty-window version of the literal displayed current-coordinate
penalty. -/
noncomputable def theorem716DisplayedEpochPenaltyEmptyWindowExtension : ℝ :=
  fs.theorem716EpochPenaltyEmptyWindowExtension

/-- Compatibility name for the corrected epoch-difference empty-window penalty. -/
noncomputable def theorem716EpochDifferencePenaltyEmptyWindowExtension : ℝ :=
  fs.theorem716EpochPenaltyWithMax
    (fun s => fs.paperMaxEpochDifferenceAlphaEmptyWindowExtension s)

/-- Conditional finite-sum output mass from Theorem 7.16 under the realization
boundary for the normalized output law. -/
noncomputable def outputMassConditional
    (_h : fs.Theorem716DomainBoundary) (R : fs.OutputTime) : ℝ :=
  fs.α R.1 / fs.alphaSum

/-- Nonnegativity of the conditional finite-sum output masses. -/
theorem outputMassConditional_nonneg
    (h : fs.Theorem716DomainBoundary) (R : fs.OutputTime) :
    0 ≤ fs.outputMassConditional h R := by
  exact div_nonneg (fs.hα_nonneg R.1)
    (le_of_lt (fs.alphaSum_pos_of_theorem716Boundary h))

/-- Conditional finite-sum output masses sum to one under the realization
boundary. -/
theorem outputMassConditional_sum_one
    (h : fs.Theorem716DomainBoundary) :
    (∑ R : fs.OutputTime, fs.outputMassConditional h R) = 1 := by
  classical
  have hR : 0 < fs.alphaSum := fs.alphaSum_pos_of_theorem716Boundary h
  simpa [outputMassConditional, outputMass] using
    fs.outputMass_sum_one_of_wellDefined hR

/-- Conditional finite-sum randomized output law from Theorem 7.16. -/
noncomputable def outputPMFConditional
    (h : fs.Theorem716DomainBoundary) : PMF fs.OutputTime :=
  PMF.ofFintypeOfReal (fs.outputMassConditional h)
    (fs.outputMassConditional_nonneg h)
    (fs.outputMassConditional_sum_one h)

/-- Conditional finite-sum joint law of output index and sample stream. -/
noncomputable def outputLawConditional
    (h : fs.Theorem716DomainBoundary) : Measure (fs.OutputTime × Ω) :=
  (fs.outputPMFConditional h).toMeasure.prod fs.P

/-- Conditional finite-sum randomized output `x_R` for Theorem 7.16. -/
noncomputable def randomOutputConditional
    (_h : fs.Theorem716DomainBoundary) (q : fs.OutputTime × Ω) : E :=
  fs.iterProcess q.1.1 q.2

/-- Wolfe gap at the conditional finite-sum randomized output. -/
noncomputable def randomOutputWolfeGapConditional
    (h : fs.Theorem716DomainBoundary) (q : fs.OutputTime × Ω) : ℝ :=
  SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.randomOutputConditional h q)

/-- Conditional finite-sum expected Wolfe gap under Theorem 7.16's normalized
output law. -/
noncomputable def expectedWolfeGapConditional
    (h : fs.Theorem716DomainBoundary) : ℝ :=
  ∫ q, fs.randomOutputWolfeGapConditional h q ∂fs.outputLawConditional h

/-- Conditional finite-sum expected Wolfe gap under the single Theorem 7.16
realization boundary for the normalized output law and nonempty
`max_{j=2,...,T}` window. -/
noncomputable def paperExpectedWolfeGapConditional
    (h : fs.Theorem716DomainBoundary) : ℝ :=
  fs.expectedWolfeGapConditional h

/-- Finite-sum accumulated squared iterate differences inside an epoch. -/
noncomputable def epochDiffSum (s t : ℕ) : Ω → ℝ :=
  SOptLib.epochSquaredDifferenceSum fs.iterProcess fs.globalIndex s t

end SOptLib.FiniteSumConditionalGradientSetup

namespace StochasticNonconvexConditionalGradient

lemma rawIterProcess_measurable
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (k : ℕ) :
    Measurable (setup.rawIterProcess k) := by
  have hpair := setup.rawProcess_pair_measurable_recursive_cutoff k
  exact measurable_fst.comp (hpair.mono (setup.filtration.le _) le_rfl)

lemma rawEstimatorProcess_measurable
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (k : ℕ) :
    Measurable (setup.rawEstimatorProcess k) := by
  have hpair := setup.rawProcess_pair_measurable_recursive_cutoff k
  exact measurable_snd.comp (hpair.mono (setup.filtration.le _) le_rfl)

lemma rawIterProcess_succ_eq_rawIterUpdate
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {k : ℕ} (hk : 1 ≤ k) :
    ∀ ω,
      setup.rawIterProcess (k + 1) ω =
        setup.rawIterUpdate (setup.rawIterProcess k ω) (setup.rawEstimatorProcess k ω) k := by
  intro ω
  cases k with
  | zero =>
      omega
  | succ n =>
      simp [StochasticNonconvexConditionalGradientSetup.rawIterProcess,
        StochasticNonconvexConditionalGradientSetup.rawEstimatorProcess,
        StochasticNonconvexConditionalGradientSetup.rawProcess]

lemma rawIterDiff_sq_integrable
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hα_le_one : ∀ k, setup.rawPaperAlpha k ≤ 1) (k l : ℕ) :
    Integrable
      (fun ω => ‖setup.rawIterProcess k ω - setup.rawIterProcess l ω‖ ^ 2) setup.P := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  haveI : IsFiniteMeasure setup.P := by infer_instance
  have hk_meas : Measurable (setup.rawIterProcess k) := rawIterProcess_measurable setup k
  have hl_meas : Measurable (setup.rawIterProcess l) := rawIterProcess_measurable setup l
  have h_meas :
      AEStronglyMeasurable
        (fun ω => ‖setup.rawIterProcess k ω - setup.rawIterProcess l ω‖ ^ 2) setup.P :=
    ((hk_meas.sub hl_meas).norm.pow_const 2).aestronglyMeasurable
  refine Integrable.mono' (integrable_const (setup.barDX ^ 2)) h_meas ?_
  filter_upwards [] with ω
  have hkX : setup.rawIterProcess k ω ∈ setup.X := setup.rawIterProcess_mem hα_le_one k ω
  have hlX : setup.rawIterProcess l ω ∈ setup.X := setup.rawIterProcess_mem hα_le_one l ω
  have hnorm :
      ‖setup.rawIterProcess k ω - setup.rawIterProcess l ω‖ ≤ setup.barDX :=
    setup.barDX_bound _ _ hkX hlX
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  exact pow_le_pow_left₀ (norm_nonneg _) hnorm 2

lemma rawDeltaProcess_globalIndex_recursive
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T) :
    ∀ ω,
      setup.rawDeltaProcess (setup.globalIndex s j) ω =
        setup.rawDeltaProcess (setup.globalIndex s (j - 1)) ω +
          (setup.b : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.b)
              (fun i =>
                setup.gradF (setup.rawIterProcess (setup.globalIndex s j) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω) -
                  setup.gradF (setup.rawIterProcess (setup.globalIndex s (j - 1)) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω)) -
          (setup.gradf (setup.rawIterProcess (setup.globalIndex s j) ω) -
            setup.gradf (setup.rawIterProcess (setup.globalIndex s (j - 1)) ω)) := by
  simpa [StochasticNonconvexConditionalGradientSetup.globalIndex,
    SOptLib.global_index_def, Nat.add_assoc] using
    (SOptLib.recursive_delta_global_index_succ_eq
      (T := setup.T) (batchSize := setup.b) (sampleOffset := setup.N * setup.m)
      (delta := setup.rawDeltaProcess)
      (x := setup.rawIterProcess)
      (sample := setup.ξ)
      (oracleDiff := fun x y ξ => setup.gradF x ξ - setup.gradF y ξ)
      (meanDiff := fun x y => setup.gradf x - setup.gradf y)
      (hrec := by
        intro k hk ω
        simpa [Nat.add_assoc] using
          setup.rawDeltaProcess_recursive_of_not_epochStart k hk ω)
      (s := s) (j := j) hj2 hjT)

lemma integral_grad_diff_sqnorm_comp_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {wt : Ω → E × E} {idx : ℕ}
    (hwt_meas : Measurable wt)
    (hwt_mem : ∀ ω, (wt ω).1 ∈ setup.X ∧ (wt ω).2 ∈ setup.X)
    (hwt_sq : Integrable (fun ω => ‖(wt ω).1 - (wt ω).2‖ ^ 2) setup.P)
    (h_indep : IndepFun wt (setup.ξ idx) setup.P) :
    Integrable
        (fun ω =>
          ‖setup.gradF (wt ω).1 (setup.ξ idx ω) -
              setup.gradF (wt ω).2 (setup.ξ idx ω)‖ ^ 2) setup.P ∧
      ∫ ω,
          ‖setup.gradF (wt ω).1 (setup.ξ idx ω) -
              setup.gradF (wt ω).2 (setup.ξ idx ω)‖ ^ 2 ∂setup.P ≤
        setup.L ^ 2 * ∫ ω, ‖(wt ω).1 - (wt ω).2‖ ^ 2 ∂setup.P := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  let ν : Measure Ξ := setup.P.map (setup.ξ 0)
  have hξ0_meas : Measurable (setup.ξ 0) := setup.hξ_meas 0
  have hξidx_meas : Measurable (setup.ξ idx) := setup.hξ_meas idx
  haveI : IsProbabilityMeasure ν := by
    dsimp [ν]
    exact Measure.isProbabilityMeasure_map hξ0_meas.aemeasurable
  exact random_pair_oracle_difference_second_moment_le_of_indep
    (P := setup.P) (ν := ν) (Y := setup.ξ idx) (Zpair := wt)
    (G := setup.gradF)
    (feasible := {w : E × E | w.1 ∈ setup.X ∧ w.2 ∈ setup.X})
    (dist := fun x y : E => ‖x - y‖) (L := setup.L)
    (by
      exact
        (setup.hgradF_meas.comp
            ((measurable_fst.comp measurable_fst).prodMk measurable_snd)).sub
          (setup.hgradF_meas.comp
            ((measurable_snd.comp measurable_fst).prodMk measurable_snd)))
    (by
      exact (((measurable_fst.sub measurable_snd).norm.pow_const 2).const_mul (setup.L ^ 2)))
    (by
      exact (measurable_fst.sub measurable_snd).norm.pow_const 2)
    hwt_meas hξidx_meas h_indep
    (by simpa [ν] using (setup.hξ_ident idx).map_eq)
    (by
      have hpairs_meas :
          MeasurableSet {w : E × E | w.1 ∈ setup.X ∧ w.2 ∈ setup.X} := by
        exact (setup.hX_closed.measurableSet.preimage measurable_fst).inter
          (setup.hX_closed.measurableSet.preimage measurable_snd)
      rw [ae_map_iff hwt_meas.aemeasurable]
      · exact Filter.Eventually.of_forall hwt_mem
      · exact hpairs_meas)
    hwt_sq
    (by
      intro w hw
      have hnorm_sq_meas :
          Measurable (fun s : Ξ => ‖setup.gradF w.1 s - setup.gradF w.2 s‖ ^ 2) := by
        exact
          ((setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ξ => w.1).prodMk measurable_id)).sub
            (setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ξ => w.2).prodMk measurable_id))).norm.pow_const 2
      have hvec0_meas :
          Measurable (fun ω : Ω => setup.gradF w.1 (setup.ξ 0 ω) -
            setup.gradF w.2 (setup.ξ 0 ω)) := by
        exact
          (setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ω => w.1).prodMk hξ0_meas)).sub
            (setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ω => w.2).prodMk hξ0_meas))
      have hbase :=
        integrable_sq_norm_of_ae_bound
          (ν := setup.P)
          hvec0_meas.aestronglyMeasurable
          (by
            simpa using setup.stochasticGradient_smooth_ae w.1 w.2 hw.1 hw.2)
      exact
        (integrable_map_measure hnorm_sq_meas.aestronglyMeasurable hξ0_meas.aemeasurable).mpr
          hbase.1)
    (by
      intro w hw
      have hnorm_sq_meas :
          Measurable (fun s : Ξ => ‖setup.gradF w.1 s - setup.gradF w.2 s‖ ^ 2) := by
        exact
          ((setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ξ => w.1).prodMk measurable_id)).sub
            (setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ξ => w.2).prodMk measurable_id))).norm.pow_const 2
      have hvec0_meas :
          Measurable (fun ω : Ω => setup.gradF w.1 (setup.ξ 0 ω) -
            setup.gradF w.2 (setup.ξ 0 ω)) := by
        exact
          (setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ω => w.1).prodMk hξ0_meas)).sub
            (setup.hgradF_meas.comp
              ((measurable_const : Measurable fun _ : Ω => w.2).prodMk hξ0_meas))
      have hbase :=
        integrable_sq_norm_of_ae_bound
          (ν := setup.P)
          hvec0_meas.aestronglyMeasurable
          (by
            simpa using setup.stochasticGradient_smooth_ae w.1 w.2 hw.1 hw.2)
      have hmap :
          ∫ s, ‖setup.gradF w.1 s - setup.gradF w.2 s‖ ^ 2 ∂ν =
            ∫ ω, ‖setup.gradF w.1 (setup.ξ 0 ω) -
              setup.gradF w.2 (setup.ξ 0 ω)‖ ^ 2 ∂setup.P := by
        dsimp [ν]
        exact integral_map hξ0_meas.aemeasurable hnorm_sq_meas.aestronglyMeasurable
      calc
        ∫ s, ‖setup.gradF w.1 s - setup.gradF w.2 s‖ ^ 2 ∂ν =
            ∫ ω, ‖setup.gradF w.1 (setup.ξ 0 ω) -
              setup.gradF w.2 (setup.ξ 0 ω)‖ ^ 2 ∂setup.P := hmap
        _ ≤ (setup.L * ‖w.1 - w.2‖) ^ 2 := hbase.2
        _ = setup.L ^ 2 * ‖w.1 - w.2‖ ^ 2 := by ring)

lemma sq_norm_batch_average_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (f : Fin setup.b → E) :
    ‖(setup.b : ℝ)⁻¹ • ∑ i, f i‖ ^ 2 ≤
      (setup.b : ℝ)⁻¹ * ∑ i, ‖f i‖ ^ 2 := by
  simpa using
    norm_sq_inv_card_smul_sum_le_inv_card_mul_sum_norm_sq
      (s := (Finset.univ : Finset (Fin setup.b))) (f := f)
      (hs := by
        have hb_one : 1 ≤ setup.b := le_trans setup.hT_pos setup.hb_ge_T
        simpa using hb_one)

/-- Under the reference sample law `map (ξ 0) P`, the paired stochastic gradient
difference has the expected inner product with any fixed direction. This is the
local unbiasedness bridge needed in the centered cross-term argument. -/
lemma paired_difference_unbiased_under_map_measure
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (w : (E × E) × E)
    (hw1 : w.1.1 ∈ setup.X)
    (hw2 : w.1.2 ∈ setup.X) :
    ∫ s, ⟪w.2, setup.gradF w.1.1 s - setup.gradF w.1.2 s⟫_ℝ
      ∂Measure.map (setup.ξ 0) setup.P =
      ⟪w.2, setup.gradf w.1.1 - setup.gradf w.1.2⟫_ℝ := by
  exact
    integral_inner_oracle_difference_eq_inner_mean_difference
      (P := setup.P) (Y := setup.ξ 0) (G := setup.gradF) (target := setup.gradf)
      (x := w.1.1) (y := w.1.2) (d := w.2)
      (setup.hξ_meas 0)
      ((setup.hgradF_meas.comp
        ((measurable_const : Measurable fun _ : Ξ => w.1.1).prodMk measurable_id)).aestronglyMeasurable)
      ((setup.hgradF_meas.comp
        ((measurable_const : Measurable fun _ : Ξ => w.1.2).prodMk measurable_id)).aestronglyMeasurable)
      (setup.paperGradientOracle_integrable w.1.1 hw1)
      (setup.paperGradientOracle_integrable w.1.2 hw2)
      (setup.paperGradient_unbiased_integral w.1.1 hw1)
      (setup.paperGradient_unbiased_integral w.1.2 hw2)

lemma integral_inner_centered_grad_diff_eq_zero
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {cutoff idx : ℕ} {x y d : Ω → E}
    (hx : Measurable[setup.filtration cutoff] x)
    (hy : Measurable[setup.filtration cutoff] y)
    (hd : Measurable[setup.filtration idx] d)
    (hcut : cutoff ≤ idx)
    (hx_mem : ∀ ω, x ω ∈ setup.X)
    (hy_mem : ∀ ω, y ω ∈ setup.X)
    (hd_sq : Integrable (fun ω => ‖d ω‖ ^ 2) setup.P)
    (hxy_sq : Integrable (fun ω => ‖x ω - y ω‖ ^ 2) setup.P) :
    ∫ ω,
        ⟪d ω,
          (setup.gradF (x ω) (setup.ξ idx ω) -
              setup.gradF (y ω) (setup.ξ idx ω)) -
            (setup.gradf (x ω) - setup.gradf (y ω))⟫_ℝ ∂setup.P = 0 := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  let ν : Measure Ξ := setup.P.map (setup.ξ 0)
  haveI : IsProbabilityMeasure ν := by
    dsimp [ν]
    exact Measure.isProbabilityMeasure_map (setup.hξ_meas 0).aemeasurable
  have hx_idx : Measurable[setup.filtration idx] x :=
    hx.mono (setup.filtration.mono hcut) le_rfl
  have hy_idx : Measurable[setup.filtration idx] y :=
    hy.mono (setup.filtration.mono hcut) le_rfl
  have hx_meas : Measurable x := hx_idx.mono (setup.filtration.le idx) le_rfl
  have hy_meas : Measurable y := hy_idx.mono (setup.filtration.le idx) le_rfl
  have hd_meas : Measurable d := hd.mono (setup.filtration.le idx) le_rfl
  let wt : Ω → E × E := fun ω => (x ω, y ω)
  have hwt_meas_idx : Measurable[setup.filtration idx] wt := hx_idx.prodMk hy_idx
  have hwt_meas : Measurable wt := hwt_meas_idx.mono (setup.filtration.le idx) le_rfl
  have h_indep :
      IndepFun wt (setup.ξ idx) setup.P :=
    setup.indepFun_of_measurable_filtration (wt := wt) hwt_meas_idx le_rfl
  have h_sample :=
    integral_grad_diff_sqnorm_comp_le setup hwt_meas
      (fun ω => ⟨hx_mem ω, hy_mem ω⟩) hxy_sq h_indep
  let μfun : Ω → E := fun ω => setup.gradf (x ω) - setup.gradf (y ω)
  have hμ_meas : Measurable μfun := by
    dsimp [μfun]
    have hxX_meas : Measurable (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_meas
    have hyX_meas : Measurable (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_meas
    exact (setup.gradfOnX_measurable.comp hxX_meas).sub
      (setup.gradfOnX_measurable.comp hyX_meas)
  have hμ_sq :
      Integrable (fun ω => ‖μfun ω‖ ^ 2) setup.P := by
    refine Integrable.mono' (hxy_sq.const_mul (setup.L ^ 2))
      ((hμ_meas.norm.pow_const 2).aestronglyMeasurable) ?_
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have hgradf : ‖μfun ω‖ ≤ setup.L * ‖x ω - y ω‖ := by
      simpa [μfun] using setup.gradf_smooth (x ω) (y ω) (hx_mem ω) (hy_mem ω)
    nlinarith [hgradf, norm_nonneg (μfun ω), norm_nonneg (x ω - y ω)]
  let wt3 : Ω → (E × E) × E := fun ω => ((x ω, y ω), d ω)
  have hwt3_meas_idx : Measurable[setup.filtration idx] wt3 :=
    hwt_meas_idx.prodMk hd
  have hwt3_meas : Measurable wt3 :=
    hwt3_meas_idx.mono (setup.filtration.le idx) le_rfl
  have h_indep3 :
      IndepFun wt3 (setup.ξ idx) setup.P :=
    setup.indepFun_of_measurable_filtration (wt := wt3) hwt3_meas_idx le_rfl
  let gsample : Ω → E := fun ω =>
    setup.gradF (x ω) (setup.ξ idx ω) - setup.gradF (y ω) (setup.ξ idx ω)
  have hgsample_meas : Measurable gsample := by
    dsimp [gsample]
    exact (setup.hgradF_meas.comp (hx_meas.prodMk (setup.hξ_meas idx))).sub
      (setup.hgradF_meas.comp (hy_meas.prodMk (setup.hξ_meas idx)))
  have h_inner_sample_int :
      Integrable (fun ω => ⟪d ω, gsample ω⟫_ℝ) setup.P := by
    refine integrable_inner_of_sq_integrable hd_meas.aestronglyMeasurable
      hgsample_meas.aestronglyMeasurable hd_sq ?_
    simpa [gsample, wt] using h_sample.1
  have hwt3_mem_map :
      ∀ᵐ w ∂setup.P.map wt3, w.1.1 ∈ setup.X ∧ w.1.2 ∈ setup.X := by
    have hpairs_meas :
        MeasurableSet {w : (E × E) × E | w.1.1 ∈ setup.X ∧ w.1.2 ∈ setup.X} := by
      exact
        (setup.hX_closed.measurableSet.preimage
            (measurable_fst.comp (measurable_fst : Measurable fun w : (E × E) × E => w.1))).inter
          (setup.hX_closed.measurableSet.preimage
            (measurable_snd.comp (measurable_fst : Measurable fun w : (E × E) × E => w.1)))
    rw [ae_map_iff hwt3_meas.aemeasurable]
    · exact Filter.Eventually.of_forall fun ω => ⟨hx_mem ω, hy_mem ω⟩
    · exact hpairs_meas
  have h_unb :
      ∀ w : (E × E) × E, w.1.1 ∈ setup.X → w.1.2 ∈ setup.X →
        ∫ s, ⟪w.2, setup.gradF w.1.1 s - setup.gradF w.1.2 s⟫_ℝ ∂ν =
          ⟪w.2, setup.gradf w.1.1 - setup.gradf w.1.2⟫_ℝ := by
    intro w hw1 hw2
    simpa [ν] using paired_difference_unbiased_under_map_measure setup w hw1 hw2
  have h_rhs_aesm :
      AEStronglyMeasurable
        (fun w : (E × E) × E =>
          ⟪w.2, setup.gradf w.1.1 - setup.gradf w.1.2⟫_ℝ)
        (setup.P.map wt3) := by
    classical
    let A : Set ((E × E) × E) :=
      {w | w.1.1 ∈ setup.X ∧ w.1.2 ∈ setup.X}
    let φ : (E × E) × E → ℝ := fun w =>
      ⟪w.2, setup.gradf w.1.1 - setup.gradf w.1.2⟫_ℝ
    have hA_meas : MeasurableSet A := by
      dsimp [A]
      exact
        (setup.hX_closed.measurableSet.preimage
            (measurable_fst.comp
              (measurable_fst : Measurable fun w : (E × E) × E => w.1))).inter
          (setup.hX_closed.measurableSet.preimage
            (measurable_snd.comp
              (measurable_fst : Measurable fun w : (E × E) × E => w.1)))
    have hxA_meas :
        Measurable (fun z : A => (⟨(z : (E × E) × E).1.1, z.2.1⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact measurable_fst.comp (measurable_fst.comp measurable_subtype_coe)
    have hyA_meas :
        Measurable (fun z : A => (⟨(z : (E × E) × E).1.2, z.2.2⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact measurable_snd.comp (measurable_fst.comp measurable_subtype_coe)
    have hdA_meas : Measurable (fun z : A => (z : (E × E) × E).2) := by
      exact measurable_snd.comp measurable_subtype_coe
    have hφA_meas : Measurable (fun z : A => φ z) := by
      dsimp [φ]
      have hdiff : Measurable (fun z : A =>
          setup.gradf ((z : (E × E) × E).1.1) -
            setup.gradf ((z : (E × E) × E).1.2)) :=
        (setup.gradfOnX_measurable.comp hxA_meas).sub
          (setup.gradfOnX_measurable.comp hyA_meas)
      exact continuous_inner.measurable.comp (hdA_meas.prodMk hdiff)
    simpa [φ] using
      (aestronglyMeasurable_map_of_measurable_on_ae_support
        (P := setup.P) (wt := wt3) (A := A) (φ := φ)
        hA_meas hφA_meas.stronglyMeasurable (by simpa [A] using hwt3_mem_map))
  have hfixed_mean :
      ∀ᵐ w ∂setup.P.map wt3,
        ∫ s, ⟪w.2, setup.gradF w.1.1 s - setup.gradF w.1.2 s⟫_ℝ ∂ν =
          ⟪w.2, setup.gradf w.1.1 - setup.gradf w.1.2⟫_ℝ :=
    hwt3_mem_map.mono fun w hw => h_unb w hw.1 hw.2
  have h_inner_target_int :
      Integrable (fun ω => ⟪d ω, μfun ω⟫_ℝ) setup.P :=
    integrable_inner_of_sq_integrable hd_meas.aestronglyMeasurable
      hμ_meas.aestronglyMeasurable hd_sq hμ_sq
  exact
    integral_inner_centered_oracleDifference_eq_zero_of_indep_adapted
      (P := setup.P) (ν := ν) (sample := setup.ξ idx)
      (x := x) (y := y) (d := d) (G := setup.gradF) (target := setup.gradf)
      setup.hgradF_meas hwt3_meas (setup.hξ_meas idx) h_indep3
      (by simp [ν, (setup.hξ_ident idx).map_eq])
      (by simpa [gsample] using h_inner_sample_int)
      (by simpa [wt3] using h_rhs_aesm)
      (by simpa [μfun] using h_inner_target_int)
      (by simpa [wt3] using hfixed_mean)

lemma integral_centered_grad_diff_sqnorm_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {cutoff idx : ℕ} {x y : Ω → E}
    (hx : Measurable[setup.filtration cutoff] x)
    (hy : Measurable[setup.filtration cutoff] y)
    (hcut : cutoff ≤ idx)
    (hx_mem : ∀ ω, x ω ∈ setup.X)
    (hy_mem : ∀ ω, y ω ∈ setup.X)
    (hxy_sq : Integrable (fun ω => ‖x ω - y ω‖ ^ 2) setup.P) :
    Integrable
        (fun ω =>
          ‖(setup.gradF (x ω) (setup.ξ idx ω) -
                setup.gradF (y ω) (setup.ξ idx ω)) -
              (setup.gradf (x ω) - setup.gradf (y ω))‖ ^ 2) setup.P ∧
      ∫ ω,
          ‖(setup.gradF (x ω) (setup.ξ idx ω) -
                setup.gradF (y ω) (setup.ξ idx ω)) -
              (setup.gradf (x ω) - setup.gradf (y ω))‖ ^ 2 ∂setup.P ≤
        setup.L ^ 2 * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P := by
  haveI : IsProbabilityMeasure setup.P := setup.hP
  have hx_idx : Measurable[setup.filtration idx] x :=
    hx.mono (setup.filtration.mono hcut) le_rfl
  have hy_idx : Measurable[setup.filtration idx] y :=
    hy.mono (setup.filtration.mono hcut) le_rfl
  have hx_meas : Measurable x := hx_idx.mono (setup.filtration.le idx) le_rfl
  have hy_meas : Measurable y := hy_idx.mono (setup.filtration.le idx) le_rfl
  let wt : Ω → E × E := fun ω => (x ω, y ω)
  have hwt_meas_idx : Measurable[setup.filtration idx] wt := hx_idx.prodMk hy_idx
  have hwt_meas : Measurable wt := hwt_meas_idx.mono (setup.filtration.le idx) le_rfl
  have h_indep :
      IndepFun wt (setup.ξ idx) setup.P :=
    setup.indepFun_of_measurable_filtration (wt := wt) hwt_meas_idx le_rfl
  have h_sample :=
    integral_grad_diff_sqnorm_comp_le setup hwt_meas
      (fun ω => ⟨hx_mem ω, hy_mem ω⟩) hxy_sq h_indep
  let g : Ω → E := fun ω =>
    setup.gradF (x ω) (setup.ξ idx ω) - setup.gradF (y ω) (setup.ξ idx ω)
  let μfun : Ω → E := fun ω => setup.gradf (x ω) - setup.gradf (y ω)
  have hg_meas : Measurable g := by
    dsimp [g]
    exact (setup.hgradF_meas.comp (hx_meas.prodMk (setup.hξ_meas idx))).sub
      (setup.hgradF_meas.comp (hy_meas.prodMk (setup.hξ_meas idx)))
  have hμ_meas : Measurable μfun := by
    dsimp [μfun]
    have hxX_meas : Measurable (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_meas
    have hyX_meas : Measurable (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_meas
    exact (setup.gradfOnX_measurable.comp hxX_meas).sub
      (setup.gradfOnX_measurable.comp hyX_meas)
  have hμ_meas_idx : Measurable[setup.filtration idx] μfun := by
    dsimp [μfun]
    have hxX_idx : Measurable[setup.filtration idx]
        (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_idx
    have hyX_idx : Measurable[setup.filtration idx]
        (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_idx
    exact (setup.gradfOnX_measurable.comp hxX_idx).sub
      (setup.gradfOnX_measurable.comp hyX_idx)
  have hμ_sq :
      Integrable (fun ω => ‖μfun ω‖ ^ 2) setup.P := by
    refine Integrable.mono' (hxy_sq.const_mul (setup.L ^ 2))
      ((hμ_meas.norm.pow_const 2).aestronglyMeasurable) ?_
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have hgradf : ‖μfun ω‖ ≤ setup.L * ‖x ω - y ω‖ := by
      simpa [μfun] using setup.gradf_smooth (x ω) (y ω) (hx_mem ω) (hy_mem ω)
    nlinarith [hgradf, norm_nonneg (μfun ω), norm_nonneg (x ω - y ω)]
  have hcross_zero :
      ∫ ω, ⟪μfun ω, g ω - μfun ω⟫_ℝ ∂setup.P = 0 :=
    integral_inner_centered_grad_diff_eq_zero setup
      hx hy hμ_meas_idx hcut hx_mem hy_mem hμ_sq hxy_sq
  exact
    random_query_centered_gradient_difference_sq_integrable_and_integral_le
      (P := setup.P) (filt := setup.filtration) (sample := setup.ξ)
      (gradF := setup.gradF) (target := setup.gradf) (L := setup.L)
      (cutoff := cutoff) (idx := idx) (x := x) (y := y)
      hx hy hcut (setup.hξ_meas idx) setup.hgradF_meas hμ_meas.aestronglyMeasurable hxy_sq
      (by simpa [g, wt] using h_sample)
      (Filter.Eventually.of_forall fun ω => by
        simpa [μfun] using setup.gradf_smooth (x ω) (y ω) (hx_mem ω) (hy_mem ω))
      (by simpa [g, μfun] using hcross_zero)

/-- A single fresh recursive paired-gradient difference has the Lan Lemma 7.5
diagonal second-moment bound.

Candidate audit: checked SOptLib mini-batch variance lemmas
`miniBatchResidual_sum_secondMoment_le_card_mul_variance` and
`miniBatchAverage_secondMoment_le_variance_div_card`, plus the target-file
`integral_centered_grad_diff_sqnorm_le`. The mini-batch lemmas start after the
per-coordinate variance bound is available; this helper specializes the local
centered-gradient-difference estimate to Algorithm 7.13's recursive index
schedule and well-defined iterates. -/
lemma recursive_centered_grad_diff_sqnorm_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j i : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T) :
    Integrable
        (fun ω =>
          ‖(setup.gradF
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                (setup.ξ
                  (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω) -
              setup.gradF
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
                (setup.ξ
                  (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω)) -
            (setup.gradf
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
              setup.gradf
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω))‖ ^ 2)
          setup.P ∧
      ∫ ω,
          ‖(setup.gradF
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                (setup.ξ
                  (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω) -
              setup.gradF
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
                (setup.ξ
                  (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω)) -
            (setup.gradf
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
              setup.gradf
                (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω))‖ ^ 2
            ∂setup.P ≤
        setup.L ^ 2 *
          ∫ ω,
            ‖setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω -
              setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2
            ∂setup.P := by
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let idx := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hcut : cutoff ≤ idx := by
    dsimp [cutoff, idx]
    omega
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hxy_sq :
      Integrable (fun ω => ‖x ω - y ω‖ ^ 2) setup.P := by
    simpa [x, y] using
      setup.iterProcessOfWellDefined_diff_sq_integrable
        hαwf hα_le_one (setup.globalIndex s j) (setup.globalIndex s (j - 1))
  have hx_idx : Measurable[setup.filtration idx] x :=
    hx.mono (setup.filtration.mono hcut) le_rfl
  have hy_idx : Measurable[setup.filtration idx] y :=
    hy.mono (setup.filtration.mono hcut) le_rfl
  have hx_meas : Measurable x := hx_idx.mono (setup.filtration.le idx) le_rfl
  have hy_meas : Measurable y := hy_idx.mono (setup.filtration.le idx) le_rfl
  let wt : Ω → E × E := fun ω => (x ω, y ω)
  have hwt_meas_idx : Measurable[setup.filtration idx] wt := hx_idx.prodMk hy_idx
  have hwt_meas : Measurable wt := hwt_meas_idx.mono (setup.filtration.le idx) le_rfl
  have h_indep :
      IndepFun wt (setup.ξ idx) setup.P :=
    setup.indepFun_of_measurable_filtration (wt := wt) hwt_meas_idx le_rfl
  have h_sample :=
    integral_grad_diff_sqnorm_comp_le setup hwt_meas
      (fun ω => ⟨hx_mem ω, hy_mem ω⟩) hxy_sq h_indep
  let g : Ω → E := fun ω =>
    setup.gradF (x ω) (setup.ξ idx ω) - setup.gradF (y ω) (setup.ξ idx ω)
  let μfun : Ω → E := fun ω => setup.gradf (x ω) - setup.gradf (y ω)
  have hμ_meas : Measurable μfun := by
    dsimp [μfun]
    have hxX_meas : Measurable (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_meas
    have hyX_meas : Measurable (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_meas
    exact (setup.gradfOnX_measurable.comp hxX_meas).sub
      (setup.gradfOnX_measurable.comp hyX_meas)
  have hμ_meas_idx : Measurable[setup.filtration idx] μfun := by
    dsimp [μfun]
    have hxX_idx : Measurable[setup.filtration idx]
        (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_idx
    have hyX_idx : Measurable[setup.filtration idx]
        (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_idx
    exact (setup.gradfOnX_measurable.comp hxX_idx).sub
      (setup.gradfOnX_measurable.comp hyX_idx)
  have hμ_sq :
      Integrable (fun ω => ‖μfun ω‖ ^ 2) setup.P := by
    refine Integrable.mono' (hxy_sq.const_mul (setup.L ^ 2))
      ((hμ_meas.norm.pow_const 2).aestronglyMeasurable) ?_
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have hgradf : ‖μfun ω‖ ≤ setup.L * ‖x ω - y ω‖ := by
      simpa [μfun] using setup.gradf_smooth (x ω) (y ω) (hx_mem ω) (hy_mem ω)
    nlinarith [hgradf, norm_nonneg (μfun ω), norm_nonneg (x ω - y ω)]
  have hcross_zero :
      ∫ ω, ⟪μfun ω, g ω - μfun ω⟫_ℝ ∂setup.P = 0 :=
    integral_inner_centered_grad_diff_eq_zero setup
      hx hy hμ_meas_idx hcut hx_mem hy_mem hμ_sq hxy_sq
  simpa [x, y, idx, g, μfun] using
    random_query_centered_gradient_difference_sq_integrable_and_integral_le
      (P := setup.P) (filt := setup.filtration) (sample := setup.ξ)
      (gradF := setup.gradF) (target := setup.gradf) (L := setup.L)
      (cutoff := cutoff) (idx := idx) (x := x) (y := y)
      hx hy hcut (setup.hξ_meas idx) setup.hgradF_meas hμ_meas.aestronglyMeasurable hxy_sq
      (by simpa [g, wt] using h_sample)
      (Filter.Eventually.of_forall fun ω => by
        simpa [μfun] using setup.gradf_smooth (x ω) (y ω) (hx_mem ω) (hy_mem ω))
      (by simpa [g, μfun] using hcross_zero)

/-- Distinct fresh coordinates in one recursive mini-batch have zero covariance.

Candidate audit: `integral_inner_centered_grad_diff_eq_zero` is the matching
local martingale-cancellation lemma; SOptLib
`miniBatchResidual_sum_secondMoment_le_card_mul_variance` consumes this as its
off-diagonal hypothesis, and `oracleResidual_inner_integral_eq_zero_of_indep_distinct_samples`
is for fixed-query residuals rather than the paired gradient-difference
increment in Lan Lemma 7.5. This helper only aligns the local cancellation
lemma with Algorithm 7.13's recursive index schedule. -/
lemma recursive_centered_grad_diff_cross_zero
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j i l : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T)
    (hi : i ∈ Finset.range setup.b) (hl : l ∈ Finset.range setup.b)
    (hil : i ≠ l) :
    ∫ ω,
        ⟪(setup.gradF
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
              (setup.ξ
                (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω) -
            setup.gradF
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
              (setup.ξ
                (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω)) -
          (setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
            setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)),
          (setup.gradF
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
              (setup.ξ
                (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + l) ω) -
            setup.gradF
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
              (setup.ξ
                (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + l) ω)) -
          (setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
            setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω))⟫_ℝ
      ∂setup.P = 0 := by
  classical
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let eps : ℕ → Ω → E := fun q ω =>
    (setup.gradF (x ω) (setup.ξ (cutoff + q) ω) -
        setup.gradF (y ω) (setup.ξ (cutoff + q) ω)) -
      (setup.gradf (x ω) - setup.gradf (y ω))
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hxy_sq :
      Integrable (fun ω => ‖x ω - y ω‖ ^ 2) setup.P := by
    simpa [x, y] using
      setup.iterProcessOfWellDefined_diff_sq_integrable
        hαwf hα_le_one (setup.globalIndex s j) (setup.globalIndex s (j - 1))
  have hdiag_i :
      Integrable (fun ω => ‖eps i ω‖ ^ 2) setup.P ∧
        ∫ ω, ‖eps i ω‖ ^ 2 ∂setup.P ≤
          setup.L ^ 2 * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P := by
    simpa [eps, x, y, cutoff, Nat.add_assoc] using
      recursive_centered_grad_diff_sqnorm_le setup hαwf hα_le_one s j i hj2 hjT
  have hdiag_l :
      Integrable (fun ω => ‖eps l ω‖ ^ 2) setup.P ∧
        ∫ ω, ‖eps l ω‖ ^ 2 ∂setup.P ≤
          setup.L ^ 2 * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P := by
    simpa [eps, x, y, cutoff, Nat.add_assoc] using
      recursive_centered_grad_diff_sqnorm_le setup hαwf hα_le_one s j l hj2 hjT
  have hmeas_eps_le
      {a b : ℕ} (hab : a < b) :
      Measurable[setup.filtration (cutoff + b)] (eps a) := by
    have hcut_a : cutoff ≤ cutoff + a := by omega
    have hx_b : Measurable[setup.filtration (cutoff + b)] x :=
      hx.mono (setup.filtration.mono (by omega)) le_rfl
    have hy_b : Measurable[setup.filtration (cutoff + b)] y :=
      hy.mono (setup.filtration.mono (by omega)) le_rfl
    have hξ_a : Measurable[setup.filtration (cutoff + b)] (setup.ξ (cutoff + a)) :=
      setup.measurable_xi_of_lt (by omega)
    have hgrad_x :
        Measurable[setup.filtration (cutoff + b)]
          (fun ω => setup.gradF (x ω) (setup.ξ (cutoff + a) ω)) :=
      setup.hgradF_meas.comp (hx_b.prodMk hξ_a)
    have hgrad_y :
        Measurable[setup.filtration (cutoff + b)]
          (fun ω => setup.gradF (y ω) (setup.ξ (cutoff + a) ω)) :=
      setup.hgradF_meas.comp (hy_b.prodMk hξ_a)
    have hxX_b :
        Measurable[setup.filtration (cutoff + b)]
          (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_b
    have hyX_b :
        Measurable[setup.filtration (cutoff + b)]
          (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_b
    have hμ_b :
        Measurable[setup.filtration (cutoff + b)]
          (fun ω => setup.gradf (x ω) - setup.gradf (y ω)) :=
      (setup.gradfOnX_measurable.comp hxX_b).sub
        (setup.gradfOnX_measurable.comp hyX_b)
    dsimp [eps]
    exact (hgrad_x.sub hgrad_y).sub hμ_b
  simpa [eps, x, y, cutoff, Nat.add_assoc] using
    centeredOracleResidual_inner_integral_eq_zero_of_distinct_fresh
      (P := setup.P) (filt := setup.filtration) (eps := eps)
      (cutoff := cutoff) (i := i) (l := l)
      hmeas_eps_le
      (fun {a b} hab hmeas_a hdiag_a =>
        integral_inner_centered_grad_diff_eq_zero setup
          (cutoff := cutoff) (idx := cutoff + b) (x := x) (y := y) (d := eps a)
          hx hy hmeas_a (by omega) hx_mem hy_mem hdiag_a hxy_sq)
      hdiag_i.1 hdiag_l.1 hil

/-- The averaged fresh recursive increment has the exact mini-batch
second-moment scale `L² / b`.

Candidate audit: this is precisely the point where SOptLib
`miniBatchResidual_sum_secondMoment_le_card_mul_variance` and
`miniBatchAverage_secondMoment_le_variance_div_card` apply. The only
paper-local ingredient not in SOptLib is the recursive off-diagonal
cancellation supplied by `recursive_centered_grad_diff_cross_zero`, matching
Lan Lemma 7.5's reduction to Lemma 6.11. -/
lemma recursive_minibatch_increment_second_moment_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T) :
    ∫ ω,
        ‖(setup.b : ℝ)⁻¹ •
          Finset.sum (Finset.range setup.b)
            (fun i =>
              (setup.gradF
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                      ω) -
                  setup.gradF
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                      ω)) -
                (setup.gradf
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
                  setup.gradf
                    (setup.iterProcessOfWellDefined hαwf
                      (setup.globalIndex s (j - 1)) ω)))‖ ^ 2 ∂setup.P ≤
      setup.L ^ 2 / setup.b *
        ∫ ω,
          ‖setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω -
            setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2
          ∂setup.P := by
  classical
  let I : Finset ℕ := Finset.range setup.b
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let eps : ℕ → Ω → E := fun i ω =>
    (setup.gradF (x ω) (setup.ξ (cutoff + i) ω) -
        setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) -
      (setup.gradf (x ω) - setup.gradf (y ω))
  let σ2 : ℝ := setup.L ^ 2 * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hmeas : ∀ i ∈ I, AEStronglyMeasurable (eps i) setup.P := by
    intro i hi
    have hx_b : Measurable[setup.filtration (cutoff + setup.b)] x :=
      hx.mono (setup.filtration.mono (by omega)) le_rfl
    have hy_b : Measurable[setup.filtration (cutoff + setup.b)] y :=
      hy.mono (setup.filtration.mono (by omega)) le_rfl
    have hξ_i :
        Measurable[setup.filtration (cutoff + setup.b)] (setup.ξ (cutoff + i)) :=
      setup.measurable_xi_of_lt (by
        have hi_lt : i < setup.b := by simpa [I] using Finset.mem_range.mp hi
        omega)
    have hgrad_x :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradF (x ω) (setup.ξ (cutoff + i) ω)) :=
      setup.hgradF_meas.comp (hx_b.prodMk hξ_i)
    have hgrad_y :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) :=
      setup.hgradF_meas.comp (hy_b.prodMk hξ_i)
    have hxX_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_b
    have hyX_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_b
    have hμ_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradf (x ω) - setup.gradf (y ω)) :=
      (setup.gradfOnX_measurable.comp hxX_b).sub
        (setup.gradfOnX_measurable.comp hyX_b)
    have heps :
        Measurable[setup.filtration (cutoff + setup.b)] (eps i) := by
      dsimp [eps]
      exact (hgrad_x.sub hgrad_y).sub hμ_b
    exact (heps.mono (setup.filtration.le (cutoff + setup.b)) le_rfl).aestronglyMeasurable
  have hdiag :
      ∀ i ∈ I,
        Integrable (fun ω => ‖eps i ω‖ ^ 2) setup.P ∧
          ∫ ω, ‖eps i ω‖ ^ 2 ∂setup.P ≤ σ2 := by
    intro i hi
    have h :=
      recursive_centered_grad_diff_sqnorm_le setup hαwf hα_le_one s j i hj2 hjT
    simpa [eps, x, y, cutoff, σ2, Nat.add_assoc] using h
  have hcross :
      ∀ i ∈ I, ∀ l ∈ I, i ≠ l →
        ∫ ω, ⟪eps i ω, eps l ω⟫_ℝ ∂setup.P = 0 := by
    intro i hi l hl hil
    simpa [eps, x, y, cutoff, Nat.add_assoc] using
      recursive_centered_grad_diff_cross_zero setup hαwf hα_le_one
        s j i l hj2 hjT (by simpa [I] using hi) (by simpa [I] using hl) hil
  have hbpos : 0 < setup.b := by
    have hb_one : 1 ≤ setup.b := le_trans setup.hT_pos setup.hb_ge_T
    omega
  have hright :
      σ2 / (setup.b : ℝ) =
        setup.L ^ 2 / setup.b * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P := by
    dsimp [σ2]
    ring
  simpa [I, eps, x, y, cutoff, Nat.add_assoc, hright] using
    centeredMiniBatchAverage_secondMoment_le_variance_div_card_of_cross_zero
      setup.P I setup.b eps σ2 hbpos (by simp [I]) hmeas hdiag hcross

/-- The averaged fresh recursive mini-batch increment is square-integrable.

Candidate audit: SOptLib `integrable_sq_norm_centeredMiniBatchAverage` is the
matching finite-average L2 closure lemma. The target-file
`recursive_centered_grad_diff_sqnorm_le` supplies the Algorithm 7.13
coordinate square-integrability needed to specialize that abstract lemma to
Lan Lemma 7.5's recursive increment. -/
lemma recursive_minibatch_increment_sq_integrable
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T) :
    Integrable
      (fun ω =>
        ‖(setup.b : ℝ)⁻¹ •
          Finset.sum (Finset.range setup.b)
            (fun i =>
              (setup.gradF
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                      ω) -
                  setup.gradF
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
                    (setup.ξ
                      (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                      ω)) -
                (setup.gradf
                    (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
                  setup.gradf
                    (setup.iterProcessOfWellDefined hαwf
                      (setup.globalIndex s (j - 1)) ω)))‖ ^ 2)
      setup.P := by
  classical
  let I : Finset ℕ := Finset.range setup.b
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let eps : ℕ → Ω → E := fun i ω =>
    (setup.gradF (x ω) (setup.ξ (cutoff + i) ω) -
        setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) -
      (setup.gradf (x ω) - setup.gradf (y ω))
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hmeas : ∀ i ∈ I, AEStronglyMeasurable (eps i) setup.P := by
    intro i hi
    have hx_b : Measurable[setup.filtration (cutoff + setup.b)] x :=
      hx.mono (setup.filtration.mono (by omega)) le_rfl
    have hy_b : Measurable[setup.filtration (cutoff + setup.b)] y :=
      hy.mono (setup.filtration.mono (by omega)) le_rfl
    have hξ_i :
        Measurable[setup.filtration (cutoff + setup.b)] (setup.ξ (cutoff + i)) :=
      setup.measurable_xi_of_lt (by
        have hi_lt : i < setup.b := by simpa [I] using Finset.mem_range.mp hi
        omega)
    have hgrad_x :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradF (x ω) (setup.ξ (cutoff + i) ω)) :=
      setup.hgradF_meas.comp (hx_b.prodMk hξ_i)
    have hgrad_y :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) :=
      setup.hgradF_meas.comp (hy_b.prodMk hξ_i)
    have hxX_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_b
    have hyX_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_b
    have hμ_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradf (x ω) - setup.gradf (y ω)) :=
      (setup.gradfOnX_measurable.comp hxX_b).sub
        (setup.gradfOnX_measurable.comp hyX_b)
    have heps :
        Measurable[setup.filtration (cutoff + setup.b)] (eps i) := by
      dsimp [eps]
      exact (hgrad_x.sub hgrad_y).sub hμ_b
    exact (heps.mono (setup.filtration.le (cutoff + setup.b)) le_rfl).aestronglyMeasurable
  have hdiag :
      ∀ i ∈ I, Integrable (fun ω => ‖eps i ω‖ ^ 2) setup.P := by
    intro i hi
    have h :=
      recursive_centered_grad_diff_sqnorm_le setup hαwf hα_le_one s j i hj2 hjT
    exact (by
      simpa [eps, x, y, cutoff, Nat.add_assoc] using h.1)
  have havg_int :
      Integrable
        (fun ω =>
          ‖(setup.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)‖ ^ 2)
        setup.P :=
    integrable_sq_norm_centeredMiniBatchAverage setup.P I setup.b eps
      (fun ω => (setup.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω))
      hmeas hdiag rfl
  simpa [I, eps, x, y, cutoff, Nat.add_assoc] using havg_int

/-- The previous estimator error is orthogonal in expectation to one fresh
recursive centered gradient-difference coordinate.

Candidate audit: `integral_inner_centered_grad_diff_eq_zero` is again the
matching local cancellation primitive; SOptLib martingale inner-product lemmas
are more abstract conditional-expectation forms and do not know Algorithm
7.13's paired-gradient residual. This helper supplies the cross term needed
when expanding the recursive identity in Lan Lemma 7.5, with the previous
delta square-integrability premise that the strengthened induction must carry. -/
lemma recursive_delta_prev_centered_grad_diff_cross_zero
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j i : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T)
    (hprev_sq :
      Integrable
        (fun ω =>
          ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2)
        setup.P) :
    ∫ ω,
        ⟪setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
          (setup.gradF
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
              (setup.ξ
                (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω) -
            setup.gradF
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)
              (setup.ξ
                (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i) ω)) -
          (setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
            setup.gradf
              (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω))⟫_ℝ
      ∂setup.P = 0 := by
  classical
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let idx := cutoff + i
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let d : Ω → E :=
    setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hGprev : Measurable[setup.filtration cutoff]
      (setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))) :=
    measurable_snd.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hyX : Measurable[setup.filtration cutoff]
      (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
    refine Measurable.subtype_mk ?_
    exact hy
  have hgrad_y : Measurable[setup.filtration cutoff]
      (fun ω => setup.gradf (y ω)) :=
    setup.gradfOnX_measurable.comp hyX
  have hxy_sq :
      Integrable (fun ω => ‖x ω - y ω‖ ^ 2) setup.P := by
    simpa [x, y] using
      setup.iterProcessOfWellDefined_diff_sq_integrable
        hαwf hα_le_one (setup.globalIndex s j) (setup.globalIndex s (j - 1))
  have hzero :=
    adapted_inner_centered_gradient_difference_residual_integral_eq_zero
      (P := setup.P) (filt := setup.filtration)
      (estimate :=
        setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)))
      (targetAtPrevious := fun ω => setup.gradf (y ω))
      (freshResidual := fun ω =>
        (setup.gradF (x ω) (setup.ξ idx ω) -
            setup.gradF (y ω) (setup.ξ idx ω)) -
          (setup.gradf (x ω) - setup.gradf (y ω)))
      (cutoff := cutoff) (idx := idx)
      hGprev hgrad_y (by dsimp [idx]; omega)
      (by
        simpa [d, StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined,
          StochasticNonconvexConditionalGradientSetup.delta, y] using hprev_sq)
      (fun d hd_idx hd_sq =>
        integral_inner_centered_grad_diff_eq_zero setup
          (cutoff := cutoff) (idx := idx) (x := x) (y := y) (d := d)
          hx hy hd_idx (by dsimp [idx]; omega) hx_mem hy_mem hd_sq hxy_sq)
  simpa [x, y, d, cutoff, idx, Nat.add_assoc] using hzero

/-- The previous estimator error is orthogonal in expectation to the averaged
fresh recursive mini-batch increment.

Candidate audit: this finite-sum version is obtained from the local
coordinate cancellation `recursive_delta_prev_centered_grad_diff_cross_zero`
and SOptLib `integral_finset_sum_const_mul_eq_zero`; no separate SOptLib
primitive encodes Algorithm 7.13's paired-gradient recursive increment. -/
lemma recursive_delta_prev_minibatch_increment_cross_zero
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T)
    (hprev_sq :
      Integrable
        (fun ω =>
          ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2)
        setup.P) :
    ∫ ω,
        ⟪setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
          (setup.b : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.b)
              (fun i =>
                (setup.gradF
                      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                      (setup.ξ
                        (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                        ω) -
                    setup.gradF
                      (setup.iterProcessOfWellDefined hαwf
                        (setup.globalIndex s (j - 1)) ω)
                      (setup.ξ
                        (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                        ω)) -
                  (setup.gradf
                      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
                    setup.gradf
                      (setup.iterProcessOfWellDefined hαwf
                        (setup.globalIndex s (j - 1)) ω)))⟫_ℝ
      ∂setup.P = 0 := by
  classical
  let I : Finset ℕ := Finset.range setup.b
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let d : Ω → E :=
    setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let eps : ℕ → Ω → E := fun i ω =>
    (setup.gradF (x ω) (setup.ξ (cutoff + i) ω) -
        setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) -
      (setup.gradf (x ω) - setup.gradf (y ω))
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hGprev : Measurable[setup.filtration cutoff]
      (setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))) :=
    measurable_snd.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hyX : Measurable[setup.filtration cutoff]
      (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
    refine Measurable.subtype_mk ?_
    exact hy
  have hgrad_y : Measurable[setup.filtration cutoff]
      (fun ω => setup.gradf (y ω)) :=
    setup.gradfOnX_measurable.comp hyX
  have hd_meas : AEStronglyMeasurable d setup.P := by
    have hd_cutoff : Measurable[setup.filtration cutoff] d := by
      dsimp [d, StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined,
        StochasticNonconvexConditionalGradientSetup.delta, y]
      exact hGprev.sub hgrad_y
    exact (hd_cutoff.mono (setup.filtration.le cutoff) le_rfl).aestronglyMeasurable
  have heps_meas : ∀ i ∈ I, AEStronglyMeasurable (eps i) setup.P := by
    intro i hi
    have hx_b : Measurable[setup.filtration (cutoff + setup.b)] x :=
      hx.mono (setup.filtration.mono (by omega)) le_rfl
    have hy_b : Measurable[setup.filtration (cutoff + setup.b)] y :=
      hy.mono (setup.filtration.mono (by omega)) le_rfl
    have hξ_i :
        Measurable[setup.filtration (cutoff + setup.b)] (setup.ξ (cutoff + i)) :=
      setup.measurable_xi_of_lt (by
        have hi_lt : i < setup.b := by simpa [I] using Finset.mem_range.mp hi
        omega)
    have hgrad_x :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradF (x ω) (setup.ξ (cutoff + i) ω)) :=
      setup.hgradF_meas.comp (hx_b.prodMk hξ_i)
    have hgrad_yi :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) :=
      setup.hgradF_meas.comp (hy_b.prodMk hξ_i)
    have hxX_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hx_b
    have hyX_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
      refine Measurable.subtype_mk ?_
      exact hy_b
    have hμ_b :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => setup.gradf (x ω) - setup.gradf (y ω)) :=
      (setup.gradfOnX_measurable.comp hxX_b).sub
        (setup.gradfOnX_measurable.comp hyX_b)
    have heps :
        Measurable[setup.filtration (cutoff + setup.b)] (eps i) := by
      dsimp [eps]
      exact (hgrad_x.sub hgrad_yi).sub hμ_b
    exact (heps.mono (setup.filtration.le (cutoff + setup.b)) le_rfl).aestronglyMeasurable
  have hdiag :
      ∀ i ∈ I,
        Integrable (fun ω => ‖eps i ω‖ ^ 2) setup.P := by
    intro i hi
    have h :=
      recursive_centered_grad_diff_sqnorm_le setup hαwf hα_le_one s j i hj2 hjT
    exact (by
      simpa [eps, x, y, cutoff, Nat.add_assoc] using h.1)
  have hZ_int :
      ∀ i ∈ I, Integrable (fun ω => ⟪d ω, eps i ω⟫_ℝ) setup.P := by
    intro i hi
    exact integrable_inner_of_sq_integrable
      hd_meas (heps_meas i hi)
      (by simpa [d] using hprev_sq) (hdiag i hi)
  have hZ_zero :
      ∀ i ∈ I, ∫ ω, ⟪d ω, eps i ω⟫_ℝ ∂setup.P = 0 := by
    intro i hi
    simpa [d, eps, x, y, cutoff, Nat.add_assoc] using
      recursive_delta_prev_centered_grad_diff_cross_zero setup hαwf hα_le_one
        s j i hj2 hjT hprev_sq
  have hpoint :
      (fun ω =>
        ⟪d ω,
          (setup.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)⟫_ℝ) =
        (fun ω =>
          Finset.sum I (fun i => (setup.b : ℝ)⁻¹ * ⟪d ω, eps i ω⟫_ℝ)) := by
    funext ω
    rw [inner_smul_right, inner_sum, Finset.mul_sum]
  calc
    ∫ ω,
        ⟪setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
          (setup.b : ℝ)⁻¹ •
            Finset.sum (Finset.range setup.b)
              (fun i =>
                (setup.gradF
                      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω)
                      (setup.ξ
                        (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                        ω) -
                    setup.gradF
                      (setup.iterProcessOfWellDefined hαwf
                        (setup.globalIndex s (j - 1)) ω)
                      (setup.ξ
                        (setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b + i)
                        ω)) -
                  (setup.gradf
                      (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω) -
                    setup.gradf
                      (setup.iterProcessOfWellDefined hαwf
                        (setup.globalIndex s (j - 1)) ω)))⟫_ℝ
      ∂setup.P
        = ∫ ω, Finset.sum I
            (fun i => (setup.b : ℝ)⁻¹ * ⟪d ω, eps i ω⟫_ℝ) ∂setup.P := by
          rw [hpoint]
    _ = 0 := by
          exact integral_finset_sum_const_mul_eq_zero
            I (fun _ => (setup.b : ℝ)⁻¹)
            (fun i ω => ⟪d ω, eps i ω⟫_ℝ) hZ_int hZ_zero

/-- One recursive step of Lan Lemma 7.5: expanding
`δ_j = δ_{j-1} +` the fresh centered mini-batch increment gives the
second-moment recurrence after the martingale cross term vanishes.

Candidate audit: `norm_add_sq_real` supplies the Hilbert-space square
expansion, `integrable_inner_of_sq_integrable` supplies the scalar cross-term
integrability, and the paper-local helpers
`deltaProcessOfWellDefined_globalIndex_recursive`,
`recursive_minibatch_increment_second_moment_le`, and
`recursive_delta_prev_minibatch_increment_cross_zero` are the exact Algorithm
7.13 specializations needed for Lan Lemma 7.5. -/
lemma epochwise_delta_one_step_second_moment_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ setup.T)
    (hprev_sq :
      Integrable
        (fun ω =>
          ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2)
        setup.P) :
    Integrable
        (fun ω =>
          ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s j) ω‖ ^ 2)
        setup.P ∧
      ∫ ω, ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P ≤
        ∫ ω,
            ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2
            ∂setup.P +
          setup.L ^ 2 / setup.b *
            ∫ ω,
              ‖setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j) ω -
                setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω‖ ^ 2
              ∂setup.P := by
  classical
  let I : Finset ℕ := Finset.range setup.b
  let cutoff := setup.N * setup.m + setup.globalIndex s (j - 1) * setup.b
  let x : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s j)
  let y : Ω → E := setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let d : Ω → E :=
    setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))
  let eps : ℕ → Ω → E := fun i ω =>
    (setup.gradF (x ω) (setup.ξ (cutoff + i) ω) -
        setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) -
      (setup.gradf (x ω) - setup.gradf (y ω))
  let inc : Ω → E := fun ω =>
    (setup.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)
  have hinc_sq :
      Integrable (fun ω => ‖inc ω‖ ^ 2) setup.P := by
    simpa [inc, I, eps, x, y, cutoff, Nat.add_assoc] using
      recursive_minibatch_increment_sq_integrable setup hαwf hα_le_one
        s j hj2 hjT
  have hinc_bound :
      ∫ ω, ‖inc ω‖ ^ 2 ∂setup.P ≤
        setup.L ^ 2 / setup.b *
          ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P := by
    simpa [inc, I, eps, x, y, cutoff, Nat.add_assoc] using
      recursive_minibatch_increment_second_moment_le setup hαwf hα_le_one
        s j hj2 hjT
  have hx : Measurable[setup.filtration cutoff] x := by
    simpa [x, cutoff] using
      setup.iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff
        hαwf s j hj2 hjT
  have hpair_prev :
      Measurable[setup.filtration cutoff]
        (fun ω =>
          (setup.iterProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω,
            setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1)) ω)) := by
    simpa [cutoff] using
      setup.processOfWellDefined_pair_measurable_recursive_cutoff
        hαwf (setup.globalIndex s (j - 1))
  have hy : Measurable[setup.filtration cutoff] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hGprev : Measurable[setup.filtration cutoff]
      (setup.estimatorProcessOfWellDefined hαwf (setup.globalIndex s (j - 1))) :=
    measurable_snd.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ setup.X := by
    intro ω
    simpa [x, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s j) ω
  have hy_mem : ∀ ω, y ω ∈ setup.X := by
    intro ω
    simpa [y, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one
        hαwf hα_le_one (setup.globalIndex s (j - 1)) ω
  have hyX : Measurable[setup.filtration cutoff]
      (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
    refine Measurable.subtype_mk ?_
    exact hy
  have hgrad_y : Measurable[setup.filtration cutoff]
      (fun ω => setup.gradf (y ω)) :=
    setup.gradfOnX_measurable.comp hyX
  have hd_cutoff : Measurable[setup.filtration cutoff] d := by
    dsimp [d, StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined,
      StochasticNonconvexConditionalGradientSetup.delta, y]
    exact hGprev.sub hgrad_y
  have hd_global : Measurable d :=
    hd_cutoff.mono (setup.filtration.le cutoff) le_rfl
  have hd_meas : AEStronglyMeasurable d setup.P :=
    hd_global.aestronglyMeasurable
  have hinc_cutoff_b : Measurable[setup.filtration (cutoff + setup.b)] inc := by
    have hx_b : Measurable[setup.filtration (cutoff + setup.b)] x :=
      hx.mono (setup.filtration.mono (by omega)) le_rfl
    have hy_b : Measurable[setup.filtration (cutoff + setup.b)] y :=
      hy.mono (setup.filtration.mono (by omega)) le_rfl
    have hsum :
        Measurable[setup.filtration (cutoff + setup.b)]
          (fun ω => Finset.sum I (fun i => eps i ω)) := by
      refine Finset.measurable_sum I ?_
      intro i hi
      have hξ_i :
          Measurable[setup.filtration (cutoff + setup.b)] (setup.ξ (cutoff + i)) :=
        setup.measurable_xi_of_lt (by
          have hi_lt : i < setup.b := by simpa [I] using Finset.mem_range.mp hi
          omega)
      have hgrad_x :
          Measurable[setup.filtration (cutoff + setup.b)]
            (fun ω => setup.gradF (x ω) (setup.ξ (cutoff + i) ω)) :=
        setup.hgradF_meas.comp (hx_b.prodMk hξ_i)
      have hgrad_yi :
          Measurable[setup.filtration (cutoff + setup.b)]
            (fun ω => setup.gradF (y ω) (setup.ξ (cutoff + i) ω)) :=
        setup.hgradF_meas.comp (hy_b.prodMk hξ_i)
      have hxX_b :
          Measurable[setup.filtration (cutoff + setup.b)]
            (fun ω => (⟨x ω, hx_mem ω⟩ : setup.X)) := by
        refine Measurable.subtype_mk ?_
        exact hx_b
      have hyX_b :
          Measurable[setup.filtration (cutoff + setup.b)]
            (fun ω => (⟨y ω, hy_mem ω⟩ : setup.X)) := by
        refine Measurable.subtype_mk ?_
        exact hy_b
      have hμ_b :
          Measurable[setup.filtration (cutoff + setup.b)]
            (fun ω => setup.gradf (x ω) - setup.gradf (y ω)) :=
        (setup.gradfOnX_measurable.comp hxX_b).sub
          (setup.gradfOnX_measurable.comp hyX_b)
      dsimp [eps]
      exact (hgrad_x.sub hgrad_yi).sub hμ_b
    dsimp [inc]
    exact hsum.const_smul ((setup.b : ℝ)⁻¹)
  have hinc_meas : AEStronglyMeasurable inc setup.P :=
    (hinc_cutoff_b.mono (setup.filtration.le (cutoff + setup.b)) le_rfl).aestronglyMeasurable
  have hinc_global : Measurable inc :=
    hinc_cutoff_b.mono (setup.filtration.le (cutoff + setup.b)) le_rfl
  have hcross :
      ∫ ω, ⟪d ω, inc ω⟫_ℝ ∂setup.P = 0 := by
    simpa [d, inc, I, eps, x, y, cutoff, Nat.add_assoc] using
      recursive_delta_prev_minibatch_increment_cross_zero setup hαwf hα_le_one
        s j hj2 hjT hprev_sq
  have hrec :
      ∀ ω,
        setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s j) ω =
          d ω + inc ω := by
    intro ω
    let a : ℕ → E := fun i =>
      setup.gradF (x ω) (setup.ξ (cutoff + i) ω) -
        setup.gradF (y ω) (setup.ξ (cutoff + i) ω)
    let target : E := setup.gradf (x ω) - setup.gradf (y ω)
    have hraw :=
      setup.deltaProcessOfWellDefined_globalIndex_recursive hαwf s j hj2 hjT ω
    have hcenter :
        ((setup.b : ℝ)⁻¹) • Finset.sum I a - target =
          (setup.b : ℝ)⁻¹ • Finset.sum I (fun i => a i - target) := by
      simpa [I] using
        miniBatchAverage_sub_target_eq_average_residual
          (I := I) (m := setup.b) (by simp [I])
          (lt_of_lt_of_le Nat.zero_lt_one
            (le_trans setup.hT_pos setup.hb_ge_T))
          (a := a) (target := target)
    calc
      setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s j) ω =
          d ω + ((setup.b : ℝ)⁻¹) • Finset.sum I a - target := by
            simpa [d, x, y, cutoff, I, a, target, Nat.add_assoc] using hraw
      _ = d ω + (((setup.b : ℝ)⁻¹) • Finset.sum I a - target) := by
            abel
      _ = d ω + inc ω := by
            rw [hcenter]
  simpa [d, x, y] using
    second_moment_add_recurrence_le_of_cross_zero
      setup.P d
      (setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s j))
      inc
      (setup.L ^ 2 / setup.b *
        ∫ ω, ‖x ω - y ω‖ ^ 2 ∂setup.P)
      hd_meas hinc_meas hprev_sq hinc_sq
      (Filter.Eventually.of_forall hrec)
      (by simpa using hcross)
      hinc_bound

/-! Core martingale/conditional-expectation bound for the recursive estimator
inside one epoch. This is the reusable proof obligation shared by Lemmas 7.4
and 7.5; `setup.linearMinimizer_measurable` supplies the canonical LMO
adaptedness fact needed by this argument. -/
lemma epochwise_estimator_variance_bound_t_one
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s : ℕ) (hsN : setup.globalIndex s 1 ≤ setup.N) :
    ∫ ω, ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s 1) ω‖ ^ 2 ∂setup.P ≤
      setup.L ^ 2 / setup.b *
        ∫ ω, setup.epochDiffSumOfWellDefined hαwf s 1 ω ∂setup.P +
      setup.σ ^ 2 / setup.m := by
  have hrefresh := setup.epochRefresh_variance_floor hαwf hα_le_one s hsN
  have hdiff_zero :
      ∫ ω, setup.epochDiffSumOfWellDefined hαwf s 1 ω ∂setup.P = 0 := by
    simp [StochasticNonconvexConditionalGradientSetup.epochDiffSumOfWellDefined]
  simpa [hdiff_zero] using hrefresh

lemma epochwise_delta_sq_integrable
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s t : ℕ) (ht : 1 ≤ t) (ht_le : t ≤ setup.T)
    (hkt : setup.globalIndex s t ≤ setup.N) :
    Integrable
      (fun ω =>
        ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s t) ω‖ ^ 2)
      setup.P := by
  induction t with
  | zero =>
      omega
  | succ n ih =>
      by_cases hn0 : n = 0
      · subst n
        exact setup.epochRefresh_delta_sq_integrable hαwf hα_le_one s hkt
      · have hn_pos : 1 ≤ n := by omega
        have hj2 : 2 ≤ n + 1 := by omega
        have hn_le_t : n ≤ n + 1 := by omega
        have hn_le_T : n ≤ setup.T := by omega
        have hprevN : setup.globalIndex s n ≤ setup.N :=
          setup.globalIndex_prefix_le_of_le (s := s) (i := n) (t := n + 1)
            hn_le_t hkt
        have hprev_sq := ih hn_pos hn_le_T hprevN
        exact (epochwise_delta_one_step_second_moment_le
          setup hαwf hα_le_one s (n + 1) hj2 ht_le hprev_sq).1

lemma epochwise_estimator_variance_bound
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hαwf : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1)
    (s t : ℕ) (ht : 1 ≤ t) (ht_le : t ≤ setup.T)
    (hkt : setup.globalIndex s t ≤ setup.N) :
    ∫ ω, ‖setup.deltaProcessOfWellDefined hαwf (setup.globalIndex s t) ω‖ ^ 2 ∂setup.P ≤
      setup.L ^ 2 / setup.b *
        ∫ ω, setup.epochDiffSumOfWellDefined hαwf s t ω ∂setup.P +
      setup.σ ^ 2 / setup.m := by
  simpa [StochasticNonconvexConditionalGradientSetup.epochDiffSumOfWellDefined,
    SOptLib.epochSquaredDifferenceSum] using
    (epoch_second_moment_le_difference_sum_add_const_of_one_step_recurrence
      setup.P
      (setup.deltaProcessOfWellDefined hαwf)
      (setup.iterProcessOfWellDefined hαwf)
      setup.globalIndex
      (fun s t => setup.globalIndex s t ≤ setup.N)
      setup.T
      (setup.L ^ 2 / setup.b)
      (setup.σ ^ 2 / setup.m)
      (by
        intro s i j hij hkj
        exact setup.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hij hkj)
      (by
        intro s hsN
        simpa [StochasticNonconvexConditionalGradientSetup.epochDiffSumOfWellDefined,
          SOptLib.epochSquaredDifferenceSum] using
          epochwise_estimator_variance_bound_t_one setup hαwf hα_le_one s hsN)
      (by
        intro s j hj hT hN
        exact epochwise_delta_sq_integrable setup hαwf hα_le_one s j hj hT hN)
      (by
        intro s i _hi _hT _hN
        simpa using
          setup.iterProcessOfWellDefined_diff_sq_integrable hαwf hα_le_one
            (setup.globalIndex s i) (setup.globalIndex s (i - 1)))
      (by
        intro s j hj2 hjT _hjN hprev_sq
        exact (epochwise_delta_one_step_second_moment_le
          setup hαwf hα_le_one s j hj2 hjT hprev_sq).2)
      s t ht ht_le hkt)

/-! Generic Wolfe-gap surrogate helper behind Eq. (7.4.2), intentionally not
named by the exact paper equation because it applies to an arbitrary feasible
point and estimator.

Book citation: source_json `key_lemmas[0].statement_math` quotes
`gap(x_k) <= max_{x in X} <G_k, x_k - x> + ||delta_k|| Dbar_X`. The
paper-named generated-process specialization remains
`eq_7_4_2_conditional` because Algorithm 7.13's stepsize realization uses
non-source boundary facts. -/
theorem wolfeGap_surrogate_bound
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (x G : E) (hx : x ∈ setup.X) :
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x ≤
      setup.maxLinModel x G + ‖setup.delta G x‖ * setup.barDX := by
  simpa [StochasticNonconvexConditionalGradientSetup.delta] using
    SOptLib.ConditionalGradient.wolfeGap_le_maxLinearModel_add_estimatorError_mul_diameter
      setup.gradf setup.wolfeGapMaximizer setup.maxLinModel setup.barDX x G
      (by
        simpa using
          setup.maxLinModel_spec x G (setup.wolfeGapMaximizer x))
      (by
        simpa using
          setup.barDX_bound x ((setup.wolfeGapMaximizer x : setup.X) : E)
            hx (setup.wolfeGapMaximizer x).property)

/-- The Algorithm 7.13 iterate produced under the conditional realization
boundary is feasible. This is a derived invariant of the generated process, not
a theorem-head assumption for Eq. (7.4.2). -/
theorem algorithmIterProcessConditional_mem
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract)
    (k : ℕ) (ω : Ω) :
    setup.algorithmIterProcessConditional h k ω ∈ setup.X := by
  let hαwf := setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h
  have hα_le :
      setup.paperAlphaOfWellDefined hαwf ≤ 1 := by
    simpa [hαwf] using setup.paperAlpha_le_one_of_algorithm713Contract h
  simpa [StochasticNonconvexConditionalGradientSetup.algorithmIterProcessConditional,
    StochasticNonconvexConditionalGradientSetup.algorithmProcessConditional, hαwf] using
    setup.processOfWellDefined_mem_of_alpha_le_one hαwf hα_le k ω

/-! Conditional realization of Eq. (7.4.2) for the generated Algorithm 7.13
iterate and estimator under the single realization contract.

Book citation: source_json `key_lemmas[0].statement_math` quotes
`gap(x_k) <= max_{x in X} <G_k, x_k - x> + ||delta_k|| Dbar_X`, and
source_json `algorithm_spec.steps[0].math`/`algorithm_spec.steps[1].math`
define the generated estimator `G_k`. Feasibility is supplied by the
generated-process invariant `algorithmIterProcessConditional_mem`, so the
theorem head does not carry an extra `x_k in X` hypothesis. -/
theorem eq_7_4_2_conditional
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract)
    (k : ℕ) (ω : Ω)
    :
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.algorithmIterProcessConditional h k ω) ≤
      setup.maxLinModel
          (setup.algorithmIterProcessConditional h k ω)
          (setup.algorithmEstimatorProcessConditional h k ω) +
        ‖setup.algorithmDeltaProcessConditional h k ω‖ * setup.barDX := by
  simpa [StochasticNonconvexConditionalGradientSetup.algorithmDeltaProcessConditional]
    using wolfeGap_surrogate_bound (setup := setup)
      (x := setup.algorithmIterProcessConditional h k ω)
      (G := setup.algorithmEstimatorProcessConditional h k ω)
      (algorithmIterProcessConditional_mem (setup := setup) h k ω)

lemma lemma_7_4_t_one
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (s : ℕ) :
    ∫ ω, ‖fs.deltaProcess (fs.globalIndex s 1) ω‖ ^ 2 ∂fs.P ≤
      fs.L ^ 2 / fs.b *
        ∫ ω, fs.epochDiffSum s 1 ω ∂fs.P := by
  have hδ := fs.deltaProcess_globalIndex_one_eq_zero s
  rw [hδ]
  simp [SOptLib.FiniteSumConditionalGradientSetup.epochDiffSum]

/-! Finite-sum sample-filtration bridge for Lan Lemma 7.4.

Candidate audit: checked the pre-searched `iIndepFun.indep_past_iSup_current`
and SOptLib `samplePrefixFiltration_indep_current`; the latter is the exact
abstract sample-prefix result. This local specialization aligns it with
Algorithm 7.12's component-index stream `fs.sample` and its setup fields
`fs.hsample_meas` and `fs.hsample_iIndep`. -/
lemma finite_samplePrefixFiltration_indep_current
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω) (r : ℕ) :
    Indep ((SOptLib.filtration fs.sample fs.hsample_meas).seq r)
      (MeasurableSpace.comap (fs.sample r)
        (by infer_instance : MeasurableSpace (Fin fs.componentCount))) fs.P := by
  simpa using
    samplePrefixFiltration_indep_current
      (ξ := fs.sample) (μ := fs.P) fs.hsample_meas fs.hsample_iIndep r

/-! Prefix-measurable finite-sum quantities are independent of the current fresh
component sample.

Candidate audit: checked SOptLib `indepFun_of_past_measurable_current_iid_sample`
and the local sample-prefix specialization
`finite_samplePrefixFiltration_indep_current`; together they exactly supply the
finite-sum past/fresh bridge requested for Lemma 7.4. -/
lemma finite_indepFun_of_prefix_measurable_current_sample
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {W : Type*} [MeasurableSpace W] {X : Ω → W} {r : ℕ}
    (hX :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r] X) :
    IndepFun X (fs.sample r) fs.P := by
  exact indepFun_of_past_measurable_current_iid_sample
    (X := X) (Y := fs.sample r) hX
    (finite_samplePrefixFiltration_indep_current (fs := fs) r)

/-! A prefix-measurable random feasible pair and direction are orthogonal in
expectation to the fresh centered finite-component residual.

Candidate audit: checked target-file
`componentQ_weighted_gradDiff_residual_sum_eq_zero`,
`sample_identDistrib_componentQ`, SOptLib `randomIterate_oracleDeviation_integral_eq_zero_of_indep`,
and the prefix bridge `finite_indepFun_of_prefix_measurable_current_sample`.
The deterministic component-law identity gives each fixed-fiber zero mean; the
SOptLib random-query transfer and Algorithm 7.12 sample-prefix independence
transport that zero mean to random past-measurable queries. -/
lemma finite_component_residual_inner_cross_zero_of_prefix_measurable
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (r : ℕ) {x y d : Ω → E}
    (hx_mem : ∀ ω, x ω ∈ fs.X)
    (hy_mem : ∀ ω, y ω ∈ fs.X)
    (hquery :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r]
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)), d ω))) :
    ∫ ω,
        ⟪d ω,
          (((fs.componentQ (fs.sample r ω) * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample r ω) (x ω) -
                fs.gradFcomp (fs.sample r ω) (y ω))) -
            (fs.gradf (x ω) - fs.gradf (y ω))⟫_ℝ ∂fs.P = 0 := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  let ν : Measure (Fin fs.componentCount) := fs.componentIndexPMF.toMeasure
  haveI : IsProbabilityMeasure ν := by
    dsimp [ν]
    exact inferInstance
  let query : Ω → (fs.X × fs.X) × E := fun ω =>
    (((⟨x ω, hx_mem ω⟩ : fs.X), (⟨y ω, hy_mem ω⟩ : fs.X)), d ω)
  let residual : fs.X × fs.X → Fin fs.componentCount → E := fun q i =>
    (((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
        (fs.gradFcomp i (q.1 : E) - fs.gradFcomp i (q.2 : E))) -
      (fs.gradf (q.1 : E) - fs.gradf (q.2 : E))
  have hquery_global : Measurable query :=
    hquery.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r) le_rfl
  have hindep : IndepFun query (fs.sample r) fs.P :=
    finite_indepFun_of_prefix_measurable_current_sample
      (fs := fs) (r := r) (X := query) hquery
  have hbranch :
      ∀ i : Fin fs.componentCount, Measurable (fun q : fs.X × fs.X => residual q i) := by
    intro i
    have hxq :
        Measurable (fun q : fs.X × fs.X => (q.1 : fs.X)) :=
      measurable_fst
    have hyq :
        Measurable (fun q : fs.X × fs.X => (q.2 : fs.X)) :=
      measurable_snd
    have hgrad_x :
        Measurable (fun q : fs.X × fs.X =>
          fs.gradFcomp i ((q.1 : fs.X) : E)) :=
      (fs.gradFcompOnX_measurable i).comp hxq
    have hgrad_y :
        Measurable (fun q : fs.X × fs.X =>
          fs.gradFcomp i ((q.2 : fs.X) : E)) :=
      (fs.gradFcompOnX_measurable i).comp hyq
    have hmean :
        Measurable (fun q : fs.X × fs.X =>
          fs.gradf ((q.1 : fs.X) : E) -
            fs.gradf ((q.2 : fs.X) : E)) :=
      (fs.gradfOnX_measurable.comp hxq).sub
        (fs.gradfOnX_measurable.comp hyq)
    dsimp [residual]
    exact (((hgrad_x.sub hgrad_y).const_smul
      ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹)).sub hmean)
  have hres_meas :
      Measurable
        (fun p : (fs.X × fs.X) × Fin fs.componentCount => residual p.1 p.2) := by
    exact measurable_from_prod_countable_left hbranch
  have hfixed_int : ∀ q : fs.X × fs.X, Integrable (fun i => residual q i) ν := by
    intro q
    exact Integrable.of_finite
  have hfixed_zero :
      ∀ q : fs.X × fs.X, ∫ i, residual q i ∂ν = 0 := by
    intro q
    have hpmf_sum :
        ∫ i, residual q i ∂ν =
          Finset.sum Finset.univ
            (fun i : Fin fs.componentCount => fs.componentQ i • residual q i) := by
      calc
        ∫ i, residual q i ∂ν =
            Finset.sum Finset.univ
              (fun i : Fin fs.componentCount =>
                (fs.componentIndexPMF i).toReal • residual q i) := by
              dsimp [ν]
              rw [integral_fintype .of_finite]
              congr with i
              rw [measureReal_def]
              congr 2
              exact PMF.toMeasure_apply_singleton fs.componentIndexPMF i
                (MeasurableSet.singleton i)
        _ =
            Finset.sum Finset.univ
              (fun i : Fin fs.componentCount => fs.componentQ i • residual q i) := by
              refine Finset.sum_congr rfl ?_
              intro i _hi
              simp [fs.componentIndexPMF_apply i,
                ENNReal.toReal_ofReal (fs.componentQ_nonneg i)]
    have hsum_vec :
        Finset.sum Finset.univ
            (fun i : Fin fs.componentCount => fs.componentQ i • residual q i) = 0 := by
      simpa [residual] using
        fs.componentQ_weighted_gradDiff_residual_sum_eq_zero
          ((q.1 : fs.X) : E) ((q.2 : fs.X) : E)
    rw [hpmf_sum, hsum_vec]
  let residualQ : ((fs.X × fs.X) × E) → Fin fs.componentCount → E := fun w i =>
    residual w.1 i
  have hresQ_meas :
      Measurable
        (fun p : ((fs.X × fs.X) × E) × Fin fs.componentCount => residualQ p.1 p.2) := by
    exact hres_meas.comp ((measurable_fst.comp measurable_fst).prodMk measurable_snd)
  have hfixed_intQ :
      ∀ w : (fs.X × fs.X) × E, Integrable (fun i => residualQ w i) ν := by
    intro w
    exact hfixed_int w.1
  have hfixed_zeroQ :
      ∀ w : (fs.X × fs.X) × E, ∫ i, residualQ w i ∂ν = 0 := by
    intro w
    exact hfixed_zero w.1
  have hzero :=
    randomQuery_inner_oracleResidual_integral_eq_zero_of_fixed_zero
      (P := fs.P) (ν := ν)
      (query := query) (sample := fs.sample r) (residual := residualQ) (d := fun w => w.2)
      hresQ_meas measurable_snd hquery_global (fs.sample_measurable r) hindep
      (by
        dsimp [ν]
        simpa [SOptLib.FiniteSumConditionalGradientSetup.componentIndexPMF,
          SOptLib.FiniteSumConditionalGradientSetup.componentQ] using
          fs.hsample_componentQ r)
      hfixed_intQ hfixed_zeroQ
  simpa [query, residualQ, residual] using hzero

/-! A single fresh finite component query at a prefix-measurable feasible pair
has the diagonal residual budget used in Lan Lemma 7.4.

Candidate audit: checked target-file
`sampled_componentQ_gradDiff_residual_secondMoment_le`,
`componentQ_weighted_gradDiff_norm_le`,
`componentQ_weighted_gradDiff_residual_secondMoment_sum_le`,
and SOptLib `randomIterate_variance_bound_of_fixed_variance`. The fixed-query
transport lemma does not handle a variance budget depending on the random
pair; this helper instead uses the already proved prefix cross-zero bridge and
the pointwise importance-sampling smoothness bound. -/
lemma finite_random_query_residual_second_moment_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (r : ℕ) {x y : Ω → E}
    (hx_mem : ∀ ω, x ω ∈ fs.X)
    (hy_mem : ∀ ω, y ω ∈ fs.X)
    (hquery :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r]
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)),
          fs.gradf (x ω) - fs.gradf (y ω))))
    (hxy_sq : Integrable (fun ω => ‖x ω - y ω‖ ^ 2) fs.P) :
    Integrable
        (fun ω =>
          ‖(((fs.componentQ (fs.sample r ω) * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample r ω) (x ω) -
                fs.gradFcomp (fs.sample r ω) (y ω))) -
            (fs.gradf (x ω) - fs.gradf (y ω))‖ ^ 2)
        fs.P ∧
      ∫ ω,
          ‖(((fs.componentQ (fs.sample r ω) * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample r ω) (x ω) -
                fs.gradFcomp (fs.sample r ω) (y ω))) -
            (fs.gradf (x ω) - fs.gradf (y ω))‖ ^ 2 ∂fs.P ≤
        fs.L ^ 2 * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂fs.P := by
  classical
  let query : Ω → (fs.X × fs.X) × E := fun ω =>
    (((⟨x ω, hx_mem ω⟩ : fs.X), (⟨y ω, hy_mem ω⟩ : fs.X)),
      fs.gradf (x ω) - fs.gradf (y ω))
  let g : Ω → E := fun ω =>
    ((fs.componentQ (fs.sample r ω) * (fs.componentCount : ℝ))⁻¹) •
      (fs.gradFcomp (fs.sample r ω) (x ω) -
        fs.gradFcomp (fs.sample r ω) (y ω))
  let μfun : Ω → E := fun ω => fs.gradf (x ω) - fs.gradf (y ω)
  have hquery_global : Measurable query :=
    hquery.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r) le_rfl
  have hxX_meas : Measurable (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
    measurable_fst.comp (measurable_fst.comp hquery_global)
  have hyX_meas : Measurable (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
    measurable_snd.comp (measurable_fst.comp hquery_global)
  have hμ_meas : Measurable μfun := by
    dsimp [μfun]
    exact (fs.gradfOnX_measurable.comp hxX_meas).sub
      (fs.gradfOnX_measurable.comp hyX_meas)
  have hbranch :
      ∀ i : Fin fs.componentCount,
        Measurable
          (fun q : fs.X × fs.X =>
            ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp i (q.1 : E) - fs.gradFcomp i (q.2 : E))) := by
    intro i
    have hxq : Measurable (fun q : fs.X × fs.X => (q.1 : fs.X)) :=
      measurable_fst
    have hyq : Measurable (fun q : fs.X × fs.X => (q.2 : fs.X)) :=
      measurable_snd
    exact ((fs.gradFcompOnX_measurable i).comp hxq).sub
      ((fs.gradFcompOnX_measurable i).comp hyq) |>.const_smul
        ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹)
  have hkernel :
      Measurable
        (fun p : (fs.X × fs.X) × Fin fs.componentCount =>
          ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
            (fs.gradFcomp p.2 (p.1.1 : E) -
              fs.gradFcomp p.2 (p.1.2 : E))) :=
    measurable_from_prod_countable_left hbranch
  have hpair_sample :
      Measurable
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)), fs.sample r ω)) :=
    (hxX_meas.prodMk hyX_meas).prodMk (fs.sample_measurable r)
  have hg_meas : Measurable g := by
    simpa [g] using hkernel.comp hpair_sample
  have hcross_zero :
      ∫ ω, ⟪μfun ω, g ω - μfun ω⟫_ℝ ∂fs.P = 0 := by
    simpa [query, μfun, g] using
      finite_component_residual_inner_cross_zero_of_prefix_measurable
        (fs := fs) (r := r) (x := x) (y := y) (d := μfun)
        hx_mem hy_mem hquery
  simpa [g, μfun] using
    (randomQuery_oracleResidual_secondMoment_le_of_pointwise_bound
      (P := fs.P) (g := g) (m := μfun)
      (distanceSq := fun ω => ‖x ω - y ω‖ ^ 2) (L := fs.L)
      hg_meas.aestronglyMeasurable hμ_meas.aestronglyMeasurable hxy_sq
      (by
        filter_upwards [] with ω
        have hnorm : ‖g ω‖ ≤ fs.L * ‖x ω - y ω‖ := by
          simpa [g] using
            fs.componentQ_weighted_gradDiff_norm_le
              (fs.sample r ω) (x ω) (y ω) (hx_mem ω) (hy_mem ω)
        nlinarith [hnorm, norm_nonneg (g ω), norm_nonneg (x ω - y ω),
          le_of_lt fs.hL_pos])
      (by
        filter_upwards [] with ω
        have hgradf : ‖μfun ω‖ ≤ fs.L * ‖x ω - y ω‖ := by
          simpa [μfun] using fs.gradf_smooth (x ω) (y ω) (hx_mem ω) (hy_mem ω)
        nlinarith [hgradf, norm_nonneg (μfun ω), norm_nonneg (x ω - y ω),
          le_of_lt fs.hL_pos])
      hcross_zero)

/-! Prefix-measurable finite-sum query pairs have the mini-batch residual
second-moment scale `L² / b`.

Candidate audit: this directly specializes SOptLib
`miniBatchResidual_sum_secondMoment_le_card_mul_variance` and
`miniBatchAverage_secondMoment_le_variance_div_card`. The paper-local inputs
not supplied by SOptLib are the random-query diagonal bridge
`finite_random_query_residual_second_moment_le` and the off-diagonal
prefix/fresh cancellation
`finite_component_residual_inner_cross_zero_of_prefix_measurable`. -/
lemma finite_prefix_minibatch_increment_second_moment_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (r0 : ℕ) {x y : Ω → E}
    (hx_mem : ∀ ω, x ω ∈ fs.X)
    (hy_mem : ∀ ω, y ω ∈ fs.X)
    (hquery0 :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)),
          fs.gradf (x ω) - fs.gradf (y ω))))
    (hxy_sq : Integrable (fun ω => ‖x ω - y ω‖ ^ 2) fs.P) :
    ∫ ω,
        ‖(fs.b : ℝ)⁻¹ •
          Finset.sum (Finset.range fs.b)
            (fun i =>
              (((fs.componentQ (fs.sample (r0 + i) ω) *
                    (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
                    fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) -
                (fs.gradf (x ω) - fs.gradf (y ω)))‖ ^ 2 ∂fs.P ≤
      fs.L ^ 2 / fs.b * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂fs.P := by
  classical
  let I : Finset ℕ := Finset.range fs.b
  let μfun : Ω → E := fun ω => fs.gradf (x ω) - fs.gradf (y ω)
  let eps : ℕ → Ω → E := fun i ω =>
    (((fs.componentQ (fs.sample (r0 + i) ω) * (fs.componentCount : ℝ))⁻¹) •
      (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
        fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) -
      μfun ω
  let σ2 : ℝ := fs.L ^ 2 * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂fs.P
  have hkernel :
      Measurable
        (fun p : (fs.X × fs.X) × Fin fs.componentCount =>
          ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
            (fs.gradFcomp p.2 (p.1.1 : E) -
              fs.gradFcomp p.2 (p.1.2 : E))) := by
    have hbranch :
        ∀ i : Fin fs.componentCount,
          Measurable
            (fun q : fs.X × fs.X =>
              ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
                (fs.gradFcomp i (q.1 : E) - fs.gradFcomp i (q.2 : E))) := by
      intro i
      exact ((fs.gradFcompOnX_measurable i).comp measurable_fst).sub
        ((fs.gradFcompOnX_measurable i).comp measurable_snd) |>.const_smul
          ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹)
    exact measurable_from_prod_countable_left hbranch
  have hquery0_global :
      Measurable
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
    hquery0.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hsample_prefix
      {q R : ℕ} (hqR : q < R) :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
        (fs.sample q) := by
    have hq :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (q + 1)]
          (fs.sample q) :=
      SOptLib.measurable_sample_le_prefixFiltration fs.sample fs.hsample_meas q
    exact hq.mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : q + 1 ≤ R))
      le_rfl
  have heps_meas_at
      {i R : ℕ} (hiR : r0 + i < R) :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
        (eps i) := by
    have hqR :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
          (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
            (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
      hquery0.mono
        ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ R))
        le_rfl
    have hxR :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
          (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
      measurable_fst.comp (measurable_fst.comp hqR)
    have hyR :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
          (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
      measurable_snd.comp (measurable_fst.comp hqR)
    have hμR :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R] μfun :=
      measurable_snd.comp hqR
    have hsR :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
          (fs.sample (r0 + i)) :=
      hsample_prefix hiR
    have hgR :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq R]
          (fun ω =>
            ((fs.componentQ (fs.sample (r0 + i) ω) *
                (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
                fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) := by
      simpa using hkernel.comp ((hxR.prodMk hyR).prodMk hsR)
    dsimp [eps]
    exact hgR.sub hμR
  have hmeas : ∀ i ∈ I, AEStronglyMeasurable (eps i) fs.P := by
    intro i hi
    have hi_lt : i < fs.b := by simpa [I] using Finset.mem_range.mp hi
    have hR : r0 + i < r0 + fs.b := by omega
    exact ((heps_meas_at (i := i) (R := r0 + fs.b) hR).mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).le (r0 + fs.b)) le_rfl).aestronglyMeasurable
  have hdiag :
      ∀ i ∈ I,
        Integrable (fun ω => ‖eps i ω‖ ^ 2) fs.P ∧
          ∫ ω, ‖eps i ω‖ ^ 2 ∂fs.P ≤ σ2 := by
    intro i hi
    have hqi :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
          (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
            (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
      hquery0.mono
        ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
        le_rfl
    simpa [eps, μfun, σ2] using
      finite_random_query_residual_second_moment_le
        (fs := fs) (r := r0 + i) (x := x) (y := y)
        hx_mem hy_mem hqi hxy_sq
  have hcross :
      ∀ i ∈ I, ∀ l ∈ I, i ≠ l →
        ∫ ω, ⟪eps i ω, eps l ω⟫_ℝ ∂fs.P = 0 := by
    intro i hi l hl hil
    rcases lt_or_gt_of_ne hil with hilt | hli
    · have hquery_l :
          Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + l)]
            (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
              (⟨y ω, hy_mem ω⟩ : fs.X)), eps i ω)) := by
        have hbase_l :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + l)]
              (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
                (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
          hquery0.mono
            ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + l))
            le_rfl
        have hx_l :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + l)]
              (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
          measurable_fst.comp (measurable_fst.comp hbase_l)
        have hy_l :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + l)]
              (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
          measurable_snd.comp (measurable_fst.comp hbase_l)
        have heps_i_l :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + l)]
              (eps i) :=
          heps_meas_at (i := i) (R := r0 + l) (by omega)
        exact (hx_l.prodMk hy_l).prodMk heps_i_l
      simpa [eps, μfun] using
        finite_component_residual_inner_cross_zero_of_prefix_measurable
          (fs := fs) (r := r0 + l) (x := x) (y := y) (d := eps i)
          hx_mem hy_mem hquery_l
    · have hquery_i :
          Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
            (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
              (⟨y ω, hy_mem ω⟩ : fs.X)), eps l ω)) := by
        have hbase_i :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
              (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
                (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
          hquery0.mono
            ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
            le_rfl
        have hx_i :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
              (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
          measurable_fst.comp (measurable_fst.comp hbase_i)
        have hy_i :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
              (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
          measurable_snd.comp (measurable_fst.comp hbase_i)
        have heps_l_i :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
              (eps l) :=
          heps_meas_at (i := l) (R := r0 + i) (by omega)
        exact (hx_i.prodMk hy_i).prodMk heps_l_i
      have hzero :=
        finite_component_residual_inner_cross_zero_of_prefix_measurable
          (fs := fs) (r := r0 + i) (x := x) (y := y) (d := eps l)
          hx_mem hy_mem hquery_i
      simpa [eps, μfun, real_inner_comm] using hzero
  have hbpos : 0 < fs.b :=
    Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hb_pos
  have havg :
      ∫ ω, ‖(fs.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)‖ ^ 2 ∂fs.P ≤
        σ2 / (fs.b : ℝ) :=
    prefixMeasurable_miniBatchResidual_secondMoment_le
      (P := fs.P)
      (filt := SOptLib.filtration fs.sample fs.hsample_meas)
      (r0 := r0) (b := fs.b) (eps := eps) (σ2 := σ2)
      hbpos
      (fun {i R} hiR => heps_meas_at (i := i) (R := R) hiR)
      (fun i hi => hdiag i (by simpa [I] using hi))
      (fun i hi j hj hij =>
        hcross i (by simpa [I] using hi) j (by simpa [I] using hj) hij)
  have hright :
      σ2 / (fs.b : ℝ) =
        fs.L ^ 2 / fs.b * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂fs.P := by
    dsimp [σ2]
    ring
  simpa [I, eps, μfun, hright] using havg

/-! Finite-sum LMO updates preserve prefix measurability.

Candidate audit: considered SOptLib
`recursive_process_measurable_of_measurable_update` and target stochastic
`iterUpdateOfWellDefined_measurable`; the SOptLib helper is abstract over a
state driver, while this local lemma is the direct Algorithm 7.12 affine LMO
update used by the finite adaptedness proof. -/
lemma finite_iterUpdate_measurable
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {cutoff k : ℕ} {x G : Ω → E}
    (hx : Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq cutoff] x)
    (hG : Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq cutoff] G) :
    Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq cutoff]
      (fun ω => fs.iterUpdate (x ω) (G ω) k) := by
  have hy :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq cutoff]
        (fun ω => fs.linearMinimizer (G ω)) :=
    fs.linearMinimizer_measurable.comp hG
  simpa [SOptLib.FiniteSumConditionalGradientSetup.iterUpdate] using
    ((hx.const_smul (1 - fs.α k)).add (hy.const_smul (fs.α k)))

/-! Finite-sum process state adaptedness before the recursive mini-batch at
sample cutoff `k * b`.

Candidate audit: checked SOptLib
`process_prefix_measurable_wrt_sampleBlock`,
`recursive_process_measurable_of_measurable_update`, and the target stochastic
`processOfWellDefined_pair_measurable_recursive_cutoff`. The abstract SOptLib
lemmas do not encode Algorithm 7.12's exact-refresh branch and importance
sampled recursive estimator, so this is the finite-sum specialization needed
for Lan Lemma 7.4's prefix/fresh split. -/
lemma finite_process_pair_measurable_recursive_cutoff_of_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (k : ℕ) (hkN : k ≤ fs.N) :
    Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
      (fun ω => (fs.iterProcess k ω, fs.estimatorProcess k ω)) := by
  classical
  exact
    recursive_process_pair_measurable_at_sample_cutoff_of_refresh_or_minibatch
      (filt := SOptLib.filtration fs.sample fs.hsample_meas)
      (sampleCutoff := fun k => k * fs.b)
      (state := fun k ω => (fs.iterProcess k ω, fs.estimatorProcess k ω))
      (nextState := fun n ω =>
        fs.iterUpdate (fs.iterProcess (n + 1) ω)
          (fs.estimatorProcess (n + 1) ω) (n + 1))
      (refreshEstimator := fun n ω =>
        fs.gradf
          (fs.iterUpdate (fs.iterProcess (n + 1) ω)
            (fs.estimatorProcess (n + 1) ω) (n + 1)))
      (recursiveEstimator := fun n ω =>
        (fs.b : ℝ)⁻¹ •
            Finset.sum (Finset.range fs.b)
              (fun i =>
                let idx := fs.sample ((n + 1) * fs.b + i) ω
                ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp idx
                      (fs.iterUpdate (fs.iterProcess (n + 1) ω)
                        (fs.estimatorProcess (n + 1) ω) (n + 1)) -
                    fs.gradFcomp idx (fs.iterProcess (n + 1) ω))) +
          fs.estimatorProcess (n + 1) ω)
      (isRefresh := fun n => n % fs.T = 0)
      (N := fs.N) (k := k)
      (by
        intro n hn
        exact Nat.mul_le_mul_right fs.b (Nat.le_succ n))
      (by
        simpa [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
          SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
          SOptLib.FiniteSumConditionalGradientSetup.process] using
          (measurable_const : Measurable fun _ : Ω => (fs.x₁, (0 : E))))
      (by
        intro hN
        have hx :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (1 * fs.b)]
              (fun _ : Ω => fs.x₁) :=
          measurable_const
        have hG :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (1 * fs.b)]
              (fun _ : Ω => fs.gradf fs.x₁) :=
          measurable_const
        simpa [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
          SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
          SOptLib.FiniteSumConditionalGradientSetup.process] using
          hx.prodMk hG)
      (by
        intro n hn hpair_prev
        have hx_prev :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fs.iterProcess (n + 1)) :=
          measurable_fst.comp hpair_prev
        have hG_prev :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fs.estimatorProcess (n + 1)) :=
          measurable_snd.comp hpair_prev
        exact finite_iterUpdate_measurable (fs := fs) (cutoff := (n + 2) * fs.b)
          hx_prev hG_prev)
      (by
        intro n hn hstart hx_next
        have hx_next_mem :
            ∀ ω,
              fs.iterUpdate (fs.iterProcess (n + 1) ω)
                  (fs.estimatorProcess (n + 1) ω) (n + 1) ∈ fs.X := by
          intro ω
          have hmem :
              fs.iterProcess (n + 2) ω ∈ fs.X :=
            fs.iterProcess_mem_of_alpha_le_one hα_le_one (n + 2) ω (by omega)
          simpa [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
            SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
            SOptLib.FiniteSumConditionalGradientSetup.process] using hmem
        have hxX :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fun ω =>
                (⟨fs.iterUpdate (fs.iterProcess (n + 1) ω)
                  (fs.estimatorProcess (n + 1) ω) (n + 1),
                  hx_next_mem ω⟩ : fs.X)) :=
          hx_next.subtype_mk
        exact fs.gradfOnX_measurable.comp hxX)
      (by
        intro n hn hstart hpair_prev hx_next
        have hx_prev :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fs.iterProcess (n + 1)) :=
          measurable_fst.comp hpair_prev
        have hG_prev :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fs.estimatorProcess (n + 1)) :=
          measurable_snd.comp hpair_prev
        have hx_next_mem :
            ∀ ω,
              fs.iterUpdate (fs.iterProcess (n + 1) ω)
                  (fs.estimatorProcess (n + 1) ω) (n + 1) ∈ fs.X := by
          intro ω
          have hmem :
              fs.iterProcess (n + 2) ω ∈ fs.X :=
            fs.iterProcess_mem_of_alpha_le_one hα_le_one (n + 2) ω (by omega)
          simpa [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
            SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
            SOptLib.FiniteSumConditionalGradientSetup.process] using hmem
        have hkernel :
            Measurable
              (fun p : (fs.X × fs.X) × Fin fs.componentCount =>
                ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp p.2 (p.1.1 : E) -
                    fs.gradFcomp p.2 (p.1.2 : E))) := by
          have hbranch :
              ∀ i : Fin fs.componentCount,
                Measurable
                  (fun q : fs.X × fs.X =>
                    ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
                      (fs.gradFcomp i (q.1 : E) - fs.gradFcomp i (q.2 : E))) := by
            intro i
            exact ((fs.gradFcompOnX_measurable i).comp measurable_fst).sub
              ((fs.gradFcompOnX_measurable i).comp measurable_snd) |>.const_smul
                ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹)
          exact measurable_from_prod_countable_left hbranch
        have hx_prev_mem : ∀ ω, fs.iterProcess (n + 1) ω ∈ fs.X := by
          intro ω
          exact fs.iterProcess_mem_of_alpha_le_one hα_le_one (n + 1) ω (by omega)
        have hxNextX :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fun ω =>
                (⟨fs.iterUpdate (fs.iterProcess (n + 1) ω)
                  (fs.estimatorProcess (n + 1) ω) (n + 1),
                  hx_next_mem ω⟩ : fs.X)) :=
          hx_next.subtype_mk
        have hxPrevX :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fun ω => (⟨fs.iterProcess (n + 1) ω, hx_prev_mem ω⟩ : fs.X)) :=
          hx_prev.subtype_mk
        have hsample_i :
            ∀ i ∈ Finset.range fs.b,
              Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
                (fs.sample ((n + 1) * fs.b + i)) := by
          intro i hi
          have hi_lt : i < fs.b := by simpa using Finset.mem_range.mp hi
          have hs :
              Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (((n + 1) * fs.b + i) + 1)]
                (fs.sample ((n + 1) * fs.b + i)) :=
            SOptLib.measurable_sample_le_prefixFiltration fs.sample fs.hsample_meas
              ((n + 1) * fs.b + i)
          have hle : ((n + 1) * fs.b + i) + 1 ≤ (n + 2) * fs.b := by
            have hbpos : 0 < fs.b := Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hb_pos
            nlinarith
          exact hs.mono
            ((SOptLib.filtration fs.sample fs.hsample_meas).mono hle) le_rfl
        have hsum :
            Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
              (fun ω =>
                Finset.sum (Finset.range fs.b)
                  (fun i =>
                    let idx := fs.sample ((n + 1) * fs.b + i) ω
                    ((fs.componentQ idx * (fs.componentCount : ℝ))⁻¹) •
                      (fs.gradFcomp idx
                          (fs.iterUpdate (fs.iterProcess (n + 1) ω)
                            (fs.estimatorProcess (n + 1) ω) (n + 1)) -
                        fs.gradFcomp idx (fs.iterProcess (n + 1) ω)))) := by
          refine Finset.measurable_sum _ ?_
          intro i hi
          have hpair :
              Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq ((n + 2) * fs.b)]
                (fun ω =>
                  (((⟨fs.iterUpdate (fs.iterProcess (n + 1) ω)
                      (fs.estimatorProcess (n + 1) ω) (n + 1),
                      hx_next_mem ω⟩ : fs.X),
                    (⟨fs.iterProcess (n + 1) ω, hx_prev_mem ω⟩ : fs.X)),
                    fs.sample ((n + 1) * fs.b + i) ω)) :=
            (hxNextX.prodMk hxPrevX).prodMk (hsample_i i hi)
          simpa using hkernel.comp hpair
        exact (hsum.const_smul ((fs.b : ℝ)⁻¹)).add hG_prev)
      (by
        intro n hn hstart
        funext ω
        simp [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
          SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
          SOptLib.FiniteSumConditionalGradientSetup.process, hstart])
      (by
        intro n hn hstart
        funext ω
        simp [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
          SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
          SOptLib.FiniteSumConditionalGradientSetup.process,
          SOptLib.FiniteSumConditionalGradientSetup.recursiveGrad, hstart])
      hkN

/-- Wolfe-gap fibers along the generated finite-sum process are integrable.

Candidate audit: chose SOptLib `integrable_of_measurable_bounded_real` after
checking compact-bound candidates and the target-file adaptedness helpers.
This is the implicit well-posedness bridge behind Theorem 7.16's expected
Wolfe gap: measurability comes from the LMO value representation, and boundedness
from `fs.gradf_smooth`, `fs.barDX_bound`, and generated feasibility. -/
theorem finite_wolfeGap_iterProcess_integrable_of_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (k : ℕ) (hkN : k ≤ fs.N) :
    Integrable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)) fs.P := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  have hpair :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
        (fun ω => (fs.iterProcess k ω, fs.estimatorProcess k ω)) :=
    finite_process_pair_measurable_recursive_cutoff_of_le
      (fs := fs) hα_le_one k hkN
  have hiter_fil :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
        (fs.iterProcess k) :=
    measurable_fst.comp hpair
  have hiter : Measurable (fs.iterProcess k) :=
    hiter_fil.mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).le (k * fs.b)) le_rfl
  have hmem : ∀ ω, fs.iterProcess k ω ∈ fs.X :=
    fun ω => fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hkN
  have hiterX : Measurable
      (fun ω => (⟨fs.iterProcess k ω, hmem ω⟩ : fs.X)) :=
    hiter.subtype_mk
  have hmeas : Measurable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)) :=
    fs.wolfeGapOnX_measurable.comp hiterX
  let C : ℝ := (‖fs.gradf fs.x₁‖ + fs.L * fs.barDX) * fs.barDX
  have hbound : ∀ ω, ‖SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)‖ ≤ C := by
    intro ω
    have hx : fs.iterProcess k ω ∈ fs.X := hmem ω
    have hgap_eq := fs.wolfeGap_eq_linearMinimizer (fs.iterProcess k ω) hx
    let y : E := fs.linearMinimizer (fs.gradf (fs.iterProcess k ω))
    have hy : y ∈ fs.X := fs.linearMinimizer_mem (fs.gradf (fs.iterProcess k ω))
    have hdiam : ‖fs.iterProcess k ω - y‖ ≤ fs.barDX :=
      fs.barDX_bound (fs.iterProcess k ω) y hx hy
    have hgrad : ‖fs.gradf (fs.iterProcess k ω)‖ ≤
        ‖fs.gradf fs.x₁‖ + fs.L * fs.barDX :=
      fs.gradf_norm_bound_on_X (fs.iterProcess k ω) hx
    have hgrad_nonneg : 0 ≤ ‖fs.gradf fs.x₁‖ + fs.L * fs.barDX := by
      have hL_nonneg : 0 ≤ fs.L := le_of_lt fs.hL_pos
      have hDX_nonneg : 0 ≤ fs.barDX := norm_nonneg _
      positivity
    calc
      ‖SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)‖ =
          |⟪fs.gradf (fs.iterProcess k ω), fs.iterProcess k ω - y⟫_ℝ| := by
        rw [hgap_eq]
        simp [Real.norm_eq_abs, y]
      _ ≤ ‖fs.gradf (fs.iterProcess k ω)‖ * ‖fs.iterProcess k ω - y‖ :=
        abs_real_inner_le_norm _ _
      _ ≤ (‖fs.gradf fs.x₁‖ + fs.L * fs.barDX) * fs.barDX := by
        exact mul_le_mul hgrad hdiam (norm_nonneg _) hgrad_nonneg
      _ = C := by
        rfl
  exact integrable_of_measurable_bounded_real hmeas hbound

/-- Objective values along generated finite-sum iterates are integrable.

Candidate audit: searched compact bounded measurable real integrability and
checked SOptLib `exists_nonneg_norm_bound_of_isCompact_of_continuousOn`,
`continuous_subtype_of_continuousOn_ambient`, and
`integrable_of_measurable_bounded_real`. These generic helpers exactly cover
the technical expectation bridge; no paper-specific primitive is needed. -/
theorem finite_f_iterProcess_integrable_of_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (k : ℕ) (hkN : k ≤ fs.N) :
    Integrable (fun ω => fs.f (fs.iterProcess k ω)) fs.P := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  have hpair :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
        (fun ω => (fs.iterProcess k ω, fs.estimatorProcess k ω)) :=
    finite_process_pair_measurable_recursive_cutoff_of_le
      (fs := fs) hα_le_one k hkN
  have hiter_fil :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
        (fs.iterProcess k) :=
    measurable_fst.comp hpair
  have hiter : Measurable (fs.iterProcess k) :=
    hiter_fil.mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).le (k * fs.b)) le_rfl
  have hmem : ∀ ω, fs.iterProcess k ω ∈ fs.X :=
    fun ω => fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hkN
  have hiterX : Measurable
      (fun ω => (⟨fs.iterProcess k ω, hmem ω⟩ : fs.X)) :=
    hiter.subtype_mk
  have hcontOn : ContinuousOn fs.f fs.X := by
    intro x hx
    exact (fs.gradf_hasGradientAt x hx).continuousAt.continuousWithinAt
  have hfX_meas : Measurable (fun x : fs.X => fs.f (x : E)) :=
    (continuous_subtype_of_continuousOn_ambient
      (fun x : fs.X => fs.f (x : E)) fs.f hcontOn (fun _ => rfl)).measurable
  have hmeas : Measurable (fun ω => fs.f (fs.iterProcess k ω)) :=
    hfX_meas.comp hiterX
  obtain ⟨C, _hC_nonneg, hC⟩ :=
    exists_nonneg_norm_bound_of_isCompact_of_continuousOn fs.f fs.hX_compact hcontOn
  exact integrable_of_measurable_bounded_real hmeas
    (C := C) (fun ω => hC (fs.iterProcess k ω) (hmem ω))

/-- Successor objective values after generated output-window steps are
integrable, including the final `N+1` value obtained by one more affine update.

Candidate audit: searched target-file objective integrability and SOptLib
bounded-measurable helpers. `finite_f_iterProcess_integrable_of_le` handles
indices `≤ N`; this helper is the route-local successor bridge using
`iterProcess_succ_eq_iterUpdate` and the measurable LMO update. -/
theorem finite_f_iterProcess_succ_integrable_of_mem
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 fs.N) :
    Integrable (fun ω => fs.f (fs.iterProcess (k + 1) ω)) fs.P := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  have hk_bounds := Finset.mem_Icc.mp hk
  have hpair_fil :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
        (fun ω => (fs.iterProcess k ω, fs.estimatorProcess k ω)) :=
    finite_process_pair_measurable_recursive_cutoff_of_le
      (fs := fs) hα_le_one k hk_bounds.2
  have hpair : Measurable (fun ω => (fs.iterProcess k ω, fs.estimatorProcess k ω)) :=
    hpair_fil.mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).le (k * fs.b)) le_rfl
  have hupdate_meas : Measurable (fun q : E × E => fs.iterUpdate q.1 q.2 k) := by
    have hx : Measurable (fun q : E × E => (1 - fs.α k) • q.1) :=
      measurable_fst.const_smul (1 - fs.α k)
    have hy : Measurable (fun q : E × E =>
        fs.α k • fs.linearMinimizer q.2) :=
      (fs.linearMinimizer_measurable.comp measurable_snd).const_smul (fs.α k)
    simpa [SOptLib.FiniteSumConditionalGradientSetup.iterUpdate] using hx.add hy
  have hnext_meas : Measurable (fs.iterProcess (k + 1)) := by
    have hraw : Measurable
        (fun ω => fs.iterUpdate (fs.iterProcess k ω) (fs.estimatorProcess k ω) k) :=
      hupdate_meas.comp hpair
    convert hraw using 1
    funext ω
    exact fs.iterProcess_succ_eq_iterUpdate (k := k) hk_bounds.1 ω
  have hmem : ∀ ω, fs.iterProcess (k + 1) ω ∈ fs.X := by
    intro ω
    rw [fs.iterProcess_succ_eq_iterUpdate (k := k) hk_bounds.1 ω]
    have hx : fs.iterProcess k ω ∈ fs.X :=
      fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hk_bounds.2
    exact fs.iterUpdate_mem_of_alpha_le_one
      (x := fs.iterProcess k ω) (G := fs.estimatorProcess k ω) (k := k)
      hx hk hα_le_one
  have hiterX : Measurable
      (fun ω => (⟨fs.iterProcess (k + 1) ω, hmem ω⟩ : fs.X)) :=
    hnext_meas.subtype_mk
  have hcontOn : ContinuousOn fs.f fs.X := by
    intro x hx
    exact (fs.gradf_hasGradientAt x hx).continuousAt.continuousWithinAt
  have hfX_meas : Measurable (fun x : fs.X => fs.f (x : E)) :=
    (continuous_subtype_of_continuousOn_ambient
      (fun x : fs.X => fs.f (x : E)) fs.f hcontOn (fun _ => rfl)).measurable
  have hmeas : Measurable (fun ω => fs.f (fs.iterProcess (k + 1) ω)) :=
    hfX_meas.comp hiterX
  obtain ⟨C, _hC_nonneg, hC⟩ :=
    exists_nonneg_norm_bound_of_isCompact_of_continuousOn fs.f fs.hX_compact hcontOn
  exact integrable_of_measurable_bounded_real hmeas
    (C := C) (fun ω => hC (fs.iterProcess (k + 1) ω) (hmem ω))

/-! Current finite-sum within-epoch iterate is measurable before the fresh
recursive mini-batch that defines its estimator.

Candidate audit: this is the finite analogue of target-file
`iterProcessOfWellDefined_globalIndex_measurable_recursive_cutoff`; SOptLib's
generic recursive adaptedness candidates do not encode the Algorithm 7.12
update-before-sampling schedule. -/
lemma finite_iterProcess_globalIndex_measurable_recursive_cutoff
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T)
    (hjN : fs.globalIndex s j ≤ fs.N) :
    Measurable[
      (SOptLib.filtration fs.sample fs.hsample_meas).seq
        (fs.globalIndex s (j - 1) * fs.b)]
      (fs.iterProcess (fs.globalIndex s j)) := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.globalIndex,
    SOptLib.global_index] using
    within_epoch_iterate_measurable_before_recursive_mini_batch
      (filt := SOptLib.filtration fs.sample fs.hsample_meas)
      (T := fs.T) (b := fs.b) (N := fs.N)
      (iter := fs.iterProcess)
      (estimator := fs.estimatorProcess)
      (update := fun x G k => fs.iterUpdate x G k)
      (h_pair := by
        intro k hkN
        exact finite_process_pair_measurable_recursive_cutoff_of_le
          (fs := fs) hα_le_one k hkN)
      (h_update_meas := by
        intro cutoff n hx hG
        exact finite_iterUpdate_measurable (fs := fs) (cutoff := cutoff) hx hG)
      (h_iter_update := by
        intro n hmod
        funext ω
        simp [SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
          SOptLib.FiniteSumConditionalGradientSetup.estimatorProcess,
          SOptLib.FiniteSumConditionalGradientSetup.process, hmod])
      (s := s) (j := j) hj2 hjT hjN

/-! Algorithm 7.12's finite-sum recursive mini-batch increment has the
Lan Lemma 7.4 second-moment scale.

Candidate audit: this consumes the proved finite prefix adaptedness helper
`finite_iterProcess_globalIndex_measurable_recursive_cutoff` and the SOptLib
mini-batch bridge wrapped in
`finite_prefix_minibatch_increment_second_moment_le`; the stochastic analogue
`recursive_minibatch_increment_second_moment_le` uses a different sample stream
and therefore cannot be reused directly. -/
lemma finite_recursive_minibatch_increment_second_moment_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T)
    (hjN : fs.globalIndex s j ≤ fs.N) :
    ∫ ω,
        ‖(fs.b : ℝ)⁻¹ •
          Finset.sum (Finset.range fs.b)
            (fun i =>
              (((fs.componentQ
                    (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω) *
                    (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp
                      (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω)
                      (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.gradFcomp
                      (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω)
                      (fs.iterProcess (fs.globalIndex s (j - 1)) ω))) -
                (fs.gradf (fs.iterProcess (fs.globalIndex s j) ω) -
                  fs.gradf (fs.iterProcess (fs.globalIndex s (j - 1)) ω)))‖ ^ 2
        ∂fs.P ≤
      fs.L ^ 2 / fs.b *
        ∫ ω,
          ‖fs.iterProcess (fs.globalIndex s j) ω -
            fs.iterProcess (fs.globalIndex s (j - 1)) ω‖ ^ 2 ∂fs.P := by
  classical
  let r0 := fs.globalIndex s (j - 1) * fs.b
  let x : Ω → E := fs.iterProcess (fs.globalIndex s j)
  let y : Ω → E := fs.iterProcess (fs.globalIndex s (j - 1))
  have hprevN : fs.globalIndex s (j - 1) ≤ fs.N :=
    fs.globalIndex_prefix_le_of_le (s := s) (i := j - 1) (t := j) (by omega) hjN
  have hx_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] x := by
    simpa [x, r0] using
      finite_iterProcess_globalIndex_measurable_recursive_cutoff
        (fs := fs) hα_le_one s j hj2 hjT hjN
  have hpair_prev :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω =>
          (fs.iterProcess (fs.globalIndex s (j - 1)) ω,
            fs.estimatorProcess (fs.globalIndex s (j - 1)) ω)) := by
    simpa [r0] using
      finite_process_pair_measurable_recursive_cutoff_of_le
        (fs := fs) hα_le_one (fs.globalIndex s (j - 1)) hprevN
  have hy_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ fs.X := by
    intro ω
    simpa [x] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one (fs.globalIndex s j) ω hjN
  have hy_mem : ∀ ω, y ω ∈ fs.X := by
    intro ω
    simpa [y] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one
        (fs.globalIndex s (j - 1)) ω hprevN
  have hx_meas : Measurable x :=
    hx_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hy_meas : Measurable y :=
    hy_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hxy_sq :
      Integrable (fun ω => ‖x ω - y ω‖ ^ 2) fs.P := by
    haveI : IsProbabilityMeasure fs.P := fs.hP
    have hZ_meas : Measurable (fun ω => ‖x ω - y ω‖ ^ 2) :=
      ((hx_meas.sub hy_meas).norm.pow_const 2)
    exact integrable_of_measurable_bounded_real hZ_meas
      (C := fs.barDX ^ 2) (by
        intro ω
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        have hdiam : ‖x ω - y ω‖ ≤ fs.barDX :=
          fs.barDX_bound (x ω) (y ω) (hx_mem ω) (hy_mem ω)
        nlinarith [hdiam, norm_nonneg (x ω - y ω), sq_nonneg fs.barDX])
  have hquery0 :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)),
          fs.gradf (x ω) - fs.gradf (y ω))) := by
    have hxX :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
      hx_cut.subtype_mk
    have hyX :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
      hy_cut.subtype_mk
    have hμ :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => fs.gradf (x ω) - fs.gradf (y ω)) :=
      (fs.gradfOnX_measurable.comp hxX).sub
        (fs.gradfOnX_measurable.comp hyX)
    exact (hxX.prodMk hyX).prodMk hμ
  simpa [r0, x, y] using
    finite_prefix_minibatch_increment_second_moment_le
      (fs := fs) (r0 := r0) (x := x) (y := y)
      hx_mem hy_mem hquery0 hxy_sq

/-! The averaged fresh recursive finite-sum mini-batch increment is
square-integrable.

Candidate audit: SOptLib `integrable_sq_norm_centeredMiniBatchAverage` is the
matching abstract finite-average L2 closure lemma. The target-file stochastic
`recursive_minibatch_increment_sq_integrable` uses `setup.ξ`; the finite
Algorithm 7.12 object instead needs `finite_random_query_residual_second_moment_le`
with the component-index stream `fs.sample`. -/
lemma finite_recursive_minibatch_increment_sq_integrable
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T)
    (hjN : fs.globalIndex s j ≤ fs.N) :
    Integrable
      (fun ω =>
        ‖(fs.b : ℝ)⁻¹ •
          Finset.sum (Finset.range fs.b)
            (fun i =>
              (((fs.componentQ
                    (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω) *
                    (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp
                      (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω)
                      (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.gradFcomp
                      (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω)
                      (fs.iterProcess (fs.globalIndex s (j - 1)) ω))) -
                (fs.gradf (fs.iterProcess (fs.globalIndex s j) ω) -
                  fs.gradf (fs.iterProcess (fs.globalIndex s (j - 1)) ω)))‖ ^ 2)
      fs.P := by
  classical
  let I : Finset ℕ := Finset.range fs.b
  let r0 := fs.globalIndex s (j - 1) * fs.b
  let x : Ω → E := fs.iterProcess (fs.globalIndex s j)
  let y : Ω → E := fs.iterProcess (fs.globalIndex s (j - 1))
  let μfun : Ω → E := fun ω => fs.gradf (x ω) - fs.gradf (y ω)
  let eps : ℕ → Ω → E := fun i ω =>
    (((fs.componentQ (fs.sample (r0 + i) ω) * (fs.componentCount : ℝ))⁻¹) •
      (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
        fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) -
      μfun ω
  let avg : Ω → E := fun ω =>
    (fs.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)
  have hprevN : fs.globalIndex s (j - 1) ≤ fs.N :=
    fs.globalIndex_prefix_le_of_le (s := s) (i := j - 1) (t := j) (by omega) hjN
  have hx_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] x := by
    simpa [x, r0] using
      finite_iterProcess_globalIndex_measurable_recursive_cutoff
        (fs := fs) hα_le_one s j hj2 hjT hjN
  have hpair_prev :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω =>
          (fs.iterProcess (fs.globalIndex s (j - 1)) ω,
            fs.estimatorProcess (fs.globalIndex s (j - 1)) ω)) := by
    simpa [r0] using
      finite_process_pair_measurable_recursive_cutoff_of_le
        (fs := fs) hα_le_one (fs.globalIndex s (j - 1)) hprevN
  have hy_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ fs.X := by
    intro ω
    simpa [x] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one (fs.globalIndex s j) ω hjN
  have hy_mem : ∀ ω, y ω ∈ fs.X := by
    intro ω
    simpa [y] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one
        (fs.globalIndex s (j - 1)) ω hprevN
  have hx_meas : Measurable x :=
    hx_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hy_meas : Measurable y :=
    hy_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hxy_sq :
      Integrable (fun ω => ‖x ω - y ω‖ ^ 2) fs.P := by
    haveI : IsProbabilityMeasure fs.P := fs.hP
    have hZ_meas : Measurable (fun ω => ‖x ω - y ω‖ ^ 2) :=
      ((hx_meas.sub hy_meas).norm.pow_const 2)
    exact integrable_of_measurable_bounded_real hZ_meas
      (C := fs.barDX ^ 2) (by
        intro ω
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        have hdiam : ‖x ω - y ω‖ ≤ fs.barDX :=
          fs.barDX_bound (x ω) (y ω) (hx_mem ω) (hy_mem ω)
        nlinarith [hdiam, norm_nonneg (x ω - y ω), sq_nonneg fs.barDX])
  have hquery_mu :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) := by
    have hxX :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
      hx_cut.subtype_mk
    have hyX :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
      hy_cut.subtype_mk
    have hμ :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] μfun := by
      dsimp [μfun]
      exact (fs.gradfOnX_measurable.comp hxX).sub
        (fs.gradfOnX_measurable.comp hyX)
    exact (hxX.prodMk hyX).prodMk hμ
  have hdiag :
      ∀ i ∈ I, Integrable (fun ω => ‖eps i ω‖ ^ 2) fs.P := by
    intro i hi
    have hqi :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
          (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
            (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
      hquery_mu.mono
        ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
        le_rfl
    simpa [eps, μfun] using
      (finite_random_query_residual_second_moment_le
        (fs := fs) (r := r0 + i) (x := x) (y := y)
        hx_mem hy_mem hqi hxy_sq).1
  have hresidual_meas :
      Measurable
        (fun p : (((fs.X × fs.X) × E) × Fin fs.componentCount) =>
          ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp p.2 (p.1.1.1 : E) -
                fs.gradFcomp p.2 (p.1.1.2 : E)) -
            p.1.2) := by
    have hkernel :
        Measurable
          (fun p : (fs.X × fs.X) × Fin fs.componentCount =>
            ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp p.2 (p.1.1 : E) -
                fs.gradFcomp p.2 (p.1.2 : E))) := by
      have hbranch :
          ∀ i : Fin fs.componentCount,
            Measurable
              (fun q : fs.X × fs.X =>
                ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp i (q.1 : E) - fs.gradFcomp i (q.2 : E))) := by
        intro i
        exact ((fs.gradFcompOnX_measurable i).comp measurable_fst).sub
          ((fs.gradFcompOnX_measurable i).comp measurable_snd) |>.const_smul
            ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹)
      exact measurable_from_prod_countable_left hbranch
    have hkernel_lift :
        Measurable
          (fun p : (((fs.X × fs.X) × E) × Fin fs.componentCount) =>
            ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp p.2 (p.1.1.1 : E) -
                fs.gradFcomp p.2 (p.1.1.2 : E))) :=
      hkernel.comp ((measurable_fst.comp measurable_fst).prodMk measurable_snd)
    exact hkernel_lift.sub (measurable_snd.comp measurable_fst)
  have havg_sq : Integrable (fun ω => ‖avg ω‖ ^ 2) fs.P := by
    simpa [avg, eps, μfun] using
      (integrable_sq_norm_randomQuery_centeredMiniBatchAverage
        (P := fs.P)
        (filt := SOptLib.filtration fs.sample fs.hsample_meas)
        (I := I) (m := fs.b) (r0 := r0)
        (query := fun ω =>
          (((⟨x ω, hx_mem ω⟩ : fs.X), (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω))
        (sample := fun i ω => fs.sample (r0 + i) ω)
        (centeredResidual := fun q z =>
          ((fs.componentQ q * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp q (z.1.1 : E) -
                fs.gradFcomp q (z.1.2 : E)) -
            z.2)
        hquery_mu
        (fun i _hi => fs.sample_measurable (r0 + i))
        hresidual_meas
        (by
          intro i hi
          simpa [eps, μfun] using hdiag i hi))
  simpa [avg, I, eps, μfun, r0, x, y] using havg_sq

/-! The previous finite-sum estimator error is orthogonal in expectation to
the fresh recursive mini-batch increment.

Candidate audit: this specializes the proved prefix/fresh bridge
`finite_component_residual_inner_cross_zero_of_prefix_measurable` and SOptLib
`integral_finset_sum_const_mul_eq_zero`; stochastic helpers with
`setup.ξ` cannot be reused for Algorithm 7.12's component-index stream
`fs.sample`. -/
lemma finite_delta_prev_minibatch_increment_cross_zero
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T)
    (hjN : fs.globalIndex s j ≤ fs.N)
    (hprev_sq :
      Integrable
        (fun ω =>
          ‖fs.deltaProcess (fs.globalIndex s (j - 1)) ω‖ ^ 2) fs.P) :
    ∫ ω,
        ⟪fs.deltaProcess (fs.globalIndex s (j - 1)) ω,
          (fs.b : ℝ)⁻¹ •
            Finset.sum (Finset.range fs.b)
              (fun i =>
                (((fs.componentQ
                      (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω) *
                      (fs.componentCount : ℝ))⁻¹) •
                    (fs.gradFcomp
                        (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω)
                        (fs.iterProcess (fs.globalIndex s j) ω) -
                      fs.gradFcomp
                        (fs.sample (fs.globalIndex s (j - 1) * fs.b + i) ω)
                        (fs.iterProcess (fs.globalIndex s (j - 1)) ω))) -
                  (fs.gradf (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.gradf (fs.iterProcess (fs.globalIndex s (j - 1)) ω)))⟫_ℝ
      ∂fs.P = 0 := by
  classical
  let I : Finset ℕ := Finset.range fs.b
  let r0 := fs.globalIndex s (j - 1) * fs.b
  let x : Ω → E := fs.iterProcess (fs.globalIndex s j)
  let y : Ω → E := fs.iterProcess (fs.globalIndex s (j - 1))
  let d : Ω → E := fs.deltaProcess (fs.globalIndex s (j - 1))
  let μfun : Ω → E := fun ω => fs.gradf (x ω) - fs.gradf (y ω)
  let eps : ℕ → Ω → E := fun i ω =>
    (((fs.componentQ (fs.sample (r0 + i) ω) * (fs.componentCount : ℝ))⁻¹) •
      (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
        fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) -
      μfun ω
  have hprevN : fs.globalIndex s (j - 1) ≤ fs.N :=
    fs.globalIndex_prefix_le_of_le (s := s) (i := j - 1) (t := j) (by omega) hjN
  have hx_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] x := by
    simpa [x, r0] using
      finite_iterProcess_globalIndex_measurable_recursive_cutoff
        (fs := fs) hα_le_one s j hj2 hjT hjN
  have hpair_prev :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω =>
          (fs.iterProcess (fs.globalIndex s (j - 1)) ω,
            fs.estimatorProcess (fs.globalIndex s (j - 1)) ω)) := by
    simpa [r0] using
      finite_process_pair_measurable_recursive_cutoff_of_le
        (fs := fs) hα_le_one (fs.globalIndex s (j - 1)) hprevN
  have hy_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hGprev_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fs.estimatorProcess (fs.globalIndex s (j - 1))) :=
    measurable_snd.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ fs.X := by
    intro ω
    simpa [x] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one (fs.globalIndex s j) ω hjN
  have hy_mem : ∀ ω, y ω ∈ fs.X := by
    intro ω
    simpa [y] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one
        (fs.globalIndex s (j - 1)) ω hprevN
  have hyX_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
    hy_cut.subtype_mk
  have hgrad_y_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => fs.gradf (y ω)) :=
    fs.gradfOnX_measurable.comp hyX_cut
  have hd_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] d := by
    dsimp [d, SOptLib.FiniteSumConditionalGradientSetup.deltaProcess]
    exact hGprev_cut.sub hgrad_y_cut
  have hx_meas : Measurable x :=
    hx_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hy_meas : Measurable y :=
    hy_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hxy_sq :
      Integrable (fun ω => ‖x ω - y ω‖ ^ 2) fs.P := by
    haveI : IsProbabilityMeasure fs.P := fs.hP
    have hZ_meas : Measurable (fun ω => ‖x ω - y ω‖ ^ 2) :=
      ((hx_meas.sub hy_meas).norm.pow_const 2)
    exact integrable_of_measurable_bounded_real hZ_meas
      (C := fs.barDX ^ 2) (by
        intro ω
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        have hdiam : ‖x ω - y ω‖ ≤ fs.barDX :=
          fs.barDX_bound (x ω) (y ω) (hx_mem ω) (hy_mem ω)
        nlinarith [hdiam, norm_nonneg (x ω - y ω), sq_nonneg fs.barDX])
  have hquery_mu :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
          (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) := by
    have hxX :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
      hx_cut.subtype_mk
    have hyX :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
          (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
      hy_cut.subtype_mk
    have hμ :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] μfun := by
      dsimp [μfun]
      exact (fs.gradfOnX_measurable.comp hxX).sub
        (fs.gradfOnX_measurable.comp hyX)
    exact (hxX.prodMk hyX).prodMk hμ
  have hdiag :
      ∀ i ∈ I, Integrable (fun ω => ‖eps i ω‖ ^ 2) fs.P := by
    intro i hi
    have hqi :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
          (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
            (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
      hquery_mu.mono
        ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
        le_rfl
    simpa [eps, μfun] using
      (finite_random_query_residual_second_moment_le
        (fs := fs) (r := r0 + i) (x := x) (y := y)
        hx_mem hy_mem hqi hxy_sq).1
  have heps_meas :
      ∀ i ∈ I, Measurable (eps i) := by
    have hkernel :
        Measurable
          (fun p : (fs.X × fs.X) × Fin fs.componentCount =>
            ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp p.2 (p.1.1 : E) -
                fs.gradFcomp p.2 (p.1.2 : E))) := by
      simpa using
        measurable_componentWeightedGradDiffKernel
          (grad := fun i (q : fs.X) => fs.gradFcomp i (q : E))
          (weight := fun i => (fs.componentQ i * (fs.componentCount : ℝ))⁻¹)
          (fun i => fs.gradFcompOnX_measurable i)
    have hquery_global :
        Measurable
          (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
            (⟨y ω, hy_mem ω⟩ : fs.X)), μfun ω)) :=
      hquery_mu.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
    have hxX_meas : Measurable (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
      measurable_fst.comp (measurable_fst.comp hquery_global)
    have hyX_meas : Measurable (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
      measurable_snd.comp (measurable_fst.comp hquery_global)
    have hμ_meas : Measurable μfun :=
      measurable_snd.comp hquery_global
    intro i _hi
    have hg :
        Measurable
          (fun ω =>
            ((fs.componentQ (fs.sample (r0 + i) ω) * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
                fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) :=
      hkernel.comp ((hxX_meas.prodMk hyX_meas).prodMk (fs.sample_measurable (r0 + i)))
    dsimp [eps]
    exact hg.sub hμ_meas
  have hZ_zero :
      ∀ i ∈ I, ∫ ω, ⟪d ω, eps i ω⟫_ℝ ∂fs.P = 0 := by
    intro i hi
    have hqi :
        Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
          (fun ω => (((⟨x ω, hx_mem ω⟩ : fs.X),
            (⟨y ω, hy_mem ω⟩ : fs.X)), d ω)) := by
      have hx_i :
          Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
            (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
        (hx_cut.subtype_mk).mono
          ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
          le_rfl
      have hy_i :
          Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)]
            (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
        (hy_cut.subtype_mk).mono
          ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
          le_rfl
      have hd_i :
          Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (r0 + i)] d :=
        hd_cut.mono
          ((SOptLib.filtration fs.sample fs.hsample_meas).mono (by omega : r0 ≤ r0 + i))
          le_rfl
      exact (hx_i.prodMk hy_i).prodMk hd_i
    simpa [d, eps, μfun] using
      finite_component_residual_inner_cross_zero_of_prefix_measurable
        (fs := fs) (r := r0 + i) (x := x) (y := y) (d := d)
        hx_mem hy_mem hqi
  have hd_meas : AEStronglyMeasurable d fs.P :=
    (hd_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl).aestronglyMeasurable
  have hmain :
      ∫ ω, ⟪d ω, (fs.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)⟫_ℝ ∂fs.P = 0 :=
    pastResidual_inner_centeredMiniBatchAverage_integral_eq_zero
      fs.P I fs.b d eps hd_meas
      (fun i hi => (heps_meas i hi).aestronglyMeasurable)
      hprev_sq hdiag hZ_zero
  simpa [I, r0, x, y, d, μfun, eps] using hmain

/-! One recursive finite-sum step of Lan Lemma 7.4: expanding
`δ_j = δ_{j-1} +` the fresh centered Algorithm 7.12 mini-batch increment gives
the second-moment recurrence after the fresh cross term vanishes.

Candidate audit: `norm_add_sq_real` supplies the Hilbert-square expansion,
`integrable_inner_of_sq_integrable` supplies cross-term integrability, and the
paper-local finite helpers `fs.deltaProcess_globalIndex_recursive_residual`,
`finite_recursive_minibatch_increment_second_moment_le`,
`finite_recursive_minibatch_increment_sq_integrable`, and
`finite_delta_prev_minibatch_increment_cross_zero` are the exact finite-sum
Algorithm 7.12 analogues of the stochastic Lemma 7.5 one-step proof. -/
lemma finite_delta_one_step_second_moment_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (s j : ℕ) (hj2 : 2 ≤ j) (hjT : j ≤ fs.T)
    (hjN : fs.globalIndex s j ≤ fs.N)
    (hprev_sq :
      Integrable
        (fun ω =>
          ‖fs.deltaProcess (fs.globalIndex s (j - 1)) ω‖ ^ 2) fs.P) :
    Integrable
        (fun ω =>
          ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2) fs.P ∧
      ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P ≤
        ∫ ω,
            ‖fs.deltaProcess (fs.globalIndex s (j - 1)) ω‖ ^ 2 ∂fs.P +
          fs.L ^ 2 / fs.b *
            ∫ ω,
              ‖fs.iterProcess (fs.globalIndex s j) ω -
                fs.iterProcess (fs.globalIndex s (j - 1)) ω‖ ^ 2 ∂fs.P := by
  classical
  let I : Finset ℕ := Finset.range fs.b
  let r0 := fs.globalIndex s (j - 1) * fs.b
  let x : Ω → E := fs.iterProcess (fs.globalIndex s j)
  let y : Ω → E := fs.iterProcess (fs.globalIndex s (j - 1))
  let d : Ω → E := fs.deltaProcess (fs.globalIndex s (j - 1))
  let μfun : Ω → E := fun ω => fs.gradf (x ω) - fs.gradf (y ω)
  let eps : ℕ → Ω → E := fun i ω =>
    (((fs.componentQ (fs.sample (r0 + i) ω) * (fs.componentCount : ℝ))⁻¹) •
      (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
        fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) -
      μfun ω
  let inc : Ω → E := fun ω =>
    (fs.b : ℝ)⁻¹ • Finset.sum I (fun i => eps i ω)
  have hinc_sq :
      Integrable (fun ω => ‖inc ω‖ ^ 2) fs.P := by
    simpa [inc, I, eps, μfun, r0, x, y] using
      finite_recursive_minibatch_increment_sq_integrable
        (fs := fs) hα_le_one s j hj2 hjT hjN
  have hinc_bound :
      ∫ ω, ‖inc ω‖ ^ 2 ∂fs.P ≤
        fs.L ^ 2 / fs.b * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂fs.P := by
    simpa [inc, I, eps, μfun, r0, x, y] using
      finite_recursive_minibatch_increment_second_moment_le
        (fs := fs) hα_le_one s j hj2 hjT hjN
  have hprevN : fs.globalIndex s (j - 1) ≤ fs.N :=
    fs.globalIndex_prefix_le_of_le (s := s) (i := j - 1) (t := j) (by omega) hjN
  have hx_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] x := by
    simpa [x, r0] using
      finite_iterProcess_globalIndex_measurable_recursive_cutoff
        (fs := fs) hα_le_one s j hj2 hjT hjN
  have hpair_prev :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω =>
          (fs.iterProcess (fs.globalIndex s (j - 1)) ω,
            fs.estimatorProcess (fs.globalIndex s (j - 1)) ω)) := by
    simpa [r0] using
      finite_process_pair_measurable_recursive_cutoff_of_le
        (fs := fs) hα_le_one (fs.globalIndex s (j - 1)) hprevN
  have hy_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] y := by
    simpa [y] using measurable_fst.comp hpair_prev
  have hGprev_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fs.estimatorProcess (fs.globalIndex s (j - 1))) :=
    measurable_snd.comp hpair_prev
  have hx_mem : ∀ ω, x ω ∈ fs.X := by
    intro ω
    simpa [x] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one (fs.globalIndex s j) ω hjN
  have hy_mem : ∀ ω, y ω ∈ fs.X := by
    intro ω
    simpa [y] using
      fs.iterProcess_mem_of_alpha_le_one hα_le_one
        (fs.globalIndex s (j - 1)) ω hprevN
  have hyX_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
    hy_cut.subtype_mk
  have hgrad_y_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0]
        (fun ω => fs.gradf (y ω)) :=
    fs.gradfOnX_measurable.comp hyX_cut
  have hd_cut :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq r0] d := by
    dsimp [d, SOptLib.FiniteSumConditionalGradientSetup.deltaProcess]
    exact hGprev_cut.sub hgrad_y_cut
  have hd_global : Measurable d :=
    hd_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
  have hd_meas : AEStronglyMeasurable d fs.P :=
    hd_global.aestronglyMeasurable
  have hinc_global : Measurable inc := by
    have hx_meas : Measurable x :=
      hx_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
    have hy_meas : Measurable y :=
      hy_cut.mono ((SOptLib.filtration fs.sample fs.hsample_meas).le r0) le_rfl
    have hxX_meas : Measurable (fun ω => (⟨x ω, hx_mem ω⟩ : fs.X)) :=
      hx_meas.subtype_mk
    have hyX_meas : Measurable (fun ω => (⟨y ω, hy_mem ω⟩ : fs.X)) :=
      hy_meas.subtype_mk
    have hμ_meas : Measurable μfun := by
      dsimp [μfun]
      exact (fs.gradfOnX_measurable.comp hxX_meas).sub
        (fs.gradfOnX_measurable.comp hyX_meas)
    have hkernel :
        Measurable
          (fun p : (fs.X × fs.X) × Fin fs.componentCount =>
            ((fs.componentQ p.2 * (fs.componentCount : ℝ))⁻¹) •
              (fs.gradFcomp p.2 (p.1.1 : E) -
                fs.gradFcomp p.2 (p.1.2 : E))) := by
      have hbranch :
          ∀ i : Fin fs.componentCount,
            Measurable
              (fun q : fs.X × fs.X =>
                ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹) •
                  (fs.gradFcomp i (q.1 : E) - fs.gradFcomp i (q.2 : E))) := by
        intro i
        exact ((fs.gradFcompOnX_measurable i).comp measurable_fst).sub
          ((fs.gradFcompOnX_measurable i).comp measurable_snd) |>.const_smul
            ((fs.componentQ i * (fs.componentCount : ℝ))⁻¹)
      exact measurable_from_prod_countable_left hbranch
    have hsum : Measurable (fun ω => Finset.sum I (fun i => eps i ω)) := by
      refine Finset.measurable_sum I ?_
      intro i _hi
      have hg :
          Measurable
            (fun ω =>
              ((fs.componentQ (fs.sample (r0 + i) ω) * (fs.componentCount : ℝ))⁻¹) •
                (fs.gradFcomp (fs.sample (r0 + i) ω) (x ω) -
                  fs.gradFcomp (fs.sample (r0 + i) ω) (y ω))) :=
        hkernel.comp ((hxX_meas.prodMk hyX_meas).prodMk (fs.sample_measurable (r0 + i)))
      dsimp [eps]
      exact hg.sub hμ_meas
    dsimp [inc]
    exact hsum.const_smul ((fs.b : ℝ)⁻¹)
  have hinc_meas : AEStronglyMeasurable inc fs.P :=
    hinc_global.aestronglyMeasurable
  have hcross :
      ∫ ω, ⟪d ω, inc ω⟫_ℝ ∂fs.P = 0 := by
    simpa [d, inc, I, eps, μfun, r0, x, y] using
      finite_delta_prev_minibatch_increment_cross_zero
        (fs := fs) hα_le_one s j hj2 hjT hjN hprev_sq
  have hrec :
      ∀ ω,
        fs.deltaProcess (fs.globalIndex s j) ω = d ω + inc ω := by
    intro ω
    simpa [d, inc, I, eps, μfun, r0, x, y] using
      fs.deltaProcess_globalIndex_recursive_residual s j hj2 hjT ω
  have hadd_sq :
      Integrable (fun ω => ‖d ω + inc ω‖ ^ 2) fs.P := by
    refine Integrable.mono' ((hprev_sq.const_mul 2).add (hinc_sq.const_mul 2))
      (((hd_global.add hinc_global).norm.pow_const 2).aestronglyMeasurable) ?_
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    simpa using
      norm_add_sq_le_two_mul_norm_sq_add_two_mul_norm_sq (d ω) (inc ω)
  have hcurrent_sq :
      Integrable
        (fun ω =>
          ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2) fs.P := by
    refine hadd_sq.congr ?_
    filter_upwards [] with ω
    rw [hrec ω]
  have hinner_int : Integrable (fun ω => ⟪d ω, inc ω⟫_ℝ) fs.P :=
    integrable_inner_of_sq_integrable hd_meas hinc_meas hprev_sq hinc_sq
  have hintegral_expand :
      ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P =
        ∫ ω, ‖d ω‖ ^ 2 ∂fs.P +
          2 * ∫ ω, ⟪d ω, inc ω⟫_ℝ ∂fs.P +
          ∫ ω, ‖inc ω‖ ^ 2 ∂fs.P := by
    have h_expand :
        (fun ω => ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2) =ᵐ[fs.P]
          (fun ω => ‖d ω‖ ^ 2 + 2 * ⟪d ω, inc ω⟫_ℝ + ‖inc ω‖ ^ 2) := by
      filter_upwards [] with ω
      rw [hrec ω]
      simpa using norm_add_sq_real (d ω) (inc ω)
    calc
      ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P =
          ∫ ω, ‖d ω‖ ^ 2 + 2 * ⟪d ω, inc ω⟫_ℝ + ‖inc ω‖ ^ 2 ∂fs.P := by
            exact integral_congr_ae h_expand
      _ = ∫ ω, (‖d ω‖ ^ 2 + 2 * ⟪d ω, inc ω⟫_ℝ) + ‖inc ω‖ ^ 2 ∂fs.P := by
            refine integral_congr_ae (Filter.Eventually.of_forall ?_)
            intro ω
            ring
      _ =
          ∫ ω, ‖d ω‖ ^ 2 + 2 * ⟪d ω, inc ω⟫_ℝ ∂fs.P +
            ∫ ω, ‖inc ω‖ ^ 2 ∂fs.P := by
            exact integral_add (hprev_sq.add (hinner_int.const_mul 2)) hinc_sq
      _ =
          (∫ ω, ‖d ω‖ ^ 2 ∂fs.P +
            ∫ ω, 2 * ⟪d ω, inc ω⟫_ℝ ∂fs.P) +
            ∫ ω, ‖inc ω‖ ^ 2 ∂fs.P := by
            rw [integral_add hprev_sq (hinner_int.const_mul 2)]
      _ =
          ∫ ω, ‖d ω‖ ^ 2 ∂fs.P +
            2 * ∫ ω, ⟪d ω, inc ω⟫_ℝ ∂fs.P +
            ∫ ω, ‖inc ω‖ ^ 2 ∂fs.P := by
            rw [integral_const_mul 2]
  constructor
  · exact hcurrent_sq
  · calc
      ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P =
          ∫ ω, ‖d ω‖ ^ 2 ∂fs.P + ∫ ω, ‖inc ω‖ ^ 2 ∂fs.P := by
            rw [hintegral_expand, hcross]
            ring
      _ ≤ ∫ ω, ‖d ω‖ ^ 2 ∂fs.P +
          fs.L ^ 2 / fs.b * ∫ ω, ‖x ω - y ω‖ ^ 2 ∂fs.P := by
            linarith [hinc_bound]

/-! Squared differences of generated finite-sum Algorithm 7.12 iterates are
integrable, using compact feasible diameter as a dominating bound.

Candidate audit: considered target-file `iterProcessOfWellDefined_diff_sq_integrable`
and SOptLib bounded-measurable integrability helpers. The stochastic helper is
for `setup.iterProcessOfWellDefined`, while this finite theorem needs the
generated-run bounds `k,l ≤ fs.N` to invoke `fs.iterProcess_mem_of_alpha_le_one`
and the finite process adaptedness helper. -/
lemma finite_iterProcess_diff_sq_integrable_of_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (k l : ℕ) (hkN : k ≤ fs.N) (hlN : l ≤ fs.N) :
    Integrable
      (fun ω =>
        ‖fs.iterProcess k ω - fs.iterProcess l ω‖ ^ 2) fs.P := by
  haveI : IsProbabilityMeasure fs.P := fs.hP
  have hpair_k :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (k * fs.b)]
        (fun ω => (fs.iterProcess k ω, fs.estimatorProcess k ω)) :=
    finite_process_pair_measurable_recursive_cutoff_of_le (fs := fs)
      hα_le_one k hkN
  have hpair_l :
      Measurable[(SOptLib.filtration fs.sample fs.hsample_meas).seq (l * fs.b)]
        (fun ω => (fs.iterProcess l ω, fs.estimatorProcess l ω)) :=
    finite_process_pair_measurable_recursive_cutoff_of_le (fs := fs)
      hα_le_one l hlN
  have hk_meas : Measurable (fs.iterProcess k) :=
    (measurable_fst.comp hpair_k).mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).le (k * fs.b)) le_rfl
  have hl_meas : Measurable (fs.iterProcess l) :=
    (measurable_fst.comp hpair_l).mono
      ((SOptLib.filtration fs.sample fs.hsample_meas).le (l * fs.b)) le_rfl
  have hZ_meas : Measurable (fun ω => ‖fs.iterProcess k ω - fs.iterProcess l ω‖ ^ 2) :=
    ((hk_meas.sub hl_meas).norm.pow_const 2)
  exact integrable_of_measurable_bounded_real hZ_meas
    (C := fs.barDX ^ 2) (by
      intro ω
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      have hkX : fs.iterProcess k ω ∈ fs.X :=
        fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hkN
      have hlX : fs.iterProcess l ω ∈ fs.X :=
        fs.iterProcess_mem_of_alpha_le_one hα_le_one l ω hlN
      have hdiam : ‖fs.iterProcess k ω - fs.iterProcess l ω‖ ≤ fs.barDX :=
        fs.barDX_bound _ _ hkX hlX
      nlinarith [hdiam, norm_nonneg (fs.iterProcess k ω - fs.iterProcess l ω),
        sq_nonneg fs.barDX])

/-! Delta square-integrability along a generated finite-sum epoch.

Candidate audit: this is the finite analogue of target-file
`epochwise_delta_sq_integrable`; it consumes the newly proved finite one-step
recurrence rather than adding an integrability hypothesis to Lan Lemma 7.4. -/
lemma finite_delta_sq_integrable
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (s t : ℕ) (ht : 1 ≤ t) (ht_le : t ≤ fs.T)
    (hkt : fs.globalIndex s t ≤ fs.N) :
    Integrable
      (fun ω =>
        ‖fs.deltaProcess (fs.globalIndex s t) ω‖ ^ 2) fs.P := by
  exact epoch_delta_sq_integrable_of_recursive_step
    (μ := fs.P) (delta := fs.deltaProcess) (index := fs.globalIndex)
    (Valid := fun s t => fs.globalIndex s t ≤ fs.N) (T := fs.T)
    (hvalid_prefix := by
      intro s i j hij hj
      exact fs.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hij hj)
    (hbase := by
      intro s _hs
      have hδ := fs.deltaProcess_globalIndex_one_eq_zero s
      rw [hδ]
      simp)
    (hstep := by
      intro s j hj2 hjT hjN hprev_sq
      exact (finite_delta_one_step_second_moment_le
        (fs := fs) hα_le_one s j hj2 hjT hjN hprev_sq).1)
    s t ht ht_le hkt

/-! `lemma_7_4` is the finite-sum recursive estimator bound imported into the
proof of `theorem_7_16`. Unlike the stochastic `lemma_7_5`, it has no
`σ² / m` refresh floor and corresponds to (7.4.4) in the textbook proof of
Theorem 7.16.

The source statement is for an iteration `k` generated by Algorithm 7.12
(`k = 1, ..., N`) and then identified with epoch coordinates `(s,t)`. The
explicit `hkt` and `hα_le_one` premises expose exactly that generated-run and
feasibility boundary; the previous unguarded arbitrary-epoch head was stronger
than the paper statement and could not use `fs.hFcomp_smooth` on generated
iterates. -/
theorem lemma_7_4
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (s t : ℕ) (ht : 1 ≤ t) (ht_le : t ≤ fs.T)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hkt : fs.globalIndex s t ≤ fs.N) :
    ∫ ω, ‖fs.deltaProcess (fs.globalIndex s t) ω‖ ^ 2 ∂fs.P ≤
      fs.L ^ 2 / fs.b *
        ∫ ω, fs.epochDiffSum s t ω ∂fs.P := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.epochDiffSum] using
    epoch_recursive_estimator_second_moment_le_difference_sum
      (μ := fs.P) (delta := fs.deltaProcess) (x := fs.iterProcess)
      (index := fs.globalIndex)
      (Valid := fun s t => fs.globalIndex s t ≤ fs.N)
      (T := fs.T) (c := fs.L ^ 2 / fs.b)
      (hvalid_prefix := by
        intro s i j hij hj
        exact fs.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hij hj)
      (hbase := by
        intro s _hs
        exact lemma_7_4_t_one (fs := fs) s)
      (hdelta_sq_int := by
        intro s j hj hjT hjN
        exact finite_delta_sq_integrable (fs := fs) hα_le_one s j hj hjT hjN)
      (hdiff_sq_int := by
        intro s i hi2 _hiT hiN
        exact finite_iterProcess_diff_sq_integrable_of_le
          (fs := fs) hα_le_one
          (fs.globalIndex s i) (fs.globalIndex s (i - 1))
          hiN
          (fs.globalIndex_prefix_le_of_le (s := s) (i := i - 1) (t := i)
            (by omega) hiN))
      (hstep := by
        intro s j hj2 hjT hjN hprev_sq
        exact (finite_delta_one_step_second_moment_le
          (fs := fs) hα_le_one s j hj2 hjT hjN hprev_sq).2)
      s t ht ht_le hkt

/-! Active-coordinate specialization of Lan Lemma 7.4 for Theorem 7.16.

Candidate audit: searched `one step expected gap descent Wolfe gap smooth`,
`finite sum active epoch partition global index`, and checked target-file
`lemma_7_4`, `activeEpochSteps_mem_epoch`, and
`activeEpochSteps_globalIndex_le`. No SOptLib helper specializes the paper's
finite estimator variance bound to Algorithm 7.12 active epoch coordinates. -/
private lemma finite_active_delta_second_moment_le
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P ≤
      fs.L ^ 2 / fs.b *
        ∫ ω, fs.epochDiffSum s j ω ∂fs.P := by
  have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
  exact lemma_7_4 (fs := fs) s j hj_epoch.1 hj_epoch.2 hα_le_one
    (fs.activeEpochSteps_globalIndex_le (s := s) (j := j) hj)

/-- Feasible-segment smooth quadratic upper bound for the finite-sum objective.

Candidate audit: checked SOptLib `Convex.carrier_smooth_quadratic_upper_bound`
and `le_value_add_of_hasDerivWithinAt_le_affine_on_Icc`. The carrier theorem
requires a `ContDiffOn`/`fderivWithin` package not exposed by this finite setup,
while the scalar integration lemma exactly matches Lan Theorem 7.16's
Eq. (7.4.5) derivation from pointwise `HasGradientAt` plus Lipschitz gradient. -/
private lemma finite_smooth_quadratic_upper_bound_of_hasGradientAt
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {x y : E} (hx : x ∈ fs.X) (hy : y ∈ fs.X) :
    fs.f y ≤ fs.f x + ⟪fs.gradf x, y - x⟫_ℝ +
      (fs.L / 2) * ‖y - x‖ ^ 2 := by
  exact smooth_quadratic_upper_bound_of_hasGradientAt_lipschitzOn_convex
    (X := fs.X) (f := fs.f) (grad := fs.gradf) (L := fs.L)
    fs.hX_convex
    (fun z hz => fs.gradf_hasGradientAt z hz)
    (fun z hz w hw => fs.gradf_smooth z w hz hw)
    hx hy

/-- Smooth descent premise specialized to the Algorithm 7.12 affine LMO update.

Candidate audit: checked SOptLib `Convex.carrier_smooth_quadratic_upper_bound`,
`le_value_add_of_hasDerivWithinAt_le_affine_on_Icc`, and target-file
`finite_smooth_quadratic_upper_bound_of_hasGradientAt`. The SOptLib carrier
lemma is too strong for the exposed finite setup, while the local segment
lemma gives exactly the smooth Eq. (7.4.5) premise after LMO feasibility,
diameter control, and Young absorption. -/
private lemma finite_one_step_smooth_descent_premise
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 fs.N) {x G : E} (hx : x ∈ fs.X) :
    fs.f (fs.iterUpdate x G k) ≤
      fs.f x + fs.α k * ⟪G, fs.linearMinimizer G - x⟫_ℝ +
        (1 / (2 * fs.L)) * ‖G - fs.gradf x‖ ^ 2 +
        fs.L * fs.α k ^ 2 * fs.barDX ^ 2 := by
  exact conditional_gradient_smooth_descent_premise_of_estimator
    (f := fs.f) (grad := fs.gradf) (lmo := fs.linearMinimizer)
    (x := x) (G := G) (xnext := fs.iterUpdate x G k)
    (alpha := fs.α k) (L := fs.L) (D := fs.barDX)
    fs.hL_pos (fs.hα_nonneg k)
    (by
      simp only [SOptLib.FiniteSumConditionalGradientSetup.iterUpdate,
        SOptLib.conditionalGradientIterUpdate]
      module)
    (finite_smooth_quadratic_upper_bound_of_hasGradientAt
      (fs := fs) (x := x) (y := fs.iterUpdate x G k) hx
      (fs.iterUpdate_mem_of_alpha_le_one (x := x) (G := G) (k := k)
        hx hk hα_le_one))
    (fs.barDX_bound (fs.linearMinimizer G) x (fs.linearMinimizer_mem G) hx)

/-- Deterministic one-step Wolfe-gap algebra after the smooth descent estimate.

Candidate audit: searched `one step expected gap descent Wolfe gap smooth`,
checked SOptLib `Convex.carrier_smooth_quadratic_upper_bound`, and reused the
finite local `finite_wolfeGap_surrogate_bound`. The SOptLib theorem supplies
the missing smooth descent premise under a stronger carrier-smoothness API;
this helper proves the subsequent paper algebra and Young absorption in the
finite Algorithm 7.12 LMO notation. -/
private lemma finite_one_step_gap_descent_pointwise_of_smooth
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {x G xnext : E} {a : ℝ} (hx : x ∈ fs.X) (ha_nonneg : 0 ≤ a)
    (hdescent :
      fs.f xnext ≤
        fs.f x + a * ⟪G, fs.linearMinimizer G - x⟫_ℝ +
          (1 / (2 * fs.L)) * ‖G - fs.gradf x‖ ^ 2 +
          fs.L * a ^ 2 * fs.barDX ^ 2) :
    a * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x ≤
      fs.f x - fs.f xnext +
        (1 / fs.L) * ‖G - fs.gradf x‖ ^ 2 +
        (3 / 2) * fs.L * a ^ 2 * fs.barDX ^ 2 := by
  exact conditional_gradient_wolfe_gap_one_step_descent_of_smooth
    (gap := SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer)
    (f := fs.f) (targetGrad := fs.gradf) (lmo := fs.linearMinimizer)
    (x := x) (G := G) (xnext := xnext) (alpha := a) (L := fs.L) (D := fs.barDX)
    fs.hL_pos ha_nonneg
    (SOptLib.FiniteSumConditionalGradientSetup.finite_wolfeGap_surrogate_bound
      (fs := fs) x G hx)
    hdescent

/-- Deterministic source-form one-step Wolfe-gap algebra before the final
Young absorption of the estimator-error norm.

Candidate audit: reused target-file `finite_one_step_smooth_descent_premise`
and `finite_wolfeGap_surrogate_bound`. SOptLib telescope/descent helpers start
after one-step inequalities are already packaged and do not expose Lan
Eq. (7.4.6)'s unabsorbed `α D_X ‖δ_k‖` term, so this local bridge preserves the
paper's scalar budget for Theorem 7.16. -/
private lemma finite_one_step_gap_descent_pointwise_source_form
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {x G xnext : E} {a : ℝ} (hx : x ∈ fs.X) (ha_nonneg : 0 ≤ a)
    (hdescent :
      fs.f xnext ≤
        fs.f x + a * ⟪G, fs.linearMinimizer G - x⟫_ℝ +
          (1 / (2 * fs.L)) * ‖G - fs.gradf x‖ ^ 2 +
          fs.L * a ^ 2 * fs.barDX ^ 2) :
    a * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x ≤
      fs.f x - fs.f xnext +
        (1 / (2 * fs.L)) * ‖G - fs.gradf x‖ ^ 2 +
        fs.L * a ^ 2 * fs.barDX ^ 2 +
        a * fs.barDX * ‖G - fs.gradf x‖ := by
  classical
  let y : E := fs.linearMinimizer G
  let d : ℝ := ‖G - fs.gradf x‖
  let D : ℝ := fs.barDX
  have hgap :=
    SOptLib.FiniteSumConditionalGradientSetup.finite_wolfeGap_surrogate_bound
      (fs := fs) x G hx
  have hgap_mul :
      a * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x ≤ a * (⟪G, x - y⟫_ℝ + d * D) := by
    exact mul_le_mul_of_nonneg_left hgap ha_nonneg
  have hinner_neg : ⟪G, x - y⟫_ℝ = -⟪G, y - x⟫_ℝ := by
    have hxy : x - y = -(y - x) := by abel
    rw [hxy, inner_neg_right]
  have hdesc_rearr :
      -a * ⟪G, y - x⟫_ℝ ≤
        fs.f x - fs.f xnext +
          (1 / (2 * fs.L)) * d ^ 2 +
          fs.L * a ^ 2 * D ^ 2 := by
    linarith
  calc
    a * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer x
        ≤ a * (⟪G, x - y⟫_ℝ + d * D) := hgap_mul
    _ = -a * ⟪G, y - x⟫_ℝ + a * (d * D) := by
        rw [hinner_neg]
        ring
    _ ≤
        (fs.f x - fs.f xnext +
          (1 / (2 * fs.L)) * d ^ 2 +
          fs.L * a ^ 2 * D ^ 2) + a * (d * D) := by
        linarith
    _ =
      fs.f x - fs.f xnext +
        (1 / (2 * fs.L)) * ‖G - fs.gradf x‖ ^ 2 +
        fs.L * a ^ 2 * fs.barDX ^ 2 +
        a * fs.barDX * ‖G - fs.gradf x‖ := by
        dsimp [d, D]
        ring

/-- Generated finite-sum iterates satisfy the one-step Wolfe-gap descent bound.

Candidate audit: reused target-file `finite_one_step_gap_descent_pointwise_of_smooth`
after proving the missing smooth premise in
`finite_one_step_smooth_descent_premise`; no SOptLib telescope helper applies at
this point because this is the Algorithm 7.12 process-index specialization
before summation/integration. -/
private lemma finite_generated_one_step_gap_descent_pointwise
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 fs.N) (ω : Ω) :
    fs.α k * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ≤
      fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω) +
        (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 +
        (3 / 2) * fs.L * fs.α k ^ 2 * fs.barDX ^ 2 := by
  have hk_bounds := Finset.mem_Icc.mp hk
  have hx : fs.iterProcess k ω ∈ fs.X :=
    fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hk_bounds.2
  have ha_nonneg : 0 ≤ fs.α k := fs.hα_nonneg k
  have hnext :
      fs.iterProcess (k + 1) ω =
        fs.iterUpdate (fs.iterProcess k ω) (fs.estimatorProcess k ω) k :=
    fs.iterProcess_succ_eq_iterUpdate (k := k) hk_bounds.1 ω
  have hdescent :
      fs.f (fs.iterProcess (k + 1) ω) ≤
        fs.f (fs.iterProcess k ω) +
          fs.α k *
            ⟪fs.estimatorProcess k ω,
              fs.linearMinimizer (fs.estimatorProcess k ω) -
                fs.iterProcess k ω⟫_ℝ +
          (1 / (2 * fs.L)) *
            ‖fs.estimatorProcess k ω - fs.gradf (fs.iterProcess k ω)‖ ^ 2 +
          fs.L * fs.α k ^ 2 * fs.barDX ^ 2 := by
    rw [hnext]
    exact finite_one_step_smooth_descent_premise
      (fs := fs) hα_le_one (k := k) hk
      (x := fs.iterProcess k ω) (G := fs.estimatorProcess k ω) hx
  have hgap :=
    finite_one_step_gap_descent_pointwise_of_smooth
      (fs := fs) (x := fs.iterProcess k ω) (G := fs.estimatorProcess k ω)
      (xnext := fs.iterProcess (k + 1) ω) (a := fs.α k)
      hx ha_nonneg hdescent
  simpa [SOptLib.FiniteSumConditionalGradientSetup.deltaProcess] using hgap

/-- Generated finite-sum iterates satisfy the source-form one-step Wolfe-gap
descent bound before absorbing `α_k D_X ‖δ_k‖`.

Candidate audit: reused the newly proved target-file
`finite_one_step_gap_descent_pointwise_source_form`. SOptLib descent/telescope
helpers considered above do not preserve the unabsorbed estimator norm term
needed in Lan Eq. (7.4.6). -/
private lemma finite_generated_one_step_gap_descent_pointwise_source_form
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 fs.N) (ω : Ω) :
    fs.α k * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ≤
      fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω) +
        (1 / (2 * fs.L)) * ‖fs.deltaProcess k ω‖ ^ 2 +
        fs.L * fs.α k ^ 2 * fs.barDX ^ 2 +
        fs.α k * fs.barDX * ‖fs.deltaProcess k ω‖ := by
  have hk_bounds := Finset.mem_Icc.mp hk
  have hx : fs.iterProcess k ω ∈ fs.X :=
    fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hk_bounds.2
  have ha_nonneg : 0 ≤ fs.α k := fs.hα_nonneg k
  have hnext :
      fs.iterProcess (k + 1) ω =
        fs.iterUpdate (fs.iterProcess k ω) (fs.estimatorProcess k ω) k :=
    fs.iterProcess_succ_eq_iterUpdate (k := k) hk_bounds.1 ω
  have hdescent :
      fs.f (fs.iterProcess (k + 1) ω) ≤
        fs.f (fs.iterProcess k ω) +
          fs.α k *
            ⟪fs.estimatorProcess k ω,
              fs.linearMinimizer (fs.estimatorProcess k ω) -
                fs.iterProcess k ω⟫_ℝ +
          (1 / (2 * fs.L)) *
            ‖fs.estimatorProcess k ω - fs.gradf (fs.iterProcess k ω)‖ ^ 2 +
          fs.L * fs.α k ^ 2 * fs.barDX ^ 2 := by
    rw [hnext]
    exact finite_one_step_smooth_descent_premise
      (fs := fs) hα_le_one (k := k) hk
      (x := fs.iterProcess k ω) (G := fs.estimatorProcess k ω) hx
  have hgap :=
    finite_one_step_gap_descent_pointwise_source_form
      (fs := fs) (x := fs.iterProcess k ω) (G := fs.estimatorProcess k ω)
      (xnext := fs.iterProcess (k + 1) ω) (a := fs.α k)
      hx ha_nonneg hdescent
  simpa [SOptLib.FiniteSumConditionalGradientSetup.deltaProcess] using hgap

/-- Expected active-coordinate one-step Wolfe-gap bound after integrating the
generated pointwise descent inequality.

Candidate audit: searched SOptLib/target for integral monotonicity and
telescope helpers. This uses the generic integral APIs plus the local
integrability bridges; SOptLib `integral_sum_telescope_bound_of_pointwise_lower_bound`
starts after summing objective drops, so it is a later consumer rather than a
replacement for this active one-step integration. -/
private lemma finite_active_expected_one_step_gap_bound
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    fs.α (fs.globalIndex s j) *
        ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess (fs.globalIndex s j) ω) ∂fs.P ≤
      ∫ ω,
          (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
            fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
        (1 / fs.L) *
          ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
        (3 / 2) * fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  let k : ℕ := fs.globalIndex s j
  have hk : k ∈ Finset.Icc 1 fs.N := by
    dsimp [k]
    exact fs.globalIndex_mem_output_of_active hj
  have hk_bounds := Finset.mem_Icc.mp hk
  have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
  have hjN := fs.activeEpochSteps_globalIndex_le (s := s) (j := j) hj
  have hgap_int :
      Integrable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)) fs.P :=
    finite_wolfeGap_iterProcess_integrable_of_le
      (fs := fs) hα_le_one k hk_bounds.2
  have hleft_int :
      Integrable
        (fun ω => fs.α k * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)) fs.P :=
    hgap_int.const_mul (fs.α k)
  have hf_int :
      Integrable (fun ω => fs.f (fs.iterProcess k ω)) fs.P :=
    finite_f_iterProcess_integrable_of_le (fs := fs) hα_le_one k hk_bounds.2
  have hfnext_int :
      Integrable (fun ω => fs.f (fs.iterProcess (k + 1) ω)) fs.P :=
    finite_f_iterProcess_succ_integrable_of_mem
      (fs := fs) hα_le_one (k := k) hk
  have hdrop_int :
      Integrable
        (fun ω =>
          fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) fs.P :=
    hf_int.sub hfnext_int
  have hdelta_int :
      Integrable (fun ω => ‖fs.deltaProcess k ω‖ ^ 2) fs.P := by
    dsimp [k]
    exact finite_delta_sq_integrable (fs := fs) hα_le_one s j
      hj_epoch.1 hj_epoch.2 hjN
  let c : ℝ := (3 / 2) * fs.L * fs.α k ^ 2 * fs.barDX ^ 2
  have hrhs_int :
      Integrable
        (fun ω =>
          (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
            (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 + c) fs.P :=
    (hdrop_int.add (hdelta_int.const_mul (1 / fs.L))).add (integrable_const c)
  have hmono :
      ∫ ω, fs.α k * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P ≤
        ∫ ω,
          (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
            (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 + c ∂fs.P := by
    refine integral_mono hleft_int hrhs_int ?_
    intro ω
    exact finite_generated_one_step_gap_descent_pointwise
      (fs := fs) hα_le_one (k := k) hk ω
  have hrhs_expand :
      ∫ ω,
          (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
            (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 + c ∂fs.P =
        ∫ ω,
          (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P +
          (1 / fs.L) * ∫ ω, ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P + c := by
    calc
      ∫ ω,
          (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
            (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 + c ∂fs.P
          =
          ∫ ω,
            ((fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
              (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2) + c ∂fs.P := by
            rfl
      _ =
          ∫ ω,
            ((fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
              (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2) ∂fs.P +
            ∫ _ω, c ∂fs.P := by
            simpa [Pi.add_apply] using
              (integral_add (hdrop_int.add (hdelta_int.const_mul (1 / fs.L)))
                (integrable_const (c := c) : Integrable (fun _ω : Ω => c) fs.P))
      _ =
          (∫ ω,
            (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P +
            ∫ ω, (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P) +
            c := by
            rw [show
              (∫ ω,
                ((fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
                  (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2) ∂fs.P) =
                (∫ ω,
                  (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P +
                  ∫ ω, (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P) by
                simpa [Pi.add_apply] using
                  (integral_add hdrop_int (hdelta_int.const_mul (1 / fs.L)))]
            simp
      _ =
          ∫ ω,
            (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P +
            (1 / fs.L) * ∫ ω, ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P + c := by
            rw [integral_const_mul]
  calc
    fs.α (fs.globalIndex s j) *
        ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess (fs.globalIndex s j) ω) ∂fs.P
        = ∫ ω, fs.α k * SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P := by
          dsimp [k]
          rw [integral_const_mul]
    _ ≤ ∫ ω,
          (fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) +
            (1 / fs.L) * ‖fs.deltaProcess k ω‖ ^ 2 + c ∂fs.P := hmono
    _ =
      ∫ ω,
          (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
            fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
        (1 / fs.L) *
          ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
        (3 / 2) * fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 := by
          dsimp [k, c] at hrhs_expand ⊢
          simpa using hrhs_expand

/-- Active one-step expected Wolfe-gap bound in the source form used just before
Lan Eq. (7.4.6)'s epoch scalar estimates.

Candidate audit: reused target-file
`finite_generated_one_step_gap_descent_pointwise_source_form`,
`finite_delta_sq_integrable`, and the imported SOptLib scalar L2-to-L1 helper
`integrable_of_nonneg_sq_integrable_integral_le_sq_bound_add_one`. Generic
SOptLib telescope lemmas considered earlier do not retain the unabsorbed
`α_{s,j} D_X E‖δ_{s,j}‖` term that drives the paper maximum budget. -/
private lemma finite_active_expected_one_step_gap_bound_source_form
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    fs.α (fs.globalIndex s j) *
        ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess (fs.globalIndex s j) ω) ∂fs.P ≤
      ∫ ω,
          (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
            fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
        (1 / (2 * fs.L)) *
          ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
        fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
        fs.α (fs.globalIndex s j) * fs.barDX *
          ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  let k : ℕ := fs.globalIndex s j
  have hk : k ∈ Finset.Icc 1 fs.N := by
    dsimp [k]
    exact fs.globalIndex_mem_output_of_active hj
  have hk_bounds := Finset.mem_Icc.mp hk
  have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
  have hjN := fs.activeEpochSteps_globalIndex_le (s := s) (j := j) hj
  have hgap_int :
      Integrable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω)) fs.P :=
    finite_wolfeGap_iterProcess_integrable_of_le
      (fs := fs) hα_le_one k hk_bounds.2
  have hf_int :
      Integrable (fun ω => fs.f (fs.iterProcess k ω)) fs.P :=
    finite_f_iterProcess_integrable_of_le (fs := fs) hα_le_one k hk_bounds.2
  have hfnext_int :
      Integrable (fun ω => fs.f (fs.iterProcess (k + 1) ω)) fs.P :=
    finite_f_iterProcess_succ_integrable_of_mem
      (fs := fs) hα_le_one (k := k) hk
  have hdrop_int :
      Integrable
        (fun ω =>
          fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω)) fs.P :=
    hf_int.sub hfnext_int
  have hdelta_sq_int :
      Integrable (fun ω => ‖fs.deltaProcess k ω‖ ^ 2) fs.P := by
    dsimp [k]
    exact finite_delta_sq_integrable (fs := fs) hα_le_one s j
      hj_epoch.1 hj_epoch.2 hjN
  have hdelta_norm_int :
      Integrable (fun ω => ‖fs.deltaProcess k ω‖) fs.P := by
    have hnorm_nonneg : ∀ᵐ ω ∂fs.P, 0 ≤ ‖fs.deltaProcess k ω‖ :=
      Filter.Eventually.of_forall (fun ω => norm_nonneg _)
    have hle :
        ∫ ω, ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P ≤
          ∫ ω, ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P := le_rfl
    exact
      (integrable_of_nonneg_sq_integrable_integral_le_sq_bound_add_one
        (μ := fs.P) (Z := fun ω => ‖fs.deltaProcess k ω‖)
        (C := ∫ ω, ‖fs.deltaProcess k ω‖ ^ 2 ∂fs.P)
        hdelta_sq_int hnorm_nonneg hle).1
  have hpoint :
      ∀ ω,
        fs.α k *
            SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
              (fs.iterProcess k ω) ≤
          ((fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω) +
              (1 / (2 * fs.L)) * ‖fs.deltaProcess k ω‖ ^ 2) +
            fs.L * fs.α k ^ 2 * fs.barDX ^ 2) +
            (fs.α k * fs.barDX) * ‖fs.deltaProcess k ω‖ := by
    intro ω
    have hpt :=
      finite_generated_one_step_gap_descent_pointwise_source_form
        (fs := fs) hα_le_one (k := k) hk ω
    dsimp [k] at hpt ⊢
    linarith
  have hbridge :=
    integral_one_step_gap_source_form_of_pointwise
      (P := fs.P)
      (gap := fun ω =>
        SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
          (fs.iterProcess k ω))
      (drop := fun ω =>
        fs.f (fs.iterProcess k ω) - fs.f (fs.iterProcess (k + 1) ω))
      (sq := fun ω => ‖fs.deltaProcess k ω‖ ^ 2)
      (norm_term := fun ω => ‖fs.deltaProcess k ω‖)
      (alpha := fs.α k)
      (sq_coeff := 1 / (2 * fs.L))
      (c_sq := fs.L * fs.α k ^ 2 * fs.barDX ^ 2)
      (c_norm := fs.α k * fs.barDX)
      hgap_int hdrop_int hdelta_sq_int hdelta_norm_int hpoint
  dsimp [k] at hbridge ⊢
  simpa [add_assoc] using hbridge

/-- Active one-step expected Wolfe-gap bound after substituting finite-sum
Lemma 7.4's estimator second-moment estimate.

Candidate audit: reused target-file `finite_active_expected_one_step_gap_bound`
and `finite_active_delta_second_moment_le`. No SOptLib lemma knows the paper's
`epochDiffSum` or Algorithm 7.12 active-coordinate map, so this is the
source-local bridge from Eq. (7.4.5) to the epoch budget. -/
private lemma finite_active_expected_one_step_gap_bound_with_epochDiff
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    fs.α (fs.globalIndex s j) *
        ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess (fs.globalIndex s j) ω) ∂fs.P ≤
      ∫ ω,
          (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
            fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
        (fs.L / fs.b) *
          ∫ ω, fs.epochDiffSum s j ω ∂fs.P +
        (3 / 2) * fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 := by
  exact one_step_gap_bound_with_epoch_budget_of_delta_second_moment
    (lhs :=
      fs.α (fs.globalIndex s j) *
        ∫ ω,
          SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
            (fs.iterProcess (fs.globalIndex s j) ω) ∂fs.P)
    (drop :=
      ∫ ω,
        (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
          fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P)
    (deltaSq :=
      ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P)
    (epochBudget := ∫ ω, fs.epochDiffSum s j ω ∂fs.P)
    (alphaSqTerm :=
      (3 / 2) * fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2)
    (L := fs.L) (b := (fs.b : ℝ))
    (finite_active_expected_one_step_gap_bound
      (fs := fs) hα_le_one (s := s) (j := j) hj)
    (finite_active_delta_second_moment_le
      (fs := fs) hα_le_one (s := s) (j := j) hj)
    fs.hL_pos
    (by
      exact_mod_cast (ne_of_gt (Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hb_pos)))

/-! Active-epoch reindexing helpers for the finite Theorem 7.16 telescope.

Candidate audit: searched target/SOptLib for `finite sum active epoch partition
global index` and `telescope weighted sum descent`. Existing hits
`activeEpochSteps_globalIndex_le`, `globalIndex_epochOfIndex_stepOfIndex`, and
`stepOfIndex_mem_activeEpochSteps` provide the two local directions, while
SOptLib `summed_one_step_gap_bound_of_telescope` only sums an already-indexed
window and does not encode Algorithm 7.12's partial final epoch. -/
private lemma finite_epochOfIndex_globalIndex_eq
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {s j : ℕ} (hj1 : 1 ≤ j) (hjT : j ≤ fs.T) :
    fs.epochOfIndex (fs.globalIndex s j) = s := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.epochOfIndex,
    SOptLib.FiniteSumConditionalGradientSetup.globalIndex] using
    SOptLib.epochOfIndex_mul_add_eq_of_pos_le (T := fs.T) (s := s) (j := j)
      hj1 hjT

private lemma finite_stepOfIndex_globalIndex_eq
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    {s j : ℕ} (hj1 : 1 ≤ j) (hjT : j ≤ fs.T) :
    fs.stepOfIndex (fs.globalIndex s j) = j := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.stepOfIndex,
    SOptLib.FiniteSumConditionalGradientSetup.globalIndex] using
    SOptLib.nat_stepOf_globalIndex_eq (T := fs.T) (s := s) (j := j) hj1 hjT

/-- Reindex the generated output window `{1, ..., N}` by Algorithm 7.12's
active epoch coordinates.

No SOptLib match: searched `finite sum active epoch partition global index`,
`telescope weighted sum descent`, and checked SOptLib
`summed_one_step_gap_bound_of_telescope`; the generic telescope helper has no
partial-final-epoch map, while the target-file active-step lemmas supply exactly
the paper epoch decoder used here. -/
private lemma finite_global_index_active_epoch_partition_sum
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (A : ℕ → ℝ) :
    Finset.sum (Finset.Icc 1 fs.N) A =
      Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j => A (fs.globalIndex s j))) := by
  classical
  rw [Finset.sum_sigma']
  refine Finset.sum_bij
    (fun k hk => Sigma.mk (fs.epochOfIndex k) (fs.stepOfIndex k))
    ?mem ?inj ?surj ?sum_eq
  · intro k hk
    have hk_bounds := Finset.mem_Icc.mp hk
    rw [Finset.mem_sigma]
    constructor
    · refine Finset.mem_Icc.mpr ⟨Nat.zero_le _, ?_⟩
      have hnum : k - 1 ≤ fs.N + fs.T - 1 := by omega
      have hdiv := Nat.div_le_div_right (c := fs.T) hnum
      simpa [SOptLib.FiniteSumConditionalGradientSetup.epochOfIndex,
        SOptLib.FiniteSumConditionalGradientSetup.S] using hdiv
    · exact fs.stepOfIndex_mem_activeEpochSteps hk
  · intro k₁ hk₁ k₂ hk₂ hpair
    have hk₁pos : 1 ≤ k₁ := (Finset.mem_Icc.mp hk₁).1
    have hk₂pos : 1 ≤ k₂ := (Finset.mem_Icc.mp hk₂).1
    have hglobal := congrArg
      (fun p : Sigma fun _s : ℕ => ℕ => fs.globalIndex p.1 p.2) hpair
    simpa [fs.globalIndex_epochOfIndex_stepOfIndex hk₁pos,
      fs.globalIndex_epochOfIndex_stepOfIndex hk₂pos] using hglobal
  · intro p hp
    rcases p with ⟨s, j⟩
    have hj : j ∈ fs.activeEpochSteps s := (Finset.mem_sigma.mp hp).2
    have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
    refine ⟨fs.globalIndex s j, fs.globalIndex_mem_output_of_active hj, ?_⟩
    apply Sigma.ext
    · exact finite_epochOfIndex_globalIndex_eq (fs := fs)
        hj_epoch.1 hj_epoch.2
    · simp [finite_stepOfIndex_globalIndex_eq (fs := fs)
        hj_epoch.1 hj_epoch.2]
  · intro k hk
    have hkpos : 1 ≤ k := (Finset.mem_Icc.mp hk).1
    simp [fs.globalIndex_epochOfIndex_stepOfIndex hkpos]

/-! `lemma_7_5` controls the second moment of the estimator error `δ_k` within
one epoch via the accumulated squared iterate differences and the batch-refresh
variance floor σ²/m. It corresponds to (7.4.14) in Lan's book and is the key
stochastic ingredient that differentiates Theorem 7.17 from Theorem 7.16. -/
theorem lemma_7_5_conditional
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract)
    (s t : ℕ) (hs : 0 ≤ s) (ht : 1 ≤ t) (ht_le : t ≤ setup.T)
    (hkt : setup.globalIndex s t ≤ setup.N) :
    ∫ ω, ‖setup.algorithmDeltaProcessConditional h (setup.globalIndex s t) ω‖ ^ 2 ∂setup.P ≤
      setup.L ^ 2 / setup.b *
        ∫ ω, setup.epochDiffSumOfWellDefined
          (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h) s t ω ∂setup.P +
      setup.σ ^ 2 / setup.m := by
  let hαwf := setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h
  have hα_le_one : setup.paperAlphaOfWellDefined hαwf ≤ 1 := by
    simpa [hαwf] using setup.paperAlpha_le_one_of_algorithm713Contract h
  simpa [StochasticNonconvexConditionalGradientSetup.algorithmDeltaProcessConditional,
    StochasticNonconvexConditionalGradientSetup.algorithmEstimatorProcessConditional,
    StochasticNonconvexConditionalGradientSetup.algorithmIterProcessConditional,
    StochasticNonconvexConditionalGradientSetup.algorithmProcessConditional, hαwf] using
    epochwise_estimator_variance_bound (setup := setup) hαwf hα_le_one s t ht ht_le hkt

/-- Active-window objective drops telescope to the initial optimality gap.

Candidate audit: searched SOptLib/target for `Finset sum telescope objective
drop fStar lower bound`. SOptLib's `sum_Icc_sub_succ` gives the scalar
telescoping core, while the target-file active partition and generated
feasibility/integrability helpers provide the paper-specific active-window
transport and terminal lower bound. -/
private lemma finite_active_objective_drop_telescope
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            ∫ ω,
              (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P)) ≤
      fs.f fs.x₁ - fs.fStar := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  rw [← finite_global_index_active_epoch_partition_sum (fs := fs)
    (A := fun k =>
      ∫ ω,
        (fs.f (fs.iterProcess k ω) -
          fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P)]
  let A : ℕ → ℝ := fun k => ∫ ω, fs.f (fs.iterProcess k ω) ∂fs.P
  have hdrop_eq :
      ∀ k ∈ Finset.Icc 1 fs.N,
        (∫ ω,
          (fs.f (fs.iterProcess k ω) -
            fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P) =
          A k - A (k + 1) := by
    intro k hk
    have hk_bounds := Finset.mem_Icc.mp hk
    have hf_int :
        Integrable (fun ω => fs.f (fs.iterProcess k ω)) fs.P :=
      finite_f_iterProcess_integrable_of_le (fs := fs) hα_le_one k hk_bounds.2
    have hfnext_int :
        Integrable (fun ω => fs.f (fs.iterProcess (k + 1) ω)) fs.P :=
      finite_f_iterProcess_succ_integrable_of_mem
        (fs := fs) hα_le_one (k := k) hk
    dsimp [A]
    rw [integral_sub hf_int hfnext_int]
  have htel :
      Finset.sum (Finset.Icc 1 fs.N)
          (fun k =>
            ∫ ω,
              (fs.f (fs.iterProcess k ω) -
                fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P) =
        A 1 - A (fs.N + 1) := by
    calc
      Finset.sum (Finset.Icc 1 fs.N)
          (fun k =>
            ∫ ω,
              (fs.f (fs.iterProcess k ω) -
                fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P)
          = Finset.sum (Finset.Icc 1 fs.N) (fun k => A k - A (k + 1)) := by
              refine Finset.sum_congr rfl ?_
              intro k hk
              exact hdrop_eq k hk
      _ = A 1 - A (fs.N + 1) := by
              exact sum_Icc_sub_succ A 1 fs.N fs.hN_pos
  have hA1 : A 1 = fs.f fs.x₁ := by
    dsimp [A, SOptLib.FiniteSumConditionalGradientSetup.iterProcess,
      SOptLib.FiniteSumConditionalGradientSetup.process]
    simp [integral_const, probReal_univ]
  have hterminal_int :
      Integrable (fun ω => fs.f (fs.iterProcess (fs.N + 1) ω)) fs.P := by
    have hNmem : fs.N ∈ Finset.Icc 1 fs.N := Finset.mem_Icc.mpr ⟨fs.hN_pos, le_rfl⟩
    exact finite_f_iterProcess_succ_integrable_of_mem
      (fs := fs) hα_le_one (k := fs.N) hNmem
  have hterminal_lb :
      fs.fStar ≤ A (fs.N + 1) := by
    have hconst_int : Integrable (fun _ω : Ω => fs.fStar) fs.P := integrable_const _
    have hle_int :
        ∫ _ω, fs.fStar ∂fs.P ≤
          ∫ ω, fs.f (fs.iterProcess (fs.N + 1) ω) ∂fs.P := by
      refine integral_mono hconst_int hterminal_int ?_
      intro ω
      have hNmem : fs.N ∈ Finset.Icc 1 fs.N :=
        Finset.mem_Icc.mpr ⟨fs.hN_pos, le_rfl⟩
      change fs.fStar ≤ fs.f (fs.iterProcess (fs.N + 1) ω)
      rw [fs.iterProcess_succ_eq_iterUpdate (k := fs.N) fs.hN_pos ω]
      have hx : fs.iterProcess fs.N ω ∈ fs.X :=
        fs.iterProcess_mem_of_alpha_le_one hα_le_one fs.N ω le_rfl
      exact fs.fStar_lb
        (fs.iterUpdate (fs.iterProcess fs.N ω) (fs.estimatorProcess fs.N ω) fs.N)
        (fs.iterUpdate_mem_of_alpha_le_one
          (x := fs.iterProcess fs.N ω) (G := fs.estimatorProcess fs.N ω)
          (k := fs.N) hx hNmem hα_le_one)
    simpa [A, integral_const, probReal_univ] using hle_int
  calc
    Finset.sum (Finset.Icc 1 fs.N)
        (fun k =>
          ∫ ω,
            (fs.f (fs.iterProcess k ω) -
              fs.f (fs.iterProcess (k + 1) ω)) ∂fs.P)
        = A 1 - A (fs.N + 1) := htel
    _ ≤ fs.f fs.x₁ - fs.fStar := by
        rw [hA1]
        linarith

/-- Generated within-epoch iterate differences are controlled by the displayed
stepsize and the feasible-set diameter.

Candidate audit: searched `iterate update difference norm alpha diameter` and
`epoch difference sum alpha square diameter bound`; target-file
`iterProcess_succ_eq_iterUpdate`, `iterUpdate_mem_of_alpha_le_one`, and
`barDX_bound` provide the needed Algorithm 7.12 pieces, while SOptLib iterate
helpers are generic adaptedness/reindexing facts and do not expose this
paper-specific Frank-Wolfe update identity. -/
private lemma finite_active_iter_diff_sq_le_alpha_sq_barDX_sq
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s i : ℕ} (hi2 : 2 ≤ i)
    (hiN : fs.globalIndex s i ≤ fs.N) (ω : Ω) :
    ‖fs.iterProcess (fs.globalIndex s i) ω -
        fs.iterProcess (fs.globalIndex s (i - 1)) ω‖ ^ 2 ≤
      fs.α (fs.globalIndex s (i - 1)) ^ 2 * fs.barDX ^ 2 := by
  classical
  let k : ℕ := fs.globalIndex s (i - 1)
  let x : E := fs.iterProcess k ω
  let G : E := fs.estimatorProcess k ω
  let y : E := fs.linearMinimizer G
  have hk_pos : 1 ≤ k := by
    dsimp [k]
    unfold SOptLib.FiniteSumConditionalGradientSetup.globalIndex
    omega
  have hkN : k ≤ fs.N := by
    dsimp [k] at *
    exact fs.globalIndex_prefix_le_of_le
      (s := s) (i := i - 1) (t := i) (by omega) hiN
  have hk_mem : k ∈ Finset.Icc 1 fs.N := Finset.mem_Icc.mpr ⟨hk_pos, hkN⟩
  have hsucc : fs.globalIndex s i = k + 1 := by
    dsimp [k]
    unfold SOptLib.FiniteSumConditionalGradientSetup.globalIndex
    omega
  have hiter :
      fs.iterProcess (fs.globalIndex s i) ω =
        fs.iterUpdate x G k := by
    rw [hsucc]
    exact fs.iterProcess_succ_eq_iterUpdate (k := k) hk_pos ω
  have hdiff :
      fs.iterProcess (fs.globalIndex s i) ω -
          fs.iterProcess (fs.globalIndex s (i - 1)) ω =
        fs.α k • (y - x) := by
    rw [hiter]
    dsimp [x, G, y, k]
    simp only [SOptLib.FiniteSumConditionalGradientSetup.iterUpdate,
      SOptLib.conditionalGradientIterUpdate]
    module
  have hx : x ∈ fs.X :=
    fs.iterProcess_mem_of_alpha_le_one hα_le_one k ω hkN
  have hy : y ∈ fs.X := fs.linearMinimizer_mem G
  have hdiam : ‖y - x‖ ≤ fs.barDX :=
    fs.barDX_bound y x hy hx
  have hα_nonneg : 0 ≤ fs.α k := fs.hα_nonneg k
  have hstep :
      ‖fs.iterProcess (fs.globalIndex s i) ω - x‖ ^ 2 ≤
        fs.α k ^ 2 * fs.barDX ^ 2 :=
    conditional_gradient_update_diff_sq_le_alpha_sq_diameter_sq
      (x := x) (xnext := fs.iterProcess (fs.globalIndex s i) ω)
      (y := y) (α := fs.α k) (D := fs.barDX)
      hα_nonneg (by simpa [x, k] using hdiff) hdiam
  simpa [x, k] using hstep

/-- Integrated epoch-difference budget obtained from the generated update and
diameter bound.

Candidate audit: searched `epoch difference sum alpha square diameter bound`,
`iterate update difference norm alpha diameter`, and `integral finite sum
equals sum integrals`. The previous helper
`finite_active_iter_diff_sq_le_alpha_sq_barDX_sq` gives the pointwise paper
update estimate; Mathlib `integral_finset_sum` supplies the finite integral
commutation, and no SOptLib theorem combines these Algorithm 7.12 epoch
coordinates. -/
private lemma finite_active_epochDiff_integral_le_scalar_sum
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    ∫ ω, fs.epochDiffSum s j ω ∂fs.P ≤
      fs.barDX ^ 2 *
        Finset.sum (Finset.Icc 2 j)
          (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2) := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  have hjN : fs.globalIndex s j ≤ fs.N :=
    fs.activeEpochSteps_globalIndex_le (s := s) (j := j) hj
  exact integral_epochSquaredDifferenceSum_le_diameter_sq_mul_alpha_sq_sum
    (x := fs.iterProcess) (index := fs.globalIndex) (s := s) (t := j)
    (alphaIndex := fs.epochDifferenceAlphaIndex s) (alpha := fs.α)
    (D := fs.barDX)
    (hterm_int := by
      intro i hi
      have hi_le : i ≤ j := (Finset.mem_Icc.mp hi).2
      exact finite_iterProcess_diff_sq_integrable_of_le
        (fs := fs) hα_le_one
        (fs.globalIndex s i) (fs.globalIndex s (i - 1))
        (fs.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hi_le hjN)
        (fs.globalIndex_prefix_le_of_le (s := s) (i := i - 1) (t := j)
          (by omega) hjN))
    (hpoint := by
      intro i hi
      have hi2 : 2 ≤ i := (Finset.mem_Icc.mp hi).1
      have hi_le : i ≤ j := (Finset.mem_Icc.mp hi).2
      have hiN : fs.globalIndex s i ≤ fs.N :=
        fs.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hi_le hjN
      intro ω
      simpa [SOptLib.FiniteSumConditionalGradientSetup.epochDifferenceAlphaIndex] using
        finite_active_iter_diff_sq_le_alpha_sq_barDX_sq
          (fs := fs) hα_le_one (s := s) (i := i) hi2 hiN ω)

/-- The Lemma 7.4 delta-square term is bounded by the triangular alpha-square
epoch budget after substituting the generated-iterate diameter estimate.

Candidate audit: searched `epoch difference sum alpha square diameter bound`
and checked target-file `finite_active_delta_second_moment_le`; SOptLib has
generic finite-integral and martingale tools, but the active-coordinate
specialization and predecessor-index alpha sum are paper-local. -/
private lemma finite_active_delta_square_term_le_epoch_alpha_sum
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    (1 / (2 * fs.L)) *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P ≤
      (fs.L / (2 * fs.b)) * fs.barDX ^ 2 *
        Finset.sum (Finset.Icc 2 j)
          (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2) := by
  exact delta_square_scaled_le_epoch_alpha_square_budget
    (deltaSq := ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P)
    (epochBudget := ∫ ω, fs.epochDiffSum s j ω ∂fs.P)
    (alphaSqSum :=
      Finset.sum (Finset.Icc 2 j)
        (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2))
    (L := fs.L) (b := (fs.b : ℝ)) (D := fs.barDX)
    (finite_active_delta_second_moment_le (fs := fs) hα_le_one
      (s := s) (j := j) hj)
    (finite_active_epochDiff_integral_le_scalar_sum (fs := fs) hα_le_one
      (s := s) (j := j) hj)
    fs.hL_pos
    (by
      exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hb_pos))

/-- The explicit `L α_k^2 D_X^2` term in Lan Theorem 7.16 reindexes exactly
from active epoch coordinates back to the generated global window.

Candidate audit: searched `active epoch difference alpha triangular budget
finite sum` and checked the target-file hit
`finite_global_index_active_epoch_partition_sum`; no SOptLib lemma specializes
the Algorithm 7.12 active-coordinate partition, while that local helper is
precisely the paper's generated epoch/window alignment. -/
private lemma active_alpha_square_sum_eq_global_theorem716
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j => fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2)) =
      fs.L * fs.barDX ^ 2 *
        Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) := by
  simpa [Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm] using
    (active_partition_sum_mul_const_eq_global_sum
      (epochs := Finset.Icc 0 fs.S) (active := fs.activeEpochSteps)
      (idx := fs.globalIndex) (window := Finset.Icc 1 fs.N)
      (w := fun k => fs.barDX ^ 2 * fs.α k ^ 2) (c := fs.L)
      (finite_global_index_active_epoch_partition_sum
        (fs := fs) (A := fun k => fs.barDX ^ 2 * fs.α k ^ 2)).symm)

/-- Eq. (7.4.6) per-step epoch-difference square budget is controlled by the
finite source maximum and the mini-batch size.

Candidate audit: searched `active epoch difference alpha triangular budget
finite sum`, `Finset card Icc count sum constant`, and
`Finset sum_le_card_nsmul real`; target-file
`epochDifferenceAlpha_le_paperMaxEpochDifferenceAlpha` gives the finite max
comparison, while Mathlib `Finset.sum_le_card_nsmul` and `Nat.card_Icc` supply
the generic finite counting. No SOptLib lemma combines this paper-local
epoch-difference index with the Theorem 7.16 `b ≥ T` hypothesis. -/
private lemma active_epoch_difference_alpha_square_sum_le_batch_max
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined) (hb_ge_T : fs.T ≤ fs.b)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    Finset.sum (Finset.Icc 2 j)
        (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2) ≤
      (fs.b : ℝ) * (fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s) ^ 2 := by
  refine sum_sq_le_card_mul_max_sq
    (s := Finset.Icc 2 j)
    (a := fun i => fs.α (fs.epochDifferenceAlphaIndex s i))
    (M := fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s)
    (b := fs.b) ?_ ?_ ?_
  · have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
    rw [Nat.card_Icc]
    omega
  · intro i hi
    exact fs.epochDifferenceAlpha_nonneg s i
  · intro i hi
    have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
    have hi_bounds := Finset.mem_Icc.mp hi
    have hi_inner : i ∈ fs.innerAlphaWindow := by
      simp [SOptLib.FiniteSumConditionalGradientSetup.innerAlphaWindow]
      exact ⟨hi_bounds.1, le_trans hi_bounds.2 hj_epoch.2⟩
    exact fs.epochDifferenceAlpha_le_paperMaxEpochDifferenceAlpha s i hT hi_inner

/-- Pointwise L1 absorption of the finite-sum estimator error by the corrected
epoch-difference maximum, matching the square-root estimate after Eq. (7.4.6).

Candidate audit: searched `L2 to L1 integral square root probability norm` and
`integrable square implies aestronglymeasurable real`; reused SOptLib
`sq_integral_le_integral_sq` and
`AEStronglyMeasurable.of_integrable_sq_of_nonneg`. The target-file helpers
`finite_active_delta_second_moment_le`,
`finite_active_epochDiff_integral_le_scalar_sum`, and
`active_epoch_difference_alpha_square_sum_le_batch_max` provide the paper-local
Lemma 7.4 and finite max/counting content. -/
private lemma active_delta_l1_pointwise_le_epoch_difference_max
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b)
    {s j : ℕ} (hj : j ∈ fs.activeEpochSteps s) :
    ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P ≤
      fs.L * fs.barDX * fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s := by
  classical
  haveI : IsProbabilityMeasure fs.P := fs.hP
  let Z : Ω → ℝ := fun ω => ‖fs.deltaProcess (fs.globalIndex s j) ω‖
  let M : ℝ := fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s
  have hj_epoch := fs.activeEpochSteps_mem_epoch (s := s) (j := j) hj
  have hjN := fs.activeEpochSteps_globalIndex_le (s := s) (j := j) hj
  have hZ_sq_int : Integrable (fun ω => Z ω ^ 2) fs.P := by
    dsimp [Z]
    exact finite_delta_sq_integrable (fs := fs) hα_le_one s j
      hj_epoch.1 hj_epoch.2 hjN
  have hZ_nonneg : ∀ᵐ ω ∂fs.P, 0 ≤ Z ω :=
    Filter.Eventually.of_forall (fun ω => norm_nonneg _)
  have hdelta_second :
      ∫ ω, Z ω ^ 2 ∂fs.P ≤
        fs.L ^ 2 / fs.b * ∫ ω, fs.epochDiffSum s j ω ∂fs.P := by
    dsimp [Z]
    exact finite_active_delta_second_moment_le
      (fs := fs) hα_le_one (s := s) (j := j) hj
  have hepoch :
      ∫ ω, fs.epochDiffSum s j ω ∂fs.P ≤
        fs.barDX ^ 2 *
          Finset.sum (Finset.Icc 2 j)
            (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2) :=
    finite_active_epochDiff_integral_le_scalar_sum
      (fs := fs) hα_le_one (s := s) (j := j) hj
  have hmax_sum :
      Finset.sum (Finset.Icc 2 j)
          (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2) ≤
        (fs.b : ℝ) * M ^ 2 := by
    dsimp [M]
    exact active_epoch_difference_alpha_square_sum_le_batch_max
      (fs := fs) hT hb_ge_T (s := s) (j := j) hj
  have hbar_nonneg : 0 ≤ fs.barDX := by
    exact norm_nonneg _
  have hM_nonneg : 0 ≤ M := by
    dsimp [M]
    exact fs.paperMaxEpochDifferenceAlphaOfWellDefined_nonneg s hT
  have hb_pos_real : 0 < (fs.b : ℝ) := by
    exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one fs.hb_pos)
  have hb_ne : (fs.b : ℝ) ≠ 0 := ne_of_gt hb_pos_real
  have hcoef_nonneg : 0 ≤ fs.L ^ 2 / fs.b := by
    exact div_nonneg (sq_nonneg fs.L) (le_of_lt hb_pos_real)
  have hepoch_max :
      ∫ ω, fs.epochDiffSum s j ω ∂fs.P ≤
        fs.barDX ^ 2 * ((fs.b : ℝ) * M ^ 2) :=
    le_trans hepoch (mul_le_mul_of_nonneg_left hmax_sum (sq_nonneg fs.barDX))
  have hdelta_square_bound :
      ∫ ω, Z ω ^ 2 ∂fs.P ≤ (fs.L * fs.barDX * M) ^ 2 := by
    calc
      ∫ ω, Z ω ^ 2 ∂fs.P
          ≤ fs.L ^ 2 / fs.b * ∫ ω, fs.epochDiffSum s j ω ∂fs.P :=
            hdelta_second
      _ ≤ fs.L ^ 2 / fs.b * (fs.barDX ^ 2 * ((fs.b : ℝ) * M ^ 2)) :=
            mul_le_mul_of_nonneg_left hepoch_max hcoef_nonneg
      _ = (fs.L * fs.barDX * M) ^ 2 := by
            field_simp [hb_ne]
  have hright_nonneg : 0 ≤ fs.L * fs.barDX * M := by
    exact mul_nonneg (mul_nonneg (le_of_lt fs.hL_pos) hbar_nonneg) hM_nonneg
  simpa [Z, M] using
    (integral_nonneg_le_of_integral_sq_le_sq
      (μ := fs.P) (Z := Z) (C := fs.L * fs.barDX * M)
      hZ_sq_int hZ_nonneg hright_nonneg hdelta_square_bound)

/-- The full `α D_X E‖δ‖` contribution in Lan Theorem 7.16 is absorbed by
the corrected epoch-difference penalty.

Candidate audit: searched `L2 to L1 integral square root probability norm` and
`active epoch difference alpha triangular budget finite sum`. The pointwise
absorption is supplied by
`active_delta_l1_pointwise_le_epoch_difference_max`; no SOptLib theorem unfolds
the paper-local `theorem716EpochDifferencePenalty` active epoch notation. -/
private lemma active_delta_l1_sum_le_epoch_difference_penalty
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            fs.α (fs.globalIndex s j) * fs.barDX *
              ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) ≤
      fs.L * fs.barDX ^ 2 * fs.theorem716EpochDifferencePenalty hT := by
  classical
  let M : ℕ → ℝ := fun s => fs.paperMaxEpochDifferenceAlphaOfWellDefined hT s
  have hterm :
      ∀ s ∈ Finset.Icc 0 fs.S, ∀ j ∈ fs.activeEpochSteps s,
        fs.α (fs.globalIndex s j) * fs.barDX *
            ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P ≤
          fs.L * fs.barDX ^ 2 * (fs.α (fs.globalIndex s j) * M s) := by
    intro s hs j hj
    have hpoint :=
      active_delta_l1_pointwise_le_epoch_difference_max
        (fs := fs) hT hα_le_one hb_ge_T (s := s) (j := j) hj
    have hcoef_nonneg : 0 ≤ fs.α (fs.globalIndex s j) * fs.barDX := by
      exact mul_nonneg (fs.hα_nonneg (fs.globalIndex s j)) (norm_nonneg _)
    have hmul :=
      mul_le_mul_of_nonneg_left hpoint hcoef_nonneg
    dsimp [M] at hmul ⊢
    nlinarith
  calc
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            fs.α (fs.globalIndex s j) * fs.barDX *
              ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P))
        ≤
      Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j => fs.L * fs.barDX ^ 2 * (fs.α (fs.globalIndex s j) * M s))) := by
          exact Finset.sum_le_sum (fun s hs =>
            Finset.sum_le_sum (fun j hj => hterm s hs j hj))
    _ =
      fs.L * fs.barDX ^ 2 * fs.theorem716EpochDifferencePenalty hT := by
        simp [SOptLib.FiniteSumConditionalGradientSetup.theorem716EpochDifferencePenalty,
          SOptLib.FiniteSumConditionalGradientSetup.theorem716EpochPenaltyWithMax,
          M, Finset.mul_sum, Finset.sum_mul, mul_assoc, mul_left_comm, mul_comm]

/-- One epoch of the triangular `α_{s,i}^2` sum is bounded by `T` copies of
the active generated stepsizes.

No SOptLib match: searched `active epoch difference alpha triangular budget
finite sum` and `Finset image sum subset nonnegative active epoch counting`,
then checked the target-file partition and active-step helpers. Mathlib supplies
the generic `Finset.sum_image`, `Finset.sum_le_sum_of_subset_of_nonneg`, and
`Finset.sum_le_card_nsmul`; the predecessor map
`epochDifferenceAlphaIndex s i = globalIndex s (i - 1)` and the partial final
epoch prefix closure are paper-local to Eq. (7.4.6). -/
private lemma one_epoch_triangular_budget_le_epoch_sum
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω) (s : ℕ) :
    Finset.sum (fs.activeEpochSteps s)
        (fun j => Finset.sum (Finset.Icc 2 j)
          (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2)) ≤
      (fs.T : ℝ) *
        Finset.sum (fs.activeEpochSteps s)
          (fun r => fs.α (fs.globalIndex s r) ^ 2) := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.epochDifferenceAlphaIndex] using
    (sum_active_triangular_predecessor_le_card_mul_sum
      (active := fs.activeEpochSteps s)
      (T := fs.T)
      (a := fun r => fs.α (fs.globalIndex s r) ^ 2)
      (P := fun r => fs.globalIndex s r ≤ fs.N)
      (hactive := by
        intro r
        exact fs.mem_activeEpochSteps)
      (hprefix := by
        intro i j hij hj
        exact fs.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hij hj)
      (ha_nonneg := by
        intro r _hr
        exact sq_nonneg (fs.α (fs.globalIndex s r))))

/-- The triangular epoch-difference alpha-square budget from Eq. (7.4.6), after
reindexing generated active epochs into the global output window.

Candidate audit: searched `active epoch difference alpha triangular budget
finite sum`, `Finset card Icc count sum constant`, and the target-file active
partition helpers. The available lemmas prove pointwise max/counting and the
global active partition, but no SOptLib/Mathlib lemma packages this paper-local
three-index triangular counting fact: each generated predecessor coefficient is
counted at most `T` times and `T ≤ b`. -/
private lemma active_triangular_epoch_difference_alpha_budget
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hb_ge_T : fs.T ≤ fs.b) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j => Finset.sum (Finset.Icc 2 j)
            (fun i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2))) ≤
      (fs.b : ℝ) *
        Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) := by
  classical
  simpa [SOptLib.FiniteSumConditionalGradientSetup.epochDifferenceAlphaIndex] using
    (active_triangular_predecessor_sum_le_batch_mul_global_sum
      (epochs := Finset.Icc 0 fs.S)
      (active := fs.activeEpochSteps)
      (T := fs.T) (b := fs.b)
      (idx := fun s r => fs.globalIndex s r)
      (a := fun k => fs.α k ^ 2)
      (P := fun s r => fs.globalIndex s r ≤ fs.N)
      (window := Finset.Icc 1 fs.N)
      (hactive := by
        intro s r
        exact fs.mem_activeEpochSteps)
      (hprefix := by
        intro s i j hij hj
        exact fs.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hij hj)
      (hactive_eq_global := by
        exact (finite_global_index_active_epoch_partition_sum
          (fs := fs) (A := fun k => fs.α k ^ 2)).symm)
      (hb_ge_T := hb_ge_T)
      (ha_nonneg := by
        intro s _hs r _hr
        exact sq_nonneg (fs.α (fs.globalIndex s r))))

/-- The delta-square component of Lan Theorem 7.16 contributes only the
`1/2 ∑ α_k^2` scalar budget.

Candidate audit: reused target-file
`finite_active_delta_square_term_le_epoch_alpha_sum` for the pointwise Lemma 7.4
substitution. The only remaining nonlocal ingredient is the paper-specific
triangular counting helper
`active_triangular_epoch_difference_alpha_budget`; no SOptLib theorem knows this
Algorithm 7.12 active epoch predecessor-index map. -/
private lemma active_delta_square_sum_le_half_alpha_square_budget
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            (1 / (2 * fs.L)) *
              ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P)) ≤
      fs.L * fs.barDX ^ 2 *
        (1 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2)) := by
  classical
  exact residual_second_moment_sum_le_half_alpha_square_budget
    (epochs := Finset.Icc 0 fs.S)
    (active := fs.activeEpochSteps)
    (secondMoment := fun s j =>
      (1 / (2 * fs.L)) *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P)
    (predecessorSq := fun s i => fs.α (fs.epochDifferenceAlphaIndex s i) ^ 2)
    (L := fs.L) (D := fs.barDX) (b := fs.b)
    (globalSq := Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2))
    (hb_pos := fs.hb_pos)
    (hL_nonneg := le_of_lt fs.hL_pos)
    (hpoint := by
      intro s _hs j hj
      exact finite_active_delta_square_term_le_epoch_alpha_sum
        (fs := fs) hα_le_one (s := s) (j := j) hj)
    (htri := active_triangular_epoch_difference_alpha_budget (fs := fs) hb_ge_T)

/-- Objective-free scalar budget for the corrected generated-process variant of
Lan Theorem 7.16.

Candidate audit: no pre-searched SOptLib/Mathlib candidate was listed for
`finite_source_form_budget_and_telescope_theorem716`; additionally searched
`epoch difference sum alpha square diameter bound`, `iterate update difference
norm alpha diameter`, and `Finset sum add distribute inequality split`, scanned
the target-file active epoch helpers and SOptLib finite-sum/integrability hits.
Those provide update, reindexing, and API fragments but no lemma packages the
paper's Eq. (7.4.6) scalar budget with the active epoch maximum. -/
private lemma finite_epochDifference_form_scalar_budget_theorem716
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            (1 / (2 * fs.L)) *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
              fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
              fs.α (fs.globalIndex s j) * fs.barDX *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) ≤
      fs.L * fs.barDX ^ 2 *
        (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochDifferencePenalty hT) := by
  classical
  exact three_term_scalar_budget_le_alpha_square_add_penalty
    (epochs := Finset.Icc 0 fs.S) (active := fs.activeEpochSteps)
    (A := fun s j =>
      (1 / (2 * fs.L)) *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P)
    (B := fun s j => fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2)
    (C := fun s j =>
      fs.α (fs.globalIndex s j) * fs.barDX *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)
    (alphaSq := Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2))
    (penalty := fs.theorem716EpochDifferencePenalty hT) (L := fs.L)
    (D := fs.barDX)
    (hA := active_delta_square_sum_le_half_alpha_square_budget
      (fs := fs) hα_le_one hb_ge_T)
    (hB := le_of_eq (active_alpha_square_sum_eq_global_theorem716 (fs := fs)))
    (hC := active_delta_l1_sum_le_epoch_difference_penalty
      (fs := fs) hT hα_le_one hb_ge_T)

/-- Source-corrected scalar budget for Lan Theorem 7.16, stated with the
literal displayed maximum from the printed theorem statement.

The PDF line after Eq. (7.4.6) writes
`x_{s,i} - x_{s,i-1} = α_{s,i}(y_{s,i}-x_{s,i})`, while Algorithm 7.12 and the
paper's own `k = sT + t` convention make this update coefficient
`α (globalIndex s (i - 1))`. The formal counterexample
`theorem716_literal_max_not_general_epoch_difference_bound` records why this
literal helper is not Fill-ready from the generated-process delta-square
evidence; the corrected generated-process scalar route is
`finite_epochDifference_form_scalar_budget_theorem716`. -/
private lemma finite_source_form_scalar_budget_theorem716
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b)
    (hbridge : fs.Theorem716LiteralCoordinateBridge hT) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            (1 / (2 * fs.L)) *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
              fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
              fs.α (fs.globalIndex s j) * fs.barDX *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) ≤
      fs.L * fs.barDX ^ 2 *
        (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty hT) := by
  exact scalar_budget_mono_penalty
    (hscalar := finite_epochDifference_form_scalar_budget_theorem716
      (fs := fs) hT hα_le_one hb_ge_T)
    (hcoef_nonneg := mul_nonneg (le_of_lt fs.hL_pos) (sq_nonneg fs.barDX))
    (hpenalty := hbridge)

/-- Source-form finite-sum assembly after Eq. (7.4.6) for Lan Theorem 7.16.

Book/PDF citation: Theorem 7.16 proof, lines after Eq. (7.4.6), bound the
`(1 / (2b))` double sum and the square-root estimator term, then telescope the
objective values and use `f*` as a terminal lower bound. Candidate audit:
searched SOptLib/target for `finite sum epoch difference active steps paper
maximum alpha square` and `Finset sum telescope objective drop fStar lower
bound`; existing hits (`finite_global_index_active_epoch_partition_sum`,
`sum_Icc_sub_succ`, `integral_sum_telescope_bound_of_pointwise_lower_bound`)
cover reindex/telescope fragments but no lemma packages the paper-specific
active-window scalar budget with `paperMaxInnerAlphaOfWellDefined`. -/
private lemma finite_source_form_budget_and_telescope_theorem716
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b)
    (hbridge : fs.Theorem716LiteralCoordinateBridge hT) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            ∫ ω,
                (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                  fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
              (1 / (2 * fs.L)) *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
              fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
              fs.α (fs.globalIndex s j) * fs.barDX *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) ≤
      fs.f fs.x₁ - fs.fStar +
        fs.L * fs.barDX ^ 2 *
          (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
            fs.theorem716EpochPenalty hT) := by
  classical
  have hobj := finite_active_objective_drop_telescope
    (fs := fs) hα_le_one
  have hscalar := finite_source_form_scalar_budget_theorem716
    (fs := fs) hT hα_le_one hb_ge_T hbridge
  let Obj : ℕ → ℕ → ℝ := fun s j =>
    ∫ ω,
      (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
        fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P
  let Scal : ℕ → ℕ → ℝ := fun s j =>
    (1 / (2 * fs.L)) *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
      fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
      fs.α (fs.globalIndex s j) * fs.barDX *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P
  simpa [Obj, Scal, add_assoc] using
    active_sum_objective_drop_add_scalar_budget
      (epochs := Finset.Icc 0 fs.S) (active := fs.activeEpochSteps)
      (Obj := Obj) (Scal := Scal) (init := fs.f fs.x₁) (lower := fs.fStar)
      (scalarBound :=
        fs.L * fs.barDX ^ 2 *
          (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
            fs.theorem716EpochPenalty hT))
      (hobj := by simpa [Obj] using hobj)
      (hscalar := by simpa [Scal, add_assoc] using hscalar)

/-- Source-form finite-sum assembly for the statement-corrected generated
process version of Lan Theorem 7.16.

This is the same telescope assembly as
`finite_source_form_budget_and_telescope_theorem716`, but it consumes the
epoch-difference scalar budget matching the generated update convention
`x_k -> x_{k+1}` with coefficient `α_k`. -/
private lemma finite_epochDifference_form_budget_and_telescope_theorem716
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b) :
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            ∫ ω,
                (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                  fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
              (1 / (2 * fs.L)) *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
              fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
              fs.α (fs.globalIndex s j) * fs.barDX *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) ≤
      fs.f fs.x₁ - fs.fStar +
        fs.L * fs.barDX ^ 2 *
          (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
            fs.theorem716EpochDifferencePenalty hT) := by
  classical
  have hobj := finite_active_objective_drop_telescope
    (fs := fs) hα_le_one
  have hscalar := finite_epochDifference_form_scalar_budget_theorem716
    (fs := fs) hT hα_le_one hb_ge_T
  let Obj : ℕ → ℕ → ℝ := fun s j =>
    ∫ ω,
      (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
        fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P
  let Scal : ℕ → ℕ → ℝ := fun s j =>
    (1 / (2 * fs.L)) *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
      fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
      fs.α (fs.globalIndex s j) * fs.barDX *
        ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P
  have hsplit :
      Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s)
            (fun j =>
              ∫ ω,
                  (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
                (1 / (2 * fs.L)) *
                  ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
                fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
                fs.α (fs.globalIndex s j) * fs.barDX *
                  ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) =
        Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s) (fun j => Obj s j)) +
        Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s) (fun j => Scal s j)) := by
    calc
      Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s)
            (fun j =>
              ∫ ω,
                  (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
                (1 / (2 * fs.L)) *
                  ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
                fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
                fs.α (fs.globalIndex s j) * fs.barDX *
                  ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P))
          =
          Finset.sum (Finset.Icc 0 fs.S)
            (fun s => Finset.sum (fs.activeEpochSteps s)
              (fun j => Obj s j + Scal s j)) := by
            refine Finset.sum_congr rfl ?_
            intro s hs
            refine Finset.sum_congr rfl ?_
            intro j hj
            dsimp [Obj, Scal]
            ring
      _ =
          Finset.sum (Finset.Icc 0 fs.S)
            (fun s =>
              Finset.sum (fs.activeEpochSteps s) (fun j => Obj s j) +
              Finset.sum (fs.activeEpochSteps s) (fun j => Scal s j)) := by
            refine Finset.sum_congr rfl ?_
            intro s hs
            rw [Finset.sum_add_distrib]
      _ =
          Finset.sum (Finset.Icc 0 fs.S)
            (fun s => Finset.sum (fs.activeEpochSteps s) (fun j => Obj s j)) +
          Finset.sum (Finset.Icc 0 fs.S)
            (fun s => Finset.sum (fs.activeEpochSteps s) (fun j => Scal s j)) := by
            rw [Finset.sum_add_distrib]
  calc
    Finset.sum (Finset.Icc 0 fs.S)
        (fun s => Finset.sum (fs.activeEpochSteps s)
          (fun j =>
            ∫ ω,
                (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                  fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
              (1 / (2 * fs.L)) *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
              fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
              fs.α (fs.globalIndex s j) * fs.barDX *
                ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P))
        =
        Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s) (fun j => Obj s j)) +
        Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s) (fun j => Scal s j)) := hsplit
    _ ≤
        (fs.f fs.x₁ - fs.fStar) +
          fs.L * fs.barDX ^ 2 *
            (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
              fs.theorem716EpochDifferencePenalty hT) := by
        exact add_le_add hobj hscalar

/-- Finite weighted Wolfe-gap telescope core for Lan Theorem 7.16.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json`
`key_lemmas[2].proof` says to combine the smooth one-step descent, the
finite-sum Lemma 7.4 estimator bound, epoch summation, and the objective-value
telescope. Candidate audit: checked SOptLib `summed_one_step_gap_bound_of_telescope`,
target-file `lemma_7_4`, and finite one-step second-moment helpers; no existing
theorem packages this paper-specific active-epoch reindexing and
`theorem716EpochPenalty` scalar budget, so this helper isolates the source
proof core after output-law/integrability bookkeeping has been discharged. -/
theorem finite_weighted_gap_sum_bound_theorem716
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b)
    (hbridge : fs.Theorem716LiteralCoordinateBridge hT) :
    Finset.sum (Finset.Icc 1 fs.N)
        (fun k => fs.α k *
          ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P) ≤
      fs.f fs.x₁ - fs.fStar +
        fs.L * fs.barDX ^ 2 *
          (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
            fs.theorem716EpochPenalty hT) := by
  classical
  exact
    weighted_gap_sum_bound_of_active_one_step_and_budget
      (window := Finset.Icc 1 fs.N) (epochs := Finset.Icc 0 fs.S)
      (active := fs.activeEpochSteps) (idx := fun s j => fs.globalIndex s j)
      (gapWeight := fun k =>
        fs.α k *
          ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
            (fs.iterProcess k ω) ∂fs.P)
      (stepRhs := fun s j =>
        ∫ ω,
            (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
              fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
          (1 / (2 * fs.L)) *
            ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
          fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
          fs.α (fs.globalIndex s j) * fs.barDX *
            ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)
      (finalBound :=
        fs.f fs.x₁ - fs.fStar +
          fs.L * fs.barDX ^ 2 *
            (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
              fs.theorem716EpochPenalty hT))
      (finite_global_index_active_epoch_partition_sum (fs := fs)
        (A := fun k =>
          fs.α k *
            ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
              (fs.iterProcess k ω) ∂fs.P))
      (by
        intro s _hs j hj
        exact finite_active_expected_one_step_gap_bound_source_form
          (fs := fs) hα_le_one (s := s) (j := j) hj)
      (finite_source_form_budget_and_telescope_theorem716
        (fs := fs) hT hα_le_one hb_ge_T hbridge)

/-- Corrected generated-process finite weighted Wolfe-gap telescope core for
Lan Theorem 7.16.

This theorem is the public handoff point for the statement-corrected finite-sum
route: its RHS uses `theorem716EpochDifferencePenalty`, while the literal
printed theorem remains under `finite_weighted_gap_sum_bound_theorem716`. -/
theorem finite_weighted_gap_sum_bound_theorem716_epochDifference
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b) :
    Finset.sum (Finset.Icc 1 fs.N)
        (fun k => fs.α k *
          ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P) ≤
      fs.f fs.x₁ - fs.fStar +
        fs.L * fs.barDX ^ 2 *
          (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
            fs.theorem716EpochDifferencePenalty hT) := by
  classical
  rw [finite_global_index_active_epoch_partition_sum (fs := fs)
    (A := fun k =>
      fs.α k * ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P)]
  have hstep_sum :
      Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s)
            (fun j =>
              fs.α (fs.globalIndex s j) *
                ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer
                  (fs.iterProcess (fs.globalIndex s j) ω) ∂fs.P)) ≤
        Finset.sum (Finset.Icc 0 fs.S)
          (fun s => Finset.sum (fs.activeEpochSteps s)
            (fun j =>
              ∫ ω,
                  (fs.f (fs.iterProcess (fs.globalIndex s j) ω) -
                    fs.f (fs.iterProcess (fs.globalIndex s j + 1) ω)) ∂fs.P +
                (1 / (2 * fs.L)) *
                  ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ^ 2 ∂fs.P +
                fs.L * fs.α (fs.globalIndex s j) ^ 2 * fs.barDX ^ 2 +
                fs.α (fs.globalIndex s j) * fs.barDX *
                  ∫ ω, ‖fs.deltaProcess (fs.globalIndex s j) ω‖ ∂fs.P)) := by
    refine Finset.sum_le_sum ?_
    intro s hs
    refine Finset.sum_le_sum ?_
    intro j hj
    exact finite_active_expected_one_step_gap_bound_source_form
      (fs := fs) hα_le_one (s := s) (j := j) hj
  exact le_trans hstep_sum
    (finite_epochDifference_form_budget_and_telescope_theorem716
      (fs := fs) hT hα_le_one hb_ge_T)

/-! Guarded domain-aware finite-sum nonconvex variance-reduced
conditional-gradient bound with the literal printed current-coordinate
maximum. This declaration is audit-only unless the coordinate bridge
`Theorem716LiteralCoordinateBridge` is supplied; the active generated-process
replacement is `theorem_7_16_general_epochDifference_domainAware`. -/
theorem theorem_7_16_general_domainAware
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hR : 0 < fs.alphaSum) (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b)
    (hbridge : fs.Theorem716LiteralCoordinateBridge hT) :
    fs.expectedWolfeGapOfWellDefined hR ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty hT) := by
  classical
  have hgap_int :
      ∀ R : fs.OutputTime,
        Integrable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess R.1 ω)) fs.P := by
    intro R
    exact finite_wolfeGap_iterProcess_integrable_of_le
      (fs := fs) hα_le_one R.1 (Finset.mem_Icc.mp R.2).2
  have hexpand :=
    fs.expectedWolfeGapOfWellDefined_eq_weighted_sum_of_integrable hR hgap_int
  rw [hexpand]
  have hsum :=
    finite_weighted_gap_sum_bound_theorem716
      (fs := fs) hT hα_le_one hb_ge_T hbridge
  have hinv_nonneg : 0 ≤ (fs.alphaSum)⁻¹ :=
    inv_nonneg.mpr (le_of_lt hR)
  calc
    (fs.alphaSum)⁻¹ *
        Finset.sum (Finset.Icc 1 fs.N)
          (fun k => fs.α k *
            ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P)
        ≤ (fs.alphaSum)⁻¹ *
          (fs.f fs.x₁ - fs.fStar +
            fs.L * fs.barDX ^ 2 *
              (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
                fs.theorem716EpochPenalty hT)) :=
      mul_le_mul_of_nonneg_left hsum hinv_nonneg
    _ =
        (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
          fs.L * fs.barDX ^ 2 / fs.alphaSum *
            (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
              fs.theorem716EpochPenalty hT) := by
      field_simp [ne_of_gt hR]

/-! Statement-corrected domain-aware finite-sum nonconvex variance-reduced
conditional-gradient bound.

The printed Theorem 7.16 maximum is retained by
`theorem_7_16_general_domainAware`. This corrected generated-process theorem
uses the epoch-difference maximum matching the formal update convention
`x_k -> x_{k+1}` with coefficient `α_k`; it is the FILL-ready public route
created by the statement-correction protocol. -/
theorem theorem_7_16_general_epochDifference_domainAware
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (hR : 0 < fs.alphaSum) (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b) :
    fs.expectedWolfeGapOfWellDefined hR ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochDifferencePenalty hT) := by
  classical
  have hgap_int :
      ∀ R : fs.OutputTime,
        Integrable (fun ω => SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess R.1 ω)) fs.P := by
    intro R
    exact finite_wolfeGap_iterProcess_integrable_of_le
      (fs := fs) hα_le_one R.1 (Finset.mem_Icc.mp R.2).2
  have hexpand :=
    fs.expectedWolfeGapOfWellDefined_eq_weighted_sum_of_integrable hR hgap_int
  rw [hexpand]
  have hsum :=
    finite_weighted_gap_sum_bound_theorem716_epochDifference
      (fs := fs) hT hα_le_one hb_ge_T
  have hinv_nonneg : 0 ≤ (fs.alphaSum)⁻¹ :=
    inv_nonneg.mpr (le_of_lt hR)
  calc
    (fs.alphaSum)⁻¹ *
        Finset.sum (Finset.Icc 1 fs.N)
          (fun k => fs.α k *
            ∫ ω, SOptLib.ConditionalGradient.wolfeGap fs.gradf fs.wolfeGapMaximizer (fs.iterProcess k ω) ∂fs.P)
        ≤ (fs.alphaSum)⁻¹ *
          (fs.f fs.x₁ - fs.fStar +
            fs.L * fs.barDX ^ 2 *
              (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
                fs.theorem716EpochDifferencePenalty hT)) :=
      mul_le_mul_of_nonneg_left hsum hinv_nonneg
    _ =
        (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
          fs.L * fs.barDX ^ 2 / fs.alphaSum *
            (3 / 2 * Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
              fs.theorem716EpochDifferencePenalty hT) := by
      field_simp [ne_of_gt hR]

/-! Conditional form of the statement-corrected finite-sum Theorem 7.16. -/
theorem theorem_7_16_general_epochDifference_conditional
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (h : fs.Theorem716DomainBoundary) :
    fs.paperExpectedWolfeGapConditional h ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochDifferencePenalty
            (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) := by
  let hR := fs.alphaSum_pos_of_theorem716Boundary h
  let hT := fs.innerAlphaWindowWellDefined_of_theorem716Boundary h
  let hα_le_one := fs.alpha_le_one_of_theorem716Boundary h
  let hb_ge_T := fs.batch_ge_epoch_of_theorem716Boundary h
  simpa [SOptLib.FiniteSumConditionalGradientSetup.paperExpectedWolfeGapConditional]
    using theorem_7_16_general_epochDifference_domainAware
      (fs := fs) hR hT hα_le_one hb_ge_T

/-! Guarded compatibility helper for Theorem 7.16 using the documented
empty-window convention for the corrected epoch-difference maximum.

This theorem is intentionally not an unguarded paper-facing statement. The
empty-window convention is a Lean totalization, while the source proof still
uses the normalized output law, nonempty source epoch-difference maximum, and
feasible convex-combination iterates supplied by `Theorem716DomainBoundary`. -/
theorem theorem_7_16_general_emptyWindowExtension
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (h : fs.Theorem716DomainBoundary)
    (hbridge :
      fs.Theorem716LiteralCoordinateBridge
        (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) :
    fs.paperExpectedWolfeGapConditional h ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenaltyEmptyWindowExtension) := by
  classical
  let hR := fs.alphaSum_pos_of_theorem716Boundary h
  let hT := fs.innerAlphaWindowWellDefined_of_theorem716Boundary h
  let hα_le_one := fs.alpha_le_one_of_theorem716Boundary h
  let hb_ge_T := fs.batch_ge_epoch_of_theorem716Boundary h
  have hmax :
      ∀ s,
        fs.paperMaxInnerAlphaEmptyWindowExtension s =
          fs.paperMaxInnerAlphaOfWellDefined hT s := by
    intro s
    rw [fs.paperMaxInnerAlphaEmptyWindowExtension_eq_sup_of_wellDefined s hT,
      fs.paperMaxInnerAlphaOfWellDefined_eq_sup s hT]
  have hpenalty :
      fs.theorem716EpochPenaltyEmptyWindowExtension =
        fs.theorem716EpochPenalty hT := by
    simp [SOptLib.FiniteSumConditionalGradientSetup.theorem716EpochPenalty,
      SOptLib.FiniteSumConditionalGradientSetup.theorem716DisplayedEpochPenalty,
      SOptLib.FiniteSumConditionalGradientSetup.theorem716EpochPenaltyEmptyWindowExtension,
      SOptLib.FiniteSumConditionalGradientSetup.theorem716EpochPenaltyWithMax, hmax]
  rw [hpenalty]
  simpa [SOptLib.FiniteSumConditionalGradientSetup.paperExpectedWolfeGapConditional]
    using theorem_7_16_general_domainAware
      (fs := fs) hR hT hα_le_one hb_ge_T hbridge

/-! Guarded finite-sum nonconvex variance-reduced conditional-gradient bound
with the literal displayed current-coordinate maximum from the JSON/PDF. It
requires the explicit coordinate bridge because the generated process natively
proves the epoch-difference maximum bound.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json
key_lemmas[2].statement_math`: `"\mathbb{E}[\operatorname{gap}(x_R)] \le ... \sum_{s=0}^{S} ((\sum_{j=1}^{T} \alpha_{s,j} \max_{j=2,\ldots,T} \alpha_{s,j}))"`. -/
theorem theorem_7_16_general_conditional
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (h : fs.Theorem716DomainBoundary)
    (hbridge :
      fs.Theorem716LiteralCoordinateBridge
        (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) :
    fs.paperExpectedWolfeGapConditional h ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty
            (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) := by
  let hR := fs.alphaSum_pos_of_theorem716Boundary h
  let hT := fs.innerAlphaWindowWellDefined_of_theorem716Boundary h
  let hα_le_one := fs.alpha_le_one_of_theorem716Boundary h
  let hb_ge_T := fs.batch_ge_epoch_of_theorem716Boundary h
  simpa [SOptLib.FiniteSumConditionalGradientSetup.paperExpectedWolfeGapConditional]
    using theorem_7_16_general_domainAware
      (fs := fs) hR hT hα_le_one hb_ge_T hbridge

/-! Conditional alias for the finite-sum nonconvex variance-reduced
conditional-gradient bound under the single Algorithm 7.13 realization
boundary. It is deliberately not the paper-named `theorem_7_16_general`,
because the contract carries facts not asserted in the current JSON. -/
theorem theorem_7_16_general_conditional_alias
    (fs : FiniteSumNonconvexConditionalGradientSetup E Ω)
    (h : fs.Theorem716DomainBoundary)
    (hbridge :
      fs.Theorem716LiteralCoordinateBridge
        (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) :
    fs.paperExpectedWolfeGapConditional h ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty
            (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) := by
  simpa [SOptLib.FiniteSumConditionalGradientSetup.paperExpectedWolfeGapConditional]
    using theorem_7_16_general_conditional (fs := fs) h hbridge

/-! Guarded domain-aware Euclidean helper for the literal Theorem 7.16
penalty. -/
theorem theorem_7_16_domainAware
    (n : ℕ)
    (fs : FiniteSumNonconvexConditionalGradientSetup (PaperVariableSpace n) Ω)
    (hR : 0 < fs.alphaSum) (hT : fs.InnerAlphaWindowWellDefined)
    (hα_le_one : ∀ k, k ∈ Finset.Icc 1 fs.N → fs.α k ≤ 1)
    (hb_ge_T : fs.T ≤ fs.b)
    (hbridge : fs.Theorem716LiteralCoordinateBridge hT) :
    fs.expectedWolfeGapOfWellDefined hR ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty hT) := by
  simpa using theorem_7_16_general_domainAware
    (fs := fs) hR hT hα_le_one hb_ge_T hbridge

/-! Euclidean conditional helper for Theorem 7.16 over `X ⊆ ℝ^n`, with the
displayed `max_{j=2,...,T}` represented by the genuine finite maximum supplied
by the realization contract.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json
key_lemmas[2].statement_math`: `"\mathbb{E}[\operatorname{gap}(x_R)] \le ... \max_{j=2,\ldots,T} \alpha_{s,j}"`. -/
theorem theorem_7_16_conditional
    (n : ℕ)
    (fs : FiniteSumNonconvexConditionalGradientSetup (PaperVariableSpace n) Ω)
    (h : fs.Theorem716DomainBoundary)
    (hbridge :
      fs.Theorem716LiteralCoordinateBridge
        (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) :
    fs.expectedWolfeGapConditional h ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty
            (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) := by
  simpa using theorem_7_16_general_conditional (fs := fs) h hbridge

/-! Conditional Euclidean form of Theorem 7.16 over `X ⊆ ℝ^n`, stated against
the single finite-sum realization contract for the normalized output law and
nonempty `max_{j=2,...,T}` window. It is not exported under the paper theorem
name because the contract facts are not source-derived in the current JSON. -/
theorem theorem_7_16_conditional_euclidean
    (n : ℕ)
    (fs : FiniteSumNonconvexConditionalGradientSetup (PaperVariableSpace n) Ω)
    (h : fs.Theorem716DomainBoundary)
    (hbridge :
      fs.Theorem716LiteralCoordinateBridge
        (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) :
    fs.paperExpectedWolfeGapConditional h ≤
      (fs.f fs.x₁ - fs.fStar) / fs.alphaSum +
        fs.L * fs.barDX ^ 2 / fs.alphaSum *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 fs.N) (fun k => fs.α k ^ 2) +
          fs.theorem716EpochPenalty
            (fs.innerAlphaWindowWellDefined_of_theorem716Boundary h)) := by
  simpa using theorem_7_16_general_conditional_alias (fs := fs) h hbridge

/-- Active within-epoch steps generated by stochastic Algorithm 7.13.

No SOptLib match: searched `active epoch partition global index stochastic
setup` and checked the target-file finite analogue
`finite_global_index_active_epoch_partition_sum`. The finite helper has the
wrong setup type, while abstract telescope helpers do not encode the partial
final epoch `globalIndex s j ≤ N`. -/
private noncomputable def stochasticActiveEpochSteps
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) (s : ℕ) :
    Finset ℕ :=
  (Finset.Icc 1 setup.T).filter (fun j => setup.globalIndex s j ≤ setup.N)

private lemma stochastic_mem_activeEpochSteps
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) {s j : ℕ} :
    j ∈ stochasticActiveEpochSteps setup s ↔
      j ∈ Finset.Icc 1 setup.T ∧ setup.globalIndex s j ≤ setup.N := by
  simp [stochasticActiveEpochSteps]

private lemma stochastic_activeEpochSteps_mem_epoch
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) {s j : ℕ}
    (hj : j ∈ stochasticActiveEpochSteps setup s) :
    1 ≤ j ∧ j ≤ setup.T := by
  exact SOptLib.activeEpochStepsOfGlobalIndex_mem_epoch
    (activeEpochSteps := stochasticActiveEpochSteps setup)
    (hmem_epoch := by
      intro s j hj
      exact ((stochastic_mem_activeEpochSteps setup).mp hj).1)
    (s := s) (j := j) hj

private lemma stochastic_activeEpochSteps_globalIndex_le
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) {s j : ℕ}
    (hj : j ∈ stochasticActiveEpochSteps setup s) :
    setup.globalIndex s j ≤ setup.N :=
  ((stochastic_mem_activeEpochSteps setup).mp hj).2

private lemma stochastic_globalIndex_mem_output_of_active
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) {s j : ℕ}
    (hj : j ∈ stochasticActiveEpochSteps setup s) :
    setup.globalIndex s j ∈ Finset.Icc 1 setup.N := by
  exact SOptLib.global_index_mem_output_window_of_mem_active_epoch_steps
    (N := setup.N) (T := setup.T) (globalIndex := setup.globalIndex)
    (activeSteps := stochasticActiveEpochSteps setup)
    (hactive_epoch := fun {s j} hj =>
      stochastic_activeEpochSteps_mem_epoch (setup := setup) hj)
    (hglobalIndex_pos_of_epoch := fun {s j} hj_epoch => by
      unfold StochasticNonconvexConditionalGradientSetup.globalIndex
      omega)
    (hglobalIndex_le_of_mem := fun {s j} hj =>
      stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj)
    hj

private noncomputable def stochasticEpochOfIndex
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) (k : ℕ) : ℕ :=
  (k - 1) / setup.T

private noncomputable def stochasticStepOfIndex
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) (k : ℕ) : ℕ :=
  (k - 1) % setup.T + 1

private lemma stochastic_globalIndex_epochOfIndex_stepOfIndex
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {k : ℕ} (hk : 1 ≤ k) :
    setup.globalIndex (stochasticEpochOfIndex setup k)
        (stochasticStepOfIndex setup k) = k := by
  unfold StochasticNonconvexConditionalGradientSetup.globalIndex
    stochasticEpochOfIndex stochasticStepOfIndex
  rw [Nat.mul_comm ((k - 1) / setup.T) setup.T, ← Nat.add_assoc, Nat.div_add_mod]
  omega

private lemma stochastic_stepOfIndex_mem_activeEpochSteps
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 setup.N) :
    stochasticStepOfIndex setup k ∈
      stochasticActiveEpochSteps setup (stochasticEpochOfIndex setup k) := by
  exact SOptLib.stepOfGlobalIndex_mem_activeEpochSteps
    (T := setup.T) (N := setup.N) (globalIndex := setup.globalIndex)
    (activeSteps := stochasticActiveEpochSteps setup)
    (epochOfIndex := stochasticEpochOfIndex setup)
    (stepOfIndex := stochasticStepOfIndex setup)
    (hactive := by
      intro s j
      exact stochastic_mem_activeEpochSteps setup)
    (hstep_mem_epoch := by
      intro k hk
      have hTpos : 0 < setup.T := Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hT_pos
      have hmod : (k - 1) % setup.T < setup.T := Nat.mod_lt _ hTpos
      unfold stochasticStepOfIndex
      exact Finset.mem_Icc.mpr ⟨by omega, by omega⟩)
    (hglobalIndex_decode := by
      intro k hk
      exact stochastic_globalIndex_epochOfIndex_stepOfIndex setup (Finset.mem_Icc.mp hk).1)
    hk

private lemma stochastic_epochOfIndex_globalIndex_eq
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {s j : ℕ} (hj1 : 1 ≤ j) (hjT : j ≤ setup.T) :
    stochasticEpochOfIndex setup (setup.globalIndex s j) = s := by
  have hjsub : j - 1 < setup.T := by omega
  unfold stochasticEpochOfIndex StochasticNonconvexConditionalGradientSetup.globalIndex
  have hsub : s * setup.T + j - 1 = j - 1 + setup.T * s := by
    rw [Nat.mul_comm]
    omega
  rw [hsub]
  rw [Nat.add_mul_div_left, Nat.div_eq_of_lt hjsub]
  simp
  all_goals omega

private lemma stochastic_stepOfIndex_globalIndex_eq
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {s j : ℕ} (hj1 : 1 ≤ j) (hjT : j ≤ setup.T) :
    stochasticStepOfIndex setup (setup.globalIndex s j) = j := by
  have hjsub : j - 1 < setup.T := by omega
  unfold stochasticStepOfIndex StochasticNonconvexConditionalGradientSetup.globalIndex
  have hsub : s * setup.T + j - 1 = j - 1 + setup.T * s := by
    rw [Nat.mul_comm]
    omega
  rw [hsub]
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hjsub]
  omega

/-- Reindex the stochastic output window `{1, ..., N}` by Algorithm 7.13's
active epoch coordinates.

No SOptLib match: searched `active epoch partition global index stochastic
setup`, `stochastic weighted gap sum theorem 7.17 active epoch partition output
normalization`, and checked the finite target-file partition helper. The
finite helper cannot be specialized across setup types, while SOptLib telescope
lemmas do not encode Algorithm 7.13's partial final epoch. -/
private lemma stochastic_global_index_active_epoch_partition_sum
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (A : ℕ → ℝ) :
    Finset.sum (Finset.Icc 1 setup.N) A =
      Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j => A (setup.globalIndex s j))) := by
  exact SOptLib.sum_output_window_eq_sum_active_epoch_steps
    (N := setup.N) (S := setup.S) (A := A)
    (activeEpochSteps := stochasticActiveEpochSteps setup)
    (globalIndex := setup.globalIndex)
    (epochOfIndex := stochasticEpochOfIndex setup)
    (stepOfIndex := stochasticStepOfIndex setup)
    (by
      intro k hk
      refine ⟨?_, stochastic_stepOfIndex_mem_activeEpochSteps (setup := setup) hk⟩
      have hk_bounds := Finset.mem_Icc.mp hk
      have hnum : k - 1 ≤ setup.N + setup.T - 1 := by omega
      have hdiv := Nat.div_le_div_right (c := setup.T) hnum
      refine Finset.mem_Icc.mpr ⟨Nat.zero_le _, ?_⟩
      simpa [stochasticEpochOfIndex, StochasticNonconvexConditionalGradientSetup.S]
        using hdiv)
    (by
      intro k hk
      exact stochastic_globalIndex_epochOfIndex_stepOfIndex setup
        (Finset.mem_Icc.mp hk).1)
    (by
      intro s j _hs hj
      have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
      exact ⟨stochastic_globalIndex_mem_output_of_active (setup := setup) hj,
        stochastic_epochOfIndex_globalIndex_eq (setup := setup)
          hj_epoch.1 hj_epoch.2,
        stochastic_stepOfIndex_globalIndex_eq (setup := setup)
          hj_epoch.1 hj_epoch.2⟩)

/-- The active stochastic epoch windows contain exactly the `N` generated
global steps, counted as real weights.

Candidate audit: this is a direct specialization of
`stochastic_global_index_active_epoch_partition_sum` with constant weight one;
no separate SOptLib counting lemma knows Algorithm 7.13's partial final epoch. -/
private lemma stochastic_active_step_count_real
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => ((stochasticActiveEpochSteps setup s).card : ℝ)) =
      (setup.N : ℝ) := by
  exact SOptLib.sum_card_active_epoch_steps_eq_output_window_card_real
    (N := setup.N) (S := setup.S)
    (activeEpochSteps := stochasticActiveEpochSteps setup)
    (globalIndex := setup.globalIndex)
    (epochOfIndex := stochasticEpochOfIndex setup)
    (stepOfIndex := stochasticStepOfIndex setup)
    (by
      intro k hk
      refine ⟨?_, stochastic_stepOfIndex_mem_activeEpochSteps (setup := setup) hk⟩
      have hk_bounds := Finset.mem_Icc.mp hk
      have hnum : k - 1 ≤ setup.N + setup.T - 1 := by omega
      have hdiv := Nat.div_le_div_right (c := setup.T) hnum
      refine Finset.mem_Icc.mpr ⟨Nat.zero_le _, ?_⟩
      simpa [stochasticEpochOfIndex, StochasticNonconvexConditionalGradientSetup.S]
        using hdiv)
    (by
      intro k hk
      exact stochastic_globalIndex_epochOfIndex_stepOfIndex setup
        (Finset.mem_Icc.mp hk).1)
    (by
      intro s j _hs hj
      have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
      exact ⟨stochastic_globalIndex_mem_output_of_active (setup := setup) hj,
        stochastic_epochOfIndex_globalIndex_eq (setup := setup)
          hj_epoch.1 hj_epoch.2,
        stochastic_stepOfIndex_globalIndex_eq (setup := setup)
          hj_epoch.1 hj_epoch.2⟩)

/-- Stochastic Lemma 7.5 substitution for the active delta-square term, with
one variance-floor contribution per generated step.

Candidate audit: no pre-searched candidate was listed for
`stochastic_weighted_gap_sum_bound_theorem717`. I additionally searched
`stochastic scalar budget variance active epoch delta square` and checked the
finite target-file helpers `active_delta_square_sum_le_half_alpha_square_budget`
and `finite_epochDifference_form_scalar_budget_theorem716`; those helpers use
the finite-sum setup and have no `σ² / m` floor. SOptLib telescope/scalar
helpers do not encode Algorithm 7.13's active epoch partition, so this local
bridge is the source-specific Lemma 7.5 variance substitution from Lan
Theorem 7.17. -/
private lemma stochastic_active_delta_square_term_le_epochDiff_plus_variance
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX)
    (hlemma75_active :
      ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
        ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
            ∂setup.P ≤
          setup.L ^ 2 / setup.b *
              ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
            setup.σ ^ 2 / setup.m)
    (hactive_count :
      Finset.sum (Finset.Icc 0 setup.S)
          (fun s => ((stochasticActiveEpochSteps setup s).card : ℝ)) =
        (setup.N : ℝ)) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j =>
            (1 / (2 * setup.L)) *
              ∫ ω,
                ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                ∂setup.P)) ≤
      Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j =>
            (setup.L / (2 * setup.b)) *
              ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P)) +
        (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) := by
  exact sum_scaled_l2_error_le_epoch_budget_add_variance_floor
    (outer := Finset.Icc 0 setup.S)
    (inner := fun s => stochasticActiveEpochSteps setup s)
    (L := setup.L) (b := (setup.b : ℝ)) (m := (setup.m : ℝ))
    (sigma := setup.σ) (N := (setup.N : ℝ))
    (deltaSq := fun s j =>
      ∫ ω,
        ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
        ∂setup.P)
    (epoch := fun s j =>
      ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P)
    setup.hL_pos
    (by
      exact ne_of_gt (by
        exact_mod_cast
          (Nat.lt_of_lt_of_le Nat.zero_lt_one
            (le_trans setup.hT_pos setup.hb_ge_T))))
    (by
      exact ne_of_gt (by
        exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos)))
    hlemma75_active hactive_count

/-- The explicit `L α_k^2 D̄_X^2` term in Lan Theorem 7.17 reindexes from
stochastic active epoch coordinates back to the global output window.

Candidate audit: searched `stochastic active epoch scalar budget delta square`,
`active epoch partition global index stochastic setup`, and checked the finite
target-file helper `active_alpha_square_sum_eq_global_theorem716`. The finite
helper has the wrong setup type; the stochastic partition helper
`stochastic_global_index_active_epoch_partition_sum` is the matching local
primitive, and no SOptLib theorem encodes Algorithm 7.13's active epoch map. -/
private lemma stochastic_active_alpha_square_sum_eq_global
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j =>
            setup.L * setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
              setup.barDX ^ 2)) =
      setup.L * setup.barDX ^ 2 *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k => setup.αOfWellDefined hDX k ^ 2) := by
  exact active_sum_scaled_alpha_sq_eq_global
    (outer := Finset.Icc 0 setup.S)
    (inner := fun s => stochasticActiveEpochSteps setup s)
    (globalIndex := setup.globalIndex)
    (alpha := setup.αOfWellDefined hDX)
    (L := setup.L) (D := setup.barDX) (N := setup.N)
    (by
      rw [stochastic_global_index_active_epoch_partition_sum (setup := setup)
        (A := fun k => setup.αOfWellDefined hDX k ^ 2)])

/-- Eq. (7.4.15) absorbs the Lemma 7.5 variance floor into the squared
`L D̄_X α` scale.

No SOptLib match: searched `alpha formula sqrt variance floor sigma m L D` and
found the target-file `rawAlphaFormula` plus unrelated mini-batch variance
lemmas; no reusable SOptLib scalar parameter-choice lemma states this SNCCG
Eq. (7.4.15) arithmetic. -/
private lemma sigma_floor_le_LD_alpha_square_of_formula
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) :
    setup.σ ^ 2 / (setup.m : ℝ) ≤
      (setup.L * setup.barDX * setup.paperAlphaOfWellDefined hDX) ^ 2 := by
  simpa [StochasticNonconvexConditionalGradientSetup.rawAlphaFormula,
    SOptLib.varianceBalancedStepSize, setup.paperAlphaOfWellDefined_eq_formula hDX] using
    SOptLib.varianceBalancedStepSize_variance_floor_le_sq_smoothness_diameter_mul
      setup.N setup.m setup.L setup.barDX setup.σ setup.hN_pos setup.hm_pos
      setup.hL_pos hDX

/-- Active-window objective drops for stochastic Algorithm 7.13 telescope to the
initial optimality gap.

Candidate audit: searched `active objective drop telescope sum Icc sub succ`.
The usable pieces are SOptLib `sum_Icc_sub_succ`, the target-file stochastic
active partition `stochastic_global_index_active_epoch_partition_sum`, and the
new stochastic objective integrability bridge
`f_iterProcessOfWellDefined_integrable`; no SOptLib theorem packages Algorithm
7.13's partial active epoch coordinates. This aligns with Lan Theorem 7.17's
instruction to reuse the Theorem 7.16 proof template and then apply Lemma 7.5. -/
private lemma stochastic_active_objective_drop_telescope
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j =>
            ∫ ω,
              (setup.f
                  (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
                setup.f
                  (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
              ∂setup.P)) ≤
      setup.f setup.x₁ - setup.fStar := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  rw [← stochastic_global_index_active_epoch_partition_sum (setup := setup)
    (A := fun k =>
      ∫ ω,
        (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
          setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P)]
  let A : ℕ → ℝ := fun k =>
    ∫ ω, setup.f (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P
  have hdrop_eq :
      ∀ k ∈ Finset.Icc 1 setup.N,
        (∫ ω,
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P) =
          A k - A (k + 1) := by
    intro k _hk
    have hf_int :
        Integrable
          (fun ω => setup.f (setup.iterProcessOfWellDefined hDX k ω)) setup.P :=
      setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one k
    have hfnext_int :
        Integrable
          (fun ω => setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω))
          setup.P :=
      setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one (k + 1)
    dsimp [A]
    rw [integral_sub hf_int hfnext_int]
  have htel :
      Finset.sum (Finset.Icc 1 setup.N)
          (fun k =>
            ∫ ω,
              (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P) =
        A 1 - A (setup.N + 1) := by
    calc
      Finset.sum (Finset.Icc 1 setup.N)
          (fun k =>
            ∫ ω,
              (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P)
          = Finset.sum (Finset.Icc 1 setup.N) (fun k => A k - A (k + 1)) := by
              refine Finset.sum_congr rfl ?_
              intro k hk
              exact hdrop_eq k hk
      _ = A 1 - A (setup.N + 1) := by
              exact sum_Icc_sub_succ A 1 setup.N setup.hN_pos
  have hA1 : A 1 = setup.f setup.x₁ := by
    dsimp [A, StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined]
    simp [integral_const, probReal_univ]
  have hterminal_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX (setup.N + 1) ω))
        setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one (setup.N + 1)
  have hterminal_lb :
      setup.fStar ≤ A (setup.N + 1) := by
    have hconst_int : Integrable (fun _ω : Ω => setup.fStar) setup.P :=
      integrable_const _
    have hle_int :
        ∫ _ω, setup.fStar ∂setup.P ≤
          ∫ ω,
            setup.f (setup.iterProcessOfWellDefined hDX (setup.N + 1) ω)
            ∂setup.P := by
      refine integral_mono hconst_int hterminal_int ?_
      intro ω
      change setup.fStar ≤
        setup.f (setup.iterProcessOfWellDefined hDX (setup.N + 1) ω)
      have hx :
          setup.iterProcessOfWellDefined hDX (setup.N + 1) ω ∈ setup.X := by
        simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
          setup.processOfWellDefined_mem_of_alpha_le_one
            hDX hα_le_one (setup.N + 1) ω
      exact setup.fStar_lb
        (setup.iterProcessOfWellDefined hDX (setup.N + 1) ω) hx
    simpa [A, integral_const, probReal_univ] using hle_int
  calc
    Finset.sum (Finset.Icc 1 setup.N)
        (fun k =>
          ∫ ω,
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P)
        = A 1 - A (setup.N + 1) := htel
    _ ≤ setup.f setup.x₁ - setup.fStar := by
        rw [hA1]
        linarith

/-- The well-defined stochastic iterate advances by Algorithm 7.13's affine LMO
update after the initialized first step.

Candidate audit: searched `iterProcessOfWellDefined succ iterUpdate`. The
target file had the finite `iterProcess_succ_eq_iterUpdate` and the raw
stochastic analogue, but no well-defined-process theorem; SOptLib recursive
iterate views do not encode Algorithm 7.13's two initial states and epoch
refresh branch. -/
private lemma stochastic_iterProcessOfWellDefined_succ_eq_iterUpdate
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    {k : ℕ} (hk : 1 ≤ k) (ω : Ω) :
    setup.iterProcessOfWellDefined hDX (k + 1) ω =
      setup.iterUpdateOfWellDefined hDX
        (setup.iterProcessOfWellDefined hDX k ω)
        (setup.estimatorProcessOfWellDefined hDX k ω) k := by
  simpa [-SOptLib.varianceReducedConditionalGradientProcess_def,
    StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.estimatorProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined] using
    (SOptLib.iterProcess_succ_eq_update_of_process_recursion
      (fun x G s => ({ x := x, G := G, s := s } : StochasticNonconvexCGState E))
      (fun state : StochasticNonconvexCGState E => state.x)
      (fun state : StochasticNonconvexCGState E => state.G)
      (fun state : StochasticNonconvexCGState E => state.s)
      setup.x₁ setup.m setup.b setup.T setup.N setup.ξ setup.gradF
      (setup.iterUpdateOfWellDefined hDX)
      (by intro x G s; rfl) hk ω)

/-- Generated stochastic within-epoch iterate differences are controlled by
Eq. (7.4.15)'s constant stepsize and the feasible-set diameter.

No SOptLib match: searched `iterate update difference norm alpha diameter` and
the target-file finite analogue `finite_active_iter_diff_sq_le_alpha_sq_barDX_sq`.
The finite helper has the wrong setup type, while SOptLib iterate primitives do
not expose Algorithm 7.13's well-defined stochastic process and active epoch
coordinates. -/
private lemma stochastic_active_iter_diff_sq_le_alpha_sq_barDX_sq
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {s i : ℕ} (hi2 : 2 ≤ i)
    (hiN : setup.globalIndex s i ≤ setup.N) (ω : Ω) :
    ‖setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω -
        setup.iterProcessOfWellDefined hDX (setup.globalIndex s (i - 1)) ω‖ ^ 2 ≤
      setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2 *
        setup.barDX ^ 2 := by
  classical
  let k : ℕ := setup.globalIndex s (i - 1)
  let x : E := setup.iterProcessOfWellDefined hDX k ω
  let G : E := setup.estimatorProcessOfWellDefined hDX k ω
  let y : E := setup.linearMinimizer G
  have hk_pos : 1 ≤ k := by
    dsimp [k]
    unfold StochasticNonconvexConditionalGradientSetup.globalIndex
    omega
  have hkN : k ≤ setup.N := by
    dsimp [k] at *
    exact setup.globalIndex_prefix_le_of_le
      (s := s) (i := i - 1) (t := i) (by omega) hiN
  have hsucc : setup.globalIndex s i = k + 1 := by
    dsimp [k]
    unfold StochasticNonconvexConditionalGradientSetup.globalIndex
    omega
  have hiter :
      setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω =
        setup.iterUpdateOfWellDefined hDX x G k := by
    rw [hsucc]
    exact stochastic_iterProcessOfWellDefined_succ_eq_iterUpdate
      (setup := setup) hDX hk_pos ω
  have hdiff :
      setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω -
          setup.iterProcessOfWellDefined hDX (setup.globalIndex s (i - 1)) ω =
        setup.αOfWellDefined hDX k • (y - x) := by
    rw [hiter]
    dsimp [x, G, y, k]
    simp only [StochasticNonconvexConditionalGradientSetup.iterUpdateOfWellDefined]
    module
  have hx : x ∈ setup.X := by
    dsimp [x]
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hDX hα_le_one k ω
  have hy : y ∈ setup.X := by
    dsimp [y]
    exact setup.linearMinimizer_mem G
  have hdiam : ‖y - x‖ ≤ setup.barDX :=
    setup.barDX_bound y x hy hx
  have hα_nonneg : 0 ≤ setup.αOfWellDefined hDX k :=
    le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  have hbarDX_nonneg : 0 ≤ setup.barDX := by
    exact norm_nonneg _
  have hnorm :
      ‖setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω -
          setup.iterProcessOfWellDefined hDX (setup.globalIndex s (i - 1)) ω‖ ≤
        setup.αOfWellDefined hDX k * setup.barDX := by
    calc
      ‖setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω -
          setup.iterProcessOfWellDefined hDX (setup.globalIndex s (i - 1)) ω‖
          = ‖setup.αOfWellDefined hDX k • (y - x)‖ := by rw [hdiff]
      _ = setup.αOfWellDefined hDX k * ‖y - x‖ := by
          rw [norm_smul, Real.norm_of_nonneg hα_nonneg]
      _ ≤ setup.αOfWellDefined hDX k * setup.barDX :=
          mul_le_mul_of_nonneg_left hdiam hα_nonneg
  have hright_nonneg : 0 ≤ setup.αOfWellDefined hDX k * setup.barDX :=
    mul_nonneg hα_nonneg hbarDX_nonneg
  have hleft_nonneg :
      0 ≤
        ‖setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω -
          setup.iterProcessOfWellDefined hDX (setup.globalIndex s (i - 1)) ω‖ :=
    norm_nonneg _
  dsimp [k] at hnorm ⊢
  nlinarith

/-- Integrated stochastic epoch-difference budget obtained from the generated
update and the diameter bound.

No SOptLib match: searched `epoch difference sum alpha square diameter bound`
and checked the finite helper `finite_active_epochDiff_integral_le_scalar_sum`.
The finite helper has the wrong setup type; Mathlib supplies
`integral_finset_sum`, while Algorithm 7.13's well-defined process and active
epoch domain are local. -/
private lemma stochastic_active_epochDiff_integral_le_scalar_sum
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {s j : ℕ} (hj : j ∈ stochasticActiveEpochSteps setup s) :
    ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P ≤
      setup.barDX ^ 2 *
        Finset.sum (Finset.Icc 2 j)
          (fun i => setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2) := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  have hjN : setup.globalIndex s j ≤ setup.N :=
    stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
  exact integral_finset_epoch_difference_le_sum_bounds
    (idxs := Finset.Icc 2 j)
    (epoch_diff := fun i ω =>
      ‖setup.iterProcessOfWellDefined hDX (setup.globalIndex s i) ω -
        setup.iterProcessOfWellDefined hDX (setup.globalIndex s (i - 1)) ω‖ ^ 2)
    (bound := fun i =>
      setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2)
    (D := setup.barDX)
    (hterm_int := by
      intro i _hi
      exact setup.iterProcessOfWellDefined_diff_sq_integrable
        hDX hα_le_one (setup.globalIndex s i) (setup.globalIndex s (i - 1)))
    (hpoint := by
      intro i hi ω
      have hi2 : 2 ≤ i := (Finset.mem_Icc.mp hi).1
      have hi_le : i ≤ j := (Finset.mem_Icc.mp hi).2
      have hiN : setup.globalIndex s i ≤ setup.N :=
        setup.globalIndex_prefix_le_of_le (s := s) (i := i) (t := j) hi_le hjN
      exact stochastic_active_iter_diff_sq_le_alpha_sq_barDX_sq
        (setup := setup) hDX hα_le_one (s := s) (i := i) hi2 hiN ω)

/-- One stochastic epoch of the triangular predecessor-alpha budget is bounded
by `T` copies of the active generated stepsizes.

No SOptLib match: searched `active epoch difference alpha triangular budget
finite sum` and checked the finite helper
`one_epoch_triangular_budget_le_epoch_sum`. Mathlib supplies the finite image,
subset, and card estimates; the Algorithm 7.13 active epoch map is local. -/
private lemma stochastic_one_epoch_triangular_budget_le_epoch_sum
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined) (s : ℕ) :
    Finset.sum (stochasticActiveEpochSteps setup s)
        (fun j => Finset.sum (Finset.Icc 2 j)
          (fun i =>
            setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2)) ≤
      (setup.T : ℝ) *
        Finset.sum (stochasticActiveEpochSteps setup s)
          (fun r => setup.αOfWellDefined hDX (setup.globalIndex s r) ^ 2) := by
  classical
  let epochSq : ℝ :=
    Finset.sum (stochasticActiveEpochSteps setup s)
      (fun r => setup.αOfWellDefined hDX (setup.globalIndex s r) ^ 2)
  have hinner :
      ∀ j ∈ stochasticActiveEpochSteps setup s,
        Finset.sum (Finset.Icc 2 j)
            (fun i =>
              setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2) ≤
          epochSq := by
    intro j hj
    let predSet : Finset ℕ := (Finset.Icc 2 j).image (fun i : ℕ => i - 1)
    have hinj : Set.InjOn (fun i : ℕ => i - 1) (Finset.Icc 2 j) := by
      intro a ha b hb hab
      have ha2 : 2 ≤ a := (Finset.mem_Icc.mp ha).1
      have hb2 : 2 ≤ b := (Finset.mem_Icc.mp hb).1
      calc
        a = (a - 1) + 1 := (Nat.sub_add_cancel (by omega : 1 ≤ a)).symm
        _ = (b - 1) + 1 := by
          simpa [Nat.succ_eq_add_one] using congrArg Nat.succ hab
        _ = b := Nat.sub_add_cancel (by omega : 1 ≤ b)
    have hsum_image :
        Finset.sum predSet
            (fun r => setup.αOfWellDefined hDX (setup.globalIndex s r) ^ 2) =
          Finset.sum (Finset.Icc 2 j)
            (fun i =>
              setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2) := by
      dsimp [predSet]
      rw [Finset.sum_image hinj]
    have hsubset : predSet ⊆ stochasticActiveEpochSteps setup s := by
      intro r hr
      rcases Finset.mem_image.mp hr with ⟨i, hi, rfl⟩
      have hi_bounds := Finset.mem_Icc.mp hi
      have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
      have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
      rw [stochastic_mem_activeEpochSteps]
      constructor
      · exact Finset.mem_Icc.mpr ⟨by omega, by omega⟩
      · exact setup.globalIndex_prefix_le_of_le
          (s := s) (i := i - 1) (t := j) (by omega) hjN
    have himage_le :
        Finset.sum predSet
            (fun r => setup.αOfWellDefined hDX (setup.globalIndex s r) ^ 2) ≤
          epochSq := by
      dsimp [epochSq]
      exact Finset.sum_le_sum_of_subset_of_nonneg hsubset
        (fun r _hr_active _hr_not_pred =>
          sq_nonneg (setup.αOfWellDefined hDX (setup.globalIndex s r)))
    exact hsum_image ▸ himage_le
  have hsum_card :
      Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j => Finset.sum (Finset.Icc 2 j)
            (fun i =>
              setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2)) ≤
        ((stochasticActiveEpochSteps setup s).card : ℝ) * epochSq := by
    have h :=
      Finset.sum_le_card_nsmul (stochasticActiveEpochSteps setup s)
        (fun j => Finset.sum (Finset.Icc 2 j)
          (fun i =>
            setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2))
        epochSq hinner
    simpa [nsmul_eq_mul] using h
  have hcard_le_T_nat : (stochasticActiveEpochSteps setup s).card ≤ setup.T := by
    have hsubset_epoch :
        stochasticActiveEpochSteps setup s ⊆ Finset.Icc 1 setup.T := by
      intro j hj
      exact ((stochastic_mem_activeEpochSteps setup).mp hj).1
    calc
      (stochasticActiveEpochSteps setup s).card ≤ (Finset.Icc 1 setup.T).card :=
        Finset.card_le_card hsubset_epoch
      _ = setup.T := by
        rw [Nat.card_Icc]
        omega
  have hcard_le_T : ((stochasticActiveEpochSteps setup s).card : ℝ) ≤ setup.T := by
    exact_mod_cast hcard_le_T_nat
  have hepoch_nonneg : 0 ≤ epochSq := by
    dsimp [epochSq]
    exact Finset.sum_nonneg
      (fun r _hr => sq_nonneg (setup.αOfWellDefined hDX (setup.globalIndex s r)))
  exact le_trans hsum_card
    (mul_le_mul_of_nonneg_right hcard_le_T hepoch_nonneg)

/-- The stochastic triangular epoch-difference alpha-square budget reindexes
back to the global output window.

No SOptLib match: searched `active epoch scalar budget stochastic variance
floor` and checked the finite `active_triangular_epoch_difference_alpha_budget`.
The finite helper has the wrong setup type; the matching stochastic ingredients
are the local active partition and one-epoch triangular bound. -/
private lemma stochastic_active_triangular_epoch_difference_alpha_budget
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j => Finset.sum (Finset.Icc 2 j)
            (fun i =>
              setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2))) ≤
      (setup.b : ℝ) *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k => setup.αOfWellDefined hDX k ^ 2) := by
  exact active_triangular_prefix_budget_le_global_sq_sum
    (outer := Finset.Icc 0 setup.S)
    (inner := fun s => stochasticActiveEpochSteps setup s)
    (globalIndex := fun s r => setup.globalIndex s r)
    (alpha := fun k => setup.αOfWellDefined hDX k)
    (T := (setup.T : ℝ))
    (b := (setup.b : ℝ))
    (N := setup.N)
    (hlocal := by
      intro s _hs
      exact stochastic_one_epoch_triangular_budget_le_epoch_sum
        (setup := setup) hDX s)
    (hactive_eq_global := by
      exact (stochastic_global_index_active_epoch_partition_sum
        (setup := setup) (A := fun k => setup.αOfWellDefined hDX k ^ 2)).symm)
    (hT_le_b := by
      exact_mod_cast setup.hb_ge_T)

/-- The epoch-difference component produced by Lemma 7.5 contributes the
`1/2 ∑ α_k²` part of the stochastic Theorem 7.17 scalar budget.

No SOptLib match: searched `active epoch scalar budget stochastic variance
floor` and checked the finite `active_delta_square_sum_le_half_alpha_square_budget`.
The finite lemma has the wrong setup type and bundles a no-floor estimator
bound; this stochastic helper uses the local epochDiff integral and triangular
counting lemmas. -/
private lemma stochastic_active_epochDiff_term_le_half_alpha_square_budget
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j =>
            (setup.L / (2 * setup.b)) *
              ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P)) ≤
      setup.L * setup.barDX ^ 2 *
        (1 / 2 *
          Finset.sum (Finset.Icc 1 setup.N)
            (fun k => setup.αOfWellDefined hDX k ^ 2)) := by
  classical
  exact active_epoch_difference_scaled_sum_le_half_global_alpha_sq
    (outer := Finset.Icc 0 setup.S)
    (inner := fun s => stochasticActiveEpochSteps setup s)
    (epoch := fun s j =>
      ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P)
    (localBudget := fun s j =>
      Finset.sum (Finset.Icc 2 j)
        (fun i => setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2))
    (globalSq :=
      Finset.sum (Finset.Icc 1 setup.N)
        (fun k => setup.αOfWellDefined hDX k ^ 2))
    (L := setup.L)
    (b := (setup.b : ℝ))
    (D := setup.barDX)
    (hb_pos := by
      exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one
        (le_trans setup.hT_pos setup.hb_ge_T)))
    (hL_nonneg := le_of_lt setup.hL_pos)
    (hpoint := by
      intro s _hs j hj
      exact stochastic_active_epochDiff_integral_le_scalar_sum
        (setup := setup) hDX hα_le_one (s := s) (j := j) hj)
    (hbudget :=
      stochastic_active_triangular_epoch_difference_alpha_budget
        (setup := setup) hDX)

/-- The stochastic `α D̄_X E‖δ‖` contribution in Lan Theorem 7.17 splits into
the finite active epoch-maximum penalty plus the explicit Lemma 7.5 variance
floor contribution.

No SOptLib match: searched `stochastic active delta l1 norm integral sigma
floor alpha square`, `sq integral le integral sq probability`, and checked the
finite helpers `active_delta_l1_pointwise_le_epoch_difference_max` and
`active_delta_l1_sum_le_epoch_difference_penalty`. Those finite helpers have
the wrong setup type and no Lemma 7.5 variance floor; the stochastic target-file
helpers `sigma_floor_le_LD_alpha_square_of_formula` and
`stochastic_active_epochDiff_integral_le_scalar_sum` supply only separate scalar
ingredients. The source-faithful combination keeps the `σ²/m` floor visible as
`D̄_X σ / sqrt(m)` times the active stepsize mass. -/
private lemma stochastic_active_delta_l1_pointwise_le_epoch_penalty_with_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    (hlemma75_active :
      ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
        ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
            ∂setup.P ≤
          setup.L ^ 2 / setup.b *
              ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
            setup.σ ^ 2 / setup.m)
    {s j : ℕ} (hs : s ∈ Finset.Icc 0 setup.S)
    (hj : j ∈ stochasticActiveEpochSteps setup s) :
    ∫ ω,
        ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
        ∂setup.P ≤
      setup.L * setup.barDX * setup.paperMaxInnerAlphaOfWellDefined hDX hT s +
        setup.σ / Real.sqrt (setup.m : ℝ) := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  let Z : Ω → ℝ := fun ω =>
    ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
  let A : ℝ := setup.L * setup.barDX * setup.paperAlphaOfWellDefined hDX
  let C : ℝ := setup.σ / Real.sqrt (setup.m : ℝ)
  have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
  have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
  have hZ_sq_int : Integrable (fun ω => Z ω ^ 2) setup.P := by
    dsimp [Z]
    exact epochwise_delta_sq_integrable
      setup hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
  have hZ_nonneg : ∀ᵐ ω ∂setup.P, 0 ≤ Z ω :=
    Filter.Eventually.of_forall (fun ω => norm_nonneg _)
  have hα_const :
      ∀ k, setup.αOfWellDefined hDX k = setup.paperAlphaOfWellDefined hDX := by
    intro k
    exact setup.alphaOfWellDefined_eq_paperAlphaOfWellDefined hDX k
  have hmax_const :
      setup.paperMaxInnerAlphaOfWellDefined hDX hT s =
        setup.paperAlphaOfWellDefined hDX :=
    setup.paperMaxInnerAlphaOfWellDefined_eq_paperAlphaOfWellDefined hDX hT s
  have hepoch :
      ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P ≤
        setup.barDX ^ 2 *
          Finset.sum (Finset.Icc 2 j)
            (fun i =>
              setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2) :=
    stochastic_active_epochDiff_integral_le_scalar_sum
      (setup := setup) hDX hα_le_one (s := s) (j := j) hj
  have hcard_le_b_nat : (Finset.Icc 2 j).card ≤ setup.b := by
    have hj_le_T : j ≤ setup.T := hj_epoch.2
    have hT_le_b : setup.T ≤ setup.b := setup.hb_ge_T
    rw [Nat.card_Icc]
    omega
  have hcard_le_b : ((Finset.Icc 2 j).card : ℝ) ≤ setup.b := by
    exact_mod_cast hcard_le_b_nat
  have hpred_sum :
      Finset.sum (Finset.Icc 2 j)
          (fun i =>
            setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2) ≤
        (setup.b : ℝ) * setup.paperAlphaOfWellDefined hDX ^ 2 := by
    calc
      Finset.sum (Finset.Icc 2 j)
          (fun i =>
            setup.αOfWellDefined hDX (setup.globalIndex s (i - 1)) ^ 2)
          =
        ((Finset.Icc 2 j).card : ℝ) *
          setup.paperAlphaOfWellDefined hDX ^ 2 := by
            simp [hα_const, Finset.sum_const, nsmul_eq_mul]
      _ ≤ (setup.b : ℝ) * setup.paperAlphaOfWellDefined hDX ^ 2 :=
            mul_le_mul_of_nonneg_right hcard_le_b
              (sq_nonneg (setup.paperAlphaOfWellDefined hDX))
  have hepoch_max :
      ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P ≤
        setup.barDX ^ 2 *
          ((setup.b : ℝ) * setup.paperAlphaOfWellDefined hDX ^ 2) :=
    le_trans hepoch
      (mul_le_mul_of_nonneg_left hpred_sum (sq_nonneg setup.barDX))
  have hb_pos_real : 0 < (setup.b : ℝ) := by
    exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one
      (le_trans setup.hT_pos setup.hb_ge_T))
  have hb_ne : (setup.b : ℝ) ≠ 0 := ne_of_gt hb_pos_real
  have hcoef_nonneg : 0 ≤ setup.L ^ 2 / (setup.b : ℝ) := by
    exact div_nonneg (sq_nonneg setup.L) (le_of_lt hb_pos_real)
  have hdelta_square_bound :
      ∫ ω, Z ω ^ 2 ∂setup.P ≤ A ^ 2 + setup.σ ^ 2 / (setup.m : ℝ) := by
    calc
      ∫ ω, Z ω ^ 2 ∂setup.P
          ≤ setup.L ^ 2 / setup.b *
              ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
            setup.σ ^ 2 / setup.m := by
            dsimp [Z]
            exact hlemma75_active s hs j hj
      _ ≤ setup.L ^ 2 / setup.b *
              (setup.barDX ^ 2 *
                ((setup.b : ℝ) * setup.paperAlphaOfWellDefined hDX ^ 2)) +
            setup.σ ^ 2 / setup.m := by
            exact add_le_add
              (mul_le_mul_of_nonneg_left hepoch_max hcoef_nonneg) le_rfl
      _ = A ^ 2 + setup.σ ^ 2 / (setup.m : ℝ) := by
            dsimp [A]
            field_simp [hb_ne]
  have hm_pos_real : 0 < (setup.m : ℝ) := by
    exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos)
  have hsqrt_m_pos : 0 < Real.sqrt (setup.m : ℝ) :=
    Real.sqrt_pos.2 hm_pos_real
  have hsqrt_m_ne : Real.sqrt (setup.m : ℝ) ≠ 0 := ne_of_gt hsqrt_m_pos
  have hfloor_sq :
      setup.σ ^ 2 / (setup.m : ℝ) = C ^ 2 := by
    dsimp [C]
    field_simp [hsqrt_m_ne]
    rw [Real.sq_sqrt (le_of_lt hm_pos_real)]
  have hdelta_square_bound_floor :
      ∫ ω, Z ω ^ 2 ∂setup.P ≤ A ^ 2 + C ^ 2 := by
    calc
      ∫ ω, Z ω ^ 2 ∂setup.P
          ≤ A ^ 2 + setup.σ ^ 2 / (setup.m : ℝ) := hdelta_square_bound
      _ = A ^ 2 + C ^ 2 := by rw [hfloor_sq]
  have hA_nonneg : 0 ≤ A := by
    dsimp [A]
    exact mul_nonneg
      (mul_nonneg (le_of_lt setup.hL_pos) (le_of_lt hDX))
      (le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX 1))
  have hC_nonneg : 0 ≤ C := by
    dsimp [C]
    exact div_nonneg setup.hσ_nonneg (le_of_lt hsqrt_m_pos)
  have hmain : ∫ ω, Z ω ∂setup.P ≤ A + C :=
    integral_nonneg_le_add_of_integral_sq_le_add_sq
      hZ_sq_int hZ_nonneg hA_nonneg hC_nonneg hdelta_square_bound_floor
  simpa [Z, A, C, hmax_const] using hmain

private lemma stochastic_active_delta_l1_sum_le_epoch_penalty_with_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    (hlemma75_active :
      ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
        ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
            ∂setup.P ≤
          setup.L ^ 2 / setup.b *
              ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
            setup.σ ^ 2 / setup.m) :
    Finset.sum (Finset.Icc 0 setup.S)
        (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
          (fun j =>
            setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
              ∫ ω,
                ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
                ∂setup.P)) ≤
      setup.L * setup.barDX ^ 2 *
        Finset.sum (Finset.Icc 0 setup.S)
          (fun s =>
            Finset.sum (Finset.Icc 1 setup.T)
              (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
            setup.paperMaxInnerAlphaOfWellDefined hDX hT s) +
        (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
              (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j))) := by
  classical
  have hα_nonneg :
      ∀ k, 0 ≤ setup.αOfWellDefined hDX k := by
    intro k
    exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  have hDX_nonneg : 0 ≤ setup.barDX := le_of_lt hDX
  exact active_l1_error_sum_le_epoch_penalty_add_floor_mass
    (outer := Finset.Icc 0 setup.S)
    (active := stochasticActiveEpochSteps setup)
    (full := fun _s => Finset.Icc 1 setup.T)
    (alpha := fun s j => setup.αOfWellDefined hDX (setup.globalIndex s j))
    (l1 := fun s j =>
      ∫ ω, ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ∂setup.P)
    (M := fun s => setup.paperMaxInnerAlphaOfWellDefined hDX hT s)
    (L := setup.L) (D := setup.barDX)
    (floorCoeff := setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ))
    (hpoint := by
      intro s hs j hj
      have hpoint :=
        stochastic_active_delta_l1_pointwise_le_epoch_penalty_with_floor
          (setup := setup) hDX hT hα_le_one hlemma75_active hs hj
      have hcoef_nonneg :
          0 ≤ setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX :=
        mul_nonneg (hα_nonneg _) hDX_nonneg
      have hmul := mul_le_mul_of_nonneg_left hpoint hcoef_nonneg
      calc
        setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
            (∫ ω,
              ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
              ∂setup.P)
            ≤
          setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
            (setup.L * setup.barDX *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s +
              setup.σ / Real.sqrt (setup.m : ℝ)) := hmul
        _ =
          setup.L * setup.barDX ^ 2 *
              (setup.αOfWellDefined hDX (setup.globalIndex s j) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s) +
            setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) *
              setup.αOfWellDefined hDX (setup.globalIndex s j) := by
            ring)
    (hsubset := by
      intro s _hs j hj
      exact ((stochastic_mem_activeEpochSteps setup).mp hj).1)
    (halpha_missing_nonneg := by
      intro s _hs j _hj_full _hj_not_active
      exact hα_nonneg (setup.globalIndex s j))
    (hM_nonneg := by
      intro s _hs
      change 0 ≤ setup.paperMaxInnerAlphaOfWellDefined hDX hT s
      rw [setup.paperMaxInnerAlphaOfWellDefined_eq_paperAlphaOfWellDefined hDX hT s]
      exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX 1))
    (hpenalty_nonneg := mul_nonneg (le_of_lt setup.hL_pos) (sq_nonneg setup.barDX))

/-- Scalar sanity check for the failed no-floor absorption route.

With one unit of epoch contribution and one unit of variance floor, the shape
`sqrt (epoch + floor) ≤ alphaScale` already fails at `alphaScale = 1`. This is
the route-level counterexample behind keeping the Lemma 7.5 floor explicit in
`stochastic_active_delta_l1_sum_le_epoch_penalty_with_floor` instead of asking
FILL to re-prove the old finite-penalty-only helper. -/
private lemma stochastic_l1_floor_absorption_scalar_counterexample :
    ¬ Real.sqrt ((((2 : ℝ) - 1) / 1) * (1 : ℝ) ^ 2 + 1 * (1 : ℝ) ^ 2) ≤
      (1 : ℝ) := by
  norm_num [Real.sqrt_le_one]

/-- Nondegenerate scalar form of the missing Theorem 7.17 floor-erasure step.

If the diameter, stochastic variance level, stepsize mass, and minibatch size
are all positive, the normalized L1 floor contribution cannot be nonpositive.
This is the formal obstruction to using a nonpositive floor-absorption premise
as a source-facing theorem hypothesis. -/
private lemma stochastic_l1_floor_absorption_nonpositive_false
    {D σ m αsum : ℝ} (hD : 0 < D) (hσ : 0 < σ) (hm : 0 < m)
    (hαsum : 0 < αsum) :
    ¬ (D * σ / Real.sqrt m) * αsum ≤ 0 := by
  exact not_mul_div_sqrt_mul_le_zero_of_pos hD hσ hm hαsum

/-- Setup-level version of the nondegenerate floor-erasure obstruction.

For any realized stochastic setup with positive diameter and positive `sigma`,
the normalized L1 floor mass appearing in the corrected Theorem 7.17 route is
strictly positive, so it cannot be erased by assuming it is nonpositive. -/
private lemma stochastic_l1_floor_absorption_setup_nonpositive_false
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hσ : 0 < setup.σ) :
    ¬ (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k => setup.αOfWellDefined hDX k) ≤ 0 := by
  have hm : 0 < (setup.m : ℝ) := by
    exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos)
  have hαsum :
      0 <
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k => setup.αOfWellDefined hDX k) := by
    simpa [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
      using setup.alphaSum_pos_of_nonzeroDiameter hDX
  exact stochastic_l1_floor_absorption_nonpositive_false hDX hσ hm hαsum

/-- The theorem-level RHS obstruction behind the statement-corrected Theorem
7.17 route: in every nondegenerate stochastic regime, the proved RHS with the
explicit normalized L1 floor is strictly larger than the printed RHS.

This does not introduce a new assumption. It records the formal migration debt:
recovering the printed theorem from the corrected proof route would require
erasing the positive term `D̄_X σ / sqrt(m)`, not merely simplifying scalars. -/
private lemma theorem717_l1_floor_rhs_not_le_printed_rhs
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hσ : 0 < setup.σ) :
    ¬
      (let printed : ℝ :=
        (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
          setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 /
            (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX);
       printed + setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) ≤ printed) := by
  dsimp
  intro hle
  have hm : 0 < (setup.m : ℝ) := by
    exact_mod_cast (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos)
  have hfloor_pos :
      0 < setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
    exact div_pos (mul_pos hDX hσ) (Real.sqrt_pos.2 hm)
  linarith

/-- Scalar Step-0 witness for retiring the exact printed Theorem 7.17 bound.

The previous obstruction only said the corrected RHS is strictly larger than
the printed RHS. This statement includes a possible left-hand side: a quantity
can satisfy the corrected inequality while violating the printed one. It is a
route-local scalar countermodel for monotone-erasing the explicit L1 floor, not
a new algorithmic assumption. -/
private theorem theorem717_printed_bound_scalar_countermodel :
    ∃ weightedGap printed corrected : ℝ,
      weightedGap ≤ corrected ∧ ¬ weightedGap ≤ printed ∧
        corrected = printed + (1 : ℝ) := by
  refine ⟨1, 0, 1, ?_, ?_, ?_⟩ <;> norm_num

/-- Scalar obstruction for treating the corrected Corollary 7.12 leaf as a
pure algebraic consequence of the `_l1_floor` Theorem 7.17 route.

Even with the displayed Eq. (7.4.15) alpha formula, `α ≤ 1`, positive
parameters, and zero variance, the normalized theorem-level RHS can exceed the
coefficient-5 corollary RHS when the initial objective gap is large. A proof of
the corollary therefore needs an additional source-backed scalar premise, a
different source route, or formal statement correction; FILL should not attack
the current leaf as routine `ring_nf` cleanup. -/
private theorem corollary712_l1_floor_scalar_countermodel :
    ∃ fGap L D α σ m N : ℝ,
      0 < fGap ∧ 0 < L ∧ 0 < D ∧ 0 ≤ σ ∧ 0 < m ∧ 0 < N ∧
      α ≤ 1 ∧
      α = Real.sqrt ((1 / N + σ ^ 2 / (L * m)) / (L * D ^ 2)) ∧
      ¬
        fGap / (N * α) + (7 / 2) * L * D ^ 2 * α +
            σ ^ 2 / (2 * L * m * α) + D * σ / Real.sqrt m ≤
          fGap / Real.sqrt N + 7 * L * D ^ 2 / (2 * Real.sqrt N) +
            5 * σ * D / Real.sqrt m := by
  refine ⟨20, 4, 1, 1 / 4, 0, 1, 4, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  ·
    have hsqrt : Real.sqrt ((1 : ℝ) / 16) = 1 / 4 := by
      have hsq : (Real.sqrt ((1 : ℝ) / 16)) ^ 2 = (1 / 4 : ℝ) ^ 2 := by
        rw [Real.sq_sqrt]
        · norm_num
        · norm_num
      have hcases := sq_eq_sq_iff_eq_or_eq_neg.mp hsq
      rcases hcases with h | h
      · exact h
      · have hnonneg : 0 ≤ Real.sqrt ((1 : ℝ) / 16) := Real.sqrt_nonneg _
        nlinarith
    norm_num [hsqrt]
  · norm_num

/-- Feasible-segment smooth quadratic upper bound for the stochastic objective.

Candidate audit: searched `smooth quadratic upper bound stochastic setup f
gradf`; SOptLib `Convex.carrier_smooth_quadratic_upper_bound` was considered
but requires a carrier derivative package not exposed by this setup. This
local bridge copies the finite proof from `gradf_hasGradientAt` and
`gradf_smooth`, matching Lan Eq. (7.4.5). -/
private lemma stochastic_smooth_quadratic_upper_bound_of_hasGradientAt
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {x y : E} (hx : x ∈ setup.X) (hy : y ∈ setup.X) :
    setup.f y ≤ setup.f x + ⟪setup.gradf x, y - x⟫_ℝ +
      (setup.L / 2) * ‖y - x‖ ^ 2 := by
  let d : E := y - x
  let s : Set ℝ := Set.Icc (0 : ℝ) 1
  let line : ℝ → E := fun t => AffineMap.lineMap x y t
  let Fseg : ℝ → ℝ := fun t => setup.f (line t)
  have hF0 : Fseg 0 = setup.f x := by
    simp [Fseg, line]
  have hF1 : Fseg 1 = setup.f y := by
    simp [Fseg, line]
  have hline_mem : ∀ t ∈ s, line t ∈ setup.X := by
    intro t ht
    exact setup.hX_convex.lineMap_mem hx hy ht
  have hderiv : ∀ (t : ℝ) (ht : t ∈ s),
      HasDerivWithinAt Fseg ⟪setup.gradf (line t), d⟫_ℝ s t := by
    intro t ht
    have hline_deriv : HasDerivWithinAt line d s t := by
      simpa [line, d] using
        (AffineMap.hasDerivWithinAt_lineMap (a := x) (b := y)
          (s := s) (x := t))
    have hfderiv :
        HasFDerivAt setup.f
          (InnerProductSpace.toDual ℝ E (setup.gradf (line t))) (line t) :=
      (setup.gradf_hasGradientAt (line t) (hline_mem t ht)).hasFDerivAt
    have hfseg : HasDerivWithinAt Fseg
        ((InnerProductSpace.toDual ℝ E (setup.gradf (line t))) d) s t := by
      simpa [Fseg, Function.comp_def] using
        hfderiv.comp_hasDerivWithinAt t hline_deriv
    simpa using hfseg
  have hbound : ∀ (t : ℝ) (ht : t ∈ s),
      ⟪setup.gradf (line t), d⟫_ℝ ≤
        ⟪setup.gradf x, d⟫_ℝ + setup.L * t * ‖d‖ ^ 2 := by
    intro t ht
    have ht_nonneg : 0 ≤ t := ht.1
    have hseg_sub : line t - x = t • d := by
      simp [line, d, AffineMap.lineMap_apply_module']
    have hnorm_line : ‖line t - x‖ = t * ‖d‖ := by
      rw [hseg_sub, norm_smul, Real.norm_of_nonneg ht_nonneg]
    have hlip : ‖setup.gradf (line t) - setup.gradf x‖ ≤ setup.L * (t * ‖d‖) := by
      calc
        ‖setup.gradf (line t) - setup.gradf x‖
            ≤ setup.L * ‖line t - x‖ :=
          setup.gradf_smooth (line t) x (hline_mem t ht) hx
        _ = setup.L * (t * ‖d‖) := by rw [hnorm_line]
    have hinner_diff :
        ⟪setup.gradf (line t), d⟫_ℝ - ⟪setup.gradf x, d⟫_ℝ =
          ⟪setup.gradf (line t) - setup.gradf x, d⟫_ℝ := by
      rw [inner_sub_left]
    have hcs :
        ⟪setup.gradf (line t) - setup.gradf x, d⟫_ℝ ≤
          ‖setup.gradf (line t) - setup.gradf x‖ * ‖d‖ := by
      calc
        ⟪setup.gradf (line t) - setup.gradf x, d⟫_ℝ
            ≤ |⟪setup.gradf (line t) - setup.gradf x, d⟫_ℝ| := le_abs_self _
        _ ≤ ‖setup.gradf (line t) - setup.gradf x‖ * ‖d‖ :=
          abs_real_inner_le_norm (setup.gradf (line t) - setup.gradf x) d
    have hmul :
        ‖setup.gradf (line t) - setup.gradf x‖ * ‖d‖ ≤
          setup.L * t * ‖d‖ ^ 2 := by
      have hmul' :
          ‖setup.gradf (line t) - setup.gradf x‖ * ‖d‖ ≤
            (setup.L * (t * ‖d‖)) * ‖d‖ :=
        mul_le_mul_of_nonneg_right hlip (norm_nonneg d)
      calc
        ‖setup.gradf (line t) - setup.gradf x‖ * ‖d‖
            ≤ (setup.L * (t * ‖d‖)) * ‖d‖ := hmul'
        _ = setup.L * t * ‖d‖ ^ 2 := by ring
    have hdiff_le :
        ⟪setup.gradf (line t), d⟫_ℝ - ⟪setup.gradf x, d⟫_ℝ ≤
          setup.L * t * ‖d‖ ^ 2 := by
      calc
        ⟪setup.gradf (line t), d⟫_ℝ - ⟪setup.gradf x, d⟫_ℝ
            = ⟪setup.gradf (line t) - setup.gradf x, d⟫_ℝ := hinner_diff
        _ ≤ ‖setup.gradf (line t) - setup.gradf x‖ * ‖d‖ := hcs
        _ ≤ setup.L * t * ‖d‖ ^ 2 := hmul
    linarith
  let phi : ℝ → ℝ := fun t =>
    if ht : t ∈ s then ⟪setup.gradf (line t), d⟫_ℝ else 0
  have hderiv_phi : ∀ (t : ℝ) (ht : t ∈ s),
      HasDerivWithinAt Fseg (phi t) s t := by
    intro t ht
    rw [show phi t = ⟪setup.gradf (line t), d⟫_ℝ by
      dsimp [phi]
      rw [if_pos ht]]
    exact hderiv t ht
  have hbound_phi : ∀ (t : ℝ) (ht : t ∈ s),
      phi t ≤ ⟪setup.gradf x, d⟫_ℝ + (setup.L * ‖d‖ ^ 2) * t := by
    intro t ht
    have hb := hbound t ht
    rw [show phi t = ⟪setup.gradf (line t), d⟫_ℝ by
      dsimp [phi]
      rw [if_pos ht]]
    simpa [mul_assoc, mul_comm, mul_left_comm] using hb
  have hscalar :
      Fseg 1 ≤ Fseg 0 + ⟪setup.gradf x, d⟫_ℝ + (setup.L * ‖d‖ ^ 2) / 2 := by
    exact le_value_add_of_hasDerivWithinAt_le_affine_on_Icc Fseg phi
      ⟪setup.gradf x, d⟫_ℝ (setup.L * ‖d‖ ^ 2) hderiv_phi hbound_phi
  rw [hF0, hF1] at hscalar
  change setup.f y ≤ setup.f x + ⟪setup.gradf x, y - x⟫_ℝ +
    (setup.L / 2) * ‖y - x‖ ^ 2
  rw [show d = y - x by rfl] at hscalar
  calc
    setup.f y ≤ setup.f x + ⟪setup.gradf x, y - x⟫_ℝ +
        (setup.L * ‖y - x‖ ^ 2) / 2 := hscalar
    _ = setup.f x + ⟪setup.gradf x, y - x⟫_ℝ +
        (setup.L / 2) * ‖y - x‖ ^ 2 := by ring

/-- Smooth descent premise specialized to Algorithm 7.13's well-defined affine
LMO update.

Candidate audit: searched `smooth quadratic upper bound stochastic setup f
gradf`, checked SOptLib `Convex.carrier_smooth_quadratic_upper_bound`, and used
the local stochastic quadratic bridge above. This is the stochastic Eq. (7.4.5)
premise before Wolfe-gap algebra. -/
private lemma stochastic_one_step_smooth_descent_premise
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {k : ℕ} {x G : E} (hx : x ∈ setup.X) :
    setup.f (setup.iterUpdateOfWellDefined hDX x G k) ≤
      setup.f x +
        setup.αOfWellDefined hDX k *
          ⟪G, setup.linearMinimizer G - x⟫_ℝ +
        (1 / (2 * setup.L)) * ‖G - setup.gradf x‖ ^ 2 +
        setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
  classical
  let a : ℝ := setup.αOfWellDefined hDX k
  let y : E := setup.linearMinimizer G
  let upd : E := setup.iterUpdateOfWellDefined hDX x G k
  let e : ℝ := ‖setup.gradf x - G‖
  let D : ℝ := setup.barDX
  have ha_nonneg : 0 ≤ a := by
    dsimp [a]
    exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  have hD_nonneg : 0 ≤ D := by
    dsimp [D]
    exact norm_nonneg _
  have hy : y ∈ setup.X := by
    dsimp [y]
    exact setup.linearMinimizer_mem G
  have hupd_mem : upd ∈ setup.X := by
    dsimp [upd]
    exact setup.iterUpdateOfWellDefined_mem_of_alpha_le_one
      (hαwf := hDX) (hα_le_one := hα_le_one) (x := x) (G := G) (k := k) hx
  have hstep : upd - x = a • (y - x) := by
    dsimp [upd, a, y]
    simp only [StochasticNonconvexConditionalGradientSetup.iterUpdateOfWellDefined]
    module
  have hquad :=
    stochastic_smooth_quadratic_upper_bound_of_hasGradientAt
      (setup := setup) (x := x) (y := upd) hx hupd_mem
  have hinner_update :
      ⟪setup.gradf x, upd - x⟫_ℝ = a * ⟪setup.gradf x, y - x⟫_ℝ := by
    rw [hstep, inner_smul_right]
  have hquad' :
      setup.f upd ≤ setup.f x + a * ⟪setup.gradf x, y - x⟫_ℝ +
        (setup.L / 2) * ‖upd - x‖ ^ 2 := by
    simpa [hinner_update] using hquad
  have hdiam : ‖y - x‖ ≤ D := by
    dsimp [D, y]
    exact setup.barDX_bound y x hy hx
  have hnorm_update : ‖upd - x‖ ≤ a * D := by
    calc
      ‖upd - x‖ = ‖a • (y - x)‖ := by rw [hstep]
      _ = a * ‖y - x‖ := by
        rw [norm_smul, Real.norm_of_nonneg ha_nonneg]
      _ ≤ a * D := mul_le_mul_of_nonneg_left hdiam ha_nonneg
  have hnorm_sq : ‖upd - x‖ ^ 2 ≤ a ^ 2 * D ^ 2 := by
    have hright_nonneg : 0 ≤ a * D := mul_nonneg ha_nonneg hD_nonneg
    have hleft_nonneg : 0 ≤ ‖upd - x‖ := norm_nonneg _
    nlinarith
  have hqbound :
      (setup.L / 2) * ‖upd - x‖ ^ 2 ≤
        (setup.L / 2) * (a ^ 2 * D ^ 2) := by
    have hLhalf_nonneg : 0 ≤ setup.L / 2 := by
      linarith [setup.hL_pos]
    exact mul_le_mul_of_nonneg_left hnorm_sq hLhalf_nonneg
  have hinner_decomp :
      a * ⟪setup.gradf x, y - x⟫_ℝ =
        a * ⟪G, y - x⟫_ℝ + a * ⟪setup.gradf x - G, y - x⟫_ℝ := by
    have hbase :
        ⟪setup.gradf x, y - x⟫_ℝ =
          ⟪G, y - x⟫_ℝ + ⟪setup.gradf x - G, y - x⟫_ℝ := by
      rw [inner_sub_left]
      ring
    rw [hbase]
    ring
  have hinner_err :
      ⟪setup.gradf x - G, y - x⟫_ℝ ≤ e * D := by
    calc
      ⟪setup.gradf x - G, y - x⟫_ℝ
          ≤ |⟪setup.gradf x - G, y - x⟫_ℝ| := le_abs_self _
      _ ≤ ‖setup.gradf x - G‖ * ‖y - x‖ :=
          abs_real_inner_le_norm (setup.gradf x - G) (y - x)
      _ ≤ e * D := by
          dsimp [e]
          exact mul_le_mul_of_nonneg_left hdiam (norm_nonneg _)
  have hcross_linear :
      a * ⟪setup.gradf x - G, y - x⟫_ℝ ≤ a * (e * D) :=
    mul_le_mul_of_nonneg_left hinner_err ha_nonneg
  have hyoung :
      a * (e * D) ≤
        (1 / (2 * setup.L)) * e ^ 2 + (setup.L / 2) * a ^ 2 * D ^ 2 := by
    have hsq : 0 ≤ (e - setup.L * a * D) ^ 2 := sq_nonneg _
    have hmain :
        2 * setup.L * (a * (e * D)) ≤ e ^ 2 + setup.L ^ 2 * a ^ 2 * D ^ 2 := by
      nlinarith [hsq]
    have hden_pos : 0 < 2 * setup.L := mul_pos two_pos setup.hL_pos
    calc
      a * (e * D) = (2 * setup.L * (a * (e * D))) / (2 * setup.L) := by
        have hden_ne : 2 * setup.L ≠ 0 := ne_of_gt hden_pos
        calc
          a * (e * D) = (a * (e * D)) * ((2 * setup.L) / (2 * setup.L)) := by
            rw [div_self hden_ne]
            ring
          _ = (2 * setup.L * (a * (e * D))) / (2 * setup.L) := by
            field_simp [hden_ne]
      _ ≤ (e ^ 2 + setup.L ^ 2 * a ^ 2 * D ^ 2) / (2 * setup.L) := by
        exact div_le_div_of_nonneg_right hmain (le_of_lt hden_pos)
      _ = (1 / (2 * setup.L)) * e ^ 2 + (setup.L / 2) * a ^ 2 * D ^ 2 := by
        field_simp [ne_of_gt setup.hL_pos, ne_of_gt hden_pos]
  have hcross :
      a * ⟪setup.gradf x - G, y - x⟫_ℝ ≤
        (1 / (2 * setup.L)) * e ^ 2 + (setup.L / 2) * a ^ 2 * D ^ 2 :=
    le_trans hcross_linear hyoung
  have hcombined :
      setup.f upd ≤
        setup.f x + a * ⟪G, y - x⟫_ℝ +
          (1 / (2 * setup.L)) * e ^ 2 + setup.L * a ^ 2 * D ^ 2 := by
    have hquad_bound :
        setup.f upd ≤ setup.f x + a * ⟪setup.gradf x, y - x⟫_ℝ +
          (setup.L / 2) * (a ^ 2 * D ^ 2) := by
      linarith
    have hhalf :
        (setup.L / 2) * (a ^ 2 * D ^ 2) +
          (setup.L / 2) * a ^ 2 * D ^ 2 =
            setup.L * a ^ 2 * D ^ 2 := by
      ring
    linarith
  dsimp [upd, a, y, D, e] at hcombined ⊢
  simpa [norm_sub_rev] using hcombined

/-- Deterministic source-form Wolfe-gap algebra for stochastic Algorithm 7.13
before absorbing the estimator-error norm term.

Candidate audit: searched `one step expected Wolfe gap source form`; finite
helpers had the wrong setup type, while SOptLib telescope lemmas start after
one-step inequalities are supplied. This uses stochastic
`wolfeGap_surrogate_bound` plus the LMO bridge `maxLinModel_le_inner_sub_lmo`,
aligning with Lan Eq. (7.4.2) and Eq. (7.4.5). -/
private lemma stochastic_one_step_gap_descent_pointwise_source_form
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {x G xnext : E} {a : ℝ} (hx : x ∈ setup.X) (ha_nonneg : 0 ≤ a)
    (hdescent :
      setup.f xnext ≤
        setup.f x + a * ⟪G, setup.linearMinimizer G - x⟫_ℝ +
          (1 / (2 * setup.L)) * ‖setup.delta G x‖ ^ 2 +
          setup.L * a ^ 2 * setup.barDX ^ 2) :
    a * SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x ≤
      setup.f x - setup.f xnext +
        (1 / (2 * setup.L)) * ‖setup.delta G x‖ ^ 2 +
        setup.L * a ^ 2 * setup.barDX ^ 2 +
        a * setup.barDX * ‖setup.delta G x‖ := by
  classical
  refine wolfe_gap_descent_source_form_of_smooth_descent
    (gap := SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer)
    (f := setup.f) (lmo := setup.linearMinimizer) (delta := setup.delta)
    (x := x) (G := G) (xnext := xnext) (alpha := a) (L := setup.L)
    (D := setup.barDX) ha_nonneg ?_ hdescent
  have hsur := wolfeGap_surrogate_bound (setup := setup) x G hx
  have hmax := setup.maxLinModel_le_inner_sub_lmo (x := x) (G := G)
  linarith

/-- Deterministic Wolfe-gap algebra after the Young absorption of the
estimator-error norm.

This is the stochastic analogue of
`finite_one_step_gap_descent_pointwise_of_smooth`. It keeps Theorem 7.17 on the
printed route: the `α D̄_X ‖δ_k‖` perturbation from Eq. (7.4.2) is absorbed into
another half of the second-moment term before Lemma 7.5 is summed, so the
Lemma 7.5 variance floor contributes only through `E‖δ_k‖²`. -/
private lemma stochastic_one_step_gap_descent_pointwise_of_smooth
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {x G xnext : E} {a : ℝ} (hx : x ∈ setup.X) (ha_nonneg : 0 ≤ a)
    (hdescent :
      setup.f xnext ≤
        setup.f x + a * ⟪G, setup.linearMinimizer G - x⟫_ℝ +
          (1 / (2 * setup.L)) * ‖setup.delta G x‖ ^ 2 +
          setup.L * a ^ 2 * setup.barDX ^ 2) :
    a * SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x ≤
      setup.f x - setup.f xnext +
        (1 / setup.L) * ‖setup.delta G x‖ ^ 2 +
        (3 / 2) * setup.L * a ^ 2 * setup.barDX ^ 2 := by
  classical
  let y : E := setup.linearMinimizer G
  let d : ℝ := ‖setup.delta G x‖
  let D : ℝ := setup.barDX
  have hsur := wolfeGap_surrogate_bound (setup := setup) x G hx
  have hmax := setup.maxLinModel_le_inner_sub_lmo (x := x) (G := G)
  have hgap :
      SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x ≤ ⟪G, x - y⟫_ℝ + d * D := by
    dsimp [y, d, D]
    linarith
  have hgap_mul :
      a * SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x ≤ a * (⟪G, x - y⟫_ℝ + d * D) := by
    exact mul_le_mul_of_nonneg_left hgap ha_nonneg
  have hinner_neg : ⟪G, x - y⟫_ℝ = -⟪G, y - x⟫_ℝ := by
    have hxy : x - y = -(y - x) := by abel
    rw [hxy, inner_neg_right]
  have hdesc_rearr :
      -a * ⟪G, y - x⟫_ℝ ≤
        setup.f x - setup.f xnext +
          (1 / (2 * setup.L)) * d ^ 2 +
          setup.L * a ^ 2 * D ^ 2 := by
    dsimp [y, d, D]
    linarith
  have hyoung :
      a * (d * D) ≤
        (1 / (2 * setup.L)) * d ^ 2 +
          (setup.L / 2) * a ^ 2 * D ^ 2 := by
    have hsq : 0 ≤ (d - setup.L * a * D) ^ 2 := sq_nonneg _
    have hmain :
        2 * setup.L * (a * (d * D)) ≤ d ^ 2 + setup.L ^ 2 * a ^ 2 * D ^ 2 := by
      nlinarith [hsq]
    have hden_pos : 0 < 2 * setup.L := mul_pos two_pos setup.hL_pos
    calc
      a * (d * D) = (2 * setup.L * (a * (d * D))) / (2 * setup.L) := by
        have hden_ne : 2 * setup.L ≠ 0 := ne_of_gt hden_pos
        calc
          a * (d * D) = (a * (d * D)) * ((2 * setup.L) / (2 * setup.L)) := by
            rw [div_self hden_ne]
            ring
          _ = (2 * setup.L * (a * (d * D))) / (2 * setup.L) := by
            field_simp [hden_ne]
      _ ≤ (d ^ 2 + setup.L ^ 2 * a ^ 2 * D ^ 2) / (2 * setup.L) := by
        exact div_le_div_of_nonneg_right hmain (le_of_lt hden_pos)
      _ = (1 / (2 * setup.L)) * d ^ 2 +
          (setup.L / 2) * a ^ 2 * D ^ 2 := by
        field_simp [ne_of_gt setup.hL_pos, ne_of_gt hden_pos]
  calc
    a * SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x
        ≤ a * (⟪G, x - y⟫_ℝ + d * D) := hgap_mul
    _ = -a * ⟪G, y - x⟫_ℝ + a * (d * D) := by
        rw [hinner_neg]
        ring
    _ ≤
        (setup.f x - setup.f xnext +
          (1 / (2 * setup.L)) * d ^ 2 +
          setup.L * a ^ 2 * D ^ 2) +
          ((1 / (2 * setup.L)) * d ^ 2 +
            (setup.L / 2) * a ^ 2 * D ^ 2) := by
        exact add_le_add hdesc_rearr hyoung
    _ =
      setup.f x - setup.f xnext +
        (1 / setup.L) * ‖setup.delta G x‖ ^ 2 +
        (3 / 2) * setup.L * a ^ 2 * setup.barDX ^ 2 := by
        dsimp [d, D]
        field_simp [ne_of_gt setup.hL_pos]
        ring

/-- Smooth descent for the affine LMO update, kept in true-gradient form.

This is the sharper route needed for the printed Theorem 7.17 coefficient:
the estimator error is not Young-absorbed inside the smoothness step. Instead it
is later combined with the Wolfe-gap perturbation as a single
`⟪δ, z - y⟫` term. -/
private lemma stochastic_one_step_smooth_descent_gradf_premise
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {k : ℕ} {x G : E} (hx : x ∈ setup.X) :
    setup.f (setup.iterUpdateOfWellDefined hDX x G k) ≤
      setup.f x +
        setup.αOfWellDefined hDX k *
          ⟪setup.gradf x, setup.linearMinimizer G - x⟫_ℝ +
        (setup.L / 2) * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
  classical
  refine affine_lmo_smooth_descent_true_gradient
    (f := setup.f) (grad := setup.gradf) (lmo := setup.linearMinimizer)
    (x := x) (G := G) (xnext := setup.iterUpdateOfWellDefined hDX x G k)
    (a := setup.αOfWellDefined hDX k) (L := setup.L) (D := setup.barDX)
    ?_ (le_of_lt setup.hL_pos) ?_ ?_ ?_
  · exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  · simp only [StochasticNonconvexConditionalGradientSetup.iterUpdateOfWellDefined]
    module
  · exact stochastic_smooth_quadratic_upper_bound_of_hasGradientAt
      (setup := setup) (x := x) (y := setup.iterUpdateOfWellDefined hDX x G k) hx
      (setup.iterUpdateOfWellDefined_mem_of_alpha_le_one
        (hαwf := hDX) (hα_le_one := hα_le_one) (x := x) (G := G) (k := k) hx)
  · exact setup.barDX_bound (setup.linearMinimizer G) x (setup.linearMinimizer_mem G) hx

/-- Deterministic combined Wolfe/descent algebra for the printed Theorem 7.17
route.

The key point is to combine the true-gradient smoothness inequality and the
Wolfe-gap LMO comparison before applying Young. The two estimator-error linear
terms collapse to `⟪δ, z - y⟫`, where both `z` and `y` are feasible, so one
Young inequality contributes only `1/(2L) * ‖δ‖²`. -/
private lemma stochastic_one_step_gap_descent_pointwise_printed
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    {x G xnext : E} {a : ℝ} (hx : x ∈ setup.X) (ha_nonneg : 0 ≤ a)
    (hdescent :
      setup.f xnext ≤
        setup.f x + a * ⟪setup.gradf x, setup.linearMinimizer G - x⟫_ℝ +
          (setup.L / 2) * a ^ 2 * setup.barDX ^ 2) :
    a * SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer x ≤
      setup.f x - setup.f xnext +
        (1 / (2 * setup.L)) * ‖setup.delta G x‖ ^ 2 +
        setup.L * a ^ 2 * setup.barDX ^ 2 := by
  simpa [StochasticNonconvexConditionalGradientSetup.delta, SOptLib.oracleEstimatorError] using
    (SOptLib.ConditionalGradient.wolfeGap_descent_of_trueGradient_linearMinimizer_descent
      (f := setup.f) (grad := setup.gradf)
      (maximizer := setup.wolfeGapMaximizer)
      (linearMinimizer := setup.linearMinimizer)
      (x := x) (G := G) (xnext := xnext)
      (a := a) (L := setup.L) (D := setup.barDX)
      setup.hL_pos ha_nonneg
      (setup.linearMinimizer_spec G (setup.wolfeGapMaximizer x : E)
        (setup.wolfeGapMaximizer x).property)
      (by
        exact setup.barDX_bound (setup.wolfeGapMaximizer x : E) (setup.linearMinimizer G)
          (setup.wolfeGapMaximizer x).property (setup.linearMinimizer_mem G))
      hdescent)

/-- Generated stochastic iterates satisfy the Young-absorbed one-step
Wolfe-gap descent bound. -/
private lemma stochastic_generated_one_step_gap_descent_pointwise
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 setup.N) (ω : Ω) :
    setup.αOfWellDefined hDX k *
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ≤
      setup.f (setup.iterProcessOfWellDefined hDX k ω) -
        setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω) +
        (1 / setup.L) *
          ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 +
        (3 / 2) * setup.L *
          setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
  have hk_bounds := Finset.mem_Icc.mp hk
  have hx : setup.iterProcessOfWellDefined hDX k ω ∈ setup.X := by
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hDX hα_le_one k ω
  have ha_nonneg : 0 ≤ setup.αOfWellDefined hDX k :=
    le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  have hnext :
      setup.iterProcessOfWellDefined hDX (k + 1) ω =
        setup.iterUpdateOfWellDefined hDX
          (setup.iterProcessOfWellDefined hDX k ω)
          (setup.estimatorProcessOfWellDefined hDX k ω) k :=
    stochastic_iterProcessOfWellDefined_succ_eq_iterUpdate
      (setup := setup) hDX hk_bounds.1 ω
  have hdescent :
      setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω) ≤
        setup.f (setup.iterProcessOfWellDefined hDX k ω) +
          setup.αOfWellDefined hDX k *
            ⟪setup.estimatorProcessOfWellDefined hDX k ω,
              setup.linearMinimizer (setup.estimatorProcessOfWellDefined hDX k ω) -
                setup.iterProcessOfWellDefined hDX k ω⟫_ℝ +
          (1 / (2 * setup.L)) *
            ‖setup.delta
                (setup.estimatorProcessOfWellDefined hDX k ω)
                (setup.iterProcessOfWellDefined hDX k ω)‖ ^ 2 +
          setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
    rw [hnext]
    simpa [StochasticNonconvexConditionalGradientSetup.delta] using
      stochastic_one_step_smooth_descent_premise
        (setup := setup) hDX hα_le_one
        (k := k) (x := setup.iterProcessOfWellDefined hDX k ω)
        (G := setup.estimatorProcessOfWellDefined hDX k ω) hx
  have hgap :=
    stochastic_one_step_gap_descent_pointwise_of_smooth
      (setup := setup) (x := setup.iterProcessOfWellDefined hDX k ω)
      (G := setup.estimatorProcessOfWellDefined hDX k ω)
      (xnext := setup.iterProcessOfWellDefined hDX (k + 1) ω)
      (a := setup.αOfWellDefined hDX k) hx ha_nonneg hdescent
  simpa [StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined]
    using hgap

/-- Generated stochastic iterates satisfy the combined printed-route one-step
Wolfe-gap descent bound with only one half-second-moment term. -/
private lemma stochastic_generated_one_step_gap_descent_pointwise_printed
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 setup.N) (ω : Ω) :
    setup.αOfWellDefined hDX k *
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ≤
      setup.f (setup.iterProcessOfWellDefined hDX k ω) -
        setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω) +
        (1 / (2 * setup.L)) *
          ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 +
        setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
  have hk_bounds := Finset.mem_Icc.mp hk
  have hx : setup.iterProcessOfWellDefined hDX k ω ∈ setup.X := by
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hDX hα_le_one k ω
  have ha_nonneg : 0 ≤ setup.αOfWellDefined hDX k :=
    le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  have hnext :
      setup.iterProcessOfWellDefined hDX (k + 1) ω =
        setup.iterUpdateOfWellDefined hDX
          (setup.iterProcessOfWellDefined hDX k ω)
          (setup.estimatorProcessOfWellDefined hDX k ω) k :=
    stochastic_iterProcessOfWellDefined_succ_eq_iterUpdate
      (setup := setup) hDX hk_bounds.1 ω
  have hdescent :
      setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω) ≤
        setup.f (setup.iterProcessOfWellDefined hDX k ω) +
          setup.αOfWellDefined hDX k *
            ⟪setup.gradf (setup.iterProcessOfWellDefined hDX k ω),
              setup.linearMinimizer (setup.estimatorProcessOfWellDefined hDX k ω) -
                setup.iterProcessOfWellDefined hDX k ω⟫_ℝ +
          (setup.L / 2) *
            setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
    rw [hnext]
    simpa using
      stochastic_one_step_smooth_descent_gradf_premise
        (setup := setup) hDX hα_le_one
        (k := k) (x := setup.iterProcessOfWellDefined hDX k ω)
        (G := setup.estimatorProcessOfWellDefined hDX k ω) hx
  have hgap :=
    stochastic_one_step_gap_descent_pointwise_printed
      (setup := setup) (x := setup.iterProcessOfWellDefined hDX k ω)
      (G := setup.estimatorProcessOfWellDefined hDX k ω)
      (xnext := setup.iterProcessOfWellDefined hDX (k + 1) ω)
      (a := setup.αOfWellDefined hDX k) hx ha_nonneg hdescent
  simpa [StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined]
    using hgap

/-- Generated stochastic iterates satisfy the source-form one-step Wolfe-gap
descent bound before the Lemma 7.5 scalar substitution.

Candidate audit: reused the route-local stochastic smooth and Wolfe algebra
above after checking target-file finite one-step helpers. SOptLib descent
helpers do not expose Algorithm 7.13's `iterProcessOfWellDefined` successor
identity or the unabsorbed `α D E‖δ‖` term from Lan Eq. (7.4.6). -/
private lemma stochastic_generated_one_step_gap_descent_pointwise_source_form
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : setup.paperAlphaFormulaWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {k : ℕ} (hk : k ∈ Finset.Icc 1 setup.N) (ω : Ω) :
    setup.αOfWellDefined hDX k *
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ≤
      setup.f (setup.iterProcessOfWellDefined hDX k ω) -
        setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω) +
        (1 / (2 * setup.L)) *
          ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 +
        setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 +
        setup.αOfWellDefined hDX k * setup.barDX *
          ‖setup.deltaProcessOfWellDefined hDX k ω‖ := by
  have hk_bounds := Finset.mem_Icc.mp hk
  have hx : setup.iterProcessOfWellDefined hDX k ω ∈ setup.X := by
    simpa [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined] using
      setup.processOfWellDefined_mem_of_alpha_le_one hDX hα_le_one k ω
  have ha_nonneg : 0 ≤ setup.αOfWellDefined hDX k :=
    le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
  have hnext :
      setup.iterProcessOfWellDefined hDX (k + 1) ω =
        setup.iterUpdateOfWellDefined hDX
          (setup.iterProcessOfWellDefined hDX k ω)
          (setup.estimatorProcessOfWellDefined hDX k ω) k :=
    stochastic_iterProcessOfWellDefined_succ_eq_iterUpdate
      (setup := setup) hDX hk_bounds.1 ω
  have hdescent :
      setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω) ≤
        setup.f (setup.iterProcessOfWellDefined hDX k ω) +
          setup.αOfWellDefined hDX k *
            ⟪setup.estimatorProcessOfWellDefined hDX k ω,
              setup.linearMinimizer (setup.estimatorProcessOfWellDefined hDX k ω) -
                setup.iterProcessOfWellDefined hDX k ω⟫_ℝ +
          (1 / (2 * setup.L)) *
            ‖setup.delta
                (setup.estimatorProcessOfWellDefined hDX k ω)
                (setup.iterProcessOfWellDefined hDX k ω)‖ ^ 2 +
          setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2 := by
    rw [hnext]
    simpa [StochasticNonconvexConditionalGradientSetup.delta] using
      stochastic_one_step_smooth_descent_premise
        (setup := setup) hDX hα_le_one
        (k := k) (x := setup.iterProcessOfWellDefined hDX k ω)
        (G := setup.estimatorProcessOfWellDefined hDX k ω) hx
  have hgap :=
    stochastic_one_step_gap_descent_pointwise_source_form
      (setup := setup) (x := setup.iterProcessOfWellDefined hDX k ω)
      (G := setup.estimatorProcessOfWellDefined hDX k ω)
      (xnext := setup.iterProcessOfWellDefined hDX (k + 1) ω)
      (a := setup.αOfWellDefined hDX k) hx ha_nonneg hdescent
  simpa [StochasticNonconvexConditionalGradientSetup.deltaProcessOfWellDefined]
    using hgap

/-- Active stochastic one-step expected Wolfe-gap bound in the source form used
before applying Lemma 7.5 and the scalar epoch budget.

Candidate audit: searched `one step expected Wolfe gap source form`; the finite
`finite_active_expected_one_step_gap_bound_source_form` has the wrong setup
type, and SOptLib telescope helpers operate after this one-step estimate is
already summed. This lemma uses the stochastic pointwise source-form descent,
`wolfeGap_iterProcessOfWellDefined_integrable`,
`f_iterProcessOfWellDefined_integrable`, and `epochwise_delta_sq_integrable`,
aligning with Lan Theorem 7.17's reuse of the Theorem 7.16 proof template. -/
private lemma stochastic_active_expected_one_step_gap_bound_source_form
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {s j : ℕ} (hj : j ∈ stochasticActiveEpochSteps setup s) :
    setup.αOfWellDefined hDX (setup.globalIndex s j) *
        ∫ ω,
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω)
          ∂setup.P ≤
      ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P +
        (1 / (2 * setup.L)) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P +
        setup.L * setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2 +
        setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
          ∂setup.P := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  let k : ℕ := setup.globalIndex s j
  have hk : k ∈ Finset.Icc 1 setup.N := by
    dsimp [k]
    exact stochastic_globalIndex_mem_output_of_active (setup := setup) hj
  have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
  have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
  have hgap_int :
      Integrable
        (fun ω => SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω))
        setup.P :=
    setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one k
  have hleft_int :
      Integrable
        (fun ω =>
          setup.αOfWellDefined hDX k *
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω))
        setup.P :=
    hgap_int.const_mul (setup.αOfWellDefined hDX k)
  have hf_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX k ω)) setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one k
  have hfnext_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω))
        setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one (k + 1)
  have hdrop_int :
      Integrable
        (fun ω =>
          setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) setup.P :=
    hf_int.sub hfnext_int
  have hdelta_sq_int :
      Integrable
        (fun ω => ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) setup.P := by
    dsimp [k]
    exact epochwise_delta_sq_integrable
      setup hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
  have hdelta_norm_int :
      Integrable
        (fun ω => ‖setup.deltaProcessOfWellDefined hDX k ω‖) setup.P := by
    have hnorm_nonneg :
        ∀ᵐ ω ∂setup.P, 0 ≤ ‖setup.deltaProcessOfWellDefined hDX k ω‖ :=
      Filter.Eventually.of_forall (fun ω => norm_nonneg _)
    have hle :
        ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P ≤
          ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P :=
      le_rfl
    exact
      (integrable_of_nonneg_sq_integrable_integral_le_sq_bound_add_one
        (μ := setup.P) (Z := fun ω => ‖setup.deltaProcessOfWellDefined hDX k ω‖)
        (C := ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P)
        hdelta_sq_int hnorm_nonneg hle).1
  let cSq : ℝ := setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2
  let cNorm : ℝ := setup.αOfWellDefined hDX k * setup.barDX
  have hrhs_int :
      Integrable
        (fun ω =>
          ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / (2 * setup.L)) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + cSq) +
            cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖) setup.P :=
    ((hdrop_int.add (hdelta_sq_int.const_mul (1 / (2 * setup.L)))).add
      (integrable_const cSq)).add (hdelta_norm_int.const_mul cNorm)
  have hmono :
      ∫ ω,
          setup.αOfWellDefined hDX k *
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P ≤
        ∫ ω,
          ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / (2 * setup.L)) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + cSq) +
            cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
    refine integral_mono hleft_int hrhs_int ?_
    intro ω
    have hpt :=
      stochastic_generated_one_step_gap_descent_pointwise_source_form
        (setup := setup) hDX hα_le_one (k := k) hk ω
    dsimp [cSq, cNorm, k] at hpt ⊢
    linarith
  have hrhs_expand :
      ∫ ω,
          ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / (2 * setup.L)) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + cSq) +
            cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P =
        ∫ ω,
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
          (1 / (2 * setup.L)) *
            ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P +
          cSq +
          cNorm *
            ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
    have hdrop_delta :
        Integrable
          (fun ω =>
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
              (1 / (2 * setup.L)) *
                ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) setup.P :=
      hdrop_int.add (hdelta_sq_int.const_mul (1 / (2 * setup.L)))
    calc
      ∫ ω,
          ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / (2 * setup.L)) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + cSq) +
            cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P
          =
          ∫ ω,
            ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
              (1 / (2 * setup.L)) *
                ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + cSq) ∂setup.P +
            ∫ ω, cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
            exact integral_add
              (hdrop_delta.add (integrable_const cSq))
              (hdelta_norm_int.const_mul cNorm)
      _ =
          (∫ ω,
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
              (1 / (2 * setup.L)) *
                ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P +
            ∫ _ω, cSq ∂setup.P) +
            cNorm *
              ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
            have hnorm_pull :
                ∫ ω,
                    cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P =
                  cNorm *
                    ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
              rw [integral_const_mul]
            rw [integral_add hdrop_delta (integrable_const (c := cSq))]
            rw [hnorm_pull]
      _ =
          ((∫ ω,
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
            ∫ ω,
              (1 / (2 * setup.L)) *
                ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P) +
            cSq) +
            cNorm *
              ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
            rw [integral_add hdrop_int
              (hdelta_sq_int.const_mul (1 / (2 * setup.L)))]
            simp
      _ =
          ∫ ω,
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
          (1 / (2 * setup.L)) *
            ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P +
          cSq +
          cNorm *
            ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ∂setup.P := by
            rw [integral_const_mul]
  calc
    setup.αOfWellDefined hDX (setup.globalIndex s j) *
        ∫ ω,
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω)
          ∂setup.P
        =
      ∫ ω,
        setup.αOfWellDefined hDX k *
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P := by
        dsimp [k]
        rw [integral_const_mul]
    _ ≤
      ∫ ω,
          ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / (2 * setup.L)) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + cSq) +
            cNorm * ‖setup.deltaProcessOfWellDefined hDX k ω‖
          ∂setup.P := hmono
    _ =
      ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P +
        (1 / (2 * setup.L)) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P +
        setup.L * setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2 +
        setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
          ∂setup.P := by
        dsimp [k, cSq, cNorm] at hrhs_expand ⊢
        simpa using hrhs_expand

/-- Active stochastic one-step expected Wolfe-gap bound after Young absorption
of the estimator-error norm.

This is the honest stochastic analogue of the already proved finite
`finite_active_expected_one_step_gap_bound`: it removes the separate L1 term but
doubles the second-moment coefficient from `1/(2L)` to `1/L`. -/
private lemma stochastic_active_expected_one_step_gap_bound_young
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {s j : ℕ} (hj : j ∈ stochasticActiveEpochSteps setup s) :
    setup.αOfWellDefined hDX (setup.globalIndex s j) *
        ∫ ω,
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω)
          ∂setup.P ≤
      ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P +
        (1 / setup.L) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P +
        (3 / 2) * setup.L *
          setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2 := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  let k : ℕ := setup.globalIndex s j
  have hk : k ∈ Finset.Icc 1 setup.N := by
    dsimp [k]
    exact stochastic_globalIndex_mem_output_of_active (setup := setup) hj
  have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
  have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
  have hgap_int :
      Integrable
        (fun ω => SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω))
        setup.P :=
    setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one k
  have hleft_int :
      Integrable
        (fun ω =>
          setup.αOfWellDefined hDX k *
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω))
        setup.P :=
    hgap_int.const_mul (setup.αOfWellDefined hDX k)
  have hf_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX k ω)) setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one k
  have hfnext_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω))
        setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one (k + 1)
  have hdrop_int :
      Integrable
        (fun ω =>
          setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) setup.P :=
    hf_int.sub hfnext_int
  have hdelta_sq_int :
      Integrable
        (fun ω => ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) setup.P := by
    dsimp [k]
    exact epochwise_delta_sq_integrable
      setup hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
  let c : ℝ := (3 / 2) * setup.L *
    setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2
  have hrhs_int :
      Integrable
        (fun ω =>
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / setup.L) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + c) setup.P :=
    (hdrop_int.add (hdelta_sq_int.const_mul (1 / setup.L))).add
      (integrable_const c)
  have hmono :
      ∫ ω,
          setup.αOfWellDefined hDX k *
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P ≤
        ∫ ω,
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / setup.L) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + c ∂setup.P := by
    refine integral_mono hleft_int hrhs_int ?_
    intro ω
    have hpt :=
      stochastic_generated_one_step_gap_descent_pointwise
        (setup := setup) hDX hα_le_one (k := k) hk ω
    dsimp [c, k] at hpt ⊢
    linarith
  have hrhs_expand :
      ∫ ω,
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / setup.L) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + c ∂setup.P =
        ∫ ω,
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
          (1 / setup.L) *
            ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P + c := by
    calc
      ∫ ω,
          (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
            (1 / setup.L) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + c ∂setup.P
          =
          ∫ ω,
            ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
              (1 / setup.L) *
                ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) + c ∂setup.P := by
            rfl
      _ =
          ∫ ω,
            ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
              (1 / setup.L) *
                ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) ∂setup.P +
            ∫ _ω, c ∂setup.P := by
            simpa [Pi.add_apply] using
              (integral_add
                (hdrop_int.add (hdelta_sq_int.const_mul (1 / setup.L)))
                (integrable_const (c := c) :
                  Integrable (fun _ω : Ω => c) setup.P))
      _ =
          (∫ ω,
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
            ∫ ω, (1 / setup.L) *
              ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P) + c := by
            rw [show
              (∫ ω,
                ((setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                    setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
                  (1 / setup.L) *
                    ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) ∂setup.P) =
                (∫ ω,
                  (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
                    setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
                  ∫ ω, (1 / setup.L) *
                    ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P) by
                simpa [Pi.add_apply] using
                  (integral_add hdrop_int
                    (hdelta_sq_int.const_mul (1 / setup.L)))]
            simp
      _ =
          ∫ ω,
            (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
              setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) ∂setup.P +
          (1 / setup.L) *
            ∫ ω, ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 ∂setup.P + c := by
            rw [integral_const_mul]
  calc
    setup.αOfWellDefined hDX (setup.globalIndex s j) *
        ∫ ω,
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω)
          ∂setup.P
        =
      ∫ ω,
        setup.αOfWellDefined hDX k *
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P := by
        dsimp [k]
        rw [integral_const_mul]
    _ ≤
      ∫ ω,
        (setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) +
          (1 / setup.L) *
            ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2 + c ∂setup.P := hmono
    _ =
      ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P +
        (1 / setup.L) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P +
        (3 / 2) * setup.L *
          setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2 := by
        dsimp [k, c] at hrhs_expand ⊢
        simpa using hrhs_expand

/-- Active stochastic one-step expected Wolfe-gap bound for the printed
Theorem 7.17 route.

This is the sharper version of `stochastic_active_expected_one_step_gap_bound_young`:
the combined deterministic bridge leaves the second-moment coefficient
`1/(2L)`, exactly as in the paper display. -/
private lemma stochastic_active_expected_one_step_gap_bound_printed
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1)
    {s j : ℕ} (hj : j ∈ stochasticActiveEpochSteps setup s) :
    setup.αOfWellDefined hDX (setup.globalIndex s j) *
        ∫ ω,
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω)
          ∂setup.P ≤
      ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P +
        (1 / (2 * setup.L)) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P +
        setup.L * setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2 := by
  classical
  haveI : IsProbabilityMeasure setup.P := setup.hP
  let k : ℕ := setup.globalIndex s j
  have hk : k ∈ Finset.Icc 1 setup.N := by
    dsimp [k]
    exact stochastic_globalIndex_mem_output_of_active (setup := setup) hj
  have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
  have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
  have hgap_int :
      Integrable
        (fun ω => SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω))
        setup.P :=
    setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one k
  have hf_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX k ω)) setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one k
  have hfnext_int :
      Integrable
        (fun ω => setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω))
        setup.P :=
    setup.f_iterProcessOfWellDefined_integrable hDX hα_le_one (k + 1)
  have hdrop_int :
      Integrable
        (fun ω =>
          setup.f (setup.iterProcessOfWellDefined hDX k ω) -
            setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω)) setup.P :=
    hf_int.sub hfnext_int
  have hdelta_sq_int :
      Integrable
        (fun ω => ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2) setup.P := by
    dsimp [k]
    exact epochwise_delta_sq_integrable
      setup hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
  let c : ℝ := setup.L * setup.αOfWellDefined hDX k ^ 2 * setup.barDX ^ 2
  simpa [k, c] using
    (integral_one_step_gap_bound_of_pointwise
      (P := setup.P)
      (gap := fun ω =>
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
          (setup.iterProcessOfWellDefined hDX k ω))
      (drop := fun ω =>
        setup.f (setup.iterProcessOfWellDefined hDX k ω) -
          setup.f (setup.iterProcessOfWellDefined hDX (k + 1) ω))
      (deltaSq := fun ω => ‖setup.deltaProcessOfWellDefined hDX k ω‖ ^ 2)
      (a := setup.αOfWellDefined hDX k)
      (cDelta := 1 / (2 * setup.L))
      (c := c)
      hgap_int hdrop_int hdelta_sq_int
      (by
        intro ω
        have hpt :=
          stochastic_generated_one_step_gap_descent_pointwise_printed
            (setup := setup) hDX hα_le_one (k := k) hk ω
        dsimp [c, k] at hpt ⊢
        linarith))

/-- Statement-corrected unnormalized stochastic weighted Wolfe-gap numerator.

This is the direct formal consequence of Eq. (7.4.5), Eq. (7.4.2), and Lemma
7.5/Eq. (7.4.14): the stochastic variance floor contributes once through the
second-moment term and once through the `α D̄_X E‖δ‖` term. -/
private theorem stochastic_weighted_gap_sum_bound_theorem717_with_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    Finset.sum (Finset.Icc 1 setup.N)
        (fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω,
              SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P) ≤
      setup.f setup.x₁ - setup.fStar +
        setup.L * setup.barDX ^ 2 *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
        (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) +
        (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
          Finset.sum (Finset.Icc 1 setup.N)
            (fun k => setup.αOfWellDefined hDX k) := by
  classical
  let globalSq : ℝ :=
    Finset.sum (Finset.Icc 1 setup.N)
      (fun k => setup.αOfWellDefined hDX k ^ 2)
  let epochPenalty : ℝ :=
    Finset.sum (Finset.Icc 0 setup.S)
      (fun s =>
        Finset.sum (Finset.Icc 1 setup.T)
          (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
        setup.paperMaxInnerAlphaOfWellDefined hDX hT s)
  let floorMass : ℝ :=
      (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k => setup.αOfWellDefined hDX k)
  exact
    weighted_gap_sum_bound_of_active_one_step_with_l1_floor
      (global := Finset.Icc 1 setup.N) (outer := Finset.Icc 0 setup.S)
      (active := stochasticActiveEpochSteps setup)
      (index := fun s j => setup.globalIndex s j)
      (weightedGap := fun k =>
        setup.αOfWellDefined hDX k *
          ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P)
      (obj := fun s j =>
        ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P)
      (sq := fun s j =>
        (1 / (2 * setup.L)) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P)
      (alpha := fun s j =>
        setup.L * setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2)
      (l1 := fun s j =>
        setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
          ∂setup.P)
      (objBudget := setup.f setup.x₁ - setup.fStar)
      (sqBudget := setup.L * setup.barDX ^ 2 * (1 / 2 * globalSq))
      (alphaBudget := setup.L * setup.barDX ^ 2 * globalSq)
      (l1Budget := setup.L * setup.barDX ^ 2 * epochPenalty)
      (varianceFloor := (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m))
      (floorMass := floorMass)
      (totalBudget := setup.L * setup.barDX ^ 2 * (3 / 2 * globalSq + epochPenalty))
      (stochastic_global_index_active_epoch_partition_sum (setup := setup)
        (A := fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
              (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P))
      (by
        intro s hs j hj
        have hstep :=
          stochastic_active_expected_one_step_gap_bound_source_form
            (setup := setup) hDX hα_le_one (s := s) (j := j) hj
        simpa [add_assoc] using hstep)
      (by
        exact stochastic_active_objective_drop_telescope
          (setup := setup) hDX hα_le_one)
      (by
        let epochDiffTerm : ℝ :=
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
              (fun j =>
                (setup.L / (2 * setup.b)) *
                  ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P))
        have hlemma75_active :
            ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
              ∫ ω,
                  ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                  ∂setup.P ≤
                setup.L ^ 2 / setup.b *
                    ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
                  setup.σ ^ 2 / setup.m := by
          intro s hs j hj
          have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
          have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
          exact epochwise_estimator_variance_bound
            (setup := setup) hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
        have hsq_raw :
            Finset.sum (Finset.Icc 0 setup.S)
                (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                  (fun j =>
                    (1 / (2 * setup.L)) *
                      ∫ ω,
                        ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                      ∂setup.P)) ≤
              epochDiffTerm + (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) := by
          dsimp [epochDiffTerm]
          exact stochastic_active_delta_square_term_le_epochDiff_plus_variance
            (setup := setup) hDX hlemma75_active
            (stochastic_active_step_count_real (setup := setup))
        have hepochDiff :
            epochDiffTerm ≤ setup.L * setup.barDX ^ 2 * (1 / 2 * globalSq) := by
          dsimp [epochDiffTerm, globalSq]
          exact stochastic_active_epochDiff_term_le_half_alpha_square_budget
            (setup := setup) hDX hα_le_one
        exact le_trans hsq_raw (add_le_add hepochDiff le_rfl))
      (by
        dsimp [globalSq]
        exact stochastic_active_alpha_square_sum_eq_global (setup := setup) hDX)
      (by
        have hlemma75_active :
            ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
              ∫ ω,
                  ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                  ∂setup.P ≤
                setup.L ^ 2 / setup.b *
                    ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
                  setup.σ ^ 2 / setup.m := by
          intro s hs j hj
          have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
          have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
          exact epochwise_estimator_variance_bound
            (setup := setup) hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
        have hl1_raw :
            Finset.sum (Finset.Icc 0 setup.S)
                (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                  (fun j =>
                    setup.αOfWellDefined hDX (setup.globalIndex s j) * setup.barDX *
                      ∫ ω,
                        ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖
                      ∂setup.P)) ≤
              setup.L * setup.barDX ^ 2 * epochPenalty +
                (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
                  Finset.sum (Finset.Icc 0 setup.S)
                    (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                      (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j))) := by
          dsimp [epochPenalty]
          exact stochastic_active_delta_l1_sum_le_epoch_penalty_with_floor
            (setup := setup) hDX hT hα_le_one hlemma75_active
        have hactive_alpha :
            Finset.sum (Finset.Icc 0 setup.S)
                (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j))) =
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k) := by
          exact (stochastic_global_index_active_epoch_partition_sum
            (setup := setup) (A := fun k => setup.αOfWellDefined hDX k)).symm
        simpa [floorMass, hactive_alpha] using hl1_raw)
      (by
        dsimp [globalSq, epochPenalty]
        linarith)

/-! Alternative no-L1 stochastic route.

The deterministic Young absorption of `α D̄_X ‖δ_k‖` before applying Lemma 7.5
avoids the explicit L1 floor, but it doubles the Lemma 7.5 variance contribution.
This is a real source-shaped route through Eq. (7.4.5)/(7.4.2), not a proof of
the printed Theorem 7.17 coefficient. -/
private theorem stochastic_weighted_gap_sum_bound_theorem717_young_variance
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    Finset.sum (Finset.Icc 1 setup.N)
        (fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω,
              SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P) ≤
      setup.f setup.x₁ - setup.fStar +
        setup.L * setup.barDX ^ 2 *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 / (setup.L * setup.m) := by
    classical
    exact
      weighted_gap_sum_bound_of_active_one_step_young_variance
        (global := Finset.Icc 1 setup.N) (outer := Finset.Icc 0 setup.S)
        (active := stochasticActiveEpochSteps setup)
        (index := fun s j => setup.globalIndex s j)
        (weightedGap := fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
              (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P)
        (obj := fun s j =>
          ∫ ω,
            (setup.f
                (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
              setup.f
                (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
            ∂setup.P)
        (sq := fun s j =>
          (1 / setup.L) *
            ∫ ω,
              ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
            ∂setup.P)
        (sqHalf := fun s j =>
          (1 / (2 * setup.L)) *
            ∫ ω,
              ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
            ∂setup.P)
        (alphaYoung := fun s j =>
          (3 / 2) * setup.L *
            setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
            setup.barDX ^ 2)
        (objBudget := setup.f setup.x₁ - setup.fStar)
        (coeff := setup.L * setup.barDX ^ 2)
        (globalSq :=
          Finset.sum (Finset.Icc 1 setup.N)
            (fun k => setup.αOfWellDefined hDX k ^ 2))
        (epochPenalty :=
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s))
        (varianceFloor := (setup.N : ℝ) * setup.σ ^ 2 / (setup.L * setup.m))
        (stochastic_global_index_active_epoch_partition_sum (setup := setup)
          (A := fun k =>
            setup.αOfWellDefined hDX k *
              ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
                (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P))
        (by
          intro s hs j hj
          have hstep :=
            stochastic_active_expected_one_step_gap_bound_young
              (setup := setup) hDX hα_le_one (s := s) (j := j) hj
          simpa [add_assoc, mul_assoc, mul_left_comm, mul_comm] using hstep)
        (by
          exact stochastic_active_objective_drop_telescope
            (setup := setup) hDX hα_le_one)
        (by
          let epochDiffTerm : ℝ :=
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                (fun j =>
                  (setup.L / (2 * setup.b)) *
                    ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P))
          have hlemma75_active :
              ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
                ∫ ω,
                    ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                    ∂setup.P ≤
                  setup.L ^ 2 / setup.b *
                      ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
                    setup.σ ^ 2 / setup.m := by
            intro s hs j hj
            have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
            have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
            exact epochwise_estimator_variance_bound
              (setup := setup) hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
          have hsq_raw :
              Finset.sum (Finset.Icc 0 setup.S)
                  (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                    (fun j =>
                      (1 / (2 * setup.L)) *
                        ∫ ω,
                          ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                        ∂setup.P)) ≤
                epochDiffTerm + (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) := by
            dsimp [epochDiffTerm]
            exact stochastic_active_delta_square_term_le_epochDiff_plus_variance
              (setup := setup) hDX hlemma75_active
              (stochastic_active_step_count_real (setup := setup))
          have hepochDiff :
              epochDiffTerm ≤ setup.L * setup.barDX ^ 2 *
                (1 / 2 *
                  Finset.sum (Finset.Icc 1 setup.N)
                    (fun k => setup.αOfWellDefined hDX k ^ 2)) := by
            dsimp [epochDiffTerm]
            exact stochastic_active_epochDiff_term_le_half_alpha_square_budget
              (setup := setup) hDX hα_le_one
          have hhalf :
              Finset.sum (Finset.Icc 0 setup.S)
                  (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                    (fun j =>
                      (1 / (2 * setup.L)) *
                        ∫ ω,
                          ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                        ∂setup.P)) ≤
                setup.L * setup.barDX ^ 2 *
                  (1 / 2 *
                    Finset.sum (Finset.Icc 1 setup.N)
                      (fun k => setup.αOfWellDefined hDX k ^ 2)) +
                  (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) :=
            le_trans hsq_raw (add_le_add hepochDiff le_rfl)
          have hL_ne : setup.L ≠ 0 := ne_of_gt setup.hL_pos
          have hm_ne : (setup.m : ℝ) ≠ 0 := by
            exact_mod_cast (ne_of_gt (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos))
          convert hhalf using 1
          field_simp [hL_ne, hm_ne])
        (by
          simp [Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm, one_div])
        (by
          have hpart :=
            (stochastic_global_index_active_epoch_partition_sum
              (setup := setup)
              (A := fun k =>
                (3 / 2) * setup.L * setup.αOfWellDefined hDX k ^ 2 *
                  setup.barDX ^ 2)).symm
          rw [hpart]
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro k hk
          ring)
        (by
          let globalSq : ℝ :=
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2)
          let epochPenalty : ℝ :=
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)
          have hglobal_le_epochPenalty : globalSq ≤ epochPenalty := by
            have hα_const :
                ∀ k, setup.αOfWellDefined hDX k = setup.paperAlphaOfWellDefined hDX := by
              intro k
              exact setup.alphaOfWellDefined_eq_paperAlphaOfWellDefined hDX k
            have hmax_const :
                ∀ s,
                  setup.paperMaxInnerAlphaOfWellDefined hDX hT s =
                    setup.paperAlphaOfWellDefined hDX := by
              intro s
              exact setup.paperMaxInnerAlphaOfWellDefined_eq_paperAlphaOfWellDefined hDX hT s
            have hα_nonneg :
                ∀ k, 0 ≤ setup.αOfWellDefined hDX k := by
              intro k
              exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX k)
            let M : ℕ → ℝ := fun s => setup.paperMaxInnerAlphaOfWellDefined hDX hT s
            have hM_nonneg : ∀ s, 0 ≤ M s := by
              intro s
              dsimp [M]
              rw [hmax_const s]
              exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX 1)
            have hactive_to_full :
                Finset.sum (Finset.Icc 0 setup.S)
                    (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                      (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j) * M s)) ≤
                  Finset.sum (Finset.Icc 0 setup.S)
                    (fun s =>
                      Finset.sum (Finset.Icc 1 setup.T)
                        (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                      M s) := by
              refine Finset.sum_le_sum ?_
              intro s hs
              have hsubset :
                  stochasticActiveEpochSteps setup s ⊆ Finset.Icc 1 setup.T := by
                intro j hj
                exact ((stochastic_mem_activeEpochSteps setup).mp hj).1
              have hsum_subset :
                  Finset.sum (stochasticActiveEpochSteps setup s)
                      (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) ≤
                    Finset.sum (Finset.Icc 1 setup.T)
                      (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) :=
                Finset.sum_le_sum_of_subset_of_nonneg hsubset
                  (fun j _hj_full _hj_not_active => hα_nonneg (setup.globalIndex s j))
              have hmul :=
                mul_le_mul_of_nonneg_right hsum_subset (hM_nonneg s)
              simpa [Finset.sum_mul] using hmul
            calc
              globalSq =
                  Finset.sum (Finset.Icc 0 setup.S)
                    (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                      (fun j =>
                        setup.αOfWellDefined hDX (setup.globalIndex s j) *
                          M s)) := by
                    dsimp [globalSq, M]
                    rw [stochastic_global_index_active_epoch_partition_sum (setup := setup)
                      (A := fun k => setup.αOfWellDefined hDX k ^ 2)]
                    refine Finset.sum_congr rfl ?_
                    intro s hs
                    refine Finset.sum_congr rfl ?_
                    intro j hj
                    rw [hmax_const s, hα_const (setup.globalIndex s j)]
                    ring
              _ ≤
                  Finset.sum (Finset.Icc 0 setup.S)
                    (fun s =>
                      Finset.sum (Finset.Icc 1 setup.T)
                        (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                      M s) := hactive_to_full
              _ = epochPenalty := by
                  dsimp [epochPenalty, M]
          have hcoef_nonneg : 0 ≤ setup.L * setup.barDX ^ 2 :=
            mul_nonneg (le_of_lt setup.hL_pos) (sq_nonneg setup.barDX)
          calc
            setup.L * setup.barDX ^ 2 *
                  Finset.sum (Finset.Icc 1 setup.N)
                    (fun k => setup.αOfWellDefined hDX k ^ 2) +
                (3 / 2) * (setup.L * setup.barDX ^ 2) *
                  Finset.sum (Finset.Icc 1 setup.N)
                    (fun k => setup.αOfWellDefined hDX k ^ 2)
                =
              setup.L * setup.barDX ^ 2 *
                ((5 / 2) *
                  Finset.sum (Finset.Icc 1 setup.N)
                    (fun k => setup.αOfWellDefined hDX k ^ 2)) := by ring
            _ ≤ setup.L * setup.barDX ^ 2 *
                (3 / 2 *
                  Finset.sum (Finset.Icc 1 setup.N)
                    (fun k => setup.αOfWellDefined hDX k ^ 2) +
                  Finset.sum (Finset.Icc 0 setup.S)
                    (fun s =>
                      Finset.sum (Finset.Icc 1 setup.T)
                        (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                      setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) := by
                refine mul_le_mul_of_nonneg_left ?_ hcoef_nonneg
                nlinarith [hglobal_le_epochPenalty])

/-- Printed unnormalized stochastic weighted Wolfe-gap numerator.

This is the sharper combined route for Theorem 7.17: the estimator-error linear
terms are combined before Young absorption, so Lemma 7.5 contributes the printed
variance coefficient `N σ^2 / (2 L m)` and no explicit L1 floor. -/
private theorem stochastic_weighted_gap_sum_bound_theorem717_printed
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    Finset.sum (Finset.Icc 1 setup.N)
        (fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω,
              SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P) ≤
      setup.f setup.x₁ - setup.fStar +
        setup.L * setup.barDX ^ 2 *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
        (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) := by
  classical
  let globalSq : ℝ :=
    Finset.sum (Finset.Icc 1 setup.N)
      (fun k => setup.αOfWellDefined hDX k ^ 2)
  let epochPenalty : ℝ :=
    Finset.sum (Finset.Icc 0 setup.S)
      (fun s =>
        Finset.sum (Finset.Icc 1 setup.T)
          (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
        setup.paperMaxInnerAlphaOfWellDefined hDX hT s)
  exact
    weighted_gap_sum_bound_of_active_one_step_with_variance_floor
      (global := Finset.Icc 1 setup.N) (outer := Finset.Icc 0 setup.S)
      (active := stochasticActiveEpochSteps setup)
      (index := fun s j => setup.globalIndex s j)
      (weightedGap := fun k =>
        setup.αOfWellDefined hDX k *
          ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P)
      (obj := fun s j =>
        ∫ ω,
          (setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j) ω) -
            setup.f
              (setup.iterProcessOfWellDefined hDX (setup.globalIndex s j + 1) ω))
          ∂setup.P)
      (sq := fun s j =>
        (1 / (2 * setup.L)) *
          ∫ ω,
            ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
          ∂setup.P)
      (alpha := fun s j =>
        setup.L * setup.αOfWellDefined hDX (setup.globalIndex s j) ^ 2 *
          setup.barDX ^ 2)
      (objBudget := setup.f setup.x₁ - setup.fStar)
      (sqBudget := setup.L * setup.barDX ^ 2 * (1 / 2 * globalSq))
      (alphaBudget := setup.L * setup.barDX ^ 2 * globalSq)
      (penaltyBudget := setup.L * setup.barDX ^ 2 * (3 / 2 * globalSq + epochPenalty))
      (varianceFloor := (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m))
      (stochastic_global_index_active_epoch_partition_sum (setup := setup)
        (A := fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω, SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
              (setup.iterProcessOfWellDefined hDX k ω) ∂setup.P))
      (by
        intro s hs j hj
        have hstep :=
          stochastic_active_expected_one_step_gap_bound_printed
            (setup := setup) hDX hα_le_one (s := s) (j := j) hj
        simpa [add_assoc, mul_assoc, mul_left_comm, mul_comm] using hstep)
      (by
        exact stochastic_active_objective_drop_telescope
          (setup := setup) hDX hα_le_one)
      (by
        let epochDiffTerm : ℝ :=
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
              (fun j =>
                (setup.L / (2 * setup.b)) *
                  ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P))
        have hlemma75_active :
            ∀ s ∈ Finset.Icc 0 setup.S, ∀ j ∈ stochasticActiveEpochSteps setup s,
              ∫ ω,
                  ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                  ∂setup.P ≤
                setup.L ^ 2 / setup.b *
                    ∫ ω, setup.epochDiffSumOfWellDefined hDX s j ω ∂setup.P +
                  setup.σ ^ 2 / setup.m := by
          intro s hs j hj
          have hj_epoch := stochastic_activeEpochSteps_mem_epoch (setup := setup) hj
          have hjN := stochastic_activeEpochSteps_globalIndex_le (setup := setup) hj
          exact epochwise_estimator_variance_bound
            (setup := setup) hDX hα_le_one s j hj_epoch.1 hj_epoch.2 hjN
        have hsq_raw :
            Finset.sum (Finset.Icc 0 setup.S)
                (fun s => Finset.sum (stochasticActiveEpochSteps setup s)
                  (fun j =>
                    (1 / (2 * setup.L)) *
                      ∫ ω,
                        ‖setup.deltaProcessOfWellDefined hDX (setup.globalIndex s j) ω‖ ^ 2
                      ∂setup.P)) ≤
              epochDiffTerm + (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) := by
          dsimp [epochDiffTerm]
          exact stochastic_active_delta_square_term_le_epochDiff_plus_variance
            (setup := setup) hDX hlemma75_active
            (stochastic_active_step_count_real (setup := setup))
        have hepochDiff :
            epochDiffTerm ≤ setup.L * setup.barDX ^ 2 * (1 / 2 * globalSq) := by
          dsimp [epochDiffTerm, globalSq]
          exact stochastic_active_epochDiff_term_le_half_alpha_square_budget
            (setup := setup) hDX hα_le_one
        exact le_trans hsq_raw (add_le_add hepochDiff le_rfl))
      (by
        dsimp [globalSq]
        exact stochastic_active_alpha_square_sum_eq_global (setup := setup) hDX)
      (by
        have hnonneg_epochPenalty : 0 ≤ epochPenalty := by
          dsimp [epochPenalty]
          refine Finset.sum_nonneg ?_
          intro s hs
          have hsum_nonneg :
              0 ≤ Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) := by
            refine Finset.sum_nonneg ?_
            intro j hj
            exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX _)
          have hmax_nonneg :
              0 ≤ setup.paperMaxInnerAlphaOfWellDefined hDX hT s := by
            rw [setup.paperMaxInnerAlphaOfWellDefined_eq_paperAlphaOfWellDefined hDX hT s]
            exact le_of_lt (setup.hα_pos_of_paperAlphaFormulaWellDefined hDX 1)
          exact mul_nonneg hsum_nonneg hmax_nonneg
        have hcoef_nonneg : 0 ≤ setup.L * setup.barDX ^ 2 :=
          mul_nonneg (le_of_lt setup.hL_pos) (sq_nonneg setup.barDX)
        calc
          setup.L * setup.barDX ^ 2 * (1 / 2 * globalSq) +
              setup.L * setup.barDX ^ 2 * globalSq
              = setup.L * setup.barDX ^ 2 * (3 / 2 * globalSq) := by ring
          _ ≤ setup.L * setup.barDX ^ 2 * (3 / 2 * globalSq + epochPenalty) := by
              exact mul_le_mul_of_nonneg_left (by linarith) hcoef_nonneg)

/-! Statement-corrected nondegenerate stochastic helper. This is the proved
Theorem 7.17 route produced by the current formal model: it keeps the Lemma 7.5
L1 floor as the explicit normalized term `D̄_X σ / sqrt(m)`. -/
theorem theorem_7_17_general_with_wellDefined_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) :
    ∀ (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined),
      setup.paperAlphaOfWellDefined hDX ≤ 1 →
      setup.expectedWolfeGapOfWellDefined hDX
          (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
        (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
          setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 /
            (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX) +
          setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
  intro hDX hT hα_le_one
  classical
  let hR : 0 < setup.alphaSumOfWellDefined hDX :=
    setup.alphaSum_pos_of_nonzeroDiameter hDX
  have hgap_int :
      ∀ R : setup.OutputTime,
        Integrable
          (fun ω =>
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX R.1 ω))
          setup.P := by
    intro R
    exact setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one R.1
  have hexpand :=
    setup.expectedWolfeGapOfWellDefined_eq_weighted_sum hDX hR hgap_int
  have hnum :=
    stochastic_weighted_gap_sum_bound_theorem717_with_l1_floor
      (setup := setup) hDX hT hα_le_one
  have hL_ne : setup.L ≠ 0 := ne_of_gt setup.hL_pos
  have hm_ne : (setup.m : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos))
  have hnormalized :
      (setup.alphaSumOfWellDefined hDX)⁻¹ *
        (setup.f setup.x₁ - setup.fStar +
          setup.L * setup.barDX ^ 2 *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) +
          (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k)) =
        (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
          setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 /
            (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX) +
          setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
    have hsum_ne :
        Finset.sum (Finset.Icc 1 setup.N)
            (fun k => setup.αOfWellDefined hDX k) ≠ 0 := by
      simpa [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
        using ne_of_gt hR
    simp only [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
    field_simp [hsum_ne, hL_ne, hm_ne]
  exact
    expected_bound_of_weighted_sum_bound
      (setup.alphaSumOfWellDefined hDX)
      (setup.expectedWolfeGapOfWellDefined hDX hR)
      (Finset.sum (Finset.Icc 1 setup.N)
        (fun k =>
          setup.αOfWellDefined hDX k *
            ∫ ω,
              SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω)
                ∂setup.P))
      (setup.f setup.x₁ - setup.fStar +
        setup.L * setup.barDX ^ 2 *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
        (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m) +
        (setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)) *
          Finset.sum (Finset.Icc 1 setup.N)
            (fun k => setup.αOfWellDefined hDX k))
      ((setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
        setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
        (setup.N : ℝ) * setup.σ ^ 2 /
          (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX) +
        setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ))
      hR hexpand hnum hnormalized

/-! Alternative normalized Theorem 7.17 route obtained by Young-absorbing the
`α D̄_X E‖δ_k‖` term before applying Lemma 7.5. This removes the explicit L1
floor but doubles the variance contribution relative to the printed theorem. -/
theorem theorem_7_17_general_with_wellDefined_young_variance
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) :
    ∀ (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined),
      setup.paperAlphaOfWellDefined hDX ≤ 1 →
      setup.expectedWolfeGapOfWellDefined hDX
          (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
        (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
          setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 /
            (setup.L * setup.m * setup.alphaSumOfWellDefined hDX) := by
  intro hDX hT hα_le_one
  classical
  let hR : 0 < setup.alphaSumOfWellDefined hDX :=
    setup.alphaSum_pos_of_nonzeroDiameter hDX
  have hgap_int :
      ∀ R : setup.OutputTime,
        Integrable
          (fun ω =>
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX R.1 ω))
          setup.P := by
    intro R
    exact setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one R.1
  have hexpand :=
    setup.expectedWolfeGapOfWellDefined_eq_weighted_sum hDX hR hgap_int
  rw [hexpand]
  have hnum :=
    stochastic_weighted_gap_sum_bound_theorem717_young_variance
      (setup := setup) hDX hT hα_le_one
  have hinv_nonneg : 0 ≤ (setup.alphaSumOfWellDefined hDX)⁻¹ :=
    inv_nonneg.mpr (le_of_lt hR)
  have hL_ne : setup.L ≠ 0 := ne_of_gt setup.hL_pos
  have hm_ne : (setup.m : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos))
  calc
    (setup.alphaSumOfWellDefined hDX)⁻¹ *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k =>
            setup.αOfWellDefined hDX k *
              ∫ ω,
                SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω)
                  ∂setup.P)
        ≤
      (setup.alphaSumOfWellDefined hDX)⁻¹ *
        (setup.f setup.x₁ - setup.fStar +
          setup.L * setup.barDX ^ 2 *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 / (setup.L * setup.m)) :=
      mul_le_mul_of_nonneg_left hnum hinv_nonneg
    _ =
        (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
          setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 /
            (setup.L * setup.m * setup.alphaSumOfWellDefined hDX) := by
      have hsum_ne :
          Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k) ≠ 0 := by
        simpa [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
          using ne_of_gt hR
      simp only [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
      field_simp [hsum_ne, hL_ne, hm_ne]

/-- Well-defined exact printed right-hand side for Theorem 7.17, without the
additional normalized L1 floor exposed by the formal Lemma 7.5 route. -/
noncomputable def theorem_7_17_rhs_withWellDefined_printed
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) : ℝ :=
  SOptLib.ConditionalGradient.expectedWolfeGapUpperBoundWithEpochPenalty
    (setup.f setup.x₁) setup.fStar setup.L setup.barDX setup.σ
    (setup.alphaSumOfWellDefined hDX) setup.N setup.m
    (Finset.sum (Finset.Icc 1 setup.N)
      (fun k => setup.αOfWellDefined hDX k ^ 2))
    (Finset.sum (Finset.Icc 0 setup.S)
      (fun s =>
        Finset.sum (Finset.Icc 1 setup.T)
          (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
        setup.paperMaxInnerAlphaOfWellDefined hDX hT s))

/-- The exact printed Theorem 7.17 expected-gap claim in the well-defined
nonzero-diameter branch.

This predicate is the source-boundary object for the displayed theorem, kept
separate from the proved explicit-L1-floor replacement. A retirement proof for
the printed theorem must negate this predicate for a concrete feasible setup,
not merely show that the corrected RHS is larger than the printed RHS. -/
def theorem_7_17_printedClaimWithWellDefined
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) : Prop :=
  setup.expectedWolfeGapOfWellDefined hDX
      (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
    theorem_7_17_rhs_withWellDefined_printed setup hDX hT

/-- Setup-level strict violation predicate for the exact printed Theorem 7.17
claim. This is the remaining falsity-retirement target if the printed theorem
cannot be proved by a sharper source route. -/
def theorem_7_17_printedClaimViolatedWithWellDefined
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) : Prop :=
  theorem_7_17_rhs_withWellDefined_printed setup hDX hT <
    setup.expectedWolfeGapOfWellDefined hDX
      (setup.alphaSum_pos_of_nonzeroDiameter hDX)

/-- Exact printed Theorem 7.17 artifact in the nonzero-diameter well-defined
branch.

The proved source-faithful route available in this file is
`theorem_7_17_general_with_wellDefined_l1_floor`, which carries the explicit
Lemma 7.5 L1 floor. This declaration keeps the paper-facing printed theorem
head intact for the post-refactor proof/retirement decision instead of hiding
the missing step behind a strict-violation premise. -/
theorem theorem_7_17_general_with_wellDefined
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) :
    ∀ (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined),
      setup.paperAlphaOfWellDefined hDX ≤ 1 →
      theorem_7_17_printedClaimWithWellDefined setup hDX hT := by
  intro hDX hT hα_le_one
  classical
  let hR : 0 < setup.alphaSumOfWellDefined hDX :=
    setup.alphaSum_pos_of_nonzeroDiameter hDX
  have hgap_int :
      ∀ R : setup.OutputTime,
        Integrable
          (fun ω =>
            SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX R.1 ω))
          setup.P := by
    intro R
    exact setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one R.1
  have hexpand :=
    setup.expectedWolfeGapOfWellDefined_eq_weighted_sum hDX hR hgap_int
  dsimp [StochasticNonconvexConditionalGradient.theorem_7_17_printedClaimWithWellDefined]
  rw [hexpand]
  have hnum :=
    stochastic_weighted_gap_sum_bound_theorem717_printed
      (setup := setup) hDX hT hα_le_one
  have hinv_nonneg : 0 ≤ (setup.alphaSumOfWellDefined hDX)⁻¹ :=
    inv_nonneg.mpr (le_of_lt hR)
  have hL_ne : setup.L ≠ 0 := ne_of_gt setup.hL_pos
  have hm_ne : (setup.m : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt (Nat.lt_of_lt_of_le Nat.zero_lt_one setup.hm_pos))
  dsimp [theorem_7_17_rhs_withWellDefined_printed]
  calc
    (setup.alphaSumOfWellDefined hDX)⁻¹ *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k =>
            setup.αOfWellDefined hDX k *
              ∫ ω,
                SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω)
                  ∂setup.P)
        ≤
      (setup.alphaSumOfWellDefined hDX)⁻¹ *
        (setup.f setup.x₁ - setup.fStar +
          setup.L * setup.barDX ^ 2 *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 / (2 * setup.L * setup.m)) :=
      mul_le_mul_of_nonneg_left hnum hinv_nonneg
    _ =
        (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
          setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
            (3 / 2 *
              Finset.sum (Finset.Icc 1 setup.N)
                (fun k => setup.αOfWellDefined hDX k ^ 2) +
            Finset.sum (Finset.Icc 0 setup.S)
              (fun s =>
                Finset.sum (Finset.Icc 1 setup.T)
                  (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
                setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
          (setup.N : ℝ) * setup.σ ^ 2 /
            (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX) := by
      have hsum_ne :
          Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k) ≠ 0 := by
        simpa [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
          using ne_of_gt hR
      simp only [StochasticNonconvexConditionalGradientSetup.alphaSumOfWellDefined]
      field_simp [hsum_ne, hL_ne, hm_ne]

/-- Well-defined statement-corrected Theorem 7.17 right-hand side, carrying the
explicit normalized L1 floor from Lemma 7.5. -/
noncomputable def theorem_7_17_rhs_withWellDefined_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) : ℝ :=
  SOptLib.ConditionalGradient.expectedWolfeGapUpperBoundWithL1Floor
    (theorem_7_17_rhs_withWellDefined_printed setup hDX hT)
    setup.barDX setup.σ setup.m

/-- The statement-corrected Theorem 7.17 RHS is exactly the printed RHS plus
the explicit normalized L1 variance floor forced by Lemma 7.5. -/
theorem theorem_7_17_rhs_withWellDefined_l1_floor_eq_printed_add_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) :
    theorem_7_17_rhs_withWellDefined_l1_floor setup hDX hT =
      theorem_7_17_rhs_withWellDefined_printed setup hDX hT +
        setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
  rfl

/-- Route obstruction for deriving the exact printed Theorem 7.17 RHS from the
proved explicit-L1-floor replacement by monotonicity alone. -/
theorem theorem_7_17_general_with_wellDefined_l1_floor_not_le_printed_rhs
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hσ : 0 < setup.σ) :
    ¬ theorem_7_17_rhs_withWellDefined_l1_floor setup hDX hT ≤
      theorem_7_17_rhs_withWellDefined_printed setup hDX hT := by
  simpa [theorem_7_17_rhs_withWellDefined_l1_floor,
    theorem_7_17_rhs_withWellDefined_printed] using
    theorem717_l1_floor_rhs_not_le_printed_rhs
      (setup := setup) hDX hT hσ

/-- Retired exact printed Theorem 7.17 artifact.

This theorem-level artifact is source-faithful about the claim being retired:
it negates the actual printed expected-gap inequality for a setup whose
canonical Algorithm 7.13 expected Wolfe gap is strictly larger than the printed
right-hand side. The remaining reconstruct boundary is to instantiate such a
setup, or else prove the printed inequality directly by a sharper source route.
-/
theorem theorem_7_17_general_with_wellDefined_printedClaim_of_strict_violation
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hbad : theorem_7_17_printedClaimViolatedWithWellDefined setup hDX hT) :
    ¬ theorem_7_17_printedClaimWithWellDefined setup hDX hT := by
  exact not_le_of_gt hbad

/-! Domain-aware stochastic nonconvex variance-reduced conditional-gradient
bound with the explicit L1 floor. It adds the stochastic variance contribution
from Lemma 7.5 relative to `theorem_7_16`. This nondegenerate helper exposes
only `0 < D̄_X`, from which the output normalizer is derived internally. -/
theorem theorem_7_17_general_domainAware_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    setup.expectedWolfeGapOfWellDefined hDX
        (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
      (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
        setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
        Finset.sum (Finset.Icc 0 setup.S)
          (fun s =>
            Finset.sum (Finset.Icc 1 setup.T)
              (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
            setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
        (setup.N : ℝ) * setup.σ ^ 2 /
          (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX) +
        setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
  exact theorem_7_17_general_with_wellDefined_l1_floor
    (setup := setup) hDX hT hα_le_one

/-! Conditional right-hand side for Theorem 7.17 using the Algorithm 7.13
stepsize/normalizer and epoch maximum under the single conditional Algorithm
7.13 realization contract. The zero-diameter and empty-window conventions live
only in declarations named `Extension` or `degenerate`. -/
noncomputable def theorem_7_17_rhs_conditional
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract) : ℝ :=
  SOptLib.ConditionalGradient.expected_wolfe_gap_upper_bound_with_epoch_schedule
    (setup.f setup.x₁) setup.fStar setup.L setup.barDX setup.σ
    (setup.alphaSumConditional h) setup.N setup.m setup.S setup.T
    (setup.algorithmAlphaConditional h) setup.globalIndex
    (setup.paperMaxInnerAlphaOfWellDefined
      (setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h)
      (setup.innerAlphaWindowWellDefined_of_algorithm713Contract h))

/-! Statement-corrected conditional right-hand side for Theorem 7.17, carrying
the explicit normalized L1 floor from Lemma 7.5. -/
noncomputable def theorem_7_17_rhs_conditional_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract) : ℝ :=
  theorem_7_17_rhs_conditional setup h +
    setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)

/-- Raw displayed right-hand side for Theorem 7.17, written with the displayed
Algorithm 7.13 stepsize expression and raw normalizer. The theorem head does
not carry the non-source Algorithm 7.13 realization contract; the only explicit
semantic boundary is the nonempty maximum window needed to interpret
`max_{j=2,...,T}` as a genuine finite maximum. -/
noncomputable def theorem_7_17_rhsDisplayedExpression
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hT : setup.InnerAlphaWindowWellDefined) : ℝ :=
  (setup.f setup.x₁ - setup.fStar) / setup.alphaSumDisplayedExpression +
    setup.L * setup.barDX ^ 2 / setup.alphaSumDisplayedExpression *
      (3 / 2 *
        Finset.sum (Finset.Icc 1 setup.N)
          (fun k => setup.algorithmAlphaDisplayedExpression k ^ 2) +
      Finset.sum (Finset.Icc 0 setup.S)
        (fun s =>
          Finset.sum (Finset.Icc 1 setup.T)
            (fun j =>
          setup.algorithmAlphaDisplayedExpression (setup.globalIndex s j)) *
          setup.rawPaperMaxInnerAlpha hT s)) +
    (setup.N : ℝ) * setup.σ ^ 2 /
      (2 * setup.L * setup.m * setup.alphaSumDisplayedExpression)

/-! Statement-corrected displayed RHS for Theorem 7.17, carrying the explicit
normalized L1 floor from Lemma 7.5. -/
noncomputable def theorem_7_17_rhsDisplayedExpression_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hT : setup.InnerAlphaWindowWellDefined) : ℝ :=
  theorem_7_17_rhsDisplayedExpression setup hT +
    setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ)

/-! Conditional stochastic nonconvex variance-reduced conditional-gradient
bound corresponding to the statement-corrected Theorem 7.17 route, stated
against the Algorithm 7.13 expected gap realized under the single contract.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json
main_theorem.proof[2].math`: `"\mathbb{E}[\operatorname{gap}(x_R)] \le ... + \frac{N \sigma^2}{2 L m \sum_{k=1}^{N} \alpha_k}"`;
`algorithm_spec.output.math`: `"x_R, \quad \text{where } \operatorname{Prob}\{R = k\} = \frac{\alpha_k}{\sum_{k=1}^{N} \alpha_k}"`. -/
theorem theorem_7_17_general_conditional_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h ≤
      theorem_7_17_rhs_conditional_l1_floor setup h := by
  let hDX := setup.paperAlphaFormulaWellDefined_of_algorithm713Contract h
  let hT := setup.innerAlphaWindowWellDefined_of_algorithm713Contract h
  let hα_le_one := setup.paperAlpha_le_one_of_algorithm713Contract h
  rw [setup.paperExpectedWolfeGapConditional_eq_of_wellDefined h]
  simpa [theorem_7_17_rhs_conditional_l1_floor, theorem_7_17_rhs_conditional,
    StochasticNonconvexConditionalGradientSetup.alphaSumConditional_eq_of_wellDefined,
    StochasticNonconvexConditionalGradientSetup.algorithmAlphaConditional_eq_of_wellDefined]
    using theorem_7_17_general_with_wellDefined_l1_floor
      (setup := setup) hDX hT hα_le_one

/-! Conditional stochastic nonconvex variance-reduced conditional-gradient
bound under the single Algorithm 7.13 realization boundary, using the corrected
explicit-L1-floor RHS. This alias is kept out of the paper theorem name because
the boundary carries facts not asserted in the current JSON. -/
theorem theorem_7_17_general_conditional_alias_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h ≤
      theorem_7_17_rhs_conditional_l1_floor setup h := by
  simpa [StochasticNonconvexConditionalGradientSetup.paperExpectedWolfeGapConditional]
    using theorem_7_17_general_conditional_l1_floor (setup := setup) h

/-! Domain-aware Euclidean helper for the corrected Theorem 7.17 route. -/
theorem theorem_7_17_domainAware_l1_floor
    (n d : ℕ)
    (setup :
      StochasticNonconvexConditionalGradientSetup
        (PaperVariableSpace n) Ω (PaperSampleSpace d))
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    setup.expectedWolfeGapOfWellDefined hDX
        (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
      (setup.f setup.x₁ - setup.fStar) / setup.alphaSumOfWellDefined hDX +
        setup.L * setup.barDX ^ 2 / setup.alphaSumOfWellDefined hDX *
          (3 / 2 *
            Finset.sum (Finset.Icc 1 setup.N)
              (fun k => setup.αOfWellDefined hDX k ^ 2) +
          Finset.sum (Finset.Icc 0 setup.S)
            (fun s =>
              Finset.sum (Finset.Icc 1 setup.T)
                (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
              setup.paperMaxInnerAlphaOfWellDefined hDX hT s)) +
        (setup.N : ℝ) * setup.σ ^ 2 /
          (2 * setup.L * setup.m * setup.alphaSumOfWellDefined hDX) +
        setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
  simpa using theorem_7_17_general_domainAware_l1_floor
    (setup := setup) hDX hT hα_le_one

/-! Euclidean form of the corrected Theorem 7.17 route over `X ⊆ ℝ^n` and `ξ ∈ Ξ ⊆ ℝ^d`, stated
against the single Algorithm 7.13 run boundary.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json
main_theorem.proof[2].math`: `"\mathbb{E}[\operatorname{gap}(x_R)] \le ... + \frac{N \sigma^2}{2 L m \sum_{k=1}^{N} \alpha_k}"`. -/
theorem theorem_7_17_conditional_euclidean_l1_floor
    (n d : ℕ)
    (setup :
      StochasticNonconvexConditionalGradientSetup
        (PaperVariableSpace n) Ω (PaperSampleSpace d))
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h ≤
      theorem_7_17_rhs_conditional_l1_floor setup h := by
  simpa using theorem_7_17_general_conditional_alias_l1_floor
    (setup := setup) h

/-! Direct RHS inherited from the proved explicit-L1-floor Theorem 7.17 route.

This is the active replacement for the previously attempted coefficient-`5`
corollary compression. The scalar compression is not source-backed under the
current setup assumptions: `corollary712_l1_floor_scalar_countermodel` shows
that the initial-gap term can defeat the displayed `1 / sqrt N` normalization.
Keeping the Theorem 7.17 RHS explicit is the FILL-ready statement. -/
noncomputable def corollary_7_12_rhs_withWellDefined_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) : ℝ :=
  SOptLib.ConditionalGradient.expectedWolfeGapUpperBoundWithEpochPenaltyAndL1Floor
    (setup.f setup.x₁) setup.fStar setup.L setup.barDX setup.σ
    (setup.alphaSumOfWellDefined hDX) setup.N setup.m
    (Finset.sum (Finset.Icc 1 setup.N)
      (fun k => setup.αOfWellDefined hDX k ^ 2))
    (Finset.sum (Finset.Icc 0 setup.S)
      (fun s =>
        Finset.sum (Finset.Icc 1 setup.T)
          (fun j => setup.αOfWellDefined hDX (setup.globalIndex s j)) *
        setup.paperMaxInnerAlphaOfWellDefined hDX hT s))

/-- The direct corrected Corollary 7.12 RHS currently available is the
well-defined Theorem 7.17 printed RHS plus the explicit L1 floor, not the
coefficient-`4` printed corollary display. -/
theorem corollary_7_12_rhs_withWellDefined_l1_floor_eq_theorem717_printed_add_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined) :
    corollary_7_12_rhs_withWellDefined_l1_floor setup hDX hT =
      theorem_7_17_rhs_withWellDefined_printed setup hDX hT +
        setup.barDX * setup.σ / Real.sqrt (setup.m : ℝ) := by
  rfl

/-- Exact printed right-hand side of Corollary 7.12, with coefficient `4`.

This definition records the source display separately from the proved
explicit-floor replacement so retirement artifacts can refer to the real printed
claim rather than to a route-level scalar obstruction. -/
noncomputable def corollary_7_12_rhs_withWellDefined_printed
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ) : ℝ :=
  SOptLib.ConditionalGradient.expectedWolfeGapUpperBound
    (setup.f setup.x₁) setup.fStar setup.L setup.barDX setup.σ setup.N setup.m

/-- The exact printed Corollary 7.12 expected-gap claim with coefficient `4`.

This is the source-boundary predicate for the displayed corollary, separated
from the proved explicit-floor Theorem 7.17 RHS. -/
def corollary_7_12_printedClaimWithWellDefined
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) : Prop :=
  setup.expectedWolfeGapOfWellDefined hDX
      (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
    corollary_7_12_rhs_withWellDefined_printed setup

/-- Setup-level strict violation predicate for the exact printed Corollary 7.12
claim. -/
def corollary_7_12_printedClaimViolatedWithWellDefined
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) : Prop :=
  corollary_7_12_rhs_withWellDefined_printed setup <
    setup.expectedWolfeGapOfWellDefined hDX
      (setup.alphaSum_pos_of_nonzeroDiameter hDX)

/-- Scalar obstruction for deriving the printed Corollary 7.12 display directly
from the now-proved printed Theorem 7.17 RHS.

This is the source-boundary scalar gap left after Theorem 7.17 was proved:
substituting Eq. (7.4.15) gives the term `fGap / (N * alpha)`. Without an
additional source-backed normalization such as `L * D^2 ≤ 1`, this need not be
bounded by the displayed `fGap / sqrt N`. -/
theorem corollary_7_12_general_with_wellDefined_theorem717_scalar_obstruction :
    ∃ fGap L D α σ m N : ℝ,
      0 < fGap ∧ 0 < L ∧ 0 < D ∧ 0 ≤ σ ∧ 0 < m ∧ 0 < N ∧
      α ≤ 1 ∧
      α = Real.sqrt ((1 / N + σ ^ 2 / (L * m)) / (L * D ^ 2)) ∧
      ¬
        fGap / (N * α) + (7 / 2) * L * D ^ 2 * α +
            σ ^ 2 / (2 * L * m * α) ≤
          fGap / Real.sqrt N + 7 * L * D ^ 2 / (2 * Real.sqrt N) +
            4 * σ * D / Real.sqrt m :=
  exists_counterexample_variance_balanced_stepsize_no_floor_scalar_compression 4

/-! Internal direct corollary route using the canonical Algorithm 7.13 expected
gap. It deliberately does not claim the unsupported coefficient-`5` display;
the active RHS is exactly the proved `_l1_floor` Theorem 7.17 RHS. -/
theorem corollary_7_12_general_with_wellDefined_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    setup.expectedWolfeGapOfWellDefined hDX
        (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
      corollary_7_12_rhs_withWellDefined_l1_floor setup hDX hT := by
  simpa [corollary_7_12_rhs_withWellDefined_l1_floor] using
    theorem_7_17_general_with_wellDefined_l1_floor
      (setup := setup) hDX hT hα_le_one

/-- Scalar route obstruction for treating the exact printed Corollary 7.12
coefficient-`4` display as a routine compression of the explicit-L1-floor
Theorem 7.17 RHS. -/
theorem corollary_7_12_general_with_wellDefined_printed_scalar_obstruction :
    ∃ fGap L D α σ m N : ℝ,
      0 < fGap ∧ 0 < L ∧ 0 < D ∧ 0 ≤ σ ∧ 0 < m ∧ 0 < N ∧
      α ≤ 1 ∧
      α = Real.sqrt ((1 / N + σ ^ 2 / (L * m)) / (L * D ^ 2)) ∧
      ¬
        fGap / (N * α) + (7 / 2) * L * D ^ 2 * α +
            σ ^ 2 / (2 * L * m * α) + D * σ / Real.sqrt m ≤
          fGap / Real.sqrt N + 7 * L * D ^ 2 / (2 * Real.sqrt N) +
            4 * σ * D / Real.sqrt m := by
  exact exists_counterexample_variance_balanced_stepsize_scalar_compression 4

/-- Linear minimization oracle for the concrete one-dimensional witness over
`X = [0,1]`. -/
private noncomputable def corollary712ConcreteLMO :
    SOptLib.LinearMinimizationOracle ℝ (Set.Icc (0 : ℝ) 1) :=
  linearMinimizationOracle_Icc_real 0 1 (by norm_num)

/-- Deterministic one-dimensional SNCCG setup used to retire the printed
Corollary 7.12 coefficient-`4` claim. -/
private noncomputable def corollary712ConcreteSetup
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀] :
    StochasticNonconvexConditionalGradientSetup ℝ Ω₀ Ξ₀ where
  X := Set.Icc (0 : ℝ) 1
  x₁ := 1
  F := fun x _ => 100 * x
  gradf := fun _ => 100
  gradF := fun _ _ => 100
  L := 4
  σ := 0
  N := 4
  m := 4
  T_choice := 2
  b_choice := 2
  ξ := fun _ _ => default
  hξ_meas := by intro k; exact measurable_const
  lmo := corollary712ConcreteLMO
  P := Measure.dirac default
  hP := inferInstance
  hX_closed := isClosed_Icc
  hX_compact := isCompact_Icc
  hX_convex := convex_Icc (0 : ℝ) 1
  hx₁_mem := by norm_num
  hL_pos := by norm_num
  hσ_nonneg := by norm_num
  hN_pos := by norm_num
  hm_pos := by norm_num
  hm_eq_T_mul_b_choice := by norm_num
  hb_choice_eq_T := by norm_num
  hF_objective_wellDefined := by
    intro x hx
    simpa [SOptLib.objectiveWellDefined, SOptLib.objectiveKernel] using
      (integrable_const (100 * x : ℝ) : Integrable (fun _ : Ω₀ => 100 * x)
        (Measure.dirac default))
  hF_hasGradientAt_ae := by
    filter_upwards with ω x hx
    simpa using ((hasDerivAt_const_mul (100 : ℝ) (x := x)).hasGradientAt)
  hF_smooth_ae := by
    filter_upwards with ω x y hx hy
    simp
  hgradF_meas := by
    exact measurable_const
  hgradF_unbiased := by
    intro x hx
    constructor
    · simpa [SOptLib.oracleWellDefined, SOptLib.oracleKernel] using
        (integrable_const (100 : ℝ) : Integrable (fun _ : Ω₀ => (100 : ℝ))
          (Measure.dirac default))
    · simp [SOptLib.oracleMean, SOptLib.oracleKernel]
  hgradf_hasGradientAt := by
    intro x hx
    have hfun :
        SOptLib.objectiveExpectation (Measure.dirac default)
          (fun x _ => (100 : ℝ) * x) (fun _ : Ω₀ => (default : Ξ₀)) =
          fun x : ℝ => 100 * x := by
      funext z
      simp [SOptLib.objectiveExpectation, SOptLib.objectiveKernel]
    rw [hfun]
    simpa using ((hasDerivAt_const_mul (100 : ℝ) (x := x)).hasGradientAt)
  hgradF_variance_bound := by
    intro x hx
    constructor
    · simp
    · simp
  hξ_indep := by
    rw [iIndepFun_iff_measure_inter_preimage_eq_mul]
    intro S sets hsets
    by_cases hAll : ∀ i, i ∈ S → (default : Ξ₀) ∈ sets i
    · have hleft_mem :
          (default : Ω₀) ∈ ⋂ i ∈ S, (fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i := by
        simp only [Set.mem_iInter, Set.mem_preimage]
        intro i hi
        exact hAll i hi
      have hleft :
          Measure.dirac (default : Ω₀)
              (⋂ i ∈ S, (fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i) = 1 :=
        Measure.dirac_apply_of_mem hleft_mem
      have hright :
          (∏ i ∈ S,
            Measure.dirac (default : Ω₀)
              ((fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i)) = 1 := by
        refine Finset.prod_eq_one ?_
        intro i hi
        exact Measure.dirac_apply_of_mem (by simpa using hAll i hi)
      rw [hleft, hright]
    · push_neg at hAll
      rcases hAll with ⟨i, hiS, hi_not⟩
      have hleft_empty :
          (⋂ i ∈ S, (fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i) = ∅ := by
        ext u
        constructor
        · intro hu
          have hui : u ∈ (fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i :=
            (Set.mem_iInter.mp (Set.mem_iInter.mp hu i) hiS)
          exact False.elim (hi_not (by simpa using hui))
        · intro hu
          simp at hu
      have hfactor :
          Measure.dirac (default : Ω₀)
              ((fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i) = 0 := by
        rw [Measure.dirac_apply]
        simp [hi_not]
      have hright :
          (∏ j ∈ S,
            Measure.dirac (default : Ω₀)
              ((fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets j)) = 0 :=
        Finset.prod_eq_zero hiS hfactor
      calc
        Measure.dirac (default : Ω₀)
            (⋂ i ∈ S, (fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets i) = 0 := by
          rw [hleft_empty]
          simp
        _ = ∏ j ∈ S,
              Measure.dirac (default : Ω₀)
                ((fun _ : Ω₀ => (default : Ξ₀)) ⁻¹' sets j) := by
          rw [hright]
  hξ_ident := by
    intro k
    exact IdentDistrib.refl measurable_const.aemeasurable

private lemma corollary712Concrete_barDX_eq_one
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀] :
    (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).barDX = 1 := by
  classical
  let setup := corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)
  have hlower : (1 : ℝ) ≤ setup.barDX := by
    have hbound := setup.barDX_bound 1 0
      (by change (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1; norm_num)
      (by change (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1; norm_num)
    simpa [setup, corollary712ConcreteSetup, Real.norm_eq_abs] using hbound
  have hupper : setup.barDX ≤ 1 := by
    let p := setup.diameterPair
    have hp1 : (p.1 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := p.1.property
    have hp2 : (p.2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := p.2.property
    have habs : |(p.1 : ℝ) - (p.2 : ℝ)| ≤ 1 := by
      rw [abs_le]
      constructor <;> nlinarith [hp1.1, hp1.2, hp2.1, hp2.2]
    simpa [setup, StochasticNonconvexConditionalGradientSetup.barDX,
      corollary712ConcreteSetup, Real.norm_eq_abs] using habs
  exact le_antisymm hupper hlower

private lemma corollary712Concrete_alpha_eq_quarter
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀]
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).barDX) :
    (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).paperAlphaOfWellDefined hDX =
      (1 / 4 : ℝ) := by
  have hsqrt : Real.sqrt ((1 : ℝ) / 16) = 1 / 4 := by
    have hsq : (Real.sqrt ((1 : ℝ) / 16)) ^ 2 = (1 / 4 : ℝ) ^ 2 := by
      rw [Real.sq_sqrt] <;> norm_num
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp hsq with h | h
    · exact h
    · have hnonneg : 0 ≤ Real.sqrt ((1 : ℝ) / 16) := Real.sqrt_nonneg _
      nlinarith
  rw [StochasticNonconvexConditionalGradientSetup.paperAlphaOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.rawAlphaFormula,
    corollary712Concrete_barDX_eq_one (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)]
  norm_num [corollary712ConcreteSetup, hsqrt]

private lemma corollary712Concrete_alpha_le_one
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀]
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).barDX) :
    (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).paperAlphaOfWellDefined hDX ≤ 1 := by
  rw [corollary712Concrete_alpha_eq_quarter hDX]
  norm_num

private lemma corollary712Concrete_lmo_hundred :
    (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).linearMinimizer
        (100 : ℝ) = 0 := by
  change corollary712ConcreteLMO.toFun (100 : ℝ) = 0
  simp [corollary712ConcreteLMO, linearMinimizationOracle_Icc_real]

private lemma corollary712Concrete_lmo_hundred_toFun :
    (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).lmo.toFun
        (100 : ℝ) = 0 := by
  change corollary712ConcreteLMO.toFun (100 : ℝ) = 0
  simp [corollary712ConcreteLMO, linearMinimizationOracle_Icc_real]

private lemma corollary712Concrete_f_eq
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀] (x : ℝ) :
    (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).f x = 100 * x := by
  simp [StochasticNonconvexConditionalGradientSetup.f, SOptLib.objectiveExpectation,
    SOptLib.objectiveKernel, corollary712ConcreteSetup]

private lemma corollary712Concrete_fStar_eq_zero
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀] :
    (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).fStar = 0 := by
  classical
  let setup := corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)
  have hle : setup.fStar ≤ 0 := by
    have h := setup.fStar_lb 0 (by change (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1; norm_num)
    simpa [setup, corollary712Concrete_f_eq (Ω₀ := Ω₀) (Ξ₀ := Ξ₀) 0] using h
  have hge : 0 ≤ setup.fStar := by
    rw [setup.fStar_eq]
    have hx : 0 ≤ ((setup.objectiveMinimum : setup.X) : ℝ) :=
      (setup.objectiveMinimum : setup.X).property.1
    rw [corollary712Concrete_f_eq (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)]
    nlinarith
  exact le_antisymm hle hge

private lemma corollary712Concrete_alphaSum_eq_one
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀]
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).barDX) :
    (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)).alphaSumOfWellDefined hDX =
      1 := by
  let setup := corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)
  have hα : ∀ k, setup.αOfWellDefined hDX k = (1 / 4 : ℝ) := by
    intro k
    simpa [setup, StochasticNonconvexConditionalGradientSetup.αOfWellDefined]
      using corollary712Concrete_alpha_eq_quarter (Ω₀ := Ω₀) (Ξ₀ := Ξ₀) hDX
  change Finset.sum (Finset.Icc 1 4) (fun k => setup.αOfWellDefined hDX k) = 1
  norm_num [hα]

private lemma corollary712Concrete_rhs_printed_eq
    {Ω₀ Ξ₀ : Type*} [Unique Ω₀] [MeasurableSpace Ω₀]
    [MeasurableSpace Ξ₀] [Inhabited Ξ₀] :
    corollary_7_12_rhs_withWellDefined_printed
      (corollary712ConcreteSetup (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)) = 57 := by
  rw [corollary_7_12_rhs_withWellDefined_printed,
    corollary712Concrete_f_eq (Ω₀ := Ω₀) (Ξ₀ := Ξ₀),
    corollary712Concrete_fStar_eq_zero (Ω₀ := Ω₀) (Ξ₀ := Ξ₀),
    corollary712Concrete_barDX_eq_one (Ω₀ := Ω₀) (Ξ₀ := Ξ₀)]
  norm_num [corollary712ConcreteSetup]

private lemma corollary712Concrete_iter_one
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX)
    (ω : PUnit) :
    (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).iterProcessOfWellDefined
        hDX 1 ω = 1 := by
  cases ω
  rfl

private lemma corollary712Concrete_iter_two
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX)
    (ω : PUnit) :
    (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).iterProcessOfWellDefined
        hDX 2 ω = (3 / 4 : ℝ) := by
  cases ω
  have hbar :
      (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX = 1 :=
    corollary712Concrete_barDX_eq_one (Ω₀ := PUnit) (Ξ₀ := PUnit)
  simp only [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_zero,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_one,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_succ_succ,
    StochasticNonconvexConditionalGradientSetup.iterUpdateOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.linearMinimizer,
    StochasticNonconvexConditionalGradientSetup.αOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.paperAlphaOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.rawAlphaFormula]
  rw [hbar]
  norm_num [corollary712ConcreteSetup, corollary712ConcreteLMO,
    corollary712Concrete_lmo_hundred_toFun]

private lemma corollary712Concrete_iter_three
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX)
    (ω : PUnit) :
    (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).iterProcessOfWellDefined
        hDX 3 ω = (9 / 16 : ℝ) := by
  cases ω
  have hbar :
      (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX = 1 :=
    corollary712Concrete_barDX_eq_one (Ω₀ := PUnit) (Ξ₀ := PUnit)
  simp only [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_zero,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_one,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_succ_succ,
    StochasticNonconvexConditionalGradientSetup.iterUpdateOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.linearMinimizer,
    StochasticNonconvexConditionalGradientSetup.αOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.paperAlphaOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.rawAlphaFormula]
  rw [hbar]
  norm_num [corollary712ConcreteSetup, corollary712ConcreteLMO,
    corollary712Concrete_lmo_hundred_toFun]

private lemma corollary712Concrete_iter_four
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX)
    (ω : PUnit) :
    (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).iterProcessOfWellDefined
        hDX 4 ω = (27 / 64 : ℝ) := by
  cases ω
  have hbar :
      (corollary712ConcreteSetup (Ω₀ := PUnit) (Ξ₀ := PUnit)).barDX = 1 :=
    corollary712Concrete_barDX_eq_one (Ω₀ := PUnit) (Ξ₀ := PUnit)
  simp only [StochasticNonconvexConditionalGradientSetup.iterProcessOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_zero,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_one,
    StochasticNonconvexConditionalGradientSetup.processOfWellDefined_succ_succ,
    StochasticNonconvexConditionalGradientSetup.iterUpdateOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.linearMinimizer,
    StochasticNonconvexConditionalGradientSetup.αOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.paperAlphaOfWellDefined,
    StochasticNonconvexConditionalGradientSetup.rawAlphaFormula]
  rw [hbar]
  norm_num [corollary712ConcreteSetup, corollary712ConcreteLMO,
    corollary712Concrete_lmo_hundred_toFun]

private abbrev corollary712Unit : Type := PUnit

private lemma corollary712Concrete_gap_lower
    (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    100 * x ≤
      SOptLib.ConditionalGradient.wolfeGap (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).gradf (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).wolfeGapMaximizer x := by
  let setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit :=
    corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
  have hspec := setup.wolfeGap_spec x
    (⟨0, by change (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1; norm_num⟩ : setup.X)
  simpa [setup, corollary712ConcreteSetup, real_inner_eq_re_inner,
    RCLike.inner_apply, mul_comm] using hspec

private lemma corollary712Concrete_gap_integral_lower_one
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).barDX) :
    100 ≤ ∫ ω,
      SOptLib.ConditionalGradient.wolfeGap (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).gradf (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).wolfeGapMaximizer
        ((corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).iterProcessOfWellDefined
          hDX 1 ω) ∂(corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).P := by
  let setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit :=
    corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
  change 100 ≤ ∫ ω,
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
      (setup.iterProcessOfWellDefined hDX 1 ω) ∂setup.P
  have hconst :
      (fun ω : corollary712Unit =>
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
          (setup.iterProcessOfWellDefined hDX 1 ω)) =
        fun _ : corollary712Unit =>
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer 1 := by
    funext ω
    rw [corollary712Concrete_iter_one hDX ω]
  rw [hconst]
  simpa [setup, corollary712ConcreteSetup] using
    corollary712Concrete_gap_lower (x := 1) (by norm_num)

private lemma corollary712Concrete_gap_integral_lower_two
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).barDX) :
    75 ≤ ∫ ω,
      SOptLib.ConditionalGradient.wolfeGap (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).gradf (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).wolfeGapMaximizer
        ((corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).iterProcessOfWellDefined
          hDX 2 ω) ∂(corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).P := by
  let setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit :=
    corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
  change 75 ≤ ∫ ω,
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
      (setup.iterProcessOfWellDefined hDX 2 ω) ∂setup.P
  have hconst :
      (fun ω : corollary712Unit =>
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
          (setup.iterProcessOfWellDefined hDX 2 ω)) =
        fun _ : corollary712Unit =>
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (3 / 4 : ℝ) := by
    funext ω
    rw [corollary712Concrete_iter_two hDX ω]
  rw [hconst]
  have hgap := corollary712Concrete_gap_lower (x := (3 / 4 : ℝ)) (by norm_num)
  norm_num at hgap ⊢
  simpa [setup, corollary712ConcreteSetup] using hgap

private lemma corollary712Concrete_gap_integral_lower_three
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).barDX) :
    225 / 4 ≤ ∫ ω,
      SOptLib.ConditionalGradient.wolfeGap (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).gradf (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).wolfeGapMaximizer
        ((corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).iterProcessOfWellDefined
          hDX 3 ω) ∂(corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).P := by
  let setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit :=
    corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
  change 225 / 4 ≤ ∫ ω,
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
      (setup.iterProcessOfWellDefined hDX 3 ω) ∂setup.P
  have hconst :
      (fun ω : corollary712Unit =>
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
          (setup.iterProcessOfWellDefined hDX 3 ω)) =
        fun _ : corollary712Unit =>
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (9 / 16 : ℝ) := by
    funext ω
    rw [corollary712Concrete_iter_three hDX ω]
  rw [hconst]
  have hgap := corollary712Concrete_gap_lower (x := (9 / 16 : ℝ)) (by norm_num)
  norm_num at hgap ⊢
  simpa [setup, corollary712ConcreteSetup] using hgap

private lemma corollary712Concrete_gap_integral_lower_four
    (hDX : 0 < (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).barDX) :
    675 / 16 ≤ ∫ ω,
      SOptLib.ConditionalGradient.wolfeGap (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).gradf (corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit).wolfeGapMaximizer
        ((corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).iterProcessOfWellDefined
          hDX 4 ω) ∂(corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)).P := by
  let setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit :=
    corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
  change 675 / 16 ≤ ∫ ω,
    SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
      (setup.iterProcessOfWellDefined hDX 4 ω) ∂setup.P
  have hconst :
      (fun ω : corollary712Unit =>
        SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
          (setup.iterProcessOfWellDefined hDX 4 ω)) =
        fun _ : corollary712Unit =>
          SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer
            (27 / 64 : ℝ) := by
    funext ω
    rw [corollary712Concrete_iter_four hDX ω]
  rw [hconst]
  have hgap := corollary712Concrete_gap_lower (x := (27 / 64 : ℝ)) (by norm_num)
  norm_num at hgap ⊢
  simpa [setup, corollary712ConcreteSetup] using hgap

/-- Concrete setup-level falsity target for the exact printed Corollary 7.12.

Prior scalar obstruction lemmas only show that the current Theorem 7.17 route
does not imply the printed coefficient-`4` display. This declaration pins down
the next source-boundary obligation requested by the refactor audit: construct
the deterministic one-dimensional Algorithm 7.13 instance over `X = [0,1]`,
`N = m = 4`, `T = b = 2`, `L = 4`, `σ = 0`, with a linear stochastic objective,
and prove that its actual normalized output violates the printed corollary.

The theorem is deliberately an existential setup-level witness rather than a
same-claim wrapper: completing it requires filling the concrete setup fields and
unfolding the generated process/output law, not assuming a violation premise. -/
theorem corollary_7_12_printedClaimViolatedWithWellDefined_concreteWitness :
    ∃ setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit,
      setup.X = Set.Icc (0 : ℝ) 1 ∧
      setup.x₁ = 1 ∧
      setup.L = 4 ∧
      setup.σ = 0 ∧
      setup.N = 4 ∧
      setup.m = 4 ∧
      setup.T = 2 ∧
      setup.b = 2 ∧
      (∃ hDX : 0 < setup.barDX,
        setup.InnerAlphaWindowWellDefined ∧
        setup.paperAlphaOfWellDefined hDX ≤ 1 ∧
        corollary_7_12_printedClaimViolatedWithWellDefined setup hDX) := by
  let setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit :=
    corollary712ConcreteSetup (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
  refine ⟨setup, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_⟩
  have hDX : 0 < setup.barDX := by
    dsimp [setup]
    rw [corollary712Concrete_barDX_eq_one (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)]
    norm_num
  refine ⟨hDX, ?_, corollary712Concrete_alpha_le_one hDX, ?_⟩
  · change 2 ≤ setup.T
    norm_num [setup, StochasticNonconvexConditionalGradientSetup.T, corollary712ConcreteSetup]
  · dsimp [corollary_7_12_printedClaimViolatedWithWellDefined]
    have hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1 :=
      corollary712Concrete_alpha_le_one hDX
    have hR : 0 < setup.alphaSumOfWellDefined hDX :=
      setup.alphaSum_pos_of_nonzeroDiameter hDX
    have hgap_int :
        ∀ R : setup.OutputTime,
          Integrable
            (fun ω =>
              SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX R.1 ω))
            setup.P := by
      intro R
      exact setup.wolfeGap_iterProcessOfWellDefined_integrable hDX hα_le_one R.1
    have hexp :=
      setup.expectedWolfeGapOfWellDefined_eq_weighted_sum hDX hR hgap_int
    have hα : ∀ k, setup.αOfWellDefined hDX k = (1 / 4 : ℝ) := by
      intro k
      simpa [setup, StochasticNonconvexConditionalGradientSetup.αOfWellDefined]
        using corollary712Concrete_alpha_eq_quarter
          (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) hDX
    have h1 := corollary712Concrete_gap_integral_lower_one hDX
    have h2 := corollary712Concrete_gap_integral_lower_two hDX
    have h3 := corollary712Concrete_gap_integral_lower_three hDX
    have h4 := corollary712Concrete_gap_integral_lower_four hDX
    have h1_raw :
        100 ≤ ∫ ω,
          ⟪setup.gradf (setup.iterProcessOfWellDefined hDX 1 ω),
            setup.iterProcessOfWellDefined hDX 1 ω -
              (setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX 1 ω) : ℝ)⟫_ℝ
          ∂setup.P := by
      simpa [setup, SOptLib.ConditionalGradient.wolfeGap] using h1
    have h2_raw :
        75 ≤ ∫ ω,
          ⟪setup.gradf (setup.iterProcessOfWellDefined hDX 2 ω),
            setup.iterProcessOfWellDefined hDX 2 ω -
              (setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX 2 ω) : ℝ)⟫_ℝ
          ∂setup.P := by
      simpa [setup, SOptLib.ConditionalGradient.wolfeGap] using h2
    have h3_raw :
        225 / 4 ≤ ∫ ω,
          ⟪setup.gradf (setup.iterProcessOfWellDefined hDX 3 ω),
            setup.iterProcessOfWellDefined hDX 3 ω -
              (setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX 3 ω) : ℝ)⟫_ℝ
          ∂setup.P := by
      simpa [setup, SOptLib.ConditionalGradient.wolfeGap] using h3
    have h4_raw :
        675 / 16 ≤ ∫ ω,
          ⟪setup.gradf (setup.iterProcessOfWellDefined hDX 4 ω),
            setup.iterProcessOfWellDefined hDX 4 ω -
              (setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX 4 ω) : ℝ)⟫_ℝ
          ∂setup.P := by
      simpa [setup, SOptLib.ConditionalGradient.wolfeGap] using h4
    have hsum_lower :
        (4375 / 64 : ℝ) ≤
          Finset.sum (Finset.Icc 1 setup.N)
            (fun k =>
              setup.αOfWellDefined hDX k *
                ∫ ω,
                  SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω)
                  ∂setup.P) := by
      change (4375 / 64 : ℝ) ≤
        Finset.sum (Finset.Icc 1 4)
          (fun k =>
            setup.αOfWellDefined hDX k *
              ∫ ω,
                SOptLib.ConditionalGradient.wolfeGap setup.gradf setup.wolfeGapMaximizer (setup.iterProcessOfWellDefined hDX k ω)
                ∂setup.P)
      rw [Finset.sum_Icc_succ_top (by norm_num : 1 ≤ 3 + 1)]
      rw [Finset.sum_Icc_succ_top (by norm_num : 1 ≤ 2 + 1)]
      rw [Finset.sum_Icc_succ_top (by norm_num : 1 ≤ 1 + 1)]
      rw [Finset.sum_Icc_succ_top (by norm_num : 1 ≤ 0 + 1)]
      simp [hα]
      nlinarith [h1_raw, h2_raw, h3_raw, h4_raw]
    have hsum_lower_raw :
        (4375 / 64 : ℝ) ≤
          Finset.sum (Finset.Icc 1 setup.N)
            (fun k =>
              setup.αOfWellDefined hDX k *
                ∫ ω,
                  ⟪setup.gradf (setup.iterProcessOfWellDefined hDX k ω),
                    setup.iterProcessOfWellDefined hDX k ω -
                      (setup.wolfeGapMaximizer
                        (setup.iterProcessOfWellDefined hDX k ω) : ℝ)⟫_ℝ
                  ∂setup.P) := by
      simpa [SOptLib.ConditionalGradient.wolfeGap] using hsum_lower
    have hrhs : corollary_7_12_rhs_withWellDefined_printed setup = 57 := by
      simpa [setup] using
        corollary712Concrete_rhs_printed_eq (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit)
    have hsum_eq_one : setup.alphaSumOfWellDefined hDX = 1 := by
      simpa [setup] using
        corollary712Concrete_alphaSum_eq_one
          (Ω₀ := corollary712Unit) (Ξ₀ := corollary712Unit) hDX
    rw [hrhs, hexp, hsum_eq_one]
    norm_num
    nlinarith [hsum_lower_raw]

/-- Retired exact printed Corollary 7.12 artifact with coefficient `4`.

The printed universal claim is not a FILL target in the current object model:
the concrete deterministic one-dimensional Algorithm 7.13 setup violates the
displayed coefficient-`4` predicate. The source-faithful positive replacement is
`corollary_7_12_general_with_wellDefined_l1_floor`, which keeps the Lemma 7.5
L1 floor in the right-hand side. -/
theorem corollary_7_12_general_with_wellDefined :
    ¬
      (∀ (setup : StochasticNonconvexConditionalGradientSetup ℝ corollary712Unit corollary712Unit),
        ∀ (hDX : 0 < setup.barDX),
          setup.InnerAlphaWindowWellDefined →
            setup.paperAlphaOfWellDefined hDX ≤ 1 →
              corollary_7_12_printedClaimWithWellDefined setup hDX) := by
  intro hprinted
  rcases corollary_7_12_printedClaimViolatedWithWellDefined_concreteWitness with
    ⟨setup, _hX, _hx₁, _hL, _hσ, _hN, _hm, _hTval, _hb, hcert⟩
  rcases hcert with ⟨hDX, hT, hα_le_one, hbad⟩
  exact (not_le_of_gt hbad) (hprinted setup hDX hT hα_le_one)

/-- Retired exact printed Corollary 7.12 artifact.

The JSON/PDF display coefficient `4`. This theorem-level artifact negates the
actual printed expected-gap inequality when a setup-level violation of that
printed bound is supplied. The scalar obstruction above remains only evidence
that the existing explicit-floor proof route cannot discharge this premise by
routine algebra. -/
theorem corollary_7_12_general_with_wellDefined_printedClaim_of_strict_violation
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX)
    (hbad : corollary_7_12_printedClaimViolatedWithWellDefined setup hDX) :
    ¬ corollary_7_12_printedClaimWithWellDefined setup hDX := by
  exact not_le_of_gt hbad

/-! Retired coefficient-`5` compressed proposition from the earlier corrected
route. The active theorem keeps the Theorem 7.17 RHS explicit; this proposition
is not FILL-ready without an additional source-backed scalar normalization
premise. -/
theorem corollary_7_12_general_with_wellDefined_l1_floor_compressedClaim :
    ∃ fGap L D α σ m N : ℝ,
      0 < fGap ∧ 0 < L ∧ 0 < D ∧ 0 ≤ σ ∧ 0 < m ∧ 0 < N ∧
      α ≤ 1 ∧
      α = Real.sqrt ((1 / N + σ ^ 2 / (L * m)) / (L * D ^ 2)) ∧
      ¬
        fGap / (N * α) + (7 / 2) * L * D ^ 2 * α +
            σ ^ 2 / (2 * L * m * α) + D * σ / Real.sqrt m ≤
          fGap / Real.sqrt N + 7 * L * D ^ 2 / (2 * Real.sqrt N) +
            5 * σ * D / Real.sqrt m :=
  exists_counterexample_l1_floor_corollary_scalar_compression

/-! Corrected domain-aware Corollary 7.12 route, left in the direct form
obtained from the explicit-L1-floor Theorem 7.17. -/
theorem corollary_7_12_general_domainAware_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    setup.expectedWolfeGapOfWellDefined hDX
        (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
      corollary_7_12_rhs_withWellDefined_l1_floor setup hDX hT := by
  exact corollary_7_12_general_with_wellDefined_l1_floor
    (setup := setup) hDX hT hα_le_one

/-! Corrected conditional Corollary 7.12 using the single Algorithm 7.13
realization contract, stated against the direct explicit-L1-floor Theorem 7.17
RHS. -/
theorem corollary_7_12_general_conditional_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h ≤
      theorem_7_17_rhs_conditional_l1_floor setup h := by
  exact theorem_7_17_general_conditional_l1_floor (setup := setup) h

/-! Conditional Corollary 7.12 with the Algorithm 7.13 expected gap from the
normalized randomized output law, in the direct explicit-L1-floor RHS form. This
alias is not the paper theorem name, because its realization contract carries
non-source facts.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json
main_theorem.statement_math` displays the coefficient `4`; retaining the Lemma
7.5 L1 floor gives the direct `_l1_floor` RHS above. -/
theorem corollary_7_12_general_conditional_alias_l1_floor
    (setup : StochasticNonconvexConditionalGradientSetup E Ω Ξ)
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h ≤
      theorem_7_17_rhs_conditional_l1_floor setup h := by
  simpa [StochasticNonconvexConditionalGradientSetup.paperExpectedWolfeGapConditional]
    using corollary_7_12_general_conditional_l1_floor (setup := setup) h

/-! Corrected domain-aware Euclidean helper for Corollary 7.12 over `X ⊆ ℝ^n`
and `ξ ∈ Ξ ⊆ ℝ^d`, in the direct explicit-L1-floor RHS form. -/
theorem corollary_7_12_domainAware_l1_floor
    (n d : ℕ)
    (setup :
      StochasticNonconvexConditionalGradientSetup
        (PaperVariableSpace n) Ω (PaperSampleSpace d))
    (hDX : 0 < setup.barDX) (hT : setup.InnerAlphaWindowWellDefined)
    (hα_le_one : setup.paperAlphaOfWellDefined hDX ≤ 1) :
    setup.expectedWolfeGapOfWellDefined hDX
        (setup.alphaSum_pos_of_nonzeroDiameter hDX) ≤
      corollary_7_12_rhs_withWellDefined_l1_floor setup hDX hT := by
  simpa using
    corollary_7_12_general_domainAware_l1_floor
      (setup := setup) hDX hT hα_le_one

/-! Corrected Euclidean form of Corollary 7.12 over `X ⊆ ℝ^n` and `ξ ∈ Ξ ⊆ ℝ^d`.

Book citation: `/root/SGD/SGD_challengeB_lanli/book/FOML/StochasticNonconvexConditionalGradient.json
main_theorem.statement_math` displays the coefficient `4`; retaining the Lemma
7.5 L1 floor gives the direct `_l1_floor` RHS above. -/
theorem corollary_7_12_conditional_euclidean_l1_floor
    (n d : ℕ)
    (setup :
      StochasticNonconvexConditionalGradientSetup
        (PaperVariableSpace n) Ω (PaperSampleSpace d))
    (h : setup.Algorithm713RealizationContract) :
    setup.paperExpectedWolfeGapConditional h ≤
      theorem_7_17_rhs_conditional_l1_floor setup h := by
  simpa using
    corollary_7_12_general_conditional_alias_l1_floor (setup := setup) h

end StochasticNonconvexConditionalGradient
