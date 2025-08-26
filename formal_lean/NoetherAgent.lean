import Mathlib
open Real

/-
Lemma: For the harmonic oscillator x'' + ω^2 x = 0 with ω > 0,
the energy E(t) = (1/2) (x')^2 + (1/2) ω^2 (x)^2 has derivative 0.
We work in a classical calculus-on-ℝ setting with differentiable x.
-/

variable {x : ℝ → ℝ} {ω : ℝ}

theorem energy_invariant
    (hω : 0 < ω)
    (hx2 : ContMDiff ℝ ℝ 2 x)
    (ode : ∀ t, deriv (deriv x) t + ω^2 * x t = 0) :
    ∀ t, deriv (fun t => (deriv x t)^2 / 2 + (ω^2) * (x t)^2 / 2) t = 0 := by
  intro t
  have hx1 : Differentiable ℝ x := (hx2.differentiable le_rfl)
  have hx1' : Differentiable ℝ fun t => deriv x t := by
    simpa using (hx2.differentiableDeriv le_rfl)
  -- d/dt [ (x')^2/2 ] = x' * x''
  have h1 : deriv (fun t => (deriv x t)^2 / 2) t = (deriv x t) * (deriv (deriv x) t) := by
    simpa using
      deriv_mul (hx1'.differentiableAt) (hx1'.differentiableAt)
  -- d/dt [ (ω^2 x^2)/2 ] = ω^2 x x'
  have h2 : deriv (fun t => (ω^2) * (x t)^2 / 2) t = (ω^2) * (x t) * (deriv x t) := by
    have := (deriv_mul_const (fun t => (x t)^2) ((ω^2)/2)).trans ?_
    · simpa [mul_comm, mul_left_comm, mul_assoc, two_mul, mul_div_cancel'] using this
    · have hx2' : Differentiable ℝ fun t => (x t)^2 :=
        (hx1.mul hx1).differentiable
      simpa using deriv_mul_const (fun t => (x t)^2) (ω^2 / 2)
  -- ODE: x'' = - ω^2 x
  have hode : deriv (deriv x) t = - (ω^2) * (x t) := by
    simpa using congrArg id (by have := ode t; simpa [sub_eq_add_neg] using this)
  -- combine: x' * x'' + ω^2 x x' = 0
  have : (deriv x t) * (deriv (deriv x) t) + (ω^2) * (x t) * (deriv x t) = 0 := by
    simpa [hode, add_comm, add_left_neg_self, mul_comm, mul_left_comm, mul_assoc]
  -- derivative of energy is zero:
  simpa [h1, h2, add_comm] 
