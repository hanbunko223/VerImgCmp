import CircuitCorrectness.Gadgets
import Mathlib.Tactic.Linarith
import Mathlib.Algebra.BigOperators.Ring.Finset

namespace CircuitCorrectness.Arithmetic
open scoped BigOperators

abbrev Matrix8 (R : Type) := Fin 8 → Fin 8 → R

def firstStage (a x : Matrix8 Int) (r k : Fin 8) : Int :=
  ∑ i, a r i * (x i k - 128)

def secondStage (a x : Matrix8 Int) (r c : Fin 8) : Int :=
  ∑ k, firstStage a x r k * a c k

def directTransform (a x : Matrix8 Int) (r c : Fin 8) : Int :=
  ∑ i, ∑ k, a r i * (x i k - 128) * a c k

/-- The second-stage row is `a c k`, implementing Aᵀ on the right. -/
theorem two_stages_eq_direct (a x : Matrix8 Int) (r c : Fin 8) :
    secondStage a x r c = directTransform a x r c := by
  simp only [secondStage, firstStage, directTransform, Finset.sum_mul]
  exact Finset.sum_comm

theorem cast_firstStage (a x : Matrix8 Int) (r k : Fin 8) :
    (firstStage a x r k : F) =
      ∑ i, (a r i : F) * ((x i k : F) - 128) := by
  simp [firstStage]

theorem cast_secondStage (a x : Matrix8 Int) (r c : Fin 8) :
    (secondStage a x r c : F) =
      ∑ k, (firstStage a x r k : F) * (a c k : F) := by
  simp [secondStage]

theorem cast_coefficient (m y : Int) : ((m*y : Int) : F) = (m : F) * (y : F) := by
  simp

/-- Each digit bound is a mathematical integer bound, not a field inequality. -/
theorem radix_packing_bound (base : Nat) (hb : 0 < base) (digits : Nat → Nat) :
    ∀ n, (∀ i < n, digits i < base) →
      (∑ i ∈ Finset.range n, digits i * base^i) < base^n := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    intro h
    have hlow := ih (fun i hi => h i (Nat.lt_succ_of_lt hi))
    have hlast := h n (Nat.lt_succ_self n)
    have hpos : 0 < base^n := pow_pos hb n
    rw [Finset.sum_range_succ, pow_succ]
    nlinarith

theorem pixel_packing_bound (r g b : Nat) (hr : r < 256) (hg : g < 256) (hb : b < 256) :
    r + 2^8*g + 2^16*b < 2^24 := by omega

theorem chunk_packing_bound (pixels : Nat → Nat) (h : ∀ i < 10, pixels i < 2^24) :
    (∑ i ∈ Finset.range 10, pixels i * (2^24)^i) < 2^240 := by
  have hh := radix_packing_bound (2^24) (by decide) pixels 10 h
  simpa [← pow_mul] using hh

end CircuitCorrectness.Arithmetic
