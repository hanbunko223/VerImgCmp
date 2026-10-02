import CircuitCorrectness.R1CS
import Mathlib.Tactic.LinearCombination

namespace CircuitCorrectness.Gadgets

theorem boolean_soundness (b : F) (h : b * (1 - b) = 0) : b = 0 ∨ b = 1 := by
  rcases mul_eq_zero.mp h with hb | hb
  · exact Or.inl hb
  · exact Or.inr (by linear_combination -hb)

theorem boolean_completeness (b : F) (h : b = 0 ∨ b = 1) : b * (1 - b) = 0 := by
  rcases h with rfl | rfl <;> norm_num

theorem linear_output (input output : F) (h : (input - output) * 1 = 0) :
    output = input := by linear_combination -h

theorem quintic_soundness (x pre post l2 l4 l5 : F)
    (h2 : (x + pre) * (x + pre) = l2)
    (h4 : l2 * l2 = l4) (h5 : l4 * (x + pre) = l5 - post) :
    l5 = (x + pre)^5 + post := by
  rw [← h2] at h4
  rw [← h4] at h5
  linear_combination -h5

theorem quintic_completeness (x pre post : F) :
    let l2 := (x + pre)^2
    let l4 := (x + pre)^4
    let l5 := (x + pre)^5 + post
    (x + pre) * (x + pre) = l2 ∧ l2 * l2 = l4 ∧
      l4 * (x + pre) = l5 - post := by
  dsimp
  constructor
  · ring
  constructor <;> ring

theorem fused_horner (a r next coefficient : F)
    (h : a * r = next - coefficient) : next = a * r + coefficient := by
  linear_combination -h

def horner (a r : F) (coefficients : List F) : F :=
  coefficients.foldl (fun acc c => acc * r + c) a

theorem horner_append (a r : F) (xs ys : List F) :
    horner a r (xs ++ ys) = horner (horner a r xs) r ys := by
  simp [horner, List.foldl_append]

/-- Generic consequence only; instantiating it for the exported step needs its soundness proof. -/
theorem connected_execution {S I : Type} (step : I → S → S)
    (relation : I → S → S → Prop)
    (sound : ∀ i s o, relation i s o → o = step i s)
    (images : Nat → I) (states : Nat → S)
    (links : ∀ n, relation (images n) (states n) (states (n+1))) :
    ∀ n, states n = (List.range n).foldl (fun s j => step (images j) s) (states 0) := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sound _ _ _ (links n), ih, List.range_succ, List.foldl_append]
    rfl

end CircuitCorrectness.Gadgets
