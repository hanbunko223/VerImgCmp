import CircuitCorrectness.R1CS

namespace CircuitCorrectness.Example

/-- Wires: 0=one, 1=x, 2=y, 3=sum, 4=product, 5=output. -/
def sumRow : Row := ⟨[(1, 1), (2, 1)], [(0, 1)], [(3, 1)]⟩
def productRow : Row := ⟨[(3, 1)], [(1, 1)], [(4, 1)]⟩
def outputRow : Row := ⟨[(4, 1), (0, 3)], [(0, 1)], [(5, 1)]⟩
def circuit : R1CS := ⟨6, 0, [sumRow, productRow, outputRow]⟩
def spec (x y : F) : F := (x + y) * x + 3

theorem soundness (w : Assignment) (h : circuit.Sat w) :
    w 5 = spec (w 1) (w 2) := by
  have hone : w 0 = 1 := h.1
  have hs := h.2 sumRow (by simp [circuit])
  have hp := h.2 productRow (by simp [circuit])
  have ho := h.2 outputRow (by simp [circuit])
  simp only [Row.Sat, evalLC, sumRow, productRow, outputRow,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Nat.cast_one, Nat.cast_ofNat, one_mul, add_zero, hone, mul_one] at hs hp ho
  simp only [spec]
  rw [← hs] at hp
  rw [← hp] at ho
  exact ho.symm

def witness (x y : F) : Assignment
  | 0 => 1
  | 1 => x
  | 2 => y
  | 3 => x + y
  | 4 => (x + y) * x
  | 5 => spec x y
  | _ => 0

theorem completeness (x y : F) : circuit.Sat (witness x y) := by
  constructor
  · rfl
  · intro row hr
    simp only [circuit, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl <;>
      simp [Row.Sat, evalLC, sumRow, productRow, outputRow, witness, spec]

theorem determinism (w v : Assignment) (hw : circuit.Sat w) (hv : circuit.Sat v)
    (hx : w 1 = v 1) (hy : w 2 = v 2) : w 5 = v 5 := by
  rw [soundness w hw, soundness v hv, hx, hy]

def brokenCircuit : R1CS := ⟨6, 0, [sumRow, productRow]⟩
def malicious : Assignment := fun i => if i = 5 then 4 else witness 0 0 i

/-- Removing the output row really admits the wrong public output. -/
theorem missing_output_constraint_counterexample :
    brokenCircuit.Sat malicious ∧ malicious 5 ≠ spec (malicious 1) (malicious 2) := by
  constructor
  · constructor
    · norm_num [malicious, brokenCircuit, witness]
    · intro row hr
      simp only [brokenCircuit, List.mem_cons, List.not_mem_nil, or_false] at hr
      rcases hr with rfl | rfl <;>
        norm_num [Row.Sat, evalLC, sumRow, productRow, malicious, witness]
  · change (4 : F) ≠ 3
    decide

end CircuitCorrectness.Example
