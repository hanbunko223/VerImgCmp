import CircuitCorrectness.Field

namespace CircuitCorrectness

/-- Canonical field coefficients are natural numbers in the exported data. -/
abbrev LinearCombination := List (Nat × Nat)
abbrev Assignment := Nat → F

def evalLC (w : Assignment) (lc : LinearCombination) : F :=
  (lc.map fun (i, c) => (c : F) * w i).sum

structure Row where
  a : LinearCombination
  b : LinearCombination
  c : LinearCombination
  deriving DecidableEq, Repr

def Row.Sat (row : Row) (w : Assignment) : Prop :=
  evalLC w row.a * evalLC w row.b = evalLC w row.c

structure R1CS where
  wireCount : Nat
  one : Nat
  rows : List Row

def R1CS.WellFormed (cs : R1CS) : Prop :=
  cs.one < cs.wireCount ∧
  ∀ row ∈ cs.rows, ∀ lc ∈ [row.a, row.b, row.c],
    ∀ t ∈ lc, t.1 < cs.wireCount ∧ t.2 < modulus

/-- `one` is explicitly constrained; it is not silently supplied by evaluation. -/
def R1CS.Sat (cs : R1CS) (w : Assignment) : Prop :=
  w cs.one = 1 ∧ ∀ row ∈ cs.rows, row.Sat w

theorem R1CS.row_satisfied {cs : R1CS} {w : Assignment}
    (h : cs.Sat w) {row : Row} (hm : row ∈ cs.rows) : row.Sat w := h.2 row hm

theorem evalLC_congr {w v : Assignment} (lc : LinearCombination)
    (h : ∀ t ∈ lc, w t.1 = v t.1) : evalLC w lc = evalLC v lc := by
  unfold evalLC
  congr 1
  apply List.map_congr_left
  intro t ht
  simp only [h t ht]

end CircuitCorrectness
