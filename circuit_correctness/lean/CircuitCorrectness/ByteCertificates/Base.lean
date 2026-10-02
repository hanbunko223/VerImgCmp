import CircuitCorrectness.Byte
import CircuitCorrectness.Exported

namespace CircuitCorrectness.ConcreteBytes
open scoped BigOperators

def valueWire (n : Nat) : Nat := 4 + 9 * n
def bitWire (n : Nat) (i : Fin 8) : Nat := valueWire n + 1 + i.val

/-- Exporter-sorted row at a zero-based byte-prefix row index. -/
def byteRow (j : Nat) : Row :=
  let n := j / 9
  let k := j % 9
  let bit := valueWire n + 1 + k
  if k < 8 then
    ⟨[(bit, modulus - 1), (97634, 1)], [(bit, 1)], []⟩
  else
    ⟨(valueWire n, modulus - 1) :: List.ofFn (fun i : Fin 8 => (bitWire n i, 2 ^ i.val)),
      [(97634, 1)], []⟩

theorem row_bit (n : Nat) (i : Fin 8) :
    byteRow (9 * n + i.val) =
      ⟨[(bitWire n i, modulus - 1), (97634, 1)], [(bitWire n i, 1)], []⟩ := by
  have hi : i.val < 9 := by omega
  simp [byteRow, Nat.add_div, Nat.add_mod, Nat.mod_eq_of_lt hi,
    Nat.div_eq_of_lt hi, i.isLt, bitWire, Nat.not_le.mpr hi]

theorem row_recompose (n : Nat) :
    byteRow (9 * n + 8) =
      ⟨(valueWire n, modulus - 1) :: List.ofFn (fun i : Fin 8 => (bitWire n i, 2 ^ i.val)),
        [(97634, 1)], []⟩ := by
  simp [byteRow, Nat.add_div, Nat.add_mod]

theorem row_bit_sat (n : Nat) (i : Fin 8) (w : Assignment) :
    (byteRow (9 * n + i.val)).Sat w ↔ (Byte.bitRow 97634 (bitWire n i)).Sat w := by
  rw [row_bit]
  simp [Row.Sat, Byte.bitRow, evalLC, add_comm]

theorem row_recompose_sat (n : Nat) (w : Assignment) :
    (byteRow (9 * n + 8)).Sat w ↔
      (Byte.recomposeRow 97634 (valueWire n) (bitWire n)).Sat w := by
  rw [row_recompose]
  simp only [Row.Sat, Byte.recomposeRow, evalLC, List.map_cons, List.map_append,
    List.sum_cons, List.sum_append, List.map_nil, List.sum_nil, add_zero]
  rw [add_comm ((↑(modulus - 1) : F) * w (valueWire n))]

/-- The sorted export shape is semantically identical to all nine generic rows. -/
theorem nine_rows_iff (n : Nat) (w : Assignment) :
    (∀ k < 9, (byteRow (9 * n + k)).Sat w) ↔
    (∀ row ∈ Byte.rows 97634 (valueWire n) (bitWire n), row.Sat w) := by
  constructor
  · intro hs row hr
    rcases List.mem_append.mp hr with hr | hr
    · obtain ⟨i, rfl⟩ := List.mem_ofFn.mp hr
      exact (row_bit_sat n i w).mp (hs i.val (by omega))
    · have he : row = Byte.recomposeRow 97634 (valueWire n) (bitWire n) := by simpa using hr
      rw [he]
      exact (row_recompose_sat n w).mp (hs 8 (by decide))
  · intro hs k hk
    by_cases h : k < 8
    · apply (row_bit_sat n ⟨k,h⟩ w).mpr
      apply hs
      apply List.mem_append_left
      exact List.mem_ofFn.mpr ⟨⟨k,h⟩,rfl⟩
    · have : k = 8 := by omega
      subst k
      apply (row_recompose_sat n w).mpr
      apply hs
      exact List.mem_append_right _ (by simp)

end CircuitCorrectness.ConcreteBytes
