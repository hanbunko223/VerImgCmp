import CircuitCorrectness.Arithmetic

/-! Byte-range constraints matching `AllocatedBit::alloc` and
`allocate_scalar_with_byte_range_check`. Row ordering and indices are parameters;
no allocation names or witness-generation claims are assumed. -/
set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

namespace CircuitCorrectness.Byte
open scoped BigOperators

/-- One Boolean constraint: `(one - bit) * bit = 0`. -/
def bitRow (one bit : Nat) : Row :=
  ⟨[(one, 1), (bit, modulus - 1)], [(bit, 1)], []⟩

/-- Recompose eight least-significant-first bits into the byte value. -/
def recomposeRow (one value : Nat) (bits : Fin 8 → Nat) : Row :=
  ⟨List.ofFn (fun i => (bits i, 2 ^ i.val)) ++ [(value, modulus - 1)],
   [(one, 1)], []⟩

def rows (one value : Nat) (bits : Fin 8 → Nat) : List Row :=
  List.ofFn (fun i => bitRow one (bits i)) ++ [recomposeRow one value bits]

/-- Exactly eight Boolean equations and one recomposition equation. -/
theorem rows_length (one value : Nat) (bits : Fin 8 → Nat) :
    (rows one value bits).length = 9 := by simp [rows]

theorem cast_modulus_sub_one : ((modulus - 1 : Nat) : F) = -1 := by
  rw [Nat.cast_sub (by decide : 1 ≤ modulus)]
  simp [F]

theorem bitRow_iff (one bit : Nat) (w : Assignment) (ho : w one = 1) :
    (bitRow one bit).Sat w ↔ w bit = 0 ∨ w bit = 1 := by
  simp only [bitRow, Row.Sat, evalLC, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.cast_one, one_mul, cast_modulus_sub_one,
    neg_one_mul, add_zero, ho]
  rw [mul_eq_zero]
  constructor
  · intro h
    rcases h with h | h
    · right; linear_combination -h
    · exact Or.inl h
  · rintro (h | h) <;> simp [h]

theorem recomposeRow_iff (one value : Nat) (bits : Fin 8 → Nat)
    (w : Assignment) (ho : w one = 1) :
    (recomposeRow one value bits).Sat w ↔
      w value = ∑ i, (2 : F)^i.val * w (bits i) := by
  simp only [recomposeRow, Row.Sat, evalLC, List.map_append, List.sum_append,
    List.map_ofFn, List.sum_ofFn, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Nat.cast_one, cast_modulus_sub_one, neg_one_mul,
    add_zero, ho, mul_one, Function.comp_apply, Nat.cast_pow, Nat.cast_ofNat]
  constructor <;> intro h <;> linear_combination -h

theorem rows_sat_iff (one value : Nat) (bits : Fin 8 → Nat)
    (w : Assignment) (ho : w one = 1) :
    (∀ row ∈ rows one value bits, row.Sat w) ↔
      (∀ i, w (bits i) = 0 ∨ w (bits i) = 1) ∧
      w value = ∑ i, (2 : F)^i.val * w (bits i) := by
  constructor
  · intro hs
    constructor
    · intro i
      apply (bitRow_iff one (bits i) w ho).mp
      apply hs
      apply List.mem_append_left
      exact List.mem_ofFn.mpr ⟨i, rfl⟩
    · apply (recomposeRow_iff one value bits w ho).mp
      apply hs
      exact List.mem_append_right _ (by simp)
  · rintro ⟨hb, hv⟩ row hr
    rcases List.mem_append.mp hr with hr | hr
    · obtain ⟨i, rfl⟩ := List.mem_ofFn.mp hr
      exact (bitRow_iff one (bits i) w ho).mpr (hb i)
    · have he : row = recomposeRow one value bits := by simpa using hr
      rw [he]
      exact (recomposeRow_iff one value bits w ho).mpr hv

/-- The natural number encoded by an arbitrary Boolean vector. -/
def natValue (b : Fin 8 → Bool) : Nat := ∑ i, 2 ^ i.val * (b i).toNat

theorem natValue_lt (b : Fin 8 → Bool) : natValue b < 256 := by
  have h : natValue b ≤ ∑ i : Fin 8, 2 ^ i.val := by
    apply Finset.sum_le_sum
    intro i _
    simpa using Nat.mul_le_mul_left (2 ^ i.val) (Bool.toNat_le (b i))
  have : (∑ i : Fin 8, 2 ^ i.val) = 255 := by decide
  omega

theorem cast_natValue (b : Fin 8 → Bool) :
    (natValue b : F) = ∑ i, (2 : F)^i.val * ((b i).toNat : F) := by
  simp [natValue]

/-- Kernel-evaluated finite bit identity; no native evaluator is trusted. -/
theorem testBit_recompose_fin : ∀ n : Fin 256,
    natValue (fun i => n.val.testBit i.val) = n.val := by decide

theorem testBit_recompose (n : Nat) (hn : n < 256) :
    natValue (fun i => n.testBit i.val) = n := testBit_recompose_fin ⟨n, hn⟩

theorem testBit_field_recompose (n : Nat) (hn : n < 256) :
    (n : F) = ∑ i : Fin 8, (2 : F)^i.val * ((n.testBit i.val).toNat : F) := by
  rw [← cast_natValue, testBit_recompose n hn]

theorem bool_cast_cases (b : Bool) : (b.toNat : F) = 0 ∨ (b.toNat : F) = 1 := by
  cases b <;> simp

/-- Global witnesses can use this version without constructing a local override.
Only the values on the ten referenced wires matter. -/
theorem complete_of_values (one value : Nat) (bits : Fin 8 → Nat)
    (w : Assignment) (n : Nat) (hn : n < 256) (ho : w one = 1)
    (hv : w value = (n : F))
    (hb : ∀ i, w (bits i) = ((n.testBit i.val).toNat : F)) :
    ∀ row ∈ rows one value bits, row.Sat w := by
  apply (rows_sat_iff one value bits w ho).mpr
  constructor
  · intro i
    rw [hb i]
    exact bool_cast_cases _
  · simp only [hv, hb]
    exact testBit_field_recompose n hn

/-- Every satisfying assignment has a genuine natural-byte value. -/
theorem soundness (one value : Nat) (bits : Fin 8 → Nat)
    (w : Assignment) (ho : w one = 1)
    (hs : ∀ row ∈ rows one value bits, row.Sat w) :
    ∃ n : Nat, n < 256 ∧ w value = (n : F) := by
  classical
  obtain ⟨hb, hv⟩ := (rows_sat_iff one value bits w ho).mp hs
  let b : Fin 8 → Bool := fun i => decide (w (bits i) = 1)
  have he : ∀ i, ((b i).toNat : F) = w (bits i) := by
    intro i
    rcases hb i with h | h <;> simp [b, h]
  refine ⟨natValue b, natValue_lt b, ?_⟩
  rw [cast_natValue, hv]
  congr 1
  funext i
  rw [he i]

/-- Conditions needed to freely assign a byte's ten wires (one, value, bits). -/
structure Nonalias (one value : Nat) (bits : Fin 8 → Nat) : Prop where
  one_ne_value : one ≠ value
  bits_injective : Function.Injective bits
  bits_ne_one : ∀ i, bits i ≠ one
  bits_ne_value : ∀ i, bits i ≠ value

/-- A mathematical completion of any background assignment. It changes only
this gadget's one, value, and bit wires. -/
noncomputable def witness (one value : Nat) (bits : Fin 8 → Nat)
    (n : Nat) (base : Assignment) : Assignment := fun j =>
  if j = one then 1 else if j = value then (n : F) else
    if h : ∃ i, bits i = j then ((n.testBit h.choose.val).toNat : F) else base j

theorem witness_one (one value : Nat) (bits : Fin 8 → Nat)
    (n : Nat) (base : Assignment) : witness one value bits n base one = 1 := by
  simp [witness]

theorem witness_value (one value : Nat) (bits : Fin 8 → Nat)
    (n : Nat) (base : Assignment) (ha : Nonalias one value bits) :
    witness one value bits n base value = (n : F) := by
  simp [witness, Ne.symm ha.one_ne_value]

theorem witness_bit (one value : Nat) (bits : Fin 8 → Nat)
    (n : Nat) (base : Assignment) (ha : Nonalias one value bits) (i : Fin 8) :
    witness one value bits n base (bits i) = ((n.testBit i.val).toNat : F) := by
  have h : ∃ k, bits k = bits i := ⟨i, rfl⟩
  have hc : h.choose = i := ha.bits_injective h.choose_spec
  simp [witness, ha.bits_ne_one i, ha.bits_ne_value i, h, hc]

theorem witness_other (one value : Nat) (bits : Fin 8 → Nat)
    (n : Nat) (base : Assignment) (j : Nat)
    (ho : j ≠ one) (hv : j ≠ value) (hb : ∀ i, bits i ≠ j) :
    witness one value bits n base j = base j := by
  simp [witness, ho, hv, show ¬ ∃ i, bits i = j from by simpa using hb]

/-- Every byte has a concrete bit completion satisfying all nine R1CS rows. -/
theorem completeness (one value : Nat) (bits : Fin 8 → Nat)
    (n : Nat) (hn : n < 256) (base : Assignment)
    (ha : Nonalias one value bits) :
    let w := witness one value bits n base
    w one = 1 ∧ w value = (n : F) ∧ ∀ row ∈ rows one value bits, row.Sat w := by
  dsimp only
  refine ⟨witness_one .., witness_value one value bits n base ha, ?_⟩
  apply (rows_sat_iff one value bits _ (witness_one ..)).mpr
  constructor
  · intro i
    rw [witness_bit one value bits n base ha i]
    exact bool_cast_cases _
  · simp only [witness_value one value bits n base ha,
      witness_bit one value bits n base ha]
    exact testBit_field_recompose n hn

end CircuitCorrectness.Byte
