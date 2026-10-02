import CircuitCorrectness.Spec
import CircuitCorrectness.Arithmetic
import CircuitCorrectness.ParameterChecks

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

namespace CircuitCorrectness.DctSpec
open scoped BigOperators

theorem list_range_sum_eq {R : Type*} [AddCommMonoid R] (f : Nat → R) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ, Finset.sum_range_succ, ih]

/-- The exact centered first stage at a packed-row block, evaluated over integers. -/
def first (x : Spec.Image) (r c ch k : Nat) : Int :=
  ∑ i ∈ Finset.range 8,
    Spec.matrix (r % 8) i * ((x (r / 8 * 8 + i) (c / 8 * 8 + k) ch : Int) - 128)

/-- The coefficient uses row `c % 8` in the right factor: this is Aᵀ. -/
theorem coefficient_staged (x : Spec.Image) (r c ch : Nat) :
    Spec.coefficient x r c ch =
      (Spec.multiplier ch (r%8) (c%8) : Int) *
        ∑ k ∈ Finset.range 8, first x r c ch k * Spec.matrix (c%8) k := by
  simp only [Spec.coefficient, list_range_sum_eq, first, Finset.sum_mul]
  rw [Finset.sum_comm]

theorem cast_first (x : Spec.Image) (r c ch k : Nat) :
    (first x r c ch k : F) = ∑ i ∈ Finset.range 8,
      (Spec.matrix (r%8) i : F) *
        ((x (r/8*8+i) (c/8*8+k) ch : F) - 128) := by
  simp [first]

/-- This form is exactly the centered linear combination used for a first-stage wire. -/
theorem cast_first_linear (x : Spec.Image) (r c ch k : Nat) :
    (first x r c ch k : F) =
      (∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) *
        (x (r/8*8+i) (c/8*8+k) ch : F)) -
      128 * ∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) := by
  rw [cast_first]
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul]
  ring

/-- Integer computation embeds homomorphically into the circuit field. -/
theorem cast_coefficient_staged (x : Spec.Image) (r c ch : Nat) :
    (Spec.coefficient x r c ch : F) =
      (Spec.multiplier ch (r%8) (c%8) : F) *
        ∑ k ∈ Finset.range 8, (first x r c ch k : F) * (Spec.matrix (c%8) k : F) := by
  rw [coefficient_staged]
  simp

/-- Combining multiplier and right-matrix entry gives the fused Horner linear form. -/
theorem cast_coefficient_fused (x : Spec.Image) (r c ch : Nat) :
    (Spec.coefficient x r c ch : F) =
      ∑ k ∈ Finset.range 8,
        ((Spec.multiplier ch (r%8) (c%8) : F) * (Spec.matrix (c%8) k : F)) *
          (first x r c ch k : F) := by
  rw [cast_coefficient_staged, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k hk
  ring

theorem retained_needs_first_row (ch r c : Nat)
    (hc : c < 8) (h : Spec.retained ch r c = true) :
    ∃ k ∈ Finset.range 8, Spec.retained ch r k = true :=
  ⟨c, Finset.mem_range.mpr hc, h⟩

/-- Exactly the first-stage rows retained by the implementation's pruning predicate. -/
theorem first_row_pruning : ∀ ch : Fin 3, ∀ r : Fin 8,
    (∃ c : Fin 8, Spec.retained ch.val r.val c.val = true) ↔
      ch.val = 0 ∨ r.val < 4 := by decide

theorem omitted_zero (x : Spec.Image) (r c ch : Nat)
    (h : Spec.retained ch (r%8) (c%8) = false) : Spec.coefficient x r c ch = 0 := by
  have hm : Spec.multiplier ch (r%8) (c%8) = 0 := by simpa [Spec.retained] using h
  simp [Spec.coefficient, hm]

/-- Pruning omitted positions does not change the mathematical coefficient. -/
theorem coefficient_pruned (x : Spec.Image) (r c ch : Nat) :
    Spec.coefficient x r c ch =
      if Spec.retained ch (r%8) (c%8) then
        (Spec.multiplier ch (r%8) (c%8) : Int) *
          ∑ k ∈ Finset.range 8, first x r c ch k * Spec.matrix (c%8) k
      else 0 := by
  split
  · exact coefficient_staged x r c ch
  · exact omitted_zero x r c ch (by simpa using ‹¬Spec.retained ch (r%8) (c%8) = true›)

def weight (ch r c i j : Nat) : Int :=
  (Spec.multiplier ch r c : Int) * Spec.matrix r i * Spec.matrix c j

def lower (ch r c : Nat) : Int :=
  ∑ i ∈ Finset.range 8, ∑ j ∈ Finset.range 8,
    min (-128 * weight ch r c i j) (127 * weight ch r c i j)

def upper (ch r c : Nat) : Int :=
  ∑ i ∈ Finset.range 8, ∑ j ∈ Finset.range 8,
    max (-128 * weight ch r c i j) (127 * weight ch r c i j)

theorem coefficient_weighted (x : Spec.Image) (r c ch : Nat) :
    Spec.coefficient x r c ch =
      ∑ i ∈ Finset.range 8, ∑ j ∈ Finset.range 8,
        weight ch (r%8) (c%8) i j *
          ((x (r/8*8+i) (c/8*8+j) ch : Int) - 128) := by
  simp only [Spec.coefficient, list_range_sum_eq, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  unfold weight
  ring

theorem centered_bounds (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c ch i j : Nat) (hr : r < 16) (hc : c < 160) (hch : ch < 3)
    (hi : i < 8) (hj : j < 8) :
    -128 ≤ (x (r/8*8+i) (c/8*8+j) ch : Int) - 128 ∧
      (x (r/8*8+i) (c/8*8+j) ch : Int) - 128 ≤ 127 := by
  have hp := hv (r/8*8+i) (by omega) (c/8*8+j) (by omega) ch hch
  omega

theorem scaled_centered_bounds (w z : Int) (hz : -128 ≤ z ∧ z ≤ 127) :
    min (-128*w) (127*w) ≤ w*z ∧ w*z ≤ max (-128*w) (127*w) := by
  by_cases hw : 0 ≤ w
  · constructor
    · exact le_trans (min_le_left _ _) (by nlinarith)
    · exact le_trans (by nlinarith : w*z ≤ 127*w) (le_max_right _ _)
  · constructor
    · exact le_trans (min_le_right _ _) (by nlinarith)
    · exact le_trans (by nlinarith : w*z ≤ -128*w) (le_max_left _ _)

/-- Attainable per-position endpoint bounds, matching the native reference formula. -/
theorem coefficient_bounds (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c ch : Nat) (hr : r < 16) (hc : c < 160) (hch : ch < 3) :
    lower ch (r%8) (c%8) ≤ Spec.coefficient x r c ch ∧
      Spec.coefficient x r c ch ≤ upper ch (r%8) (c%8) := by
  rw [coefficient_weighted]
  constructor
  · apply Finset.sum_le_sum
    intro i hi
    apply Finset.sum_le_sum
    intro j hj
    exact (scaled_centered_bounds _ _
      (centered_bounds x hv r c ch i j hr hc hch
        (Finset.mem_range.mp hi) (Finset.mem_range.mp hj))).1
  · apply Finset.sum_le_sum
    intro i hi
    apply Finset.sum_le_sum
    intro j hj
    exact (scaled_centered_bounds _ _
      (centered_bounds x hv r c ch i j hr hc hch
        (Finset.mem_range.mp hi) (Finset.mem_range.mp hj))).2

/-- A finite, kernel-reduced check of all 192 fixed position bounds. -/
theorem fixed_bounds : ∀ ch : Fin 3, ∀ r c : Fin 8,
    -1554357600 ≤ lower ch.val r.val c.val ∧ upper ch.val r.val c.val ≤ 1554357600 := by
  decide

theorem coefficient_global_bounds (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c ch : Nat) (hr : r < 16) (hc : c < 160) (hch : ch < 3) :
    -1554357600 ≤ Spec.coefficient x r c ch ∧
      Spec.coefficient x r c ch ≤ 1554357600 := by
  have hb := coefficient_bounds x hv r c ch hr hc hch
  have hf := fixed_bounds ⟨ch,hch⟩ ⟨r%8,Nat.mod_lt _ (by decide)⟩
    ⟨c%8,Nat.mod_lt _ (by decide)⟩
  exact ⟨le_trans hf.1 hb.1, le_trans hb.2 hf.2⟩

/-- The coefficient interval cannot contain two distinct integers with the same field image. -/
theorem bounded_cast_injective (a b : Int)
    (ha : -1554357600 ≤ a ∧ a ≤ 1554357600)
    (hb : -1554357600 ≤ b ∧ b ≤ 1554357600)
    (h : (a : F) = (b : F)) : a = b := by
  have hd := (ZMod.intCast_eq_intCast_iff_dvd_sub a b modulus).mp h
  rcases hd with ⟨k, hk⟩
  have hq : (3108715200 : Int) < (modulus : Int) := by decide
  have hk0 : k = 0 := by
    by_contra hn
    have hs : k ≤ -1 ∨ 1 ≤ k := by omega
    rcases hs with hs | hs <;> nlinarith
  subst k
  simp only [mul_zero] at hk
  omega

theorem coefficient_cast_injective (x y : Spec.Image)
    (hx : Spec.ValidImage x) (hy : Spec.ValidImage y)
    (r c ch : Nat) (hr : r < 16) (hc : c < 160) (hch : ch < 3)
    (h : (Spec.coefficient x r c ch : F) = (Spec.coefficient y r c ch : F)) :
    Spec.coefficient x r c ch = Spec.coefficient y r c ch :=
  bounded_cast_injective _ _ (coefficient_global_bounds x hx r c ch hr hc hch)
    (coefficient_global_bounds y hy r c ch hr hc hch) h

theorem packedPixel_bound (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c : Nat) (hr : r < 16) (hc : c < 160) : Spec.packedPixel x r c < 2^24 := by
  exact Arithmetic.pixel_packing_bound _ _ _
    (hv r hr c hc 0 (by decide)) (hv r hr c hc 1 (by decide))
    (hv r hr c hc 2 (by decide))

theorem cast_packedPixel (x : Spec.Image) (r c : Nat) :
    (Spec.packedPixel x r c : F) =
      (x r c 0 : F) + 256 * (x r c 1 : F) + 65536 * (x r c 2 : F) := by
  simp [Spec.packedPixel]

theorem cast_packedChunk (x : Spec.Image) (r c : Nat) :
    (Spec.packedChunk x r c : F) =
      ∑ j ∈ Finset.range 10, (Spec.packedPixel x r (10*c+j) : F) * (2:F)^(24*j) := by
  simp [Spec.packedChunk, list_range_sum_eq]

theorem packedChunk_bound (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c : Nat) (hr : r < 16) (hc : c < 16) : Spec.packedChunk x r c < 2^240 := by
  unfold Spec.packedChunk
  rw [list_range_sum_eq]
  simp_rw [show ∀ j : Nat, (2:Nat)^(24*j) = (2^24)^j from fun j => pow_mul _ _ _]
  exact Arithmetic.chunk_packing_bound _ (fun j hj => packedPixel_bound x hv r _ hr (by omega))

theorem packedChunk_field_bound (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c : Nat) (hr : r < 16) (hc : c < 16) : Spec.packedChunk x r c < modulus :=
  lt_trans (packedChunk_bound x hv r c hr hc) chunk_fits

/-- The field cast of a chunk preserves all 240 packed bits. -/
theorem packedChunk_cast_value (x : Spec.Image) (hv : Spec.ValidImage x)
    (r c : Nat) (hr : r < 16) (hc : c < 16) :
    (Spec.packedChunk x r c : F).val = Spec.packedChunk x r c := by
  rw [ZMod.val_natCast, Nat.mod_eq_of_lt (packedChunk_field_bound x hv r c hr hc)]

end CircuitCorrectness.DctSpec
