import CircuitCorrectness.DctSpec
import CircuitCorrectness.Affine
import CircuitCorrectness.StraightLine

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

namespace CircuitCorrectness.DctProgram
open scoped BigOperators

def pixelWire (r c ch : Nat) : Nat := 4 + 9 * ((r*160+c)*3+ch)
def rowCount (ch : Nat) : Nat := if ch = 0 then 8 else 4
def channelOffset (ch : Nat) : Nat := if ch = 0 then 0 else 2560+(ch-1)*1280

/-- Production channel-major, block-major, active-row-major first-stage allocation. -/
def firstWire (r c ch : Nat) : Nat :=
  69124 + channelOffset ch + ((r/8*20+c/8)*rowCount ch+r%8)*8+c%8

def accWire (n : Nat) : Nat := if n = 0 then 1 else 74243+n
def coordinate (n : Nat) : Nat × Nat × Nat := Spec.retainedCoordinates[n]!

def PixelsMatch (w : Assignment) (x : Spec.Image) : Prop :=
  ∀ r < 16, ∀ c < 160, ∀ ch < 3, w (pixelWire r c ch) = (x r c ch : F)

/-- Only allocated active first-stage rows occur in this relation. -/
def FirstStageRows (w : Assignment) : Prop :=
  ∀ r < 16, ∀ c < 160, ∀ ch < 3, ch = 0 ∨ r%8 < 4 →
    w (firstWire r c ch) =
      (∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) *
        w (pixelWire (r/8*8+i) c ch)) -
      128 * ∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F)

def FirstStageWires (w : Assignment) (x : Spec.Image) : Prop :=
  ∀ r < 16, ∀ c < 160, ∀ ch < 3, ch = 0 ∨ r%8 < 4 →
    w (firstWire r c ch) = (DctSpec.first x r c ch (c%8) : F)

theorem firstStage_sound (w : Assignment) (x : Spec.Image)
    (hp : PixelsMatch w x) (hf : FirstStageRows w) : FirstStageWires w x := by
  intro r hr c hc ch hch ha
  rw [hf r hr c hc ch hch ha, DctSpec.cast_first_linear]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  have hm : c/8*8+c%8 = c := by omega
  rw [hm, hp _ (by have := Finset.mem_range.mp hi; omega) c hc ch hch]

theorem retainedRow_valid (r : Nat) (c ch : Nat)
    (h : (c,ch) ∈ Spec.retainedRow r) :
    c < 8 ∧ ch < 3 ∧ Spec.retained ch r c = true := by
  simp only [Spec.retainedRow, List.mem_flatMap, List.mem_map,
    List.mem_filter, List.mem_range] at h
  rcases h with ⟨c',hc',ch',⟨hch',ha⟩,he⟩
  cases he
  exact ⟨hc',hch',ha⟩

theorem retainedCoordinates_valid (r c ch : Nat)
    (h : (r,c,ch) ∈ Spec.retainedCoordinates) :
    r < 16 ∧ c < 160 ∧ ch < 3 ∧ Spec.retained ch (r%8) (c%8) = true := by
  simp only [Spec.retainedCoordinates, List.mem_flatMap, List.mem_map, List.mem_range] at h
  rcases h with ⟨r',hr',b,hb,⟨c',ch'⟩,hm,he⟩
  have hv := retainedRow_valid (r'%8) c' ch' hm
  cases he
  refine ⟨hr', by omega, hv.2.1, ?_⟩
  have hm : (8*b+c')%8 = c' := by omega
  simpa [hm] using hv.2.2

theorem coordinate_mem (n : Nat) (hn : n < 3080) :
    coordinate n ∈ Spec.retainedCoordinates := by
  have hlen : n < Spec.retainedCoordinates.length := by rw [ParameterChecks.retained_count]; exact hn
  unfold coordinate
  simpa [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hlen] using
    List.getElem_mem hlen

theorem coordinate_valid (n : Nat) (hn : n < 3080) :
    let (r,c,ch) := coordinate n
    r < 16 ∧ c < 160 ∧ ch < 3 ∧ Spec.retained ch (r%8) (c%8) = true := by
  exact retainedCoordinates_valid _ _ _ (coordinate_mem n hn)

def coefficientFromWires (w : Assignment) (r c ch : Nat) : F :=
  ∑ k ∈ Finset.range 8,
    ((Spec.multiplier ch (r%8) (c%8) : F) * (Spec.matrix (c%8) k : F)) *
      w (firstWire r (c/8*8+k) ch)

theorem coefficientFromWires_eq (w : Assignment) (x : Spec.Image)
    (hf : FirstStageWires w x) (r c ch : Nat)
    (hr : r < 16) (hc : c < 160) (hch : ch < 3)
    (ha : Spec.retained ch (r%8) (c%8) = true) :
    coefficientFromWires w r c ch = (Spec.coefficient x r c ch : F) := by
  rw [DctSpec.cast_coefficient_fused]
  unfold coefficientFromWires
  apply Finset.sum_congr rfl
  intro k hk
  have hk8 : k < 8 := Finset.mem_range.mp hk
  have hrow : ch = 0 ∨ r%8 < 4 := by
    apply (DctSpec.first_row_pruning ⟨ch,hch⟩ ⟨r%8,Nat.mod_lt _ (by decide)⟩).mp
    exact ⟨⟨c%8,Nat.mod_lt _ (by decide)⟩,ha⟩
  rw [hf r hr (c/8*8+k) (by omega) ch hch hrow]
  have hb : (c/8*8+k)/8*8 = c/8*8 := by omega
  have hm : (c/8*8+k)%8 = k := by omega
  simp only [DctSpec.first, hb, hm]

/-- The actual contiguous Horner output wires, in canonical retained order. -/
def HornerRows (w : Assignment) : Prop :=
  ∀ n < 3080, let (r,c,ch) := coordinate n
    w (accWire (n+1)) = w (accWire n) * w 2 + coefficientFromWires w r c ch

def HornerWires (w : Assignment) (s : Spec.State) (x : Spec.Image) : Prop :=
  ∀ n ≤ 3080, w (accWire n) =
    Gadgets.horner s.a s.r ((Spec.coefficients x).take n)

theorem coefficients_length (x : Spec.Image) : (Spec.coefficients x).length = 3080 := by
  simp [Spec.coefficients, ParameterChecks.retained_count]

theorem coefficient_get (x : Spec.Image) (n : Nat) (hn : n < 3080) :
    (Spec.coefficients x)[n]! =
      (Spec.coefficient x (coordinate n).1 (coordinate n).2.1 (coordinate n).2.2 : F) := by
  have hlen : n < Spec.retainedCoordinates.length := by rw [ParameterChecks.retained_count]; exact hn
  simp [Spec.coefficients, coordinate, hlen]

theorem hornerWires_sound (w : Assignment) (x : Spec.Image) (s : Spec.State)
    (hf : FirstStageWires w x) (hh : HornerRows w)
    (ha : w 1 = s.a) (hr : w 2 = s.r) : HornerWires w s x := by
  intro n hn
  induction n with
  | zero => simpa [accWire, Gadgets.horner] using ha
  | succ n ih =>
    have hn' : n < 3080 := by omega
    have hv := coordinate_valid n hn'
    have he := hh n hn'
    dsimp only at hv he
    rw [he, coefficientFromWires_eq w x hf _ _ _ hv.1 hv.2.1 hv.2.2.1 hv.2.2.2,
      ih (by omega), hr]
    have hlen : n < (Spec.coefficients x).length := by rw [coefficients_length]; exact hn'
    rw [List.take_succ_eq_append_getElem hlen, Gadgets.horner_append]
    simp only [Gadgets.horner, List.foldl_cons, List.foldl_nil]
    have hget := coefficient_get x n hn'
    simpa [hlen] using congrArg
      (fun z => ((Spec.coefficients x).take n).foldl (fun acc c => acc * s.r + c) s.a * s.r + z)
      hget.symm

theorem final_horner (w : Assignment) (x : Spec.Image) (s : Spec.State)
    (hf : FirstStageWires w x) (hh : HornerRows w)
    (ha : w 1 = s.a) (hr : w 2 = s.r) :
    w 77323 = Gadgets.horner s.a s.r (Spec.coefficients x) := by
  have h := hornerWires_sound w x s hf hh ha hr 3080 (by omega)
  simpa [accWire, ← coefficients_length x] using h

/-- Canonical exported scalar representation of a signed integer constant. -/
def encodeInt (z : Int) : Nat := (z % (modulus : Int)).toNat

@[simp] theorem cast_encodeInt (z : Int) : (encodeInt z : F) = (z : F) := by
  have hv := congrArg Int.toNat (ZMod.val_intCast (n := modulus) z)
  have he : encodeInt z = (z : F).val := by simpa [encodeInt] using hv.symm
  rw [he]
  exact ZMod.natCast_zmod_val (z : F)

def sumLC : List LinearCombination → LinearCombination
  | [] => []
  | lc :: tail => Affine.add lc (sumLC tail)

@[simp] theorem eval_sumLC (w : Assignment) (xs : List LinearCombination) :
    evalLC w (sumLC xs) = (xs.map (evalLC w)).sum := by
  induction xs with
  | nil => rfl
  | cons lc tail ih => simp [sumLC, ih]

def firstLC (r c ch : Nat) : LinearCombination :=
  Affine.add
    (sumLC ((List.range 8).map fun i =>
      Affine.scale (encodeInt (Spec.matrix (r%8) i))
        (Affine.wire (pixelWire (r/8*8+i) c ch))))
    (Affine.constant 97634 (encodeInt (-128 *
      ((List.range 8).map fun i => Spec.matrix (r%8) i).sum)))

def firstInstruction (r c ch : Nat) : StraightLine.Instruction :=
  ⟨firstWire r c ch, firstLC r c ch, Affine.wire 97634, []⟩

theorem firstLC_value (w : Assignment) (h1 : w 97634 = 1) (r c ch : Nat) :
    evalLC w (firstLC r c ch) =
      (∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) *
        w (pixelWire (r/8*8+i) c ch)) -
      128 * ∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) := by
  simp [firstLC, List.map_map, DctSpec.list_range_sum_eq, h1]
  ring

theorem firstInstruction_sat (w : Assignment) (h1 : w 97634 = 1) (r c ch : Nat) :
    (firstInstruction r c ch).Sat w ↔
      w (firstWire r c ch) =
        (∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) *
          w (pixelWire (r/8*8+i) c ch)) -
        128 * ∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) := by
  simp [firstInstruction, StraightLine.Instruction.Sat, StraightLine.Instruction.value,
    firstLC_value, h1]

def coefficientLC (r c ch : Nat) : LinearCombination :=
  sumLC ((List.range 8).map fun k =>
    Affine.scale (encodeInt ((Spec.multiplier ch (r%8) (c%8) : Int) * Spec.matrix (c%8) k))
      (Affine.wire (firstWire r (c/8*8+k) ch)))

theorem coefficientLC_value (w : Assignment) (r c ch : Nat) :
    evalLC w (coefficientLC r c ch) = coefficientFromWires w r c ch := by
  simp [coefficientLC, coefficientFromWires, List.map_map, DctSpec.list_range_sum_eq]

def hornerInstruction (n : Nat) : StraightLine.Instruction :=
  let (r,c,ch) := coordinate n
  ⟨accWire (n+1), Affine.wire (accWire n), Affine.wire 2, coefficientLC r c ch⟩

theorem hornerInstruction_sat (w : Assignment) (n : Nat) :
    (hornerInstruction n).Sat w ↔
      w (accWire (n+1)) = w (accWire n) * w 2 +
        coefficientFromWires w (coordinate n).1 (coordinate n).2.1 (coordinate n).2.2 := by
  simp [hornerInstruction, StraightLine.Instruction.Sat, StraightLine.Instruction.value,
    coefficientLC_value]

def firstCoordinates : List (Nat × Nat × Nat) :=
  (List.range 3).flatMap fun ch =>
    (List.range 2).flatMap fun br =>
      (List.range 20).flatMap fun bc =>
        (List.range (rowCount ch)).flatMap fun r =>
          (List.range 8).map fun c => (8*br+r,8*bc+c,ch)

def firstProgram : StraightLine.Program :=
  firstCoordinates.map fun (r,c,ch) => firstInstruction r c ch

def hornerProgram : StraightLine.Program :=
  (List.range 3080).map hornerInstruction

theorem firstCoordinates_mem (r c ch : Nat)
    (hr : r < 16) (hc : c < 160) (hch : ch < 3) (ha : ch = 0 ∨ r%8 < 4) :
    (r,c,ch) ∈ firstCoordinates := by
  simp only [firstCoordinates, List.mem_flatMap, List.mem_map, List.mem_range]
  refine ⟨ch,hch,r/8,by omega,c/8,by omega,r%8,?_,c%8,Nat.mod_lt _ (by decide),?_⟩
  · unfold rowCount
    split
    · exact Nat.mod_lt _ (by decide)
    · omega
  · apply Prod.ext
    · omega
    · apply Prod.ext <;> dsimp <;> omega

theorem firstProgram_sound (w : Assignment) (h1 : w 97634 = 1)
    (h : StraightLine.Satisfies firstProgram w) : FirstStageRows w := by
  intro r hr c hc ch hch ha
  apply (firstInstruction_sat w h1 r c ch).mp
  apply h
  exact List.mem_map.mpr ⟨(r,c,ch),firstCoordinates_mem r c ch hr hc hch ha,rfl⟩

theorem hornerProgram_sound (w : Assignment)
    (h : StraightLine.Satisfies hornerProgram w) : HornerRows w := by
  intro n hn
  apply (hornerInstruction_sat w n).mp
  exact h _ (List.mem_map.mpr ⟨n,List.mem_range.mpr hn,rfl⟩)

theorem dctProgram_sound (w : Assignment) (x : Spec.Image) (s : Spec.State)
    (h1 : w 97634 = 1) (hp : PixelsMatch w x)
    (hf : StraightLine.Satisfies firstProgram w)
    (hh : StraightLine.Satisfies hornerProgram w)
    (ha : w 1 = s.a) (hr : w 2 = s.r) :
    w 77323 = Gadgets.horner s.a s.r (Spec.coefficients x) :=
  final_horner w x s (firstStage_sound w x hp (firstProgram_sound w h1 hf))
    (hornerProgram_sound w hh) ha hr

end CircuitCorrectness.DctProgram
