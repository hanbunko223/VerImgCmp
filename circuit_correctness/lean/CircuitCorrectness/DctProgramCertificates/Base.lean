import CircuitCorrectness.DctProgramCertificates.Fused0
import CircuitCorrectness.DctProgramCertificates.Fused1
import CircuitCorrectness.DctProgramCertificates.Fused2
import CircuitCorrectness.ConcreteProgram
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram ConcreteBytes.Codes
open scoped BigOperators

theorem fusedCode_expand : ∀ ch : Fin 3, ∀ r c k : Fin 8,
    Spec.retained ch.val r.val c.val = true →
    ExportedData.coefficientPool[fusedCode ch.val r.val c.val k.val]! =
      encodeInt (-((Spec.multiplier ch.val r.val c.val : Int) * Spec.matrix c.val k.val)) := by
  intro ch r
  fin_cases ch <;> fin_cases r
  · exact fusedCode_expand_0_0
  · exact fusedCode_expand_0_1
  · exact fusedCode_expand_0_2
  · exact fusedCode_expand_0_3
  · exact fusedCode_expand_0_4
  · exact fusedCode_expand_0_5
  · exact fusedCode_expand_0_6
  · exact fusedCode_expand_0_7
  · exact fusedCode_expand_1_0
  · exact fusedCode_expand_1_1
  · exact fusedCode_expand_1_2
  · exact fusedCode_expand_1_3
  · exact fusedCode_expand_1_4
  · exact fusedCode_expand_1_5
  · exact fusedCode_expand_1_6
  · exact fusedCode_expand_1_7
  · exact fusedCode_expand_2_0
  · exact fusedCode_expand_2_1
  · exact fusedCode_expand_2_2
  · exact fusedCode_expand_2_3
  · exact fusedCode_expand_2_4
  · exact fusedCode_expand_2_5
  · exact fusedCode_expand_2_6
  · exact fusedCode_expand_2_7

def firstRow (r c ch : Nat) : Row :=
  ⟨((List.range 8).map fun i => (pixelWire (r/8*8+i) c ch, matrixCode (r%8) i)) ++
    [(firstWire r c ch,1218)] ++ (if r%8 = 0 then [(97634,centerCode)] else []),
    [(97634,0)], []⟩

def hornerAt (n : Nat) (p : Nat × Nat × Nat) : Row :=
  let (r,c,ch) := p
  ⟨[(accWire n,0)], [(2,0)],
    ((List.range 8).map fun k =>
      (firstWire r (c/8*8+k) ch,fusedCode ch (r%8) (c%8) k)) ++ [(accWire (n+1),0)]⟩

def hornerRow (n : Nat) : Row := hornerAt n (coordinate n)

def firstRows : List Row := firstCoordinates.map fun (r,c,ch) => firstRow r c ch
def hornerRows : List Row := List.zipWith hornerAt (List.range 3080) Spec.retainedCoordinates

theorem eval_append (w : Assignment) (xs ys : LinearCombination) :
    evalLC w (xs ++ ys) = evalLC w xs + evalLC w ys := by
  simp [evalLC,List.map_append,List.sum_append]

theorem eval_first_terms (w : Assignment) (r c ch : Nat) :
    evalLC w (expandLC ((List.range 8).map fun i =>
      (pixelWire (r/8*8+i) c ch,matrixCode (r%8) i))) =
    ∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F) * w (pixelWire (r/8*8+i) c ch) := by
  simp only [expandLC,evalLC,List.map_map,Function.comp_def,DctSpec.list_range_sum_eq]
  apply Finset.sum_congr rfl
  intro i hi
  rw [matrixCode_expand ⟨r%8,Nat.mod_lt _ (by decide)⟩ ⟨i,Finset.mem_range.mp hi⟩]
  simp

theorem center_value (w : Assignment) (h1 : w 97634 = 1) (r : Nat) :
    evalLC w (expandLC (if r%8 = 0 then [(97634,centerCode)] else [])) =
      -(128 * ∑ i ∈ Finset.range 8, (Spec.matrix (r%8) i : F)) := by
  have hs := congrArg (fun z : Int => (z : F))
    (matrix_row_sum ⟨r%8,Nat.mod_lt _ (by decide)⟩)
  simp only [DctSpec.list_range_sum_eq,Int.cast_sum] at hs
  by_cases hr : r%8 = 0
  · simp only [hr,↓reduceIte] at hs ⊢
    rw [hs]
    norm_num [expandLC,evalLC,centerCode_expand,h1]
  · simp only [hr,↓reduceIte] at hs ⊢
    rw [hs]
    simp [expandLC,evalLC]

theorem firstRow_sat (w : Assignment) (h1 : w 97634 = 1) (r c ch : Nat) :
    (expandRow (firstRow r c ch)).Sat w ↔ (firstInstruction r c ch).Sat w := by
  rw [firstInstruction_sat w h1]
  unfold firstRow expandRow Row.Sat
  simp only [expandLC,List.map_append]
  rw [eval_append,eval_append]
  change (evalLC w (expandLC ((List.range 8).map fun i =>
    (pixelWire (r/8*8+i) c ch,matrixCode (r%8) i))) +
    evalLC w [(firstWire r c ch,ExportedData.coefficientPool[1218]!)] +
    evalLC w (expandLC (if r%8 = 0 then [(97634,centerCode)] else []))) *
    evalLC w [(97634,ExportedData.coefficientPool[0]!)] = evalLC w [] ↔ _
  rw [eval_first_terms,center_value w h1,neg_one_expand,one_expand]
  simp only [Affine.eval_cons,Affine.eval_nil,Nat.cast_one,h1,mul_one,add_zero,
    ConcreteProgram.cast_neg_one,neg_one_mul]
  constructor <;> intro h <;> linear_combination -h

theorem eval_fused_terms (w : Assignment) (r c ch : Nat)
    (hch : ch < 3) (ha : Spec.retained ch (r%8) (c%8) = true) :
    evalLC w (expandLC ((List.range 8).map fun k =>
      (firstWire r (c/8*8+k) ch,fusedCode ch (r%8) (c%8) k))) =
      -coefficientFromWires w r c ch := by
  simp only [expandLC,evalLC,List.map_map,Function.comp_def,DctSpec.list_range_sum_eq,
    coefficientFromWires,← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro k hk
  rw [fusedCode_expand ⟨ch,hch⟩ ⟨r%8,Nat.mod_lt _ (by decide)⟩
    ⟨c%8,Nat.mod_lt _ (by decide)⟩ ⟨k,Finset.mem_range.mp hk⟩ ha]
  simp

theorem hornerRow_sat (w : Assignment) (n : Nat) (hn : n < 3080) :
    (expandRow (hornerRow n)).Sat w ↔ (hornerInstruction n).Sat w := by
  have hv := coordinate_valid n hn
  rw [hornerInstruction_sat]
  unfold hornerRow hornerAt expandRow Row.Sat
  dsimp only
  simp only [expandLC,List.map_append]
  rw [eval_append]
  change evalLC w [(accWire n,ExportedData.coefficientPool[0]!)] *
    evalLC w [(2,ExportedData.coefficientPool[0]!)] =
    evalLC w (expandLC ((List.range 8).map fun k =>
      (firstWire (coordinate n).1 ((coordinate n).2.1/8*8+k) (coordinate n).2.2,
        fusedCode (coordinate n).2.2 ((coordinate n).1%8) ((coordinate n).2.1%8) k))) +
    evalLC w [(accWire (n+1),ExportedData.coefficientPool[0]!)] ↔ _
  rw [eval_fused_terms w (coordinate n).1 (coordinate n).2.1 (coordinate n).2.2 hv.2.2.1 hv.2.2.2,one_expand]
  simp only [Affine.eval_cons,Affine.eval_nil,Nat.cast_one,one_mul,add_zero]
  constructor <;> intro h <;> linear_combination -h

end CircuitCorrectness.DctProgramCertificates
