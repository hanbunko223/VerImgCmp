import CircuitCorrectness.PackingCertificates.Data
import CircuitCorrectness.ConcreteProgram
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes HashProgram StraightLine

def pixelRow (r c : Nat) : Row :=
  ⟨((List.range 3).map fun ch => (pixelWire r c ch,pixelCodes[ch]!)) ++
    [(rowStart r+c,1218)], [(97634,0)], []⟩
def chunkRow (r c : Nat) : Row :=
  ⟨((List.range 10).map fun j => (rowStart r+10*c+j,chunkCodes[j]!)) ++
    [(rowStart r+160+c,1218)], [(97634,0)], []⟩
def packingRows (r : Nat) : List Row :=
  (List.range 160).map (pixelRow r) ++ (List.range 16).map (chunkRow r)

theorem eval_append (w : Assignment) (xs ys : LinearCombination) :
    evalLC w (xs ++ ys) = evalLC w xs + evalLC w ys := by
  simp only [evalLC,List.map_append,List.sum_append]

theorem linear_row_sat (w : Assignment) (h1 : w 97634 = 1)
    (dst : Nat) (terms : LinearCombination) :
    (expandRow ⟨terms ++ [(dst,1218)],[(97634,0)],[]⟩).Sat w ↔
      w dst = evalLC w (expandLC terms) := by
  change evalLC w (expandLC (terms ++ [(dst,1218)])) *
    evalLC w [(97634,ExportedData.coefficientPool[0]!)] = 0 ↔ _
  rw [show expandLC (terms ++ [(dst,1218)]) =
    expandLC terms ++ [(dst,ExportedData.coefficientPool[1218]!)] by
      simp only [expandLC,List.map_append,List.map_cons,List.map_nil]]
  rw [eval_append,one_expand,neg_one_expand]
  simp only [Affine.eval_cons,Affine.eval_nil,ConcreteProgram.cast_neg_one,
    Nat.cast_one,one_mul,neg_one_mul,h1,add_zero,mul_one]
  constructor <;> intro h <;> linear_combination -h

theorem linear_sat (w : Assignment) (h1 : w 97634 = 1)
    (dst : Nat) (terms : LinearCombination) :
    (linear dst terms).Sat w ↔ w dst = evalLC w terms := by
  simp [linear,Instruction.Sat,Instruction.value,HashProgram.one,h1]

theorem eval_pixel_terms (w : Assignment) (r c : Nat) :
    evalLC w (expandLC ((List.range 3).map fun ch => (pixelWire r c ch,pixelCodes[ch]!))) =
      evalLC w (Affine.sum ((List.range 3).map fun ch =>
        Affine.scale (2^(8*ch)) (Affine.wire (pixelWire r c ch)))) := by
  simp only [Affine.eval_sum,List.map_map,Function.comp_def,
    Affine.eval_scale,Affine.eval_variable]
  simp only [expandLC,evalLC,List.map_map,Function.comp_def]
  apply congrArg List.sum
  apply List.map_congr_left
  intro ch hc
  rw [pixelCode_expand ⟨ch,List.mem_range.mp hc⟩]

theorem eval_chunk_terms (w : Assignment) (r c : Nat) :
    evalLC w (expandLC ((List.range 10).map fun j => (rowStart r+10*c+j,chunkCodes[j]!))) =
      evalLC w (Affine.sum ((List.range 10).map fun j =>
        Affine.scale (2^(24*j)) (Affine.wire (rowStart r+10*c+j)))) := by
  simp only [Affine.eval_sum,List.map_map,Function.comp_def,
    Affine.eval_scale,Affine.eval_variable]
  simp only [expandLC,evalLC,List.map_map,Function.comp_def]
  apply congrArg List.sum
  apply List.map_congr_left
  intro j hj
  rw [chunkCode_expand ⟨j,List.mem_range.mp hj⟩]

theorem pixelRow_sat (w : Assignment) (h1 : w 97634 = 1) (r c : Nat) :
    (expandRow (pixelRow r c)).Sat w ↔ (pixelOp r c).Sat w := by
  rw [pixelRow,linear_row_sat w h1,eval_pixel_terms,pixelOp,linear_sat w h1]

theorem chunkRow_sat (w : Assignment) (h1 : w 97634 = 1) (r c : Nat) :
    (expandRow (chunkRow r c)).Sat w ↔ (chunkOp r c).Sat w := by
  rw [chunkRow,linear_row_sat w h1,eval_chunk_terms,chunkOp,linear_sat w h1]

/-- Inclusion uses the actual export chunks and the verified coefficient decoder. -/
theorem coded_row_mem (c : Nat) (hc : c < 763) (row : Row)
    (hr : let chunk := ExportedData.chunks[c]!
      row ∈ rows chunk.1 chunk.2) : expandRow row ∈ Exported.rows := by
  have hc' : c < ExportedData.chunks.size := by
    have : ExportedData.chunks.size = 763 := by decide
    omega
  apply List.mem_flatMap.mpr
  refine ⟨ExportedData.chunks[c]!, ?_, ?_⟩
  · simpa only [getElem!_pos ExportedData.chunks c hc']
      using Array.getElem_mem_toList hc'
  · change expandRow row ∈ Exported.decodeRows (ExportedData.chunks[c]!).1 (ExportedData.chunks[c]!).2
    rw [rows_expand]
    exact List.mem_map.mpr ⟨row,hr,rfl⟩

end CircuitCorrectness.PackingCertificates
