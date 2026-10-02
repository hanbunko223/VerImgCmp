import CircuitCorrectness.PackingCertificates.Row15
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes HashProgram StraightLine

theorem packing_rows_mem (r : Fin 16) (row : Row) (hr : row ∈ packingRows r.val) :
    expandRow row ∈ Exported.rows := by
  fin_cases r
  · exact packing_mem_00 row hr
  · exact packing_mem_01 row hr
  · exact packing_mem_02 row hr
  · exact packing_mem_03 row hr
  · exact packing_mem_04 row hr
  · exact packing_mem_05 row hr
  · exact packing_mem_06 row hr
  · exact packing_mem_07 row hr
  · exact packing_mem_08 row hr
  · exact packing_mem_09 row hr
  · exact packing_mem_10 row hr
  · exact packing_mem_11 row hr
  · exact packing_mem_12 row hr
  · exact packing_mem_13 row hr
  · exact packing_mem_14 row hr
  · exact packing_mem_15 row hr

/-- Every RGB packing and every 240-bit chunk packing equation is an actual
exported row, including the original uncentered-byte input wiring. -/
theorem satisfied (w : Assignment) (hs : Exported.circuit.Sat w) (r : Nat) (hr : r < 16) :
    Satisfies ((List.range 160).map (pixelOp r)) w ∧
      Satisfies ((List.range 16).map (chunkOp r)) w := by
  constructor
  · intro op hop
    obtain ⟨c,hc,rfl⟩ := List.mem_map.mp hop
    apply (pixelRow_sat w hs.1 r c).mp
    apply hs.2
    apply packing_rows_mem ⟨r,hr⟩
    apply List.mem_append_left
    exact List.mem_map.mpr ⟨c,hc,rfl⟩
  · intro op hop
    obtain ⟨c,hc,rfl⟩ := List.mem_map.mp hop
    apply (chunkRow_sat w hs.1 r c).mp
    apply hs.2
    apply packing_rows_mem ⟨r,hr⟩
    apply List.mem_append_right
    exact List.mem_map.mpr ⟨c,hc,rfl⟩

end CircuitCorrectness.PackingCertificates
