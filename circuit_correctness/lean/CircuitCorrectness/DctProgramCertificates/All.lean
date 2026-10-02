import CircuitCorrectness.DctProgramCertificates.Chunks
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram ConcreteBytes.Codes

/-- A coefficient-coded row of any actual chunk is an actual exported row
once its coefficients are expanded with the proved decoder. -/
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

theorem slices (xs : List Row) (k : Nat) :
    (List.range k).flatMap (fun j => (xs.drop (128*j)).take 128) = xs.take (128*k) := by
  induction k with
  | zero => simp
  | succ k ih =>
    simp only [List.range_succ,List.flatMap_append,List.flatMap_cons,List.flatMap_nil,
      List.append_nil,ih,Nat.mul_succ,List.take_add]

theorem mem_slice (xs : List Row) (k : Nat) (hlen : xs.length ≤ 128*k)
    (row : Row) (hr : row ∈ xs) :
    ∃ j < k, row ∈ (xs.drop (128*j)).take 128 := by
  have he := slices xs k
  rw [(List.take_eq_self_iff _).mpr hlen] at he
  rw [← he] at hr
  obtain ⟨j,hj,hr⟩ := List.mem_flatMap.mp hr
  exact ⟨j,List.mem_range.mp hj,hr⟩

theorem firstRows_length : firstRows.length = 5120 := by
  simp only [firstRows,List.length_map]
  decide

theorem hornerRows_length : hornerRows.length = 3080 := by
  simp [hornerRows,List.length_zipWith,ParameterChecks.retained_count]

theorem firstRows_mem_exported (row : Row) (hr : row ∈ firstRows) :
    expandRow row ∈ Exported.rows := by
  obtain ⟨c,hc,hr⟩ := mem_slice firstRows 40 (by rw [firstRows_length]) row hr
  rw [← first_chunk ⟨c,hc⟩] at hr
  exact coded_row_mem (540+c) (by omega) row hr

theorem hornerRows_mem_exported (row : Row) (hr : row ∈ hornerRows) :
    expandRow row ∈ Exported.rows := by
  obtain ⟨c,hc,hr⟩ := mem_slice hornerRows 25 (by rw [hornerRows_length]; decide) row hr
  exact coded_row_mem (580+c) (by omega) row (horner_chunk ⟨c,hc⟩ row hr)

theorem hornerRow_mem (n : Nat) (hn : n < 3080) : hornerRow n ∈ hornerRows := by
  have hlen : n < hornerRows.length := by rw [hornerRows_length]; exact hn
  have hc : n < Spec.retainedCoordinates.length := by
    rw [ParameterChecks.retained_count]; exact hn
  have he : hornerRows[n] = hornerRow n := by
    simp only [hornerRows,List.getElem_zipWith,List.getElem_range,hornerRow,coordinate,
      List.getElem!_eq_getElem?_getD,List.getElem?_eq_getElem hc,Option.getD_some]
  rw [← he]
  exact List.getElem_mem hlen

/-- Every allocated first stage wire is forced by an actual exported row. -/
theorem firstProgram_satisfied (w : Assignment) (hs : Exported.circuit.Sat w) :
    StraightLine.Satisfies firstProgram w := by
  intro op hop
  obtain ⟨⟨r,c,ch⟩,hcoord,rfl⟩ := List.mem_map.mp hop
  apply (firstRow_sat w hs.1 r c ch).mp
  exact hs.2 _ (firstRows_mem_exported _ (List.mem_map.mpr ⟨(r,c,ch),hcoord,rfl⟩))

/-- Every Horner update is forced by an actual exported row, in the independent
specification's canonical retained order. -/
theorem hornerProgram_satisfied (w : Assignment) (hs : Exported.circuit.Sat w) :
    StraightLine.Satisfies hornerProgram w := by
  intro op hop
  obtain ⟨n,hn,rfl⟩ := List.mem_map.mp hop
  have hn' := List.mem_range.mp hn
  apply (hornerRow_sat w n hn').mp
  exact hs.2 _ (hornerRows_mem_exported _ (hornerRow_mem n hn'))

/-- Actual exported DCT/Horner rows imply the exact unnormalized integer
transform, embedded into the actual scalar field. -/
theorem exported_dct_sound (w : Assignment) (x : Spec.Image)
    (hs : Exported.circuit.Sat w) (hp : PixelsMatch w x) :
    w 77323 = Gadgets.horner (w 1) (w 2) (Spec.coefficients x) := by
  exact dctProgram_sound w x ⟨w 0,w 1,w 2,w 3⟩ hs.1 hp
    (firstProgram_satisfied w hs) (hornerProgram_satisfied w hs) rfl rfl

end CircuitCorrectness.DctProgramCertificates
