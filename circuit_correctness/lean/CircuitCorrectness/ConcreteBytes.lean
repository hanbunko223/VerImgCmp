import CircuitCorrectness.ByteCertificates.Group23
import CircuitCorrectness.ByteCertificates.Group24
import CircuitCorrectness.ByteCertificates.Group25
import CircuitCorrectness.ByteCertificates.Group26

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem all_groups (g : Fin 27) (c : Fin 20) :
    let chunk := ExportedData.chunks[20*g.val + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(20*g.val+c.val)+i)) := by
  fin_cases g
  · exact group00 c
  · exact group01 c
  · exact group02 c
  · exact group03 c
  · exact group04 c
  · exact group05 c
  · exact group06 c
  · exact group07 c
  · exact group08 c
  · exact group09 c
  · exact group10 c
  · exact group11 c
  · exact group12 c
  · exact group13 c
  · exact group14 c
  · exact group15 c
  · exact group16 c
  · exact group17 c
  · exact group18 c
  · exact group19 c
  · exact group20 c
  · exact group21 c
  · exact group22 c
  · exact group23 c
  · exact group24 c
  · exact group25 c
  · exact group26 c

theorem checked_chunk (c : Nat) (hc : c < 540) :
    let chunk := ExportedData.chunks[c]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*c+i)) := by
  have hg : c / 20 < 27 := by omega
  have hr : c % 20 < 20 := Nat.mod_lt _ (by decide)
  have he : 20*(c/20)+c%20 = c := by omega
  simpa only [he] using all_groups ⟨c/20,hg⟩ ⟨c%20,hr⟩

/-- Concatenating fixed-length blocks enumerates exactly the global range. -/
theorem flatMap_blocks {α : Type} (f : Nat → α) (m n : Nat) :
    (List.range n).flatMap (fun c => (List.range m).map (fun i => f (m*c+i))) =
      (List.range (m*n)).map f := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, ih]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [Nat.mul_succ, List.range_add, List.map_append, List.map_map]
    rfl

theorem chunks_size : ExportedData.chunks.size = 763 := by decide

theorem chunks_prefix : ExportedData.chunks.toList.take 540 =
    (List.range 540).map (fun c => ExportedData.chunks[c]!) := by
  apply List.ext_getElem
  · simp [chunks_size]
  · intro i hi hj
    have hsize : i < ExportedData.chunks.size := by
      simp only [List.length_take, Array.length_toList, chunks_size] at hi
      rw [chunks_size]
      omega
    simp [List.getElem_take, List.getElem_map, List.getElem_range,
      getElem!_pos, hsize, Array.getElem_toList]

theorem decoded_prefix :
    (ExportedData.chunks.toList.take 540).flatMap
      (fun chunk => Exported.decodeRows chunk.1 chunk.2) =
      (List.range 69120).map byteRow := by
  rw [chunks_prefix, List.flatMap_map]
  calc
    _ = (List.range 540).flatMap (fun c =>
        (List.range 128).map (fun i => byteRow (128*c+i))) := by
      apply List.flatMap_congr
      intro c hc
      exact checked_chunk c (List.mem_range.mp hc)
    _ = _ := flatMap_blocks byteRow 128 540

/-- Exact equality for the full concrete exported byte prefix; not a sampled check. -/
theorem actual_prefix : Exported.rows.take 69120 = (List.range 69120).map byteRow := by
  have hs := List.take_append_drop 540 ExportedData.chunks.toList
  unfold Exported.rows
  conv_lhs => rw [← hs, List.flatMap_append, decoded_prefix]
  rw [List.take_append_of_le_length (by simp)]
  simp

/-- Every one of the concrete byte rows is covered in both directions. -/
theorem prefix_sat_iff (w : Assignment) :
    (∀ row ∈ Exported.rows.take 69120, row.Sat w) ↔
    ∀ n < 7680, ∀ row ∈ Byte.rows 97634 (valueWire n) (bitWire n), row.Sat w := by
  rw [actual_prefix]
  constructor
  · intro h n hn
    apply (nine_rows_iff n w).mp
    intro k hk
    apply h
    apply List.mem_map.mpr
    exact ⟨9*n+k, List.mem_range.mpr (by omega), rfl⟩
  · intro h row hr
    obtain ⟨j,hj,rfl⟩ := List.mem_map.mp hr
    have hj : j < 69120 := List.mem_range.mp hj
    have hn : j/9 < 7680 := by omega
    have hk : j%9 < 9 := Nat.mod_lt _ (by decide)
    have he : 9*(j/9)+j%9 = j := by omega
    simpa only [he] using (nine_rows_iff (j/9) w).mpr (h (j/9) hn) (j%9) hk

/-- Any assignment satisfying the full actual circuit has byte-valued pixels. -/
theorem soundness (w : Assignment) (hs : Exported.circuit.Sat w) :
    ∀ n < 7680, ∃ v : Nat, v < 256 ∧ w (valueWire n) = (v : F) := by
  intro n hn
  apply Byte.soundness 97634 (valueWire n) (bitWire n) w hs.1
  apply (prefix_sat_iff w).mp _ n hn
  intro row hr
  exact hs.2 row (List.mem_of_mem_take hr)

/-- A global witness with natural bytes and their testBit values satisfies every
actual exported row in the prefix. No independent per-gadget witnesses are assumed. -/
theorem complete_of_values (w : Assignment) (pixels : Nat → Nat)
    (ho : w 97634 = 1) (hp : ∀ n < 7680, pixels n < 256)
    (hv : ∀ n < 7680, w (valueWire n) = (pixels n : F))
    (hb : ∀ n < 7680, ∀ i : Fin 8,
      w (bitWire n i) = (((pixels n).testBit i.val).toNat : F)) :
    ∀ row ∈ Exported.rows.take 69120, row.Sat w := by
  apply (prefix_sat_iff w).mpr
  intro n hn
  exact Byte.complete_of_values 97634 (valueWire n) (bitWire n) w (pixels n)
    (hp n hn) ho (hv n hn) (hb n hn)

end CircuitCorrectness.ConcreteBytes
