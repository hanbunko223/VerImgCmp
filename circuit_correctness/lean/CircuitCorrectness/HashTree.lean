import CircuitCorrectness.HashProgram
namespace CircuitCorrectness.HashTree
open HashProgram

theorem row_sound (w : Assignment) (x : Spec.Image) (r : Nat)
    (hc : ∀ i<16, w (rowStart r+160+i) = (Spec.packedChunk x r i : F))
    (hl : w (rowStart r+563) = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (rowStart r+160+i))))
    (hh : w (rowStart r+951) = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (rowStart r+168+i))))
    (hf : w (rowStart r+1189) = Spec.hash2 (w (rowStart r+563)) (w (rowStart r+951))) :
    w (rowStart r+1189) = Spec.rowHash x r := by
  have hleft : (List.range 8).toArray.map (fun i => w (rowStart r+160+i)) =
      (List.range 8).toArray.map (fun i => (Spec.packedChunk x r i : F)) := by
    apply Array.map_congr_left
    intro i hi
    apply hc
    have hi' : i<8 := by simpa using hi
    omega
  have hright : (List.range 8).toArray.map (fun i => w (rowStart r+168+i)) =
      (List.range 8).toArray.map (fun i => (Spec.packedChunk x r (8+i) : F)) := by
    apply Array.map_congr_left
    intro i hi
    have hi' : i<8 := by simpa using hi
    rw [show rowStart r+168+i = rowStart r+160+(8+i) by omega]
    exact hc (8+i) (by omega)
  rw [hleft] at hl
  rw [hright] at hh
  rw [hl,hh] at hf
  simpa only [Spec.rowHash,first_half,second_half] using hf

theorem chain_sound (w : Assignment) (x : Spec.Image)
    (hr : ∀ r<16, w (rowStart r+1189) = Spec.rowHash x r)
    (hl : w 96767 = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (rowStart i+1189))))
    (hh : w 97155 = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (rowStart (8+i)+1189))))
    (hc : w 97393 = Spec.hash2 (w 96767) (w 97155))
    (hf : w 97632 = Spec.hash2 (w 0) (w 97393)) :
    w 97632 = Spec.hash2 (w 0) (Spec.stepDigest x) := by
  have hleft : (List.range 8).toArray.map (fun i => w (rowStart i+1189)) =
      (List.range 8).toArray.map (fun i => Spec.rowHash x i) := by
    apply Array.map_congr_left
    intro i hi
    have hi' : i<8 := by simpa using hi
    exact hr i (by omega)
  have hright : (List.range 8).toArray.map (fun i => w (rowStart (8+i)+1189)) =
      (List.range 8).toArray.map (fun i => Spec.rowHash x (8+i)) := by
    apply Array.map_congr_left
    intro i hi
    have hi' : i<8 := by simpa using hi
    exact hr (8+i) (by omega)
  rw [hleft] at hl
  rw [hright] at hh
  rw [hl,hh] at hc
  rw [hc] at hf
  simpa only [Spec.stepDigest,first_half,second_half] using hf
end CircuitCorrectness.HashTree
