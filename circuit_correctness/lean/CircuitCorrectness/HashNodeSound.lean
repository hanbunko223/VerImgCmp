import CircuitCorrectness.HashWiringCertificates.All
import CircuitCorrectness.HashWiringCertificates.Transfer
import CircuitCorrectness.HashTrace8.All
import CircuitCorrectness.HashTrace2.All
import CircuitCorrectness.HashProgram

set_option maxRecDepth 20000
set_option maxHeartbeats 2000000

namespace CircuitCorrectness.HashWiring

theorem template8_sound : Template8Sound := by
  intro w h1 hs
  have hi : HashTrace8.input = (List.range 8).toArray.map
      (fun i => Affine.wire (77484+i)) := by
    apply Array.ext
    · simp [HashTrace8.input]
    · intro i hi hi'
      have hi8 : i<8 := by simpa [HashTrace8.input] using hi
      interval_cases i <;> simp [HashTrace8.input]
  have he : PoseidonProgram.evalState w HashTrace8.input =
      (List.range 8).toArray.map (fun i => w (77484+i)) := by
    rw [hi]
    simp [PoseidonProgram.evalState,Array.map_map,Function.comp_def]
  simpa only [he,Spec.hash8] using HashTrace8.template_sound w h1 hs

theorem template2_sound : Template2Sound := by
  intro w h1 hs
  have he : PoseidonProgram.evalState w HashTrace2.input = #[w 77887,w 78275] := by
    apply Array.ext
    · simp [PoseidonProgram.evalState,HashTrace2.input]
    · intro i hi hi'
      have hi2 : i<2 := by simpa using hi'
      interval_cases i <;> simp [PoseidonProgram.evalState,HashTrace2.input]
  simpa only [he,Spec.hash2] using HashTrace2.template_sound w h1 hs

theorem left_value (r : Fin 16) {w : Assignment} (hs : Exported.circuit.Sat w) :
    w ((leftNode r.val).start+387) =
      Spec.hash8 ((leftNode r.val).inputs.toArray.map w) :=
  transfer8 template8_sound (leftNode r.val) (by simp [leftNode]) hs.1 (left_sat r hs)

theorem right_value (r : Fin 16) {w : Assignment} (hs : Exported.circuit.Sat w) :
    w ((rightNode r.val).start+387) =
      Spec.hash8 ((rightNode r.val).inputs.toArray.map w) :=
  transfer8 template8_sound (rightNode r.val) (by simp [rightNode]) hs.1 (right_sat r hs)

theorem combine_value (r : Fin 16) {w : Assignment} (hs : Exported.circuit.Sat w) :
    w ((combineNode r.val).start+237) =
      Spec.hash2 (w (combineNode r.val).inputs[0]!) (w (combineNode r.val).inputs[1]!) :=
  transfer2 template2_sound (combineNode r.val) hs.1 (combine_sat r hs)

theorem digest_left_value {w : Assignment} (hs : Exported.circuit.Sat w) :
    w 96767 = Spec.hash8 (node48.inputs.toArray.map w) :=
  transfer8 template8_sound node48 (by decide) hs.1 (sat48 hs)

theorem digest_right_value {w : Assignment} (hs : Exported.circuit.Sat w) :
    w 97155 = Spec.hash8 (node49.inputs.toArray.map w) :=
  transfer8 template8_sound node49 (by decide) hs.1 (sat49 hs)

theorem digest_combine_value {w : Assignment} (hs : Exported.circuit.Sat w) :
    w 97393 = Spec.hash2 (w 96767) (w 97155) :=
  transfer2 template2_sound node50 hs.1 (sat50 hs)

theorem chain_value {w : Assignment} (hs : Exported.circuit.Sat w) :
    w 97632 = Spec.hash2 (w 0) (w 97393) :=
  transfer2 template2_sound node51 hs.1 (sat51 hs)

theorem left (w : Assignment) (hs : Exported.circuit.Sat w) (r : Nat) (hr : r<16) :
    w (HashProgram.rowStart r+563) = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (HashProgram.rowStart r+160+i))) := by
  have h := left_value ⟨r,hr⟩ hs
  have ho : (leftNode r).start+387 = HashProgram.rowStart r+563 := by
    dsimp [leftNode,HashProgram.rowStart]
  have hi : (leftNode r).inputs.toArray.map w =
      (List.range 8).toArray.map (fun i => w (HashProgram.rowStart r+160+i)) := by
    simp only [leftNode,HashProgram.rowStart,List.map_toArray,List.map_map,Function.comp_def]
  rw [ho,hi] at h
  exact h

theorem right (w : Assignment) (hs : Exported.circuit.Sat w) (r : Nat) (hr : r<16) :
    w (HashProgram.rowStart r+951) = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (HashProgram.rowStart r+168+i))) := by
  have h := right_value ⟨r,hr⟩ hs
  have ho : (rightNode r).start+387 = HashProgram.rowStart r+951 := by
    dsimp [rightNode,HashProgram.rowStart]
  have hi : (rightNode r).inputs.toArray.map w =
      (List.range 8).toArray.map (fun i => w (HashProgram.rowStart r+168+i)) := by
    simp only [rightNode,HashProgram.rowStart,List.map_toArray,List.map_map,Function.comp_def]
  rw [ho,hi] at h
  exact h

theorem combine (w : Assignment) (hs : Exported.circuit.Sat w) (r : Nat) (hr : r<16) :
    w (HashProgram.rowStart r+1189) =
      Spec.hash2 (w (HashProgram.rowStart r+563)) (w (HashProgram.rowStart r+951)) := by
  simpa [combineNode,HashProgram.rowStart,Nat.add_assoc] using combine_value ⟨r,hr⟩ hs

theorem digest_left (w : Assignment) (hs : Exported.circuit.Sat w) :
    w 96767 = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (HashProgram.rowStart i+1189))) := by
  have hi : node48.inputs = (List.range 8).map
      (fun i => HashProgram.rowStart i+1189) := by decide
  have h := digest_left_value hs
  rw [hi] at h
  simpa only [List.map_toArray,List.map_map,Function.comp_def] using h

theorem digest_right (w : Assignment) (hs : Exported.circuit.Sat w) :
    w 97155 = Spec.hash8
      ((List.range 8).toArray.map (fun i => w (HashProgram.rowStart (8+i)+1189))) := by
  have hi : node49.inputs = (List.range 8).map
      (fun i => HashProgram.rowStart (8+i)+1189) := by decide
  have h := digest_right_value hs
  rw [hi] at h
  simpa only [List.map_toArray,List.map_map,Function.comp_def] using h

end CircuitCorrectness.HashWiring
