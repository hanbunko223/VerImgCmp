import CircuitCorrectness.HashWiringCertificates.Group11
import CircuitCorrectness.HashWiringCertificates.Group12

namespace CircuitCorrectness.HashWiring

def leftNode (r : Nat) : Node := ⟨77324+1191*r+176, (List.range 8).map (fun i => 77324+1191*r+160+i)⟩
theorem left_sat (r : Fin 16) {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 (leftNode r.val)) := by
  fin_cases r
  · exact sat00 hs
  · exact sat03 hs
  · exact sat06 hs
  · exact sat09 hs
  · exact sat12 hs
  · exact sat15 hs
  · exact sat18 hs
  · exact sat21 hs
  · exact sat24 hs
  · exact sat27 hs
  · exact sat30 hs
  · exact sat33 hs
  · exact sat36 hs
  · exact sat39 hs
  · exact sat42 hs
  · exact sat45 hs

def rightNode (r : Nat) : Node := ⟨77324+1191*r+564, (List.range 8).map (fun i => 77324+1191*r+168+i)⟩
theorem right_sat (r : Fin 16) {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 (rightNode r.val)) := by
  fin_cases r
  · exact sat01 hs
  · exact sat04 hs
  · exact sat07 hs
  · exact sat10 hs
  · exact sat13 hs
  · exact sat16 hs
  · exact sat19 hs
  · exact sat22 hs
  · exact sat25 hs
  · exact sat28 hs
  · exact sat31 hs
  · exact sat34 hs
  · exact sat37 hs
  · exact sat40 hs
  · exact sat43 hs
  · exact sat46 hs

def combineNode (r : Nat) : Node := ⟨77324+1191*r+952, [77324+1191*r+563,77324+1191*r+951]⟩
theorem combine_sat (r : Fin 16) {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 (combineNode r.val)) := by
  fin_cases r
  · exact sat02 hs
  · exact sat05 hs
  · exact sat08 hs
  · exact sat11 hs
  · exact sat14 hs
  · exact sat17 hs
  · exact sat20 hs
  · exact sat23 hs
  · exact sat26 hs
  · exact sat29 hs
  · exact sat32 hs
  · exact sat35 hs
  · exact sat38 hs
  · exact sat41 hs
  · exact sat44 hs
  · exact sat47 hs

end CircuitCorrectness.HashWiring
