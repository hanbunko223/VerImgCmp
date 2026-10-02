import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group07

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node36 : Node := ⟨91792, [91776, 91777, 91778, 91779, 91780, 91781, 91782, 91783]⟩
def codes36 : List Row :=
  (([ExportedData.chunk717, ExportedData.chunk718, ExportedData.chunk719, ExportedData.chunk720].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 12).take 388
theorem window36 : codedWindow (node36.start-4) 388 = codes36 := by rfl

def node36Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk717.1 ExportedData.chunk717.2).drop 12).take 388
theorem node36Piece0_checked : node36Piece0 =
    ((literalCodes8.drop 0).take 116).map (renameRow (rename8 node36)) := by decide

def node36Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk718.1 ExportedData.chunk718.2).drop 0).take 272
theorem node36Piece1_checked : node36Piece1 =
    ((literalCodes8.drop 116).take 128).map (renameRow (rename8 node36)) := by decide

def node36Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk719.1 ExportedData.chunk719.2).drop 0).take 144
theorem node36Piece2_checked : node36Piece2 =
    ((literalCodes8.drop 244).take 128).map (renameRow (rename8 node36)) := by decide

def node36Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk720.1 ExportedData.chunk720.2).drop 0).take 16
theorem node36Piece3_checked : node36Piece3 =
    ((literalCodes8.drop 372).take 16).map (renameRow (rename8 node36)) := by decide

theorem checked36 : codes36 =
    template8Codes.map (renameRow (rename8 node36)) := by
  rw [template8_literal]
  unfold codes36
  rw [sliced_eq]
  change node36Piece0 ++ node36Piece1 ++ node36Piece2 ++ node36Piece3 ++ [] = _
  rw [node36Piece0_checked, node36Piece1_checked, node36Piece2_checked, node36Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat36 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node36) :=
  transport8 node36 (window36.trans checked36) hs

def node37 : Node := ⟨92180, [91784, 91785, 91786, 91787, 91788, 91789, 91790, 91791]⟩
def codes37 : List Row :=
  (([ExportedData.chunk720, ExportedData.chunk721, ExportedData.chunk722, ExportedData.chunk723].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 16).take 388
theorem window37 : codedWindow (node37.start-4) 388 = codes37 := by rfl

def node37Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk720.1 ExportedData.chunk720.2).drop 16).take 388
theorem node37Piece0_checked : node37Piece0 =
    ((literalCodes8.drop 0).take 112).map (renameRow (rename8 node37)) := by decide

def node37Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk721.1 ExportedData.chunk721.2).drop 0).take 276
theorem node37Piece1_checked : node37Piece1 =
    ((literalCodes8.drop 112).take 128).map (renameRow (rename8 node37)) := by decide

def node37Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk722.1 ExportedData.chunk722.2).drop 0).take 148
theorem node37Piece2_checked : node37Piece2 =
    ((literalCodes8.drop 240).take 128).map (renameRow (rename8 node37)) := by decide

def node37Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk723.1 ExportedData.chunk723.2).drop 0).take 20
theorem node37Piece3_checked : node37Piece3 =
    ((literalCodes8.drop 368).take 20).map (renameRow (rename8 node37)) := by decide

theorem checked37 : codes37 =
    template8Codes.map (renameRow (rename8 node37)) := by
  rw [template8_literal]
  unfold codes37
  rw [sliced_eq]
  change node37Piece0 ++ node37Piece1 ++ node37Piece2 ++ node37Piece3 ++ [] = _
  rw [node37Piece0_checked, node37Piece1_checked, node37Piece2_checked, node37Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat37 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node37) :=
  transport8 node37 (window37.trans checked37) hs

def node38 : Node := ⟨92568, [92179, 92567]⟩
def codes38 : List Row :=
  (([ExportedData.chunk723, ExportedData.chunk724, ExportedData.chunk725].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 20).take 238
theorem window38 : codedWindow (node38.start-4) 238 = codes38 := by rfl

def node38Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk723.1 ExportedData.chunk723.2).drop 20).take 238
theorem node38Piece0_checked : node38Piece0 =
    ((literalCodes2.drop 0).take 108).map (renameRow (rename2 node38)) := by decide

def node38Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk724.1 ExportedData.chunk724.2).drop 0).take 130
theorem node38Piece1_checked : node38Piece1 =
    ((literalCodes2.drop 108).take 128).map (renameRow (rename2 node38)) := by decide

def node38Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk725.1 ExportedData.chunk725.2).drop 0).take 2
theorem node38Piece2_checked : node38Piece2 =
    ((literalCodes2.drop 236).take 2).map (renameRow (rename2 node38)) := by decide

theorem checked38 : codes38 =
    template2Codes.map (renameRow (rename2 node38)) := by
  rw [template2_literal]
  unfold codes38
  rw [sliced_eq]
  change node38Piece0 ++ node38Piece1 ++ node38Piece2 ++ [] = _
  rw [node38Piece0_checked, node38Piece1_checked, node38Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat38 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node38) :=
  transport2 node38 (window38.trans checked38) hs

def node39 : Node := ⟨92983, [92967, 92968, 92969, 92970, 92971, 92972, 92973, 92974]⟩
def codes39 : List Row :=
  (([ExportedData.chunk726, ExportedData.chunk727, ExportedData.chunk728, ExportedData.chunk729].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 51).take 388
theorem window39 : codedWindow (node39.start-4) 388 = codes39 := by rfl

def node39Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk726.1 ExportedData.chunk726.2).drop 51).take 388
theorem node39Piece0_checked : node39Piece0 =
    ((literalCodes8.drop 0).take 77).map (renameRow (rename8 node39)) := by decide

def node39Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk727.1 ExportedData.chunk727.2).drop 0).take 311
theorem node39Piece1_checked : node39Piece1 =
    ((literalCodes8.drop 77).take 128).map (renameRow (rename8 node39)) := by decide

def node39Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk728.1 ExportedData.chunk728.2).drop 0).take 183
theorem node39Piece2_checked : node39Piece2 =
    ((literalCodes8.drop 205).take 128).map (renameRow (rename8 node39)) := by decide

def node39Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk729.1 ExportedData.chunk729.2).drop 0).take 55
theorem node39Piece3_checked : node39Piece3 =
    ((literalCodes8.drop 333).take 55).map (renameRow (rename8 node39)) := by decide

theorem checked39 : codes39 =
    template8Codes.map (renameRow (rename8 node39)) := by
  rw [template8_literal]
  unfold codes39
  rw [sliced_eq]
  change node39Piece0 ++ node39Piece1 ++ node39Piece2 ++ node39Piece3 ++ [] = _
  rw [node39Piece0_checked, node39Piece1_checked, node39Piece2_checked, node39Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat39 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node39) :=
  transport8 node39 (window39.trans checked39) hs

end CircuitCorrectness.HashWiring
