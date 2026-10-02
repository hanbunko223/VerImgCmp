import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group03

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node20 : Node := ⟨85422, [85033, 85421]⟩
def codes20 : List Row :=
  (([ExportedData.chunk667, ExportedData.chunk668, ExportedData.chunk669].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 42).take 238
theorem window20 : codedWindow (node20.start-4) 238 = codes20 := by rfl

def node20Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk667.1 ExportedData.chunk667.2).drop 42).take 238
theorem node20Piece0_checked : node20Piece0 =
    ((literalCodes2.drop 0).take 86).map (renameRow (rename2 node20)) := by decide

def node20Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk668.1 ExportedData.chunk668.2).drop 0).take 152
theorem node20Piece1_checked : node20Piece1 =
    ((literalCodes2.drop 86).take 128).map (renameRow (rename2 node20)) := by decide

def node20Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk669.1 ExportedData.chunk669.2).drop 0).take 24
theorem node20Piece2_checked : node20Piece2 =
    ((literalCodes2.drop 214).take 24).map (renameRow (rename2 node20)) := by decide

theorem checked20 : codes20 =
    template2Codes.map (renameRow (rename2 node20)) := by
  rw [template2_literal]
  unfold codes20
  rw [sliced_eq]
  change node20Piece0 ++ node20Piece1 ++ node20Piece2 ++ [] = _
  rw [node20Piece0_checked, node20Piece1_checked, node20Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat20 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node20) :=
  transport2 node20 (window20.trans checked20) hs

def node21 : Node := ⟨85837, [85821, 85822, 85823, 85824, 85825, 85826, 85827, 85828]⟩
def codes21 : List Row :=
  (([ExportedData.chunk670, ExportedData.chunk671, ExportedData.chunk672, ExportedData.chunk673].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 73).take 388
theorem window21 : codedWindow (node21.start-4) 388 = codes21 := by rfl

def node21Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk670.1 ExportedData.chunk670.2).drop 73).take 388
theorem node21Piece0_checked : node21Piece0 =
    ((literalCodes8.drop 0).take 55).map (renameRow (rename8 node21)) := by decide

def node21Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk671.1 ExportedData.chunk671.2).drop 0).take 333
theorem node21Piece1_checked : node21Piece1 =
    ((literalCodes8.drop 55).take 128).map (renameRow (rename8 node21)) := by decide

def node21Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk672.1 ExportedData.chunk672.2).drop 0).take 205
theorem node21Piece2_checked : node21Piece2 =
    ((literalCodes8.drop 183).take 128).map (renameRow (rename8 node21)) := by decide

def node21Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk673.1 ExportedData.chunk673.2).drop 0).take 77
theorem node21Piece3_checked : node21Piece3 =
    ((literalCodes8.drop 311).take 77).map (renameRow (rename8 node21)) := by decide

theorem checked21 : codes21 =
    template8Codes.map (renameRow (rename8 node21)) := by
  rw [template8_literal]
  unfold codes21
  rw [sliced_eq]
  change node21Piece0 ++ node21Piece1 ++ node21Piece2 ++ node21Piece3 ++ [] = _
  rw [node21Piece0_checked, node21Piece1_checked, node21Piece2_checked, node21Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat21 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node21) :=
  transport8 node21 (window21.trans checked21) hs

def node22 : Node := ⟨86225, [85829, 85830, 85831, 85832, 85833, 85834, 85835, 85836]⟩
def codes22 : List Row :=
  (([ExportedData.chunk673, ExportedData.chunk674, ExportedData.chunk675, ExportedData.chunk676].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 77).take 388
theorem window22 : codedWindow (node22.start-4) 388 = codes22 := by rfl

def node22Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk673.1 ExportedData.chunk673.2).drop 77).take 388
theorem node22Piece0_checked : node22Piece0 =
    ((literalCodes8.drop 0).take 51).map (renameRow (rename8 node22)) := by decide

def node22Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk674.1 ExportedData.chunk674.2).drop 0).take 337
theorem node22Piece1_checked : node22Piece1 =
    ((literalCodes8.drop 51).take 128).map (renameRow (rename8 node22)) := by decide

def node22Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk675.1 ExportedData.chunk675.2).drop 0).take 209
theorem node22Piece2_checked : node22Piece2 =
    ((literalCodes8.drop 179).take 128).map (renameRow (rename8 node22)) := by decide

def node22Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk676.1 ExportedData.chunk676.2).drop 0).take 81
theorem node22Piece3_checked : node22Piece3 =
    ((literalCodes8.drop 307).take 81).map (renameRow (rename8 node22)) := by decide

theorem checked22 : codes22 =
    template8Codes.map (renameRow (rename8 node22)) := by
  rw [template8_literal]
  unfold codes22
  rw [sliced_eq]
  change node22Piece0 ++ node22Piece1 ++ node22Piece2 ++ node22Piece3 ++ [] = _
  rw [node22Piece0_checked, node22Piece1_checked, node22Piece2_checked, node22Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat22 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node22) :=
  transport8 node22 (window22.trans checked22) hs

def node23 : Node := ⟨86613, [86224, 86612]⟩
def codes23 : List Row :=
  (([ExportedData.chunk676, ExportedData.chunk677, ExportedData.chunk678].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 81).take 238
theorem window23 : codedWindow (node23.start-4) 238 = codes23 := by rfl

def node23Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk676.1 ExportedData.chunk676.2).drop 81).take 238
theorem node23Piece0_checked : node23Piece0 =
    ((literalCodes2.drop 0).take 47).map (renameRow (rename2 node23)) := by decide

def node23Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk677.1 ExportedData.chunk677.2).drop 0).take 191
theorem node23Piece1_checked : node23Piece1 =
    ((literalCodes2.drop 47).take 128).map (renameRow (rename2 node23)) := by decide

def node23Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk678.1 ExportedData.chunk678.2).drop 0).take 63
theorem node23Piece2_checked : node23Piece2 =
    ((literalCodes2.drop 175).take 63).map (renameRow (rename2 node23)) := by decide

theorem checked23 : codes23 =
    template2Codes.map (renameRow (rename2 node23)) := by
  rw [template2_literal]
  unfold codes23
  rw [sliced_eq]
  change node23Piece0 ++ node23Piece1 ++ node23Piece2 ++ [] = _
  rw [node23Piece0_checked, node23Piece1_checked, node23Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat23 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node23) :=
  transport2 node23 (window23.trans checked23) hs

end CircuitCorrectness.HashWiring
