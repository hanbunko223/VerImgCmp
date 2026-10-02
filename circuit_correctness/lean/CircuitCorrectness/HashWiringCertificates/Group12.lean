import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group10

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node48 : Node := ⟨96380, [78513, 79704, 80895, 82086, 83277, 84468, 85659, 86850]⟩
def codes48 : List Row :=
  (([ExportedData.chunk752, ExportedData.chunk753, ExportedData.chunk754, ExportedData.chunk755].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 120).take 388
theorem window48 : codedWindow (node48.start-4) 388 = codes48 := by rfl

def node48Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk752.1 ExportedData.chunk752.2).drop 120).take 388
theorem node48Piece0_checked : node48Piece0 =
    ((literalCodes8.drop 0).take 8).map (renameRow (rename8 node48)) := by decide

def node48Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk753.1 ExportedData.chunk753.2).drop 0).take 380
theorem node48Piece1_checked : node48Piece1 =
    ((literalCodes8.drop 8).take 128).map (renameRow (rename8 node48)) := by decide

def node48Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk754.1 ExportedData.chunk754.2).drop 0).take 252
theorem node48Piece2_checked : node48Piece2 =
    ((literalCodes8.drop 136).take 128).map (renameRow (rename8 node48)) := by decide

def node48Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk755.1 ExportedData.chunk755.2).drop 0).take 124
theorem node48Piece3_checked : node48Piece3 =
    ((literalCodes8.drop 264).take 124).map (renameRow (rename8 node48)) := by decide

theorem checked48 : codes48 =
    template8Codes.map (renameRow (rename8 node48)) := by
  rw [template8_literal]
  unfold codes48
  rw [sliced_eq]
  change node48Piece0 ++ node48Piece1 ++ node48Piece2 ++ node48Piece3 ++ [] = _
  rw [node48Piece0_checked, node48Piece1_checked, node48Piece2_checked, node48Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat48 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node48) :=
  transport8 node48 (window48.trans checked48) hs

def node49 : Node := ⟨96768, [88041, 89232, 90423, 91614, 92805, 93996, 95187, 96378]⟩
def codes49 : List Row :=
  (([ExportedData.chunk755, ExportedData.chunk756, ExportedData.chunk757, ExportedData.chunk758].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 124).take 388
theorem window49 : codedWindow (node49.start-4) 388 = codes49 := by rfl

def node49Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk755.1 ExportedData.chunk755.2).drop 124).take 388
theorem node49Piece0_checked : node49Piece0 =
    ((literalCodes8.drop 0).take 4).map (renameRow (rename8 node49)) := by decide

def node49Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk756.1 ExportedData.chunk756.2).drop 0).take 384
theorem node49Piece1_checked : node49Piece1 =
    ((literalCodes8.drop 4).take 128).map (renameRow (rename8 node49)) := by decide

def node49Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk757.1 ExportedData.chunk757.2).drop 0).take 256
theorem node49Piece2_checked : node49Piece2 =
    ((literalCodes8.drop 132).take 128).map (renameRow (rename8 node49)) := by decide

def node49Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk758.1 ExportedData.chunk758.2).drop 0).take 128
theorem node49Piece3_checked : node49Piece3 =
    ((literalCodes8.drop 260).take 128).map (renameRow (rename8 node49)) := by decide

theorem checked49 : codes49 =
    template8Codes.map (renameRow (rename8 node49)) := by
  rw [template8_literal]
  unfold codes49
  rw [sliced_eq]
  change node49Piece0 ++ node49Piece1 ++ node49Piece2 ++ node49Piece3 ++ [] = _
  rw [node49Piece0_checked, node49Piece1_checked, node49Piece2_checked, node49Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat49 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node49) :=
  transport8 node49 (window49.trans checked49) hs

def node50 : Node := ⟨97156, [96767, 97155]⟩
def codes50 : List Row :=
  (([ExportedData.chunk759, ExportedData.chunk760].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 0).take 238
theorem window50 : codedWindow (node50.start-4) 238 = codes50 := by rfl

def node50Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk759.1 ExportedData.chunk759.2).drop 0).take 238
theorem node50Piece0_checked : node50Piece0 =
    ((literalCodes2.drop 0).take 128).map (renameRow (rename2 node50)) := by decide

def node50Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk760.1 ExportedData.chunk760.2).drop 0).take 110
theorem node50Piece1_checked : node50Piece1 =
    ((literalCodes2.drop 128).take 110).map (renameRow (rename2 node50)) := by decide

theorem checked50 : codes50 =
    template2Codes.map (renameRow (rename2 node50)) := by
  rw [template2_literal]
  unfold codes50
  rw [sliced_eq]
  change node50Piece0 ++ node50Piece1 ++ [] = _
  rw [node50Piece0_checked, node50Piece1_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat50 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node50) :=
  transport2 node50 (window50.trans checked50) hs

def node51 : Node := ⟨97395, [0, 97393]⟩
def codes51 : List Row :=
  (([ExportedData.chunk760, ExportedData.chunk761, ExportedData.chunk762].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 111).take 238
theorem window51 : codedWindow (node51.start-4) 238 = codes51 := by rfl

def node51Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk760.1 ExportedData.chunk760.2).drop 111).take 238
theorem node51Piece0_checked : node51Piece0 =
    ((literalCodes2.drop 0).take 17).map (renameRow (rename2 node51)) := by decide

def node51Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk761.1 ExportedData.chunk761.2).drop 0).take 221
theorem node51Piece1_checked : node51Piece1 =
    ((literalCodes2.drop 17).take 128).map (renameRow (rename2 node51)) := by decide

def node51Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk762.1 ExportedData.chunk762.2).drop 0).take 93
theorem node51Piece2_checked : node51Piece2 =
    ((literalCodes2.drop 145).take 93).map (renameRow (rename2 node51)) := by decide

theorem checked51 : codes51 =
    template2Codes.map (renameRow (rename2 node51)) := by
  rw [template2_literal]
  unfold codes51
  rw [sliced_eq]
  change node51Piece0 ++ node51Piece1 ++ node51Piece2 ++ [] = _
  rw [node51Piece0_checked, node51Piece1_checked, node51Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat51 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node51) :=
  transport2 node51 (window51.trans checked51) hs

end CircuitCorrectness.HashWiring
