import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group06

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node32 : Node := ⟨90186, [89797, 90185]⟩
def codes32 : List Row :=
  (([ExportedData.chunk704, ExportedData.chunk705, ExportedData.chunk706].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 70).take 238
theorem window32 : codedWindow (node32.start-4) 238 = codes32 := by rfl

def node32Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk704.1 ExportedData.chunk704.2).drop 70).take 238
theorem node32Piece0_checked : node32Piece0 =
    ((literalCodes2.drop 0).take 58).map (renameRow (rename2 node32)) := by decide

def node32Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk705.1 ExportedData.chunk705.2).drop 0).take 180
theorem node32Piece1_checked : node32Piece1 =
    ((literalCodes2.drop 58).take 128).map (renameRow (rename2 node32)) := by decide

def node32Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk706.1 ExportedData.chunk706.2).drop 0).take 52
theorem node32Piece2_checked : node32Piece2 =
    ((literalCodes2.drop 186).take 52).map (renameRow (rename2 node32)) := by decide

theorem checked32 : codes32 =
    template2Codes.map (renameRow (rename2 node32)) := by
  rw [template2_literal]
  unfold codes32
  rw [sliced_eq]
  change node32Piece0 ++ node32Piece1 ++ node32Piece2 ++ [] = _
  rw [node32Piece0_checked, node32Piece1_checked, node32Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat32 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node32) :=
  transport2 node32 (window32.trans checked32) hs

def node33 : Node := ⟨90601, [90585, 90586, 90587, 90588, 90589, 90590, 90591, 90592]⟩
def codes33 : List Row :=
  (([ExportedData.chunk707, ExportedData.chunk708, ExportedData.chunk709, ExportedData.chunk710].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 101).take 388
theorem window33 : codedWindow (node33.start-4) 388 = codes33 := by rfl

def node33Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk707.1 ExportedData.chunk707.2).drop 101).take 388
theorem node33Piece0_checked : node33Piece0 =
    ((literalCodes8.drop 0).take 27).map (renameRow (rename8 node33)) := by decide

def node33Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk708.1 ExportedData.chunk708.2).drop 0).take 361
theorem node33Piece1_checked : node33Piece1 =
    ((literalCodes8.drop 27).take 128).map (renameRow (rename8 node33)) := by decide

def node33Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk709.1 ExportedData.chunk709.2).drop 0).take 233
theorem node33Piece2_checked : node33Piece2 =
    ((literalCodes8.drop 155).take 128).map (renameRow (rename8 node33)) := by decide

def node33Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk710.1 ExportedData.chunk710.2).drop 0).take 105
theorem node33Piece3_checked : node33Piece3 =
    ((literalCodes8.drop 283).take 105).map (renameRow (rename8 node33)) := by decide

theorem checked33 : codes33 =
    template8Codes.map (renameRow (rename8 node33)) := by
  rw [template8_literal]
  unfold codes33
  rw [sliced_eq]
  change node33Piece0 ++ node33Piece1 ++ node33Piece2 ++ node33Piece3 ++ [] = _
  rw [node33Piece0_checked, node33Piece1_checked, node33Piece2_checked, node33Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat33 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node33) :=
  transport8 node33 (window33.trans checked33) hs

def node34 : Node := ⟨90989, [90593, 90594, 90595, 90596, 90597, 90598, 90599, 90600]⟩
def codes34 : List Row :=
  (([ExportedData.chunk710, ExportedData.chunk711, ExportedData.chunk712, ExportedData.chunk713].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 105).take 388
theorem window34 : codedWindow (node34.start-4) 388 = codes34 := by rfl

def node34Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk710.1 ExportedData.chunk710.2).drop 105).take 388
theorem node34Piece0_checked : node34Piece0 =
    ((literalCodes8.drop 0).take 23).map (renameRow (rename8 node34)) := by decide

def node34Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk711.1 ExportedData.chunk711.2).drop 0).take 365
theorem node34Piece1_checked : node34Piece1 =
    ((literalCodes8.drop 23).take 128).map (renameRow (rename8 node34)) := by decide

def node34Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk712.1 ExportedData.chunk712.2).drop 0).take 237
theorem node34Piece2_checked : node34Piece2 =
    ((literalCodes8.drop 151).take 128).map (renameRow (rename8 node34)) := by decide

def node34Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk713.1 ExportedData.chunk713.2).drop 0).take 109
theorem node34Piece3_checked : node34Piece3 =
    ((literalCodes8.drop 279).take 109).map (renameRow (rename8 node34)) := by decide

theorem checked34 : codes34 =
    template8Codes.map (renameRow (rename8 node34)) := by
  rw [template8_literal]
  unfold codes34
  rw [sliced_eq]
  change node34Piece0 ++ node34Piece1 ++ node34Piece2 ++ node34Piece3 ++ [] = _
  rw [node34Piece0_checked, node34Piece1_checked, node34Piece2_checked, node34Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat34 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node34) :=
  transport8 node34 (window34.trans checked34) hs

def node35 : Node := ⟨91377, [90988, 91376]⟩
def codes35 : List Row :=
  (([ExportedData.chunk713, ExportedData.chunk714, ExportedData.chunk715].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 109).take 238
theorem window35 : codedWindow (node35.start-4) 238 = codes35 := by rfl

def node35Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk713.1 ExportedData.chunk713.2).drop 109).take 238
theorem node35Piece0_checked : node35Piece0 =
    ((literalCodes2.drop 0).take 19).map (renameRow (rename2 node35)) := by decide

def node35Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk714.1 ExportedData.chunk714.2).drop 0).take 219
theorem node35Piece1_checked : node35Piece1 =
    ((literalCodes2.drop 19).take 128).map (renameRow (rename2 node35)) := by decide

def node35Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk715.1 ExportedData.chunk715.2).drop 0).take 91
theorem node35Piece2_checked : node35Piece2 =
    ((literalCodes2.drop 147).take 91).map (renameRow (rename2 node35)) := by decide

theorem checked35 : codes35 =
    template2Codes.map (renameRow (rename2 node35)) := by
  rw [template2_literal]
  unfold codes35
  rw [sliced_eq]
  change node35Piece0 ++ node35Piece1 ++ node35Piece2 ++ [] = _
  rw [node35Piece0_checked, node35Piece1_checked, node35Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat35 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node35) :=
  transport2 node35 (window35.trans checked35) hs

end CircuitCorrectness.HashWiring
