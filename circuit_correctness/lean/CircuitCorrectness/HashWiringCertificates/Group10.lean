import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group08

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node40 : Node := ⟨93371, [92975, 92976, 92977, 92978, 92979, 92980, 92981, 92982]⟩
def codes40 : List Row :=
  (([ExportedData.chunk729, ExportedData.chunk730, ExportedData.chunk731, ExportedData.chunk732].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 55).take 388
theorem window40 : codedWindow (node40.start-4) 388 = codes40 := by rfl

def node40Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk729.1 ExportedData.chunk729.2).drop 55).take 388
theorem node40Piece0_checked : node40Piece0 =
    ((literalCodes8.drop 0).take 73).map (renameRow (rename8 node40)) := by decide

def node40Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk730.1 ExportedData.chunk730.2).drop 0).take 315
theorem node40Piece1_checked : node40Piece1 =
    ((literalCodes8.drop 73).take 128).map (renameRow (rename8 node40)) := by decide

def node40Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk731.1 ExportedData.chunk731.2).drop 0).take 187
theorem node40Piece2_checked : node40Piece2 =
    ((literalCodes8.drop 201).take 128).map (renameRow (rename8 node40)) := by decide

def node40Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk732.1 ExportedData.chunk732.2).drop 0).take 59
theorem node40Piece3_checked : node40Piece3 =
    ((literalCodes8.drop 329).take 59).map (renameRow (rename8 node40)) := by decide

theorem checked40 : codes40 =
    template8Codes.map (renameRow (rename8 node40)) := by
  rw [template8_literal]
  unfold codes40
  rw [sliced_eq]
  change node40Piece0 ++ node40Piece1 ++ node40Piece2 ++ node40Piece3 ++ [] = _
  rw [node40Piece0_checked, node40Piece1_checked, node40Piece2_checked, node40Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat40 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node40) :=
  transport8 node40 (window40.trans checked40) hs

def node41 : Node := ⟨93759, [93370, 93758]⟩
def codes41 : List Row :=
  (([ExportedData.chunk732, ExportedData.chunk733, ExportedData.chunk734].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 59).take 238
theorem window41 : codedWindow (node41.start-4) 238 = codes41 := by rfl

def node41Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk732.1 ExportedData.chunk732.2).drop 59).take 238
theorem node41Piece0_checked : node41Piece0 =
    ((literalCodes2.drop 0).take 69).map (renameRow (rename2 node41)) := by decide

def node41Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk733.1 ExportedData.chunk733.2).drop 0).take 169
theorem node41Piece1_checked : node41Piece1 =
    ((literalCodes2.drop 69).take 128).map (renameRow (rename2 node41)) := by decide

def node41Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk734.1 ExportedData.chunk734.2).drop 0).take 41
theorem node41Piece2_checked : node41Piece2 =
    ((literalCodes2.drop 197).take 41).map (renameRow (rename2 node41)) := by decide

theorem checked41 : codes41 =
    template2Codes.map (renameRow (rename2 node41)) := by
  rw [template2_literal]
  unfold codes41
  rw [sliced_eq]
  change node41Piece0 ++ node41Piece1 ++ node41Piece2 ++ [] = _
  rw [node41Piece0_checked, node41Piece1_checked, node41Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat41 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node41) :=
  transport2 node41 (window41.trans checked41) hs

def node42 : Node := ⟨94174, [94158, 94159, 94160, 94161, 94162, 94163, 94164, 94165]⟩
def codes42 : List Row :=
  (([ExportedData.chunk735, ExportedData.chunk736, ExportedData.chunk737, ExportedData.chunk738].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 90).take 388
theorem window42 : codedWindow (node42.start-4) 388 = codes42 := by rfl

def node42Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk735.1 ExportedData.chunk735.2).drop 90).take 388
theorem node42Piece0_checked : node42Piece0 =
    ((literalCodes8.drop 0).take 38).map (renameRow (rename8 node42)) := by decide

def node42Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk736.1 ExportedData.chunk736.2).drop 0).take 350
theorem node42Piece1_checked : node42Piece1 =
    ((literalCodes8.drop 38).take 128).map (renameRow (rename8 node42)) := by decide

def node42Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk737.1 ExportedData.chunk737.2).drop 0).take 222
theorem node42Piece2_checked : node42Piece2 =
    ((literalCodes8.drop 166).take 128).map (renameRow (rename8 node42)) := by decide

def node42Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk738.1 ExportedData.chunk738.2).drop 0).take 94
theorem node42Piece3_checked : node42Piece3 =
    ((literalCodes8.drop 294).take 94).map (renameRow (rename8 node42)) := by decide

theorem checked42 : codes42 =
    template8Codes.map (renameRow (rename8 node42)) := by
  rw [template8_literal]
  unfold codes42
  rw [sliced_eq]
  change node42Piece0 ++ node42Piece1 ++ node42Piece2 ++ node42Piece3 ++ [] = _
  rw [node42Piece0_checked, node42Piece1_checked, node42Piece2_checked, node42Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat42 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node42) :=
  transport8 node42 (window42.trans checked42) hs

def node43 : Node := ⟨94562, [94166, 94167, 94168, 94169, 94170, 94171, 94172, 94173]⟩
def codes43 : List Row :=
  (([ExportedData.chunk738, ExportedData.chunk739, ExportedData.chunk740, ExportedData.chunk741].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 94).take 388
theorem window43 : codedWindow (node43.start-4) 388 = codes43 := by rfl

def node43Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk738.1 ExportedData.chunk738.2).drop 94).take 388
theorem node43Piece0_checked : node43Piece0 =
    ((literalCodes8.drop 0).take 34).map (renameRow (rename8 node43)) := by decide

def node43Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk739.1 ExportedData.chunk739.2).drop 0).take 354
theorem node43Piece1_checked : node43Piece1 =
    ((literalCodes8.drop 34).take 128).map (renameRow (rename8 node43)) := by decide

def node43Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk740.1 ExportedData.chunk740.2).drop 0).take 226
theorem node43Piece2_checked : node43Piece2 =
    ((literalCodes8.drop 162).take 128).map (renameRow (rename8 node43)) := by decide

def node43Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk741.1 ExportedData.chunk741.2).drop 0).take 98
theorem node43Piece3_checked : node43Piece3 =
    ((literalCodes8.drop 290).take 98).map (renameRow (rename8 node43)) := by decide

theorem checked43 : codes43 =
    template8Codes.map (renameRow (rename8 node43)) := by
  rw [template8_literal]
  unfold codes43
  rw [sliced_eq]
  change node43Piece0 ++ node43Piece1 ++ node43Piece2 ++ node43Piece3 ++ [] = _
  rw [node43Piece0_checked, node43Piece1_checked, node43Piece2_checked, node43Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat43 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node43) :=
  transport8 node43 (window43.trans checked43) hs

end CircuitCorrectness.HashWiring
