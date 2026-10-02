import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group02

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node16 : Node := ⟨83843, [83447, 83448, 83449, 83450, 83451, 83452, 83453, 83454]⟩
def codes16 : List Row :=
  (([ExportedData.chunk654, ExportedData.chunk655, ExportedData.chunk656, ExportedData.chunk657, ExportedData.chunk658].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 127).take 388
theorem window16 : codedWindow (node16.start-4) 388 = codes16 := by rfl

def node16Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk654.1 ExportedData.chunk654.2).drop 127).take 388
theorem node16Piece0_checked : node16Piece0 =
    ((literalCodes8.drop 0).take 1).map (renameRow (rename8 node16)) := by decide

def node16Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk655.1 ExportedData.chunk655.2).drop 0).take 387
theorem node16Piece1_checked : node16Piece1 =
    ((literalCodes8.drop 1).take 128).map (renameRow (rename8 node16)) := by decide

def node16Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk656.1 ExportedData.chunk656.2).drop 0).take 259
theorem node16Piece2_checked : node16Piece2 =
    ((literalCodes8.drop 129).take 128).map (renameRow (rename8 node16)) := by decide

def node16Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk657.1 ExportedData.chunk657.2).drop 0).take 131
theorem node16Piece3_checked : node16Piece3 =
    ((literalCodes8.drop 257).take 128).map (renameRow (rename8 node16)) := by decide

def node16Piece4 := ((ConcreteBytes.Codes.rows ExportedData.chunk658.1 ExportedData.chunk658.2).drop 0).take 3
theorem node16Piece4_checked : node16Piece4 =
    ((literalCodes8.drop 385).take 3).map (renameRow (rename8 node16)) := by decide

theorem checked16 : codes16 =
    template8Codes.map (renameRow (rename8 node16)) := by
  rw [template8_literal]
  unfold codes16
  rw [sliced_eq]
  change node16Piece0 ++ node16Piece1 ++ node16Piece2 ++ node16Piece3 ++ node16Piece4 ++ [] = _
  rw [node16Piece0_checked, node16Piece1_checked, node16Piece2_checked, node16Piece3_checked, node16Piece4_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat16 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node16) :=
  transport8 node16 (window16.trans checked16) hs

def node17 : Node := ⟨84231, [83842, 84230]⟩
def codes17 : List Row :=
  (([ExportedData.chunk658, ExportedData.chunk659].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 3).take 238
theorem window17 : codedWindow (node17.start-4) 238 = codes17 := by rfl

def node17Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk658.1 ExportedData.chunk658.2).drop 3).take 238
theorem node17Piece0_checked : node17Piece0 =
    ((literalCodes2.drop 0).take 125).map (renameRow (rename2 node17)) := by decide

def node17Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk659.1 ExportedData.chunk659.2).drop 0).take 113
theorem node17Piece1_checked : node17Piece1 =
    ((literalCodes2.drop 125).take 113).map (renameRow (rename2 node17)) := by decide

theorem checked17 : codes17 =
    template2Codes.map (renameRow (rename2 node17)) := by
  rw [template2_literal]
  unfold codes17
  rw [sliced_eq]
  change node17Piece0 ++ node17Piece1 ++ [] = _
  rw [node17Piece0_checked, node17Piece1_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat17 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node17) :=
  transport2 node17 (window17.trans checked17) hs

def node18 : Node := ⟨84646, [84630, 84631, 84632, 84633, 84634, 84635, 84636, 84637]⟩
def codes18 : List Row :=
  (([ExportedData.chunk661, ExportedData.chunk662, ExportedData.chunk663, ExportedData.chunk664].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 34).take 388
theorem window18 : codedWindow (node18.start-4) 388 = codes18 := by rfl

def node18Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk661.1 ExportedData.chunk661.2).drop 34).take 388
theorem node18Piece0_checked : node18Piece0 =
    ((literalCodes8.drop 0).take 94).map (renameRow (rename8 node18)) := by decide

def node18Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk662.1 ExportedData.chunk662.2).drop 0).take 294
theorem node18Piece1_checked : node18Piece1 =
    ((literalCodes8.drop 94).take 128).map (renameRow (rename8 node18)) := by decide

def node18Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk663.1 ExportedData.chunk663.2).drop 0).take 166
theorem node18Piece2_checked : node18Piece2 =
    ((literalCodes8.drop 222).take 128).map (renameRow (rename8 node18)) := by decide

def node18Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk664.1 ExportedData.chunk664.2).drop 0).take 38
theorem node18Piece3_checked : node18Piece3 =
    ((literalCodes8.drop 350).take 38).map (renameRow (rename8 node18)) := by decide

theorem checked18 : codes18 =
    template8Codes.map (renameRow (rename8 node18)) := by
  rw [template8_literal]
  unfold codes18
  rw [sliced_eq]
  change node18Piece0 ++ node18Piece1 ++ node18Piece2 ++ node18Piece3 ++ [] = _
  rw [node18Piece0_checked, node18Piece1_checked, node18Piece2_checked, node18Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat18 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node18) :=
  transport8 node18 (window18.trans checked18) hs

def node19 : Node := ⟨85034, [84638, 84639, 84640, 84641, 84642, 84643, 84644, 84645]⟩
def codes19 : List Row :=
  (([ExportedData.chunk664, ExportedData.chunk665, ExportedData.chunk666, ExportedData.chunk667].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 38).take 388
theorem window19 : codedWindow (node19.start-4) 388 = codes19 := by rfl

def node19Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk664.1 ExportedData.chunk664.2).drop 38).take 388
theorem node19Piece0_checked : node19Piece0 =
    ((literalCodes8.drop 0).take 90).map (renameRow (rename8 node19)) := by decide

def node19Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk665.1 ExportedData.chunk665.2).drop 0).take 298
theorem node19Piece1_checked : node19Piece1 =
    ((literalCodes8.drop 90).take 128).map (renameRow (rename8 node19)) := by decide

def node19Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk666.1 ExportedData.chunk666.2).drop 0).take 170
theorem node19Piece2_checked : node19Piece2 =
    ((literalCodes8.drop 218).take 128).map (renameRow (rename8 node19)) := by decide

def node19Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk667.1 ExportedData.chunk667.2).drop 0).take 42
theorem node19Piece3_checked : node19Piece3 =
    ((literalCodes8.drop 346).take 42).map (renameRow (rename8 node19)) := by decide

theorem checked19 : codes19 =
    template8Codes.map (renameRow (rename8 node19)) := by
  rw [template8_literal]
  unfold codes19
  rw [sliced_eq]
  change node19Piece0 ++ node19Piece1 ++ node19Piece2 ++ node19Piece3 ++ [] = _
  rw [node19Piece0_checked, node19Piece1_checked, node19Piece2_checked, node19Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat19 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node19) :=
  transport8 node19 (window19.trans checked19) hs

end CircuitCorrectness.HashWiring
