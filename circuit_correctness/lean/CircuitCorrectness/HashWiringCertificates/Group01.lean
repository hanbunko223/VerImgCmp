import CircuitCorrectness.HashWiringTemplates

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node04 : Node := ⟨79079, [78683, 78684, 78685, 78686, 78687, 78688, 78689, 78690]⟩
def codes04 : List Row :=
  (([ExportedData.chunk617, ExportedData.chunk618, ExportedData.chunk619, ExportedData.chunk620].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 99).take 388
theorem window04 : codedWindow (node04.start-4) 388 = codes04 := by rfl

def node04Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk617.1 ExportedData.chunk617.2).drop 99).take 388
theorem node04Piece0_checked : node04Piece0 =
    ((literalCodes8.drop 0).take 29).map (renameRow (rename8 node04)) := by decide

def node04Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk618.1 ExportedData.chunk618.2).drop 0).take 359
theorem node04Piece1_checked : node04Piece1 =
    ((literalCodes8.drop 29).take 128).map (renameRow (rename8 node04)) := by decide

def node04Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk619.1 ExportedData.chunk619.2).drop 0).take 231
theorem node04Piece2_checked : node04Piece2 =
    ((literalCodes8.drop 157).take 128).map (renameRow (rename8 node04)) := by decide

def node04Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk620.1 ExportedData.chunk620.2).drop 0).take 103
theorem node04Piece3_checked : node04Piece3 =
    ((literalCodes8.drop 285).take 103).map (renameRow (rename8 node04)) := by decide

theorem checked04 : codes04 =
    template8Codes.map (renameRow (rename8 node04)) := by
  rw [template8_literal]
  unfold codes04
  rw [sliced_eq]
  change node04Piece0 ++ node04Piece1 ++ node04Piece2 ++ node04Piece3 ++ [] = _
  rw [node04Piece0_checked, node04Piece1_checked, node04Piece2_checked, node04Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat04 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node04) :=
  transport8 node04 (window04.trans checked04) hs

def node05 : Node := ⟨79467, [79078, 79466]⟩
def codes05 : List Row :=
  (([ExportedData.chunk620, ExportedData.chunk621, ExportedData.chunk622].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 103).take 238
theorem window05 : codedWindow (node05.start-4) 238 = codes05 := by rfl

def node05Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk620.1 ExportedData.chunk620.2).drop 103).take 238
theorem node05Piece0_checked : node05Piece0 =
    ((literalCodes2.drop 0).take 25).map (renameRow (rename2 node05)) := by decide

def node05Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk621.1 ExportedData.chunk621.2).drop 0).take 213
theorem node05Piece1_checked : node05Piece1 =
    ((literalCodes2.drop 25).take 128).map (renameRow (rename2 node05)) := by decide

def node05Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk622.1 ExportedData.chunk622.2).drop 0).take 85
theorem node05Piece2_checked : node05Piece2 =
    ((literalCodes2.drop 153).take 85).map (renameRow (rename2 node05)) := by decide

theorem checked05 : codes05 =
    template2Codes.map (renameRow (rename2 node05)) := by
  rw [template2_literal]
  unfold codes05
  rw [sliced_eq]
  change node05Piece0 ++ node05Piece1 ++ node05Piece2 ++ [] = _
  rw [node05Piece0_checked, node05Piece1_checked, node05Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat05 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node05) :=
  transport2 node05 (window05.trans checked05) hs

def node06 : Node := ⟨79882, [79866, 79867, 79868, 79869, 79870, 79871, 79872, 79873]⟩
def codes06 : List Row :=
  (([ExportedData.chunk624, ExportedData.chunk625, ExportedData.chunk626, ExportedData.chunk627].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 6).take 388
theorem window06 : codedWindow (node06.start-4) 388 = codes06 := by rfl

def node06Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk624.1 ExportedData.chunk624.2).drop 6).take 388
theorem node06Piece0_checked : node06Piece0 =
    ((literalCodes8.drop 0).take 122).map (renameRow (rename8 node06)) := by decide

def node06Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk625.1 ExportedData.chunk625.2).drop 0).take 266
theorem node06Piece1_checked : node06Piece1 =
    ((literalCodes8.drop 122).take 128).map (renameRow (rename8 node06)) := by decide

def node06Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk626.1 ExportedData.chunk626.2).drop 0).take 138
theorem node06Piece2_checked : node06Piece2 =
    ((literalCodes8.drop 250).take 128).map (renameRow (rename8 node06)) := by decide

def node06Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk627.1 ExportedData.chunk627.2).drop 0).take 10
theorem node06Piece3_checked : node06Piece3 =
    ((literalCodes8.drop 378).take 10).map (renameRow (rename8 node06)) := by decide

theorem checked06 : codes06 =
    template8Codes.map (renameRow (rename8 node06)) := by
  rw [template8_literal]
  unfold codes06
  rw [sliced_eq]
  change node06Piece0 ++ node06Piece1 ++ node06Piece2 ++ node06Piece3 ++ [] = _
  rw [node06Piece0_checked, node06Piece1_checked, node06Piece2_checked, node06Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat06 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node06) :=
  transport8 node06 (window06.trans checked06) hs

def node07 : Node := ⟨80270, [79874, 79875, 79876, 79877, 79878, 79879, 79880, 79881]⟩
def codes07 : List Row :=
  (([ExportedData.chunk627, ExportedData.chunk628, ExportedData.chunk629, ExportedData.chunk630].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 10).take 388
theorem window07 : codedWindow (node07.start-4) 388 = codes07 := by rfl

def node07Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk627.1 ExportedData.chunk627.2).drop 10).take 388
theorem node07Piece0_checked : node07Piece0 =
    ((literalCodes8.drop 0).take 118).map (renameRow (rename8 node07)) := by decide

def node07Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk628.1 ExportedData.chunk628.2).drop 0).take 270
theorem node07Piece1_checked : node07Piece1 =
    ((literalCodes8.drop 118).take 128).map (renameRow (rename8 node07)) := by decide

def node07Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk629.1 ExportedData.chunk629.2).drop 0).take 142
theorem node07Piece2_checked : node07Piece2 =
    ((literalCodes8.drop 246).take 128).map (renameRow (rename8 node07)) := by decide

def node07Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk630.1 ExportedData.chunk630.2).drop 0).take 14
theorem node07Piece3_checked : node07Piece3 =
    ((literalCodes8.drop 374).take 14).map (renameRow (rename8 node07)) := by decide

theorem checked07 : codes07 =
    template8Codes.map (renameRow (rename8 node07)) := by
  rw [template8_literal]
  unfold codes07
  rw [sliced_eq]
  change node07Piece0 ++ node07Piece1 ++ node07Piece2 ++ node07Piece3 ++ [] = _
  rw [node07Piece0_checked, node07Piece1_checked, node07Piece2_checked, node07Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat07 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node07) :=
  transport8 node07 (window07.trans checked07) hs

end CircuitCorrectness.HashWiring
