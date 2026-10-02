import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group01

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node12 : Node := ⟨82264, [82248, 82249, 82250, 82251, 82252, 82253, 82254, 82255]⟩
def codes12 : List Row :=
  (([ExportedData.chunk642, ExportedData.chunk643, ExportedData.chunk644, ExportedData.chunk645].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 84).take 388
theorem window12 : codedWindow (node12.start-4) 388 = codes12 := by rfl

def node12Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk642.1 ExportedData.chunk642.2).drop 84).take 388
theorem node12Piece0_checked : node12Piece0 =
    ((literalCodes8.drop 0).take 44).map (renameRow (rename8 node12)) := by decide

def node12Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk643.1 ExportedData.chunk643.2).drop 0).take 344
theorem node12Piece1_checked : node12Piece1 =
    ((literalCodes8.drop 44).take 128).map (renameRow (rename8 node12)) := by decide

def node12Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk644.1 ExportedData.chunk644.2).drop 0).take 216
theorem node12Piece2_checked : node12Piece2 =
    ((literalCodes8.drop 172).take 128).map (renameRow (rename8 node12)) := by decide

def node12Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk645.1 ExportedData.chunk645.2).drop 0).take 88
theorem node12Piece3_checked : node12Piece3 =
    ((literalCodes8.drop 300).take 88).map (renameRow (rename8 node12)) := by decide

theorem checked12 : codes12 =
    template8Codes.map (renameRow (rename8 node12)) := by
  rw [template8_literal]
  unfold codes12
  rw [sliced_eq]
  change node12Piece0 ++ node12Piece1 ++ node12Piece2 ++ node12Piece3 ++ [] = _
  rw [node12Piece0_checked, node12Piece1_checked, node12Piece2_checked, node12Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat12 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node12) :=
  transport8 node12 (window12.trans checked12) hs

def node13 : Node := ⟨82652, [82256, 82257, 82258, 82259, 82260, 82261, 82262, 82263]⟩
def codes13 : List Row :=
  (([ExportedData.chunk645, ExportedData.chunk646, ExportedData.chunk647, ExportedData.chunk648].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 88).take 388
theorem window13 : codedWindow (node13.start-4) 388 = codes13 := by rfl

def node13Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk645.1 ExportedData.chunk645.2).drop 88).take 388
theorem node13Piece0_checked : node13Piece0 =
    ((literalCodes8.drop 0).take 40).map (renameRow (rename8 node13)) := by decide

def node13Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk646.1 ExportedData.chunk646.2).drop 0).take 348
theorem node13Piece1_checked : node13Piece1 =
    ((literalCodes8.drop 40).take 128).map (renameRow (rename8 node13)) := by decide

def node13Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk647.1 ExportedData.chunk647.2).drop 0).take 220
theorem node13Piece2_checked : node13Piece2 =
    ((literalCodes8.drop 168).take 128).map (renameRow (rename8 node13)) := by decide

def node13Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk648.1 ExportedData.chunk648.2).drop 0).take 92
theorem node13Piece3_checked : node13Piece3 =
    ((literalCodes8.drop 296).take 92).map (renameRow (rename8 node13)) := by decide

theorem checked13 : codes13 =
    template8Codes.map (renameRow (rename8 node13)) := by
  rw [template8_literal]
  unfold codes13
  rw [sliced_eq]
  change node13Piece0 ++ node13Piece1 ++ node13Piece2 ++ node13Piece3 ++ [] = _
  rw [node13Piece0_checked, node13Piece1_checked, node13Piece2_checked, node13Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat13 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node13) :=
  transport8 node13 (window13.trans checked13) hs

def node14 : Node := ⟨83040, [82651, 83039]⟩
def codes14 : List Row :=
  (([ExportedData.chunk648, ExportedData.chunk649, ExportedData.chunk650].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 92).take 238
theorem window14 : codedWindow (node14.start-4) 238 = codes14 := by rfl

def node14Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk648.1 ExportedData.chunk648.2).drop 92).take 238
theorem node14Piece0_checked : node14Piece0 =
    ((literalCodes2.drop 0).take 36).map (renameRow (rename2 node14)) := by decide

def node14Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk649.1 ExportedData.chunk649.2).drop 0).take 202
theorem node14Piece1_checked : node14Piece1 =
    ((literalCodes2.drop 36).take 128).map (renameRow (rename2 node14)) := by decide

def node14Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk650.1 ExportedData.chunk650.2).drop 0).take 74
theorem node14Piece2_checked : node14Piece2 =
    ((literalCodes2.drop 164).take 74).map (renameRow (rename2 node14)) := by decide

theorem checked14 : codes14 =
    template2Codes.map (renameRow (rename2 node14)) := by
  rw [template2_literal]
  unfold codes14
  rw [sliced_eq]
  change node14Piece0 ++ node14Piece1 ++ node14Piece2 ++ [] = _
  rw [node14Piece0_checked, node14Piece1_checked, node14Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat14 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node14) :=
  transport2 node14 (window14.trans checked14) hs

def node15 : Node := ⟨83455, [83439, 83440, 83441, 83442, 83443, 83444, 83445, 83446]⟩
def codes15 : List Row :=
  (([ExportedData.chunk651, ExportedData.chunk652, ExportedData.chunk653, ExportedData.chunk654].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 123).take 388
theorem window15 : codedWindow (node15.start-4) 388 = codes15 := by rfl

def node15Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk651.1 ExportedData.chunk651.2).drop 123).take 388
theorem node15Piece0_checked : node15Piece0 =
    ((literalCodes8.drop 0).take 5).map (renameRow (rename8 node15)) := by decide

def node15Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk652.1 ExportedData.chunk652.2).drop 0).take 383
theorem node15Piece1_checked : node15Piece1 =
    ((literalCodes8.drop 5).take 128).map (renameRow (rename8 node15)) := by decide

def node15Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk653.1 ExportedData.chunk653.2).drop 0).take 255
theorem node15Piece2_checked : node15Piece2 =
    ((literalCodes8.drop 133).take 128).map (renameRow (rename8 node15)) := by decide

def node15Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk654.1 ExportedData.chunk654.2).drop 0).take 127
theorem node15Piece3_checked : node15Piece3 =
    ((literalCodes8.drop 261).take 127).map (renameRow (rename8 node15)) := by decide

theorem checked15 : codes15 =
    template8Codes.map (renameRow (rename8 node15)) := by
  rw [template8_literal]
  unfold codes15
  rw [sliced_eq]
  change node15Piece0 ++ node15Piece1 ++ node15Piece2 ++ node15Piece3 ++ [] = _
  rw [node15Piece0_checked, node15Piece1_checked, node15Piece2_checked, node15Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat15 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node15) :=
  transport8 node15 (window15.trans checked15) hs

end CircuitCorrectness.HashWiring
