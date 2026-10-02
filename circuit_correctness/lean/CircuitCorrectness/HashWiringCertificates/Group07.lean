import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group05

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node28 : Node := ⟨88607, [88211, 88212, 88213, 88214, 88215, 88216, 88217, 88218]⟩
def codes28 : List Row :=
  (([ExportedData.chunk692, ExportedData.chunk693, ExportedData.chunk694, ExportedData.chunk695].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 27).take 388
theorem window28 : codedWindow (node28.start-4) 388 = codes28 := by rfl

def node28Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk692.1 ExportedData.chunk692.2).drop 27).take 388
theorem node28Piece0_checked : node28Piece0 =
    ((literalCodes8.drop 0).take 101).map (renameRow (rename8 node28)) := by decide

def node28Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk693.1 ExportedData.chunk693.2).drop 0).take 287
theorem node28Piece1_checked : node28Piece1 =
    ((literalCodes8.drop 101).take 128).map (renameRow (rename8 node28)) := by decide

def node28Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk694.1 ExportedData.chunk694.2).drop 0).take 159
theorem node28Piece2_checked : node28Piece2 =
    ((literalCodes8.drop 229).take 128).map (renameRow (rename8 node28)) := by decide

def node28Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk695.1 ExportedData.chunk695.2).drop 0).take 31
theorem node28Piece3_checked : node28Piece3 =
    ((literalCodes8.drop 357).take 31).map (renameRow (rename8 node28)) := by decide

theorem checked28 : codes28 =
    template8Codes.map (renameRow (rename8 node28)) := by
  rw [template8_literal]
  unfold codes28
  rw [sliced_eq]
  change node28Piece0 ++ node28Piece1 ++ node28Piece2 ++ node28Piece3 ++ [] = _
  rw [node28Piece0_checked, node28Piece1_checked, node28Piece2_checked, node28Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat28 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node28) :=
  transport8 node28 (window28.trans checked28) hs

def node29 : Node := ⟨88995, [88606, 88994]⟩
def codes29 : List Row :=
  (([ExportedData.chunk695, ExportedData.chunk696, ExportedData.chunk697].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 31).take 238
theorem window29 : codedWindow (node29.start-4) 238 = codes29 := by rfl

def node29Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk695.1 ExportedData.chunk695.2).drop 31).take 238
theorem node29Piece0_checked : node29Piece0 =
    ((literalCodes2.drop 0).take 97).map (renameRow (rename2 node29)) := by decide

def node29Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk696.1 ExportedData.chunk696.2).drop 0).take 141
theorem node29Piece1_checked : node29Piece1 =
    ((literalCodes2.drop 97).take 128).map (renameRow (rename2 node29)) := by decide

def node29Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk697.1 ExportedData.chunk697.2).drop 0).take 13
theorem node29Piece2_checked : node29Piece2 =
    ((literalCodes2.drop 225).take 13).map (renameRow (rename2 node29)) := by decide

theorem checked29 : codes29 =
    template2Codes.map (renameRow (rename2 node29)) := by
  rw [template2_literal]
  unfold codes29
  rw [sliced_eq]
  change node29Piece0 ++ node29Piece1 ++ node29Piece2 ++ [] = _
  rw [node29Piece0_checked, node29Piece1_checked, node29Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat29 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node29) :=
  transport2 node29 (window29.trans checked29) hs

def node30 : Node := ⟨89410, [89394, 89395, 89396, 89397, 89398, 89399, 89400, 89401]⟩
def codes30 : List Row :=
  (([ExportedData.chunk698, ExportedData.chunk699, ExportedData.chunk700, ExportedData.chunk701].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 62).take 388
theorem window30 : codedWindow (node30.start-4) 388 = codes30 := by rfl

def node30Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk698.1 ExportedData.chunk698.2).drop 62).take 388
theorem node30Piece0_checked : node30Piece0 =
    ((literalCodes8.drop 0).take 66).map (renameRow (rename8 node30)) := by decide

def node30Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk699.1 ExportedData.chunk699.2).drop 0).take 322
theorem node30Piece1_checked : node30Piece1 =
    ((literalCodes8.drop 66).take 128).map (renameRow (rename8 node30)) := by decide

def node30Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk700.1 ExportedData.chunk700.2).drop 0).take 194
theorem node30Piece2_checked : node30Piece2 =
    ((literalCodes8.drop 194).take 128).map (renameRow (rename8 node30)) := by decide

def node30Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk701.1 ExportedData.chunk701.2).drop 0).take 66
theorem node30Piece3_checked : node30Piece3 =
    ((literalCodes8.drop 322).take 66).map (renameRow (rename8 node30)) := by decide

theorem checked30 : codes30 =
    template8Codes.map (renameRow (rename8 node30)) := by
  rw [template8_literal]
  unfold codes30
  rw [sliced_eq]
  change node30Piece0 ++ node30Piece1 ++ node30Piece2 ++ node30Piece3 ++ [] = _
  rw [node30Piece0_checked, node30Piece1_checked, node30Piece2_checked, node30Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat30 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node30) :=
  transport8 node30 (window30.trans checked30) hs

def node31 : Node := ⟨89798, [89402, 89403, 89404, 89405, 89406, 89407, 89408, 89409]⟩
def codes31 : List Row :=
  (([ExportedData.chunk701, ExportedData.chunk702, ExportedData.chunk703, ExportedData.chunk704].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 66).take 388
theorem window31 : codedWindow (node31.start-4) 388 = codes31 := by rfl

def node31Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk701.1 ExportedData.chunk701.2).drop 66).take 388
theorem node31Piece0_checked : node31Piece0 =
    ((literalCodes8.drop 0).take 62).map (renameRow (rename8 node31)) := by decide

def node31Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk702.1 ExportedData.chunk702.2).drop 0).take 326
theorem node31Piece1_checked : node31Piece1 =
    ((literalCodes8.drop 62).take 128).map (renameRow (rename8 node31)) := by decide

def node31Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk703.1 ExportedData.chunk703.2).drop 0).take 198
theorem node31Piece2_checked : node31Piece2 =
    ((literalCodes8.drop 190).take 128).map (renameRow (rename8 node31)) := by decide

def node31Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk704.1 ExportedData.chunk704.2).drop 0).take 70
theorem node31Piece3_checked : node31Piece3 =
    ((literalCodes8.drop 318).take 70).map (renameRow (rename8 node31)) := by decide

theorem checked31 : codes31 =
    template8Codes.map (renameRow (rename8 node31)) := by
  rw [template8_literal]
  unfold codes31
  rw [sliced_eq]
  change node31Piece0 ++ node31Piece1 ++ node31Piece2 ++ node31Piece3 ++ [] = _
  rw [node31Piece0_checked, node31Piece1_checked, node31Piece2_checked, node31Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat31 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node31) :=
  transport8 node31 (window31.trans checked31) hs

end CircuitCorrectness.HashWiring
