import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group09

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node44 : Node := ⟨94950, [94561, 94949]⟩
def codes44 : List Row :=
  (([ExportedData.chunk741, ExportedData.chunk742, ExportedData.chunk743].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 98).take 238
theorem window44 : codedWindow (node44.start-4) 238 = codes44 := by rfl

def node44Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk741.1 ExportedData.chunk741.2).drop 98).take 238
theorem node44Piece0_checked : node44Piece0 =
    ((literalCodes2.drop 0).take 30).map (renameRow (rename2 node44)) := by decide

def node44Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk742.1 ExportedData.chunk742.2).drop 0).take 208
theorem node44Piece1_checked : node44Piece1 =
    ((literalCodes2.drop 30).take 128).map (renameRow (rename2 node44)) := by decide

def node44Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk743.1 ExportedData.chunk743.2).drop 0).take 80
theorem node44Piece2_checked : node44Piece2 =
    ((literalCodes2.drop 158).take 80).map (renameRow (rename2 node44)) := by decide

theorem checked44 : codes44 =
    template2Codes.map (renameRow (rename2 node44)) := by
  rw [template2_literal]
  unfold codes44
  rw [sliced_eq]
  change node44Piece0 ++ node44Piece1 ++ node44Piece2 ++ [] = _
  rw [node44Piece0_checked, node44Piece1_checked, node44Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat44 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node44) :=
  transport2 node44 (window44.trans checked44) hs

def node45 : Node := ⟨95365, [95349, 95350, 95351, 95352, 95353, 95354, 95355, 95356]⟩
def codes45 : List Row :=
  (([ExportedData.chunk745, ExportedData.chunk746, ExportedData.chunk747, ExportedData.chunk748].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 1).take 388
theorem window45 : codedWindow (node45.start-4) 388 = codes45 := by rfl

def node45Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk745.1 ExportedData.chunk745.2).drop 1).take 388
theorem node45Piece0_checked : node45Piece0 =
    ((literalCodes8.drop 0).take 127).map (renameRow (rename8 node45)) := by decide

def node45Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk746.1 ExportedData.chunk746.2).drop 0).take 261
theorem node45Piece1_checked : node45Piece1 =
    ((literalCodes8.drop 127).take 128).map (renameRow (rename8 node45)) := by decide

def node45Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk747.1 ExportedData.chunk747.2).drop 0).take 133
theorem node45Piece2_checked : node45Piece2 =
    ((literalCodes8.drop 255).take 128).map (renameRow (rename8 node45)) := by decide

def node45Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk748.1 ExportedData.chunk748.2).drop 0).take 5
theorem node45Piece3_checked : node45Piece3 =
    ((literalCodes8.drop 383).take 5).map (renameRow (rename8 node45)) := by decide

theorem checked45 : codes45 =
    template8Codes.map (renameRow (rename8 node45)) := by
  rw [template8_literal]
  unfold codes45
  rw [sliced_eq]
  change node45Piece0 ++ node45Piece1 ++ node45Piece2 ++ node45Piece3 ++ [] = _
  rw [node45Piece0_checked, node45Piece1_checked, node45Piece2_checked, node45Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat45 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node45) :=
  transport8 node45 (window45.trans checked45) hs

def node46 : Node := ⟨95753, [95357, 95358, 95359, 95360, 95361, 95362, 95363, 95364]⟩
def codes46 : List Row :=
  (([ExportedData.chunk748, ExportedData.chunk749, ExportedData.chunk750, ExportedData.chunk751].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 5).take 388
theorem window46 : codedWindow (node46.start-4) 388 = codes46 := by rfl

def node46Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk748.1 ExportedData.chunk748.2).drop 5).take 388
theorem node46Piece0_checked : node46Piece0 =
    ((literalCodes8.drop 0).take 123).map (renameRow (rename8 node46)) := by decide

def node46Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk749.1 ExportedData.chunk749.2).drop 0).take 265
theorem node46Piece1_checked : node46Piece1 =
    ((literalCodes8.drop 123).take 128).map (renameRow (rename8 node46)) := by decide

def node46Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk750.1 ExportedData.chunk750.2).drop 0).take 137
theorem node46Piece2_checked : node46Piece2 =
    ((literalCodes8.drop 251).take 128).map (renameRow (rename8 node46)) := by decide

def node46Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk751.1 ExportedData.chunk751.2).drop 0).take 9
theorem node46Piece3_checked : node46Piece3 =
    ((literalCodes8.drop 379).take 9).map (renameRow (rename8 node46)) := by decide

theorem checked46 : codes46 =
    template8Codes.map (renameRow (rename8 node46)) := by
  rw [template8_literal]
  unfold codes46
  rw [sliced_eq]
  change node46Piece0 ++ node46Piece1 ++ node46Piece2 ++ node46Piece3 ++ [] = _
  rw [node46Piece0_checked, node46Piece1_checked, node46Piece2_checked, node46Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat46 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node46) :=
  transport8 node46 (window46.trans checked46) hs

def node47 : Node := ⟨96141, [95752, 96140]⟩
def codes47 : List Row :=
  (([ExportedData.chunk751, ExportedData.chunk752].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 9).take 238
theorem window47 : codedWindow (node47.start-4) 238 = codes47 := by rfl

def node47Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk751.1 ExportedData.chunk751.2).drop 9).take 238
theorem node47Piece0_checked : node47Piece0 =
    ((literalCodes2.drop 0).take 119).map (renameRow (rename2 node47)) := by decide

def node47Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk752.1 ExportedData.chunk752.2).drop 0).take 119
theorem node47Piece1_checked : node47Piece1 =
    ((literalCodes2.drop 119).take 119).map (renameRow (rename2 node47)) := by decide

theorem checked47 : codes47 =
    template2Codes.map (renameRow (rename2 node47)) := by
  rw [template2_literal]
  unfold codes47
  rw [sliced_eq]
  change node47Piece0 ++ node47Piece1 ++ [] = _
  rw [node47Piece0_checked, node47Piece1_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat47 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node47) :=
  transport2 node47 (window47.trans checked47) hs

end CircuitCorrectness.HashWiring
