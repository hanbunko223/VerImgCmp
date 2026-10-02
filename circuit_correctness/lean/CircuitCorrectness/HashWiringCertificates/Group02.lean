import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group00

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node08 : Node := ⟨80658, [80269, 80657]⟩
def codes08 : List Row :=
  (([ExportedData.chunk630, ExportedData.chunk631].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 14).take 238
theorem window08 : codedWindow (node08.start-4) 238 = codes08 := by rfl

def node08Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk630.1 ExportedData.chunk630.2).drop 14).take 238
theorem node08Piece0_checked : node08Piece0 =
    ((literalCodes2.drop 0).take 114).map (renameRow (rename2 node08)) := by decide

def node08Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk631.1 ExportedData.chunk631.2).drop 0).take 124
theorem node08Piece1_checked : node08Piece1 =
    ((literalCodes2.drop 114).take 124).map (renameRow (rename2 node08)) := by decide

theorem checked08 : codes08 =
    template2Codes.map (renameRow (rename2 node08)) := by
  rw [template2_literal]
  unfold codes08
  rw [sliced_eq]
  change node08Piece0 ++ node08Piece1 ++ [] = _
  rw [node08Piece0_checked, node08Piece1_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat08 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node08) :=
  transport2 node08 (window08.trans checked08) hs

def node09 : Node := ⟨81073, [81057, 81058, 81059, 81060, 81061, 81062, 81063, 81064]⟩
def codes09 : List Row :=
  (([ExportedData.chunk633, ExportedData.chunk634, ExportedData.chunk635, ExportedData.chunk636].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 45).take 388
theorem window09 : codedWindow (node09.start-4) 388 = codes09 := by rfl

def node09Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk633.1 ExportedData.chunk633.2).drop 45).take 388
theorem node09Piece0_checked : node09Piece0 =
    ((literalCodes8.drop 0).take 83).map (renameRow (rename8 node09)) := by decide

def node09Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk634.1 ExportedData.chunk634.2).drop 0).take 305
theorem node09Piece1_checked : node09Piece1 =
    ((literalCodes8.drop 83).take 128).map (renameRow (rename8 node09)) := by decide

def node09Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk635.1 ExportedData.chunk635.2).drop 0).take 177
theorem node09Piece2_checked : node09Piece2 =
    ((literalCodes8.drop 211).take 128).map (renameRow (rename8 node09)) := by decide

def node09Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk636.1 ExportedData.chunk636.2).drop 0).take 49
theorem node09Piece3_checked : node09Piece3 =
    ((literalCodes8.drop 339).take 49).map (renameRow (rename8 node09)) := by decide

theorem checked09 : codes09 =
    template8Codes.map (renameRow (rename8 node09)) := by
  rw [template8_literal]
  unfold codes09
  rw [sliced_eq]
  change node09Piece0 ++ node09Piece1 ++ node09Piece2 ++ node09Piece3 ++ [] = _
  rw [node09Piece0_checked, node09Piece1_checked, node09Piece2_checked, node09Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat09 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node09) :=
  transport8 node09 (window09.trans checked09) hs

def node10 : Node := ⟨81461, [81065, 81066, 81067, 81068, 81069, 81070, 81071, 81072]⟩
def codes10 : List Row :=
  (([ExportedData.chunk636, ExportedData.chunk637, ExportedData.chunk638, ExportedData.chunk639].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 49).take 388
theorem window10 : codedWindow (node10.start-4) 388 = codes10 := by rfl

def node10Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk636.1 ExportedData.chunk636.2).drop 49).take 388
theorem node10Piece0_checked : node10Piece0 =
    ((literalCodes8.drop 0).take 79).map (renameRow (rename8 node10)) := by decide

def node10Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk637.1 ExportedData.chunk637.2).drop 0).take 309
theorem node10Piece1_checked : node10Piece1 =
    ((literalCodes8.drop 79).take 128).map (renameRow (rename8 node10)) := by decide

def node10Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk638.1 ExportedData.chunk638.2).drop 0).take 181
theorem node10Piece2_checked : node10Piece2 =
    ((literalCodes8.drop 207).take 128).map (renameRow (rename8 node10)) := by decide

def node10Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk639.1 ExportedData.chunk639.2).drop 0).take 53
theorem node10Piece3_checked : node10Piece3 =
    ((literalCodes8.drop 335).take 53).map (renameRow (rename8 node10)) := by decide

theorem checked10 : codes10 =
    template8Codes.map (renameRow (rename8 node10)) := by
  rw [template8_literal]
  unfold codes10
  rw [sliced_eq]
  change node10Piece0 ++ node10Piece1 ++ node10Piece2 ++ node10Piece3 ++ [] = _
  rw [node10Piece0_checked, node10Piece1_checked, node10Piece2_checked, node10Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat10 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node10) :=
  transport8 node10 (window10.trans checked10) hs

def node11 : Node := ⟨81849, [81460, 81848]⟩
def codes11 : List Row :=
  (([ExportedData.chunk639, ExportedData.chunk640, ExportedData.chunk641].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 53).take 238
theorem window11 : codedWindow (node11.start-4) 238 = codes11 := by rfl

def node11Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk639.1 ExportedData.chunk639.2).drop 53).take 238
theorem node11Piece0_checked : node11Piece0 =
    ((literalCodes2.drop 0).take 75).map (renameRow (rename2 node11)) := by decide

def node11Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk640.1 ExportedData.chunk640.2).drop 0).take 163
theorem node11Piece1_checked : node11Piece1 =
    ((literalCodes2.drop 75).take 128).map (renameRow (rename2 node11)) := by decide

def node11Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk641.1 ExportedData.chunk641.2).drop 0).take 35
theorem node11Piece2_checked : node11Piece2 =
    ((literalCodes2.drop 203).take 35).map (renameRow (rename2 node11)) := by decide

theorem checked11 : codes11 =
    template2Codes.map (renameRow (rename2 node11)) := by
  rw [template2_literal]
  unfold codes11
  rw [sliced_eq]
  change node11Piece0 ++ node11Piece1 ++ node11Piece2 ++ [] = _
  rw [node11Piece0_checked, node11Piece1_checked, node11Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat11 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node11) :=
  transport2 node11 (window11.trans checked11) hs

end CircuitCorrectness.HashWiring
