import CircuitCorrectness.HashWiringTemplates
import CircuitCorrectness.HashWiringCertificates.Group04

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node24 : Node := ⟨87028, [87012, 87013, 87014, 87015, 87016, 87017, 87018, 87019]⟩
def codes24 : List Row :=
  (([ExportedData.chunk679, ExportedData.chunk680, ExportedData.chunk681, ExportedData.chunk682].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 112).take 388
theorem window24 : codedWindow (node24.start-4) 388 = codes24 := by rfl

def node24Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk679.1 ExportedData.chunk679.2).drop 112).take 388
theorem node24Piece0_checked : node24Piece0 =
    ((literalCodes8.drop 0).take 16).map (renameRow (rename8 node24)) := by decide

def node24Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk680.1 ExportedData.chunk680.2).drop 0).take 372
theorem node24Piece1_checked : node24Piece1 =
    ((literalCodes8.drop 16).take 128).map (renameRow (rename8 node24)) := by decide

def node24Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk681.1 ExportedData.chunk681.2).drop 0).take 244
theorem node24Piece2_checked : node24Piece2 =
    ((literalCodes8.drop 144).take 128).map (renameRow (rename8 node24)) := by decide

def node24Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk682.1 ExportedData.chunk682.2).drop 0).take 116
theorem node24Piece3_checked : node24Piece3 =
    ((literalCodes8.drop 272).take 116).map (renameRow (rename8 node24)) := by decide

theorem checked24 : codes24 =
    template8Codes.map (renameRow (rename8 node24)) := by
  rw [template8_literal]
  unfold codes24
  rw [sliced_eq]
  change node24Piece0 ++ node24Piece1 ++ node24Piece2 ++ node24Piece3 ++ [] = _
  rw [node24Piece0_checked, node24Piece1_checked, node24Piece2_checked, node24Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat24 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node24) :=
  transport8 node24 (window24.trans checked24) hs

def node25 : Node := ⟨87416, [87020, 87021, 87022, 87023, 87024, 87025, 87026, 87027]⟩
def codes25 : List Row :=
  (([ExportedData.chunk682, ExportedData.chunk683, ExportedData.chunk684, ExportedData.chunk685].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 116).take 388
theorem window25 : codedWindow (node25.start-4) 388 = codes25 := by rfl

def node25Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk682.1 ExportedData.chunk682.2).drop 116).take 388
theorem node25Piece0_checked : node25Piece0 =
    ((literalCodes8.drop 0).take 12).map (renameRow (rename8 node25)) := by decide

def node25Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk683.1 ExportedData.chunk683.2).drop 0).take 376
theorem node25Piece1_checked : node25Piece1 =
    ((literalCodes8.drop 12).take 128).map (renameRow (rename8 node25)) := by decide

def node25Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk684.1 ExportedData.chunk684.2).drop 0).take 248
theorem node25Piece2_checked : node25Piece2 =
    ((literalCodes8.drop 140).take 128).map (renameRow (rename8 node25)) := by decide

def node25Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk685.1 ExportedData.chunk685.2).drop 0).take 120
theorem node25Piece3_checked : node25Piece3 =
    ((literalCodes8.drop 268).take 120).map (renameRow (rename8 node25)) := by decide

theorem checked25 : codes25 =
    template8Codes.map (renameRow (rename8 node25)) := by
  rw [template8_literal]
  unfold codes25
  rw [sliced_eq]
  change node25Piece0 ++ node25Piece1 ++ node25Piece2 ++ node25Piece3 ++ [] = _
  rw [node25Piece0_checked, node25Piece1_checked, node25Piece2_checked, node25Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat25 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node25) :=
  transport8 node25 (window25.trans checked25) hs

def node26 : Node := ⟨87804, [87415, 87803]⟩
def codes26 : List Row :=
  (([ExportedData.chunk685, ExportedData.chunk686, ExportedData.chunk687].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 120).take 238
theorem window26 : codedWindow (node26.start-4) 238 = codes26 := by rfl

def node26Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk685.1 ExportedData.chunk685.2).drop 120).take 238
theorem node26Piece0_checked : node26Piece0 =
    ((literalCodes2.drop 0).take 8).map (renameRow (rename2 node26)) := by decide

def node26Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk686.1 ExportedData.chunk686.2).drop 0).take 230
theorem node26Piece1_checked : node26Piece1 =
    ((literalCodes2.drop 8).take 128).map (renameRow (rename2 node26)) := by decide

def node26Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk687.1 ExportedData.chunk687.2).drop 0).take 102
theorem node26Piece2_checked : node26Piece2 =
    ((literalCodes2.drop 136).take 102).map (renameRow (rename2 node26)) := by decide

theorem checked26 : codes26 =
    template2Codes.map (renameRow (rename2 node26)) := by
  rw [template2_literal]
  unfold codes26
  rw [sliced_eq]
  change node26Piece0 ++ node26Piece1 ++ node26Piece2 ++ [] = _
  rw [node26Piece0_checked, node26Piece1_checked, node26Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat26 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node26) :=
  transport2 node26 (window26.trans checked26) hs

def node27 : Node := ⟨88219, [88203, 88204, 88205, 88206, 88207, 88208, 88209, 88210]⟩
def codes27 : List Row :=
  (([ExportedData.chunk689, ExportedData.chunk690, ExportedData.chunk691, ExportedData.chunk692].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 23).take 388
theorem window27 : codedWindow (node27.start-4) 388 = codes27 := by rfl

def node27Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk689.1 ExportedData.chunk689.2).drop 23).take 388
theorem node27Piece0_checked : node27Piece0 =
    ((literalCodes8.drop 0).take 105).map (renameRow (rename8 node27)) := by decide

def node27Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk690.1 ExportedData.chunk690.2).drop 0).take 283
theorem node27Piece1_checked : node27Piece1 =
    ((literalCodes8.drop 105).take 128).map (renameRow (rename8 node27)) := by decide

def node27Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk691.1 ExportedData.chunk691.2).drop 0).take 155
theorem node27Piece2_checked : node27Piece2 =
    ((literalCodes8.drop 233).take 128).map (renameRow (rename8 node27)) := by decide

def node27Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk692.1 ExportedData.chunk692.2).drop 0).take 27
theorem node27Piece3_checked : node27Piece3 =
    ((literalCodes8.drop 361).take 27).map (renameRow (rename8 node27)) := by decide

theorem checked27 : codes27 =
    template8Codes.map (renameRow (rename8 node27)) := by
  rw [template8_literal]
  unfold codes27
  rw [sliced_eq]
  change node27Piece0 ++ node27Piece1 ++ node27Piece2 ++ node27Piece3 ++ [] = _
  rw [node27Piece0_checked, node27Piece1_checked, node27Piece2_checked, node27Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat27 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node27) :=
  transport8 node27 (window27.trans checked27) hs

end CircuitCorrectness.HashWiring
