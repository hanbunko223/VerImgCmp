import CircuitCorrectness.HashWiringTemplates

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

def node00 : Node := ⟨77500, [77484, 77485, 77486, 77487, 77488, 77489, 77490, 77491]⟩
def codes00 : List Row :=
  (([ExportedData.chunk605, ExportedData.chunk606, ExportedData.chunk607, ExportedData.chunk608].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 56).take 388
theorem window00 : codedWindow (node00.start-4) 388 = codes00 := by rfl

def node00Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk605.1 ExportedData.chunk605.2).drop 56).take 388
theorem node00Piece0_checked : node00Piece0 =
    ((literalCodes8.drop 0).take 72).map (renameRow (rename8 node00)) := by decide

def node00Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk606.1 ExportedData.chunk606.2).drop 0).take 316
theorem node00Piece1_checked : node00Piece1 =
    ((literalCodes8.drop 72).take 128).map (renameRow (rename8 node00)) := by decide

def node00Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk607.1 ExportedData.chunk607.2).drop 0).take 188
theorem node00Piece2_checked : node00Piece2 =
    ((literalCodes8.drop 200).take 128).map (renameRow (rename8 node00)) := by decide

def node00Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk608.1 ExportedData.chunk608.2).drop 0).take 60
theorem node00Piece3_checked : node00Piece3 =
    ((literalCodes8.drop 328).take 60).map (renameRow (rename8 node00)) := by decide

theorem checked00 : codes00 =
    template8Codes.map (renameRow (rename8 node00)) := by
  rw [template8_literal]
  unfold codes00
  rw [sliced_eq]
  change node00Piece0 ++ node00Piece1 ++ node00Piece2 ++ node00Piece3 ++ [] = _
  rw [node00Piece0_checked, node00Piece1_checked, node00Piece2_checked, node00Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat00 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node00) :=
  transport8 node00 (window00.trans checked00) hs

def node01 : Node := ⟨77888, [77492, 77493, 77494, 77495, 77496, 77497, 77498, 77499]⟩
def codes01 : List Row :=
  (([ExportedData.chunk608, ExportedData.chunk609, ExportedData.chunk610, ExportedData.chunk611].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 60).take 388
theorem window01 : codedWindow (node01.start-4) 388 = codes01 := by rfl

def node01Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk608.1 ExportedData.chunk608.2).drop 60).take 388
theorem node01Piece0_checked : node01Piece0 =
    ((literalCodes8.drop 0).take 68).map (renameRow (rename8 node01)) := by decide

def node01Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk609.1 ExportedData.chunk609.2).drop 0).take 320
theorem node01Piece1_checked : node01Piece1 =
    ((literalCodes8.drop 68).take 128).map (renameRow (rename8 node01)) := by decide

def node01Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk610.1 ExportedData.chunk610.2).drop 0).take 192
theorem node01Piece2_checked : node01Piece2 =
    ((literalCodes8.drop 196).take 128).map (renameRow (rename8 node01)) := by decide

def node01Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk611.1 ExportedData.chunk611.2).drop 0).take 64
theorem node01Piece3_checked : node01Piece3 =
    ((literalCodes8.drop 324).take 64).map (renameRow (rename8 node01)) := by decide

theorem checked01 : codes01 =
    template8Codes.map (renameRow (rename8 node01)) := by
  rw [template8_literal]
  unfold codes01
  rw [sliced_eq]
  change node01Piece0 ++ node01Piece1 ++ node01Piece2 ++ node01Piece3 ++ [] = _
  rw [node01Piece0_checked, node01Piece1_checked, node01Piece2_checked, node01Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat01 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node01) :=
  transport8 node01 (window01.trans checked01) hs

def node02 : Node := ⟨78276, [77887, 78275]⟩
def codes02 : List Row :=
  (([ExportedData.chunk611, ExportedData.chunk612, ExportedData.chunk613].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 64).take 238
theorem window02 : codedWindow (node02.start-4) 238 = codes02 := by rfl

def node02Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk611.1 ExportedData.chunk611.2).drop 64).take 238
theorem node02Piece0_checked : node02Piece0 =
    ((literalCodes2.drop 0).take 64).map (renameRow (rename2 node02)) := by decide

def node02Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk612.1 ExportedData.chunk612.2).drop 0).take 174
theorem node02Piece1_checked : node02Piece1 =
    ((literalCodes2.drop 64).take 128).map (renameRow (rename2 node02)) := by decide

def node02Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk613.1 ExportedData.chunk613.2).drop 0).take 46
theorem node02Piece2_checked : node02Piece2 =
    ((literalCodes2.drop 192).take 46).map (renameRow (rename2 node02)) := by decide

theorem checked02 : codes02 =
    template2Codes.map (renameRow (rename2 node02)) := by
  rw [template2_literal]
  unfold codes02
  rw [sliced_eq]
  change node02Piece0 ++ node02Piece1 ++ node02Piece2 ++ [] = _
  rw [node02Piece0_checked, node02Piece1_checked, node02Piece2_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat02 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node02) :=
  transport2 node02 (window02.trans checked02) hs

def node03 : Node := ⟨78691, [78675, 78676, 78677, 78678, 78679, 78680, 78681, 78682]⟩
def codes03 : List Row :=
  (([ExportedData.chunk614, ExportedData.chunk615, ExportedData.chunk616, ExportedData.chunk617].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop 95).take 388
theorem window03 : codedWindow (node03.start-4) 388 = codes03 := by rfl

def node03Piece0 := ((ConcreteBytes.Codes.rows ExportedData.chunk614.1 ExportedData.chunk614.2).drop 95).take 388
theorem node03Piece0_checked : node03Piece0 =
    ((literalCodes8.drop 0).take 33).map (renameRow (rename8 node03)) := by decide

def node03Piece1 := ((ConcreteBytes.Codes.rows ExportedData.chunk615.1 ExportedData.chunk615.2).drop 0).take 355
theorem node03Piece1_checked : node03Piece1 =
    ((literalCodes8.drop 33).take 128).map (renameRow (rename8 node03)) := by decide

def node03Piece2 := ((ConcreteBytes.Codes.rows ExportedData.chunk616.1 ExportedData.chunk616.2).drop 0).take 227
theorem node03Piece2_checked : node03Piece2 =
    ((literalCodes8.drop 161).take 128).map (renameRow (rename8 node03)) := by decide

def node03Piece3 := ((ConcreteBytes.Codes.rows ExportedData.chunk617.1 ExportedData.chunk617.2).drop 0).take 99
theorem node03Piece3_checked : node03Piece3 =
    ((literalCodes8.drop 289).take 99).map (renameRow (rename8 node03)) := by decide

theorem checked03 : codes03 =
    template8Codes.map (renameRow (rename8 node03)) := by
  rw [template8_literal]
  unfold codes03
  rw [sliced_eq]
  change node03Piece0 ++ node03Piece1 ++ node03Piece2 ++ node03Piece3 ++ [] = _
  rw [node03Piece0_checked, node03Piece1_checked, node03Piece2_checked, node03Piece3_checked]
  simp only [List.append_nil, ← List.map_append]
  congr 1
theorem sat03 {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node03) :=
  transport8 node03 (window03.trans checked03) hs

end CircuitCorrectness.HashWiring
