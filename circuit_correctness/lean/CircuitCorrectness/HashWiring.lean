import CircuitCorrectness.ConcreteProgram

namespace CircuitCorrectness.HashWiring
open ConcreteBytes

def renameLC (f : Nat → Nat) (lc : LinearCombination) : LinearCombination :=
  lc.map fun (i,c) => (f i,c)
def renameRow (f : Nat → Nat) (row : Row) : Row :=
  ⟨renameLC f row.a,renameLC f row.b,renameLC f row.c⟩

theorem eval_renameLC (f : Nat → Nat) (w : Assignment) (lc : LinearCombination) :
    evalLC w (renameLC f lc) = evalLC (w ∘ f) lc := by
  simp only [renameLC,evalLC,List.map_map,Function.comp_def]

theorem sat_renameRow (f : Nat → Nat) (w : Assignment) (row : Row) :
    (renameRow f row).Sat w ↔ row.Sat (w ∘ f) := by
  simp only [renameRow,Row.Sat,eval_renameLC]

theorem expand_renameLC (f : Nat → Nat) (lc : LinearCombination) :
    Codes.expandLC (renameLC f lc) = renameLC f (Codes.expandLC lc) := by
  simp only [renameLC,Codes.expandLC,List.map_map,Function.comp_def]

theorem expand_renameRow (f : Nat → Nat) (row : Row) :
    Codes.expandRow (renameRow f row) = renameRow f (Codes.expandRow row) := by
  simp only [Codes.expandRow,renameRow,expand_renameLC]

/-- A small window decoded directly from the pinned compressed export. -/
def codedWindow (start count : Nat) : List Row :=
  (((ExportedData.chunks.toList.drop (start/128)).take
    ((start%128+count+127)/128)).flatMap
      (fun c => Codes.rows c.1 c.2)).drop (start%128) |>.take count

theorem codedWindow_sat (start count : Nat) {w : Assignment}
    (hs : Exported.circuit.Sat w) :
    ∀ row ∈ codedWindow start count, (Codes.expandRow row).Sat w := by
  intro row hr
  unfold codedWindow at hr
  have hr' := List.mem_of_mem_drop (List.mem_of_mem_take hr)
  obtain ⟨chunk,hc,hrow⟩ := List.mem_flatMap.mp hr'
  have hc' := List.mem_of_mem_drop (List.mem_of_mem_take hc)
  apply hs.2
  apply List.mem_flatMap.mpr
  refine ⟨chunk,hc',?_⟩
  rcases chunk with ⟨n,data⟩
  change Codes.expandRow row ∈ Exported.decodeRows n data
  rw [Codes.rows_expand]
  exact List.mem_map.mpr ⟨row,hrow,rfl⟩

def template8Codes : List Row :=
  (([ExportedData.chunk605, ExportedData.chunk606, ExportedData.chunk607, ExportedData.chunk608].flatMap
    (fun c => Codes.rows c.1 c.2)).drop 56).take 388
def template2Codes : List Row :=
  (([ExportedData.chunk611, ExportedData.chunk612, ExportedData.chunk613].flatMap
    (fun c => Codes.rows c.1 c.2)).drop 64).take 238
def template8 : List Row := template8Codes.map Codes.expandRow
def template2 : List Row := template2Codes.map Codes.expandRow

structure Node where
  start : Nat
  inputs : List Nat
  deriving DecidableEq

def rename8 (node : Node) (i : Nat) : Nat :=
  if i = 97634 then 97634
  else if 77484 ≤ i ∧ i < 77492 then node.inputs[i-77484]!
  else node.start+(i-77500)

def rename2 (node : Node) (i : Nat) : Nat :=
  if i = 97634 then 97634
  else if i = 77887 then node.inputs[0]!
  else if i = 78275 then node.inputs[1]!
  else node.start+(i-78276)

@[simp] theorem rename8_one (node : Node) : rename8 node 97634 = 97634 := rfl
@[simp] theorem rename2_one (node : Node) : rename2 node 97634 = 97634 := rfl
@[simp] theorem rename8_output (node : Node) :
    rename8 node 77887 = node.start+387 := rfl
@[simp] theorem rename2_output (node : Node) :
    rename2 node 78513 = node.start+237 := rfl
@[simp] theorem rename2_input0 (node : Node) : rename2 node 77887 = node.inputs[0]! := rfl
@[simp] theorem rename2_input1 (node : Node) : rename2 node 78275 = node.inputs[1]! := rfl
@[simp] theorem rename8_input (node : Node) (i : Nat) (hi : i<8) :
    rename8 node (77484+i) = node.inputs[i]! := by
  have hone : 77484+i ≠ 97634 := by omega
  have hlo : 77484 ≤ 77484+i := by omega
  have hhi : 77484+i < 77492 := by omega
  simp [rename8,hone,hlo,hhi]

theorem transport (template : List Row) (f : Nat → Nat) (start count : Nat)
    (he : codedWindow start count = template.map (renameRow f))
    {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template.map Codes.expandRow, row.Sat (w ∘ f) := by
  intro row hr
  obtain ⟨source,hsource,rfl⟩ := List.mem_map.mp hr
  have hw := codedWindow_sat start count hs (renameRow f source)
    (he ▸ List.mem_map.mpr ⟨source,hsource,rfl⟩)
  rw [expand_renameRow] at hw
  exact (sat_renameRow f w (Codes.expandRow source)).mp hw

theorem transport8 (node : Node)
    (he : codedWindow (node.start-4) 388 = template8Codes.map (renameRow (rename8 node)))
    {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template8, row.Sat (w ∘ rename8 node) :=
  transport template8Codes (rename8 node) (node.start-4) 388 he hs

theorem transport2 (node : Node)
    (he : codedWindow (node.start-4) 238 = template2Codes.map (renameRow (rename2 node)))
    {w : Assignment} (hs : Exported.circuit.Sat w) :
    ∀ row ∈ template2, row.Sat (w ∘ rename2 node) :=
  transport template2Codes (rename2 node) (node.start-4) 238 he hs

end CircuitCorrectness.HashWiring
