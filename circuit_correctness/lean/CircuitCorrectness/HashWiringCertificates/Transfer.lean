import CircuitCorrectness.HashWiring
import CircuitCorrectness.Spec

namespace CircuitCorrectness.HashWiring

/-- Semantic premise to be discharged by the certified arity-eight template. -/
def Template8Sound : Prop := ∀ w : Assignment, w 97634 = 1 →
  (∀ row ∈ template8, row.Sat w) →
  w 77887 = Spec.hash8 ((List.range 8).toArray.map (fun i => w (77484+i)))

/-- Semantic premise to be discharged by the certified arity-two template. -/
def Template2Sound : Prop := ∀ w : Assignment, w 97634 = 1 →
  (∀ row ∈ template2, row.Sat w) → w 78513 = Spec.hash2 (w 77887) (w 78275)

theorem inputs8_renamed (node : Node) (hn : node.inputs.length = 8) (w : Assignment) :
    (List.range 8).toArray.map (fun i => (w ∘ rename8 node) (77484+i)) =
      node.inputs.toArray.map w := by
  apply Array.ext
  · simp [hn]
  · intro i hi hi'
    have hi8 : i<8 := by simpa using hi
    have hin : i<node.inputs.length := by omega
    simp [Function.comp_def, rename8_input node i hi8, getElem!_pos, hin]

theorem transfer8 (sound : Template8Sound) (node : Node) (hn : node.inputs.length = 8)
    {w : Assignment} (h1 : w 97634 = 1)
    (hs : ∀ row ∈ template8, row.Sat (w ∘ rename8 node)) :
    w (node.start+387) = Spec.hash8 (node.inputs.toArray.map w) := by
  have h := sound (w ∘ rename8 node) (by simpa using h1) hs
  rw [inputs8_renamed node hn w] at h
  simpa only [Function.comp_apply, rename8_output] using h

theorem transfer2 (sound : Template2Sound) (node : Node)
    {w : Assignment} (h1 : w 97634 = 1)
    (hs : ∀ row ∈ template2, row.Sat (w ∘ rename2 node)) :
    w (node.start+237) = Spec.hash2 (w node.inputs[0]!) (w node.inputs[1]!) := by
  simpa only [Function.comp_apply,rename2_output,rename2_input0,rename2_input1] using
    sound (w ∘ rename2 node) (by simpa using h1) hs

end CircuitCorrectness.HashWiring
