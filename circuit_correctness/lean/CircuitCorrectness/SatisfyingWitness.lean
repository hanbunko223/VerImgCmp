import CircuitCorrectness.Seed
import CircuitCorrectness.ProgramCertificates.All

namespace CircuitCorrectness

/-- Universal mathematical witness construction covering every exported row.
The independent-spec output claim is composed with full-step soundness separately. -/
theorem exists_satisfying_step (x : Spec.Image) (s : Spec.State) (hx : Spec.ValidImage x) :
    ∃ w : Assignment, Target.matchesImage w x ∧ Target.incomingState w=s ∧
      Exported.circuit.Sat w := by
  let seed := Seed.assignment x s
  let w := StraightLine.run ProgramCertificates.program seed
  have hp := Seed.preserved x s w hx (ProgramCertificates.execution_preserves seed)
  refine ⟨w,hp.2.1,hp.2.2.1,hp.1,?_⟩
  have hsuf := ProgramCertificates.execution_satisfies seed (Seed.one x s)
  intro row hm
  have hsplit : row ∈ Exported.rows.take 69120 ++ Exported.rows.drop 69120 := by
    simpa only [List.take_append_drop] using hm
  rcases List.mem_append.mp hsplit with hb | ht
  · exact hp.2.2.2 row hb
  · exact hsuf row ht

end CircuitCorrectness
