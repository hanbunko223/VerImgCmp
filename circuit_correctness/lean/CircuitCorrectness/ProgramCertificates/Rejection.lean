import CircuitCorrectness.ProgramCertificates.Group00

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram

/-- Altering the exported destination coefficient invalidates extraction. -/
theorem rejects_destination_coefficient :
    Coded.check 97634 69124
      { codes540.headD ⟨[],[],[]⟩ with
        a := (codes540.headD ⟨[],[],[]⟩).a.map (fun t => if t.1 = 69124 then (t.1,0) else t) } = false := by
  decide

/-- An additional read from an unallocated wire invalidates the frontier check. -/
theorem rejects_future_wire :
    Coded.check 97634 69124
      { codes540.headD ⟨[],[],[]⟩ with a := (69125,0) :: (codes540.headD ⟨[],[],[]⟩).a } = false := by
  decide

/-- Reordering the first two actual rows invalidates sequential destination checks. -/
theorem rejects_row_reordering :
    Coded.checkRows 97634 69124 ((codes540.take 2).reverse) = false := by
  decide

end CircuitCorrectness.ProgramCertificates
