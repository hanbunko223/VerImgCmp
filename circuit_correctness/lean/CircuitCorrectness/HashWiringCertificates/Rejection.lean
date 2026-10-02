import CircuitCorrectness.HashWiringCertificates.Group00
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.HashWiring

/-- Swapping the two hash inputs changes the actual exported equations. -/
theorem rejects_swapped_inputs :
    codedWindow (node02.start-4) 238 ≠
      template2Codes.map (renameRow (rename2 ⟨node02.start,node02.inputs.reverse⟩)) := by
  decide

/-- Modifying a constant coefficient code is detected by row equality. -/
theorem rejects_changed_constants :
    template8Codes ≠ template8Codes.map (fun row =>
      { row with a := row.a.map (fun (i,c) => (i,if i = 97634 then c+1 else c)) }) := by
  decide

end CircuitCorrectness.HashWiring
