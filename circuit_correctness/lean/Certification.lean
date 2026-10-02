import CircuitCorrectness

-- Strict acceptance gate: these are the original universal target propositions.
-- Concrete row correspondence and independent-spec proofs are checked transitively.
example : CircuitCorrectness.Target.StepSoundness :=
  CircuitCorrectness.Target.step_soundness
example : CircuitCorrectness.Target.StepCompleteness :=
  CircuitCorrectness.Target.step_completeness
example : CircuitCorrectness.Target.OutputDeterminism :=
  CircuitCorrectness.Target.step_determinism

#print axioms CircuitCorrectness.Target.step_soundness
#print axioms CircuitCorrectness.Target.step_completeness
#print axioms CircuitCorrectness.Target.step_determinism

#check CircuitCorrectness.Target.connected_steps
#check CircuitCorrectness.Target.connected360
#print axioms CircuitCorrectness.Target.connected360
