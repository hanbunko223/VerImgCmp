import CircuitCorrectness.Target
namespace CircuitCorrectness.Target

/-- Connections are explicit mathematical hypotheses, not a claim about Nova. -/
def ConnectedSteps (images : Nat → Spec.Image) (states : Nat → Spec.State)
    (witnesses : Nat → Assignment) (count : Nat) : Prop :=
  ∀ j<count, Spec.ValidImage (images j) ∧ matchesImage (witnesses j) (images j) ∧
    incomingState (witnesses j)=states j ∧ Exported.circuit.Sat (witnesses j) ∧
    outgoingState (witnesses j)=states (j+1)

def execute (images : Nat → Spec.Image) (count : Nat) (initial : Spec.State) : Spec.State :=
  (List.range count).foldl (fun s j => Spec.step (images j) s) initial

theorem connected_steps_of_soundness (sound : StepSoundness)
    (images : Nat → Spec.Image) (states : Nat → Spec.State)
    (witnesses : Nat → Assignment) (count : Nat)
    (hc : ConnectedSteps images states witnesses count) :
    states count = execute images count (states 0) := by
  have hrun : ∀ n≤count, states n = execute images n (states 0) := by
    intro n
    induction n with
    | zero => intro _; rfl
    | succ n ih =>
      intro hn
      have hh := hc n (by omega)
      have hout := sound (images n) (states n) (witnesses n)
        hh.1 hh.2.1 hh.2.2.1 hh.2.2.2.1
      rw [hh.2.2.2.2] at hout
      rw [hout, ih (by omega)]
      simp only [execute, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
  exact hrun count (Nat.le_refl count)

end CircuitCorrectness.Target
