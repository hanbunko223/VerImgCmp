import CircuitCorrectness.ActualHash
import CircuitCorrectness.DctProgramCertificates.All
import CircuitCorrectness.SatisfyingWitness
import CircuitCorrectness.Counter
import CircuitCorrectness.Composition
namespace CircuitCorrectness.Target

theorem state_ext {s t : Spec.State} (hh : s.h=t.h) (ha : s.a=t.a)
    (hr : s.r=t.r) (ht : s.t=t.t) : s=t := by
  cases s
  cases t
  simp_all

/-- Every satisfying assignment to the pinned exported step computes the
independent specification. This theorem quantifies over arbitrary assignments. -/
theorem step_soundness : StepSoundness := by
  intro x s w _hx hm hi hs
  have hh := ActualHash.hash_transition w x (matches_hash w x hm) hs
  have ha := DctProgramCertificates.exported_dct_sound w x hs (matches_dct w x hm)
  have ht := Exported.counter_transition w hs
  have he : outgoingState w = Spec.step x (incomingState w) := by
    apply state_ext
    · exact hh
    · simpa only [outgoingState,incomingState,Spec.step,ExportedData.outgoing,
        ExportedData.incoming] using ha
    · exact actual_challenge_preserved w
    · exact ht
  simpa only [hi] using he

/-- The witness is constructed mathematically; this does not assert universal
correctness of the Rust witness generator. Every exported row is covered. -/
theorem step_completeness : StepCompleteness := by
  intro x s hx
  rcases exists_satisfying_step x s hx with ⟨w,hm,hi,hs⟩
  exact ⟨w,hm,hi,hs,step_soundness x s w hx hm hi hs⟩

theorem step_determinism : OutputDeterminism := determinism_of_soundness step_soundness

/-- Adjacent states are explicitly connected in ConnectedSteps; Nova is outside scope. -/
theorem connected_steps (images : Nat → Spec.Image) (states : Nat → Spec.State)
    (witnesses : Nat → Assignment) (count : Nat)
    (hc : ConnectedSteps images states witnesses count) :
    states count = execute images count (states 0) :=
  connected_steps_of_soundness step_soundness images states witnesses count hc

theorem connected360 (images : Nat → Spec.Image) (states : Nat → Spec.State)
    (witnesses : Nat → Assignment)
    (hc : ConnectedSteps images states witnesses 360) :
    states 360 = execute images 360 (states 0) := connected_steps images states witnesses 360 hc

end CircuitCorrectness.Target
