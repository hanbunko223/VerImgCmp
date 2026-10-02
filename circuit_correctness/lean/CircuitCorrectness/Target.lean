import CircuitCorrectness.Exported
import CircuitCorrectness.Spec

namespace CircuitCorrectness.Target

def incomingState (w : Assignment) : Spec.State :=
  ⟨w ExportedData.incoming[0]!, w ExportedData.incoming[1]!,
   w ExportedData.incoming[2]!, w ExportedData.incoming[3]!⟩

def outgoingState (w : Assignment) : Spec.State :=
  ⟨w ExportedData.outgoing[0]!, w ExportedData.outgoing[1]!,
   w ExportedData.outgoing[2]!, w ExportedData.outgoing[3]!⟩

def matchesImage (w : Assignment) (x : Spec.Image) : Prop :=
  ∀ r < 16, ∀ c < 160, ∀ ch < 3,
    w ExportedData.pixelWires[(r*160+c)*3+ch]! = (x r c ch : F)

/-- This is the required proposition, not an assumption or a theorem claiming it. -/
def StepSoundness : Prop :=
  ∀ (x : Spec.Image) (s : Spec.State) (w : Assignment),
    Spec.ValidImage x → matchesImage w x → incomingState w = s →
    Exported.circuit.Sat w → outgoingState w = Spec.step x s

def StepCompleteness : Prop :=
  ∀ (x : Spec.Image) (s : Spec.State), Spec.ValidImage x →
    ∃ w : Assignment, matchesImage w x ∧ incomingState w = s ∧
      Exported.circuit.Sat w ∧ outgoingState w = Spec.step x s

def OutputDeterminism : Prop :=
  ∀ (x : Spec.Image) (s : Spec.State) (w v : Assignment),
    Spec.ValidImage x → matchesImage w x → matchesImage v x →
    incomingState w = s → incomingState v = s →
    Exported.circuit.Sat w → Exported.circuit.Sat v → outgoingState w = outgoingState v

/-- A logical implication; the premise still needs the full exported-step proof. -/
theorem determinism_of_soundness (sound : StepSoundness) : OutputDeterminism := by
  intro x s w v hx hw hv hws hvs hsatw hsatv
  exact (sound x s w hx hw hws hsatw).trans (sound x s v hx hv hvs hsatv).symm

theorem actual_challenge_preserved (w : Assignment) :
    (outgoingState w).r = (incomingState w).r := by
  change w ExportedData.outgoing[2]! = w ExportedData.incoming[2]!
  rw [Exported.challenge_alias]

end CircuitCorrectness.Target
