import CircuitCorrectness.Seed
import CircuitCorrectness.HashProgram
import CircuitCorrectness.DctProgram
namespace CircuitCorrectness.Target

theorem matches_hash (w : Assignment) (x : Spec.Image) (h : matchesImage w x) :
    HashProgram.Matches w x := by
  intro r hr c hc ch hch
  have hp := h r hr c hc ch hch
  rw [ConcreteBytes.pixel_wire _ (by omega)] at hp
  exact hp

theorem matches_dct (w : Assignment) (x : Spec.Image) (h : matchesImage w x) :
    DctProgram.PixelsMatch w x := matches_hash w x h

end CircuitCorrectness.Target
