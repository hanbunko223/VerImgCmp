import CircuitCorrectness.Exported
import CircuitCorrectness.Gadgets

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace CircuitCorrectness.Exported

def counterRow : Row :=
  ⟨[(3,1),(97633,modulus-1),(97634,1)],[(97634,1)],[]⟩

theorem counter_row_present : counterRow ∈ rows := by
  apply List.mem_flatMap.mpr
  refine ⟨ExportedData.chunk762, ?_, ?_⟩
  · decide
  · decide

theorem counter_transition (w : Assignment) (h : circuit.Sat w) :
    w 97633 = w 3 + 1 := by
  have hrow := h.2 counterRow counter_row_present
  have hone : w 97634 = 1 := h.1
  have hneg : ((modulus-1 : Nat) : F) = -1 := by decide
  simp only [Row.Sat, counterRow, evalLC, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.cast_one, one_mul, add_zero,
    hneg, neg_one_mul, hone, mul_one] at hrow
  linear_combination -hrow

end CircuitCorrectness.Exported
