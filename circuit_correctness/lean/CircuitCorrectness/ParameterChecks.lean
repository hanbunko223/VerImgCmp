import CircuitCorrectness.Spec

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace CircuitCorrectness.ParameterChecks

theorem first_row : Parameters.dct[0]! = #[45,45,45,45,45,45,45,45] := by decide

theorem reciprocal_construction :
    ∀ ch : Fin 3, ∀ r c : Fin 8,
      let q := ((Parameters.divisors[ch.val]!)[r.val]!)[c.val]!
      Spec.multiplier ch.val r.val c.val =
        if q ≤ 97 then (2048 + q) / (2*q) else 0 := by decide

theorem red_count : ((List.range 64).filter fun n => Spec.retained 0 (n/8) (n%8)).length = 51 := by decide
theorem green_count : ((List.range 64).filter fun n => Spec.retained 1 (n/8) (n%8)).length = 13 := by decide
theorem blue_count : ((List.range 64).filter fun n => Spec.retained 2 (n/8) (n%8)).length = 13 := by decide

theorem retained_count : Spec.retainedCoordinates.length = 3080 := by
  simp only [Spec.retainedCoordinates, List.length_flatMap, List.length_map]
  try simp only [List.map_const, List.length_range, List.sum_replicate, smul_eq_mul]
  decide

end CircuitCorrectness.ParameterChecks
