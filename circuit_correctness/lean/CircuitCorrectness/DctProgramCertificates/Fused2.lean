import CircuitCorrectness.DctProgramCertificates.Fused1
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram
theorem fusedCode_expand_2_0 : ∀ c k : Fin 8,
    Spec.retained 2 0 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 0 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 0 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_1 : ∀ c k : Fin 8,
    Spec.retained 2 1 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 1 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 1 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_2 : ∀ c k : Fin 8,
    Spec.retained 2 2 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 2 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 2 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_3 : ∀ c k : Fin 8,
    Spec.retained 2 3 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 3 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 3 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_4 : ∀ c k : Fin 8,
    Spec.retained 2 4 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 4 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 4 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_5 : ∀ c k : Fin 8,
    Spec.retained 2 5 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 5 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 5 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_6 : ∀ c k : Fin 8,
    Spec.retained 2 6 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 6 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 6 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_2_7 : ∀ c k : Fin 8,
    Spec.retained 2 7 c.val = true →
    ExportedData.coefficientPool[fusedCode 2 7 c.val k.val]! =
      encodeInt (-((Spec.multiplier 2 7 c.val : Int) * Spec.matrix c.val k.val)) := by decide
end CircuitCorrectness.DctProgramCertificates
