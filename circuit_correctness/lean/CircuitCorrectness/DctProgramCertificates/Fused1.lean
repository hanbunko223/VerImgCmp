import CircuitCorrectness.DctProgramCertificates.Fused0
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram
theorem fusedCode_expand_1_0 : ∀ c k : Fin 8,
    Spec.retained 1 0 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 0 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 0 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_1 : ∀ c k : Fin 8,
    Spec.retained 1 1 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 1 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 1 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_2 : ∀ c k : Fin 8,
    Spec.retained 1 2 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 2 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 2 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_3 : ∀ c k : Fin 8,
    Spec.retained 1 3 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 3 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 3 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_4 : ∀ c k : Fin 8,
    Spec.retained 1 4 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 4 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 4 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_5 : ∀ c k : Fin 8,
    Spec.retained 1 5 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 5 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 5 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_6 : ∀ c k : Fin 8,
    Spec.retained 1 6 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 6 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 6 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_1_7 : ∀ c k : Fin 8,
    Spec.retained 1 7 c.val = true →
    ExportedData.coefficientPool[fusedCode 1 7 c.val k.val]! =
      encodeInt (-((Spec.multiplier 1 7 c.val : Int) * Spec.matrix c.val k.val)) := by decide
end CircuitCorrectness.DctProgramCertificates
