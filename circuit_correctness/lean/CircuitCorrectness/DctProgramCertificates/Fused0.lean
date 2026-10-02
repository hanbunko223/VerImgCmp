import CircuitCorrectness.DctProgramCertificates.Data
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram
theorem fusedCode_expand_0_0 : ∀ c k : Fin 8,
    Spec.retained 0 0 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 0 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 0 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_1 : ∀ c k : Fin 8,
    Spec.retained 0 1 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 1 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 1 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_2 : ∀ c k : Fin 8,
    Spec.retained 0 2 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 2 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 2 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_3 : ∀ c k : Fin 8,
    Spec.retained 0 3 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 3 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 3 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_4 : ∀ c k : Fin 8,
    Spec.retained 0 4 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 4 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 4 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_5 : ∀ c k : Fin 8,
    Spec.retained 0 5 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 5 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 5 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_6 : ∀ c k : Fin 8,
    Spec.retained 0 6 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 6 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 6 c.val : Int) * Spec.matrix c.val k.val)) := by decide
theorem fusedCode_expand_0_7 : ∀ c k : Fin 8,
    Spec.retained 0 7 c.val = true →
    ExportedData.coefficientPool[fusedCode 0 7 c.val k.val]! =
      encodeInt (-((Spec.multiplier 0 7 c.val : Int) * Spec.matrix c.val k.val)) := by decide
end CircuitCorrectness.DctProgramCertificates
