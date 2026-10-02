import CircuitCorrectness.HashProgram
import CircuitCorrectness.ByteCertificates.Decode
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
def pixelCodes : Array Nat := #[0, 22, 131]
def chunkCodes : Array Nat := #[0, 132, 133, 134, 135, 136, 137, 138, 139, 140]

theorem pixelCode_expand : ∀ i : Fin 3,
    ExportedData.coefficientPool[pixelCodes[i.val]!]! = 2^(8*i.val) := by
  intro i; fin_cases i <;> rfl
theorem chunkCode_expand : ∀ i : Fin 10,
    ExportedData.coefficientPool[chunkCodes[i.val]!]! = 2^(24*i.val) := by
  intro i; fin_cases i <;> rfl
end CircuitCorrectness.PackingCertificates
