import CircuitCorrectness.DctProgramCertificates.Coordinates
import CircuitCorrectness.DctProgramCertificates.Group07
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk585 : rows ExportedData.chunk585.1 ExportedData.chunk585.2 =
    (hornerRows.drop 640).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk586 : rows ExportedData.chunk586.1 ExportedData.chunk586.2 =
    (hornerRows.drop 768).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk587 : rows ExportedData.chunk587.1 ExportedData.chunk587.2 =
    (hornerRows.drop 896).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk588 : rows ExportedData.chunk588.1 ExportedData.chunk588.2 =
    (hornerRows.drop 1024).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk589 : rows ExportedData.chunk589.1 ExportedData.chunk589.2 =
    (hornerRows.drop 1152).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
end CircuitCorrectness.DctProgramCertificates
