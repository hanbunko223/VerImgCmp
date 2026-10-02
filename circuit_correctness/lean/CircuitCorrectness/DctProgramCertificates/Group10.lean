import CircuitCorrectness.DctProgramCertificates.Coordinates
import CircuitCorrectness.DctProgramCertificates.Group08
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk590 : rows ExportedData.chunk590.1 ExportedData.chunk590.2 =
    (hornerRows.drop 1280).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk591 : rows ExportedData.chunk591.1 ExportedData.chunk591.2 =
    (hornerRows.drop 1408).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk592 : rows ExportedData.chunk592.1 ExportedData.chunk592.2 =
    (hornerRows.drop 1536).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk593 : rows ExportedData.chunk593.1 ExportedData.chunk593.2 =
    (hornerRows.drop 1664).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk594 : rows ExportedData.chunk594.1 ExportedData.chunk594.2 =
    (hornerRows.drop 1792).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
end CircuitCorrectness.DctProgramCertificates
