import CircuitCorrectness.DctProgramCertificates.Coordinates
import CircuitCorrectness.DctProgramCertificates.Group09
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk595 : rows ExportedData.chunk595.1 ExportedData.chunk595.2 =
    (hornerRows.drop 1920).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk596 : rows ExportedData.chunk596.1 ExportedData.chunk596.2 =
    (hornerRows.drop 2048).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk597 : rows ExportedData.chunk597.1 ExportedData.chunk597.2 =
    (hornerRows.drop 2176).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk598 : rows ExportedData.chunk598.1 ExportedData.chunk598.2 =
    (hornerRows.drop 2304).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk599 : rows ExportedData.chunk599.1 ExportedData.chunk599.2 =
    (hornerRows.drop 2432).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
end CircuitCorrectness.DctProgramCertificates
