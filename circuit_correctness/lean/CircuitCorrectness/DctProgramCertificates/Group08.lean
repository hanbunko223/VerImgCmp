import CircuitCorrectness.DctProgramCertificates.Coordinates
import CircuitCorrectness.DctProgramCertificates.Group06
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk580 : rows ExportedData.chunk580.1 ExportedData.chunk580.2 =
    (hornerRows.drop 0).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk581 : rows ExportedData.chunk581.1 ExportedData.chunk581.2 =
    (hornerRows.drop 128).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk582 : rows ExportedData.chunk582.1 ExportedData.chunk582.2 =
    (hornerRows.drop 256).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk583 : rows ExportedData.chunk583.1 ExportedData.chunk583.2 =
    (hornerRows.drop 384).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk584 : rows ExportedData.chunk584.1 ExportedData.chunk584.2 =
    (hornerRows.drop 512).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
end CircuitCorrectness.DctProgramCertificates
