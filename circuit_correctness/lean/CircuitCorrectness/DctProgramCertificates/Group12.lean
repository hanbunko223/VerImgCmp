import CircuitCorrectness.DctProgramCertificates.Coordinates
import CircuitCorrectness.DctProgramCertificates.Group10
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk600 : rows ExportedData.chunk600.1 ExportedData.chunk600.2 =
    (hornerRows.drop 2560).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk601 : rows ExportedData.chunk601.1 ExportedData.chunk601.2 =
    (hornerRows.drop 2688).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk602 : rows ExportedData.chunk602.1 ExportedData.chunk602.2 =
    (hornerRows.drop 2816).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk603 : rows ExportedData.chunk603.1 ExportedData.chunk603.2 =
    (hornerRows.drop 2944).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
theorem chunk604 : (rows ExportedData.chunk604.1 ExportedData.chunk604.2).take 8 =
    (hornerRows.drop 3072).take 128 := by
  rw [hornerRows, ← retainedLiteral_eq]
  decide
end CircuitCorrectness.DctProgramCertificates
