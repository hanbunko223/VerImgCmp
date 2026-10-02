import CircuitCorrectness.DctProgramCertificates.Base
import CircuitCorrectness.DctProgramCertificates.Group03
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk565 : rows ExportedData.chunk565.1 ExportedData.chunk565.2 =
    (firstRows.drop 3200).take 128 := by decide
theorem chunk566 : rows ExportedData.chunk566.1 ExportedData.chunk566.2 =
    (firstRows.drop 3328).take 128 := by decide
theorem chunk567 : rows ExportedData.chunk567.1 ExportedData.chunk567.2 =
    (firstRows.drop 3456).take 128 := by decide
theorem chunk568 : rows ExportedData.chunk568.1 ExportedData.chunk568.2 =
    (firstRows.drop 3584).take 128 := by decide
theorem chunk569 : rows ExportedData.chunk569.1 ExportedData.chunk569.2 =
    (firstRows.drop 3712).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
