import CircuitCorrectness.DctProgramCertificates.Base
import CircuitCorrectness.DctProgramCertificates.Group00
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk550 : rows ExportedData.chunk550.1 ExportedData.chunk550.2 =
    (firstRows.drop 1280).take 128 := by decide
theorem chunk551 : rows ExportedData.chunk551.1 ExportedData.chunk551.2 =
    (firstRows.drop 1408).take 128 := by decide
theorem chunk552 : rows ExportedData.chunk552.1 ExportedData.chunk552.2 =
    (firstRows.drop 1536).take 128 := by decide
theorem chunk553 : rows ExportedData.chunk553.1 ExportedData.chunk553.2 =
    (firstRows.drop 1664).take 128 := by decide
theorem chunk554 : rows ExportedData.chunk554.1 ExportedData.chunk554.2 =
    (firstRows.drop 1792).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
