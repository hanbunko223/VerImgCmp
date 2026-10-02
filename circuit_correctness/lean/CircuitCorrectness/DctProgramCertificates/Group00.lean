import CircuitCorrectness.DctProgramCertificates.Base
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk540 : rows ExportedData.chunk540.1 ExportedData.chunk540.2 =
    (firstRows.drop 0).take 128 := by decide
theorem chunk541 : rows ExportedData.chunk541.1 ExportedData.chunk541.2 =
    (firstRows.drop 128).take 128 := by decide
theorem chunk542 : rows ExportedData.chunk542.1 ExportedData.chunk542.2 =
    (firstRows.drop 256).take 128 := by decide
theorem chunk543 : rows ExportedData.chunk543.1 ExportedData.chunk543.2 =
    (firstRows.drop 384).take 128 := by decide
theorem chunk544 : rows ExportedData.chunk544.1 ExportedData.chunk544.2 =
    (firstRows.drop 512).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
