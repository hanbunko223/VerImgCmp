import CircuitCorrectness.DctProgramCertificates.Base
import CircuitCorrectness.DctProgramCertificates.Group02
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk560 : rows ExportedData.chunk560.1 ExportedData.chunk560.2 =
    (firstRows.drop 2560).take 128 := by decide
theorem chunk561 : rows ExportedData.chunk561.1 ExportedData.chunk561.2 =
    (firstRows.drop 2688).take 128 := by decide
theorem chunk562 : rows ExportedData.chunk562.1 ExportedData.chunk562.2 =
    (firstRows.drop 2816).take 128 := by decide
theorem chunk563 : rows ExportedData.chunk563.1 ExportedData.chunk563.2 =
    (firstRows.drop 2944).take 128 := by decide
theorem chunk564 : rows ExportedData.chunk564.1 ExportedData.chunk564.2 =
    (firstRows.drop 3072).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
