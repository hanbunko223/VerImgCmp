import CircuitCorrectness.DctProgramCertificates.Base
import CircuitCorrectness.DctProgramCertificates.Group01
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk555 : rows ExportedData.chunk555.1 ExportedData.chunk555.2 =
    (firstRows.drop 1920).take 128 := by decide
theorem chunk556 : rows ExportedData.chunk556.1 ExportedData.chunk556.2 =
    (firstRows.drop 2048).take 128 := by decide
theorem chunk557 : rows ExportedData.chunk557.1 ExportedData.chunk557.2 =
    (firstRows.drop 2176).take 128 := by decide
theorem chunk558 : rows ExportedData.chunk558.1 ExportedData.chunk558.2 =
    (firstRows.drop 2304).take 128 := by decide
theorem chunk559 : rows ExportedData.chunk559.1 ExportedData.chunk559.2 =
    (firstRows.drop 2432).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
