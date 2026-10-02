import CircuitCorrectness.DctProgramCertificates.Base
import CircuitCorrectness.DctProgramCertificates.Group04
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk570 : rows ExportedData.chunk570.1 ExportedData.chunk570.2 =
    (firstRows.drop 3840).take 128 := by decide
theorem chunk571 : rows ExportedData.chunk571.1 ExportedData.chunk571.2 =
    (firstRows.drop 3968).take 128 := by decide
theorem chunk572 : rows ExportedData.chunk572.1 ExportedData.chunk572.2 =
    (firstRows.drop 4096).take 128 := by decide
theorem chunk573 : rows ExportedData.chunk573.1 ExportedData.chunk573.2 =
    (firstRows.drop 4224).take 128 := by decide
theorem chunk574 : rows ExportedData.chunk574.1 ExportedData.chunk574.2 =
    (firstRows.drop 4352).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
