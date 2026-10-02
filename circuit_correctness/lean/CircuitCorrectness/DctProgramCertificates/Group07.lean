import CircuitCorrectness.DctProgramCertificates.Base
import CircuitCorrectness.DctProgramCertificates.Group05
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk575 : rows ExportedData.chunk575.1 ExportedData.chunk575.2 =
    (firstRows.drop 4480).take 128 := by decide
theorem chunk576 : rows ExportedData.chunk576.1 ExportedData.chunk576.2 =
    (firstRows.drop 4608).take 128 := by decide
theorem chunk577 : rows ExportedData.chunk577.1 ExportedData.chunk577.2 =
    (firstRows.drop 4736).take 128 := by decide
theorem chunk578 : rows ExportedData.chunk578.1 ExportedData.chunk578.2 =
    (firstRows.drop 4864).take 128 := by decide
theorem chunk579 : rows ExportedData.chunk579.1 ExportedData.chunk579.2 =
    (firstRows.drop 4992).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
