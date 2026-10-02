import CircuitCorrectness.DctProgramCertificates.Base
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open ConcreteBytes.Codes
theorem chunk545 : rows ExportedData.chunk545.1 ExportedData.chunk545.2 =
    (firstRows.drop 640).take 128 := by decide
theorem chunk546 : rows ExportedData.chunk546.1 ExportedData.chunk546.2 =
    (firstRows.drop 768).take 128 := by decide
theorem chunk547 : rows ExportedData.chunk547.1 ExportedData.chunk547.2 =
    (firstRows.drop 896).take 128 := by decide
theorem chunk548 : rows ExportedData.chunk548.1 ExportedData.chunk548.2 =
    (firstRows.drop 1024).take 128 := by decide
theorem chunk549 : rows ExportedData.chunk549.1 ExportedData.chunk549.2 =
    (firstRows.drop 1152).take 128 := by decide
end CircuitCorrectness.DctProgramCertificates
