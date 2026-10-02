import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row00
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_01 : packingRows 1 =
    ((rows ExportedData.chunk613.1 ExportedData.chunk613.2).drop 47).take 81 ++
    (((rows ExportedData.chunk614.1 ExportedData.chunk614.2).drop 0).take 95) := by decide
theorem packing_mem_01 (row : Row) (hr : row ∈ packingRows 1) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_01] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 613 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 614 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
