import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row10
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_11 : packingRows 11 =
    ((rows ExportedData.chunk706.1 ExportedData.chunk706.2).drop 53).take 75 ++
    (((rows ExportedData.chunk707.1 ExportedData.chunk707.2).drop 0).take 101) := by decide
theorem packing_mem_11 (row : Row) (hr : row ∈ packingRows 11) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_11] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 706 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 707 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
