import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row13
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_14 : packingRows 14 =
    ((rows ExportedData.chunk734.1 ExportedData.chunk734.2).drop 42).take 86 ++
    (((rows ExportedData.chunk735.1 ExportedData.chunk735.2).drop 0).take 90) := by decide
theorem packing_mem_14 (row : Row) (hr : row ∈ packingRows 14) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_14] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 734 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 735 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
