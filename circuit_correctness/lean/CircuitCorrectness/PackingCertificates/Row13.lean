import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row12
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_13 : packingRows 13 =
    ((rows ExportedData.chunk725.1 ExportedData.chunk725.2).drop 3).take 125 ++
    (((rows ExportedData.chunk726.1 ExportedData.chunk726.2).drop 0).take 51) := by decide
theorem packing_mem_13 (row : Row) (hr : row ∈ packingRows 13) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_13] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 725 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 726 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
