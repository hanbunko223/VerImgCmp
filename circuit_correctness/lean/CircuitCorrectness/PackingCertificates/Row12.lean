import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row11
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_12 : packingRows 12 =
    ((rows ExportedData.chunk715.1 ExportedData.chunk715.2).drop 92).take 36 ++
    (((rows ExportedData.chunk716.1 ExportedData.chunk716.2).drop 0).take 128 ++
    (((rows ExportedData.chunk717.1 ExportedData.chunk717.2).drop 0).take 12)) := by decide
theorem packing_mem_12 (row : Row) (hr : row ∈ packingRows 12) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_12] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 715 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 716 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 717 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
