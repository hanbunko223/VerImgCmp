import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row14
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_15 : packingRows 15 =
    ((rows ExportedData.chunk743.1 ExportedData.chunk743.2).drop 81).take 47 ++
    (((rows ExportedData.chunk744.1 ExportedData.chunk744.2).drop 0).take 128 ++
    (((rows ExportedData.chunk745.1 ExportedData.chunk745.2).drop 0).take 1)) := by decide
theorem packing_mem_15 (row : Row) (hr : row ∈ packingRows 15) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_15] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 743 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 744 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 745 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
