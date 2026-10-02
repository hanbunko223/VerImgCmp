import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row06
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_07 : packingRows 7 =
    ((rows ExportedData.chunk669.1 ExportedData.chunk669.2).drop 25).take 103 ++
    (((rows ExportedData.chunk670.1 ExportedData.chunk670.2).drop 0).take 73) := by decide
theorem packing_mem_07 (row : Row) (hr : row ∈ packingRows 7) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_07] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 669 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 670 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
