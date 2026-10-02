import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row09
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_10 : packingRows 10 =
    ((rows ExportedData.chunk697.1 ExportedData.chunk697.2).drop 14).take 114 ++
    (((rows ExportedData.chunk698.1 ExportedData.chunk698.2).drop 0).take 62) := by decide
theorem packing_mem_10 (row : Row) (hr : row ∈ packingRows 10) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_10] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 697 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 698 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
