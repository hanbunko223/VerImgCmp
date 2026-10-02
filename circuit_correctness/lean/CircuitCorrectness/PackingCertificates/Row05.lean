import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row04
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_05 : packingRows 5 =
    ((rows ExportedData.chunk650.1 ExportedData.chunk650.2).drop 75).take 53 ++
    (((rows ExportedData.chunk651.1 ExportedData.chunk651.2).drop 0).take 123) := by decide
theorem packing_mem_05 (row : Row) (hr : row ∈ packingRows 5) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_05] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 650 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 651 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
