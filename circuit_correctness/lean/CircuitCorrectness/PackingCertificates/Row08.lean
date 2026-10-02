import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row07
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_08 : packingRows 8 =
    ((rows ExportedData.chunk678.1 ExportedData.chunk678.2).drop 64).take 64 ++
    (((rows ExportedData.chunk679.1 ExportedData.chunk679.2).drop 0).take 112) := by decide
theorem packing_mem_08 (row : Row) (hr : row ∈ packingRows 8) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_08] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 678 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 679 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
