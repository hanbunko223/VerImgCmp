import CircuitCorrectness.PackingCertificates.Base
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_00 : packingRows 0 =
    ((rows ExportedData.chunk604.1 ExportedData.chunk604.2).drop 8).take 120 ++
    (((rows ExportedData.chunk605.1 ExportedData.chunk605.2).drop 0).take 56) := by decide
theorem packing_mem_00 (row : Row) (hr : row ∈ packingRows 0) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_00] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 604 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 605 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
