import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row01
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_02 : packingRows 2 =
    ((rows ExportedData.chunk622.1 ExportedData.chunk622.2).drop 86).take 42 ++
    (((rows ExportedData.chunk623.1 ExportedData.chunk623.2).drop 0).take 128 ++
    (((rows ExportedData.chunk624.1 ExportedData.chunk624.2).drop 0).take 6)) := by decide
theorem packing_mem_02 (row : Row) (hr : row ∈ packingRows 2) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_02] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 622 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 623 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 624 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
