import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row08
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_09 : packingRows 9 =
    ((rows ExportedData.chunk687.1 ExportedData.chunk687.2).drop 103).take 25 ++
    (((rows ExportedData.chunk688.1 ExportedData.chunk688.2).drop 0).take 128 ++
    (((rows ExportedData.chunk689.1 ExportedData.chunk689.2).drop 0).take 23)) := by decide
theorem packing_mem_09 (row : Row) (hr : row ∈ packingRows 9) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_09] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 687 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 688 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 689 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
