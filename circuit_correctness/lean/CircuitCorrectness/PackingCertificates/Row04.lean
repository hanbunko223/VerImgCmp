import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row03
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_04 : packingRows 4 =
    ((rows ExportedData.chunk641.1 ExportedData.chunk641.2).drop 36).take 92 ++
    (((rows ExportedData.chunk642.1 ExportedData.chunk642.2).drop 0).take 84) := by decide
theorem packing_mem_04 (row : Row) (hr : row ∈ packingRows 4) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_04] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 641 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 642 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
