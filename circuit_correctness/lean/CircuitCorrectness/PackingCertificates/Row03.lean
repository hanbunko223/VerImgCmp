import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row02
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_03 : packingRows 3 =
    ((rows ExportedData.chunk631.1 ExportedData.chunk631.2).drop 125).take 3 ++
    (((rows ExportedData.chunk632.1 ExportedData.chunk632.2).drop 0).take 128 ++
    (((rows ExportedData.chunk633.1 ExportedData.chunk633.2).drop 0).take 45)) := by decide
theorem packing_mem_03 (row : Row) (hr : row ∈ packingRows 3) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_03] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 631 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 632 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 633 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
