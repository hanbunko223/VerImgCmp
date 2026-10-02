import CircuitCorrectness.PackingCertificates.Base
import CircuitCorrectness.PackingCertificates.Row05
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes
theorem packing_rows_06 : packingRows 6 =
    ((rows ExportedData.chunk659.1 ExportedData.chunk659.2).drop 114).take 14 ++
    (((rows ExportedData.chunk660.1 ExportedData.chunk660.2).drop 0).take 128 ++
    (((rows ExportedData.chunk661.1 ExportedData.chunk661.2).drop 0).take 34)) := by decide
theorem packing_mem_06 (row : Row) (hr : row ∈ packingRows 6) :
    expandRow row ∈ Exported.rows := by
  rw [packing_rows_06] at hr
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 659 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  rcases List.mem_append.mp hr with hs | hr
  · exact coded_row_mem 660 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))
  exact coded_row_mem 661 (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))
end CircuitCorrectness.PackingCertificates
