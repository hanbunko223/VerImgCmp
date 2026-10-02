import CircuitCorrectness.PackingCertificates.Row00
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
open ConcreteBytes.Codes

/-- These reference the actual first exported pixel and chunk packing equations. -/
def firstActualPixel : Row :=
  ((rows ExportedData.chunk604.1 ExportedData.chunk604.2).drop 8).headD ⟨[],[],[]⟩
def firstActualChunk : Row :=
  ((rows ExportedData.chunk605.1 ExportedData.chunk605.2).drop 40).headD ⟨[],[],[]⟩

theorem rejects_pixel_coefficient : firstActualPixel ≠
    { pixelRow 0 0 with a := (pixelRow 0 0).a.map fun (i,c) =>
      (i,if i = 13 then c+1 else c) } := by decide

theorem rejects_pixel_input_wire : firstActualPixel ≠
    { pixelRow 0 0 with a := (pixelRow 0 0).a.map fun (i,c) =>
      (if i = 4 then 5 else i,c) } := by decide

theorem rejects_pixel_output_wire : firstActualPixel ≠
    { pixelRow 0 0 with a := (pixelRow 0 0).a.map fun (i,c) =>
      (if i = 77324 then 77325 else i,c) } := by decide

theorem rejects_pixel_order :
    ((rows ExportedData.chunk604.1 ExportedData.chunk604.2).drop 8).take 2 ≠
      ((packingRows 0).take 2).reverse := by decide

theorem rejects_chunk_coefficient : firstActualChunk ≠
    { chunkRow 0 0 with a := (chunkRow 0 0).a.map fun (i,c) =>
      (i,if i = 77325 then c+1 else c) } := by decide

end CircuitCorrectness.PackingCertificates
