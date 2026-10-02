import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group02

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk120 :
    Exported.decodeRows ExportedData.chunk120.1 ExportedData.chunk120.2 =
      (List.range 128).map (fun i => byteRow (15360 + i)) := by
  exact Codes.certificate 120 ExportedData.chunk120 (by decide)

theorem chunk121 :
    Exported.decodeRows ExportedData.chunk121.1 ExportedData.chunk121.2 =
      (List.range 128).map (fun i => byteRow (15488 + i)) := by
  exact Codes.certificate 121 ExportedData.chunk121 (by decide)

theorem chunk122 :
    Exported.decodeRows ExportedData.chunk122.1 ExportedData.chunk122.2 =
      (List.range 128).map (fun i => byteRow (15616 + i)) := by
  exact Codes.certificate 122 ExportedData.chunk122 (by decide)

theorem chunk123 :
    Exported.decodeRows ExportedData.chunk123.1 ExportedData.chunk123.2 =
      (List.range 128).map (fun i => byteRow (15744 + i)) := by
  exact Codes.certificate 123 ExportedData.chunk123 (by decide)

theorem chunk124 :
    Exported.decodeRows ExportedData.chunk124.1 ExportedData.chunk124.2 =
      (List.range 128).map (fun i => byteRow (15872 + i)) := by
  exact Codes.certificate 124 ExportedData.chunk124 (by decide)

theorem chunk125 :
    Exported.decodeRows ExportedData.chunk125.1 ExportedData.chunk125.2 =
      (List.range 128).map (fun i => byteRow (16000 + i)) := by
  exact Codes.certificate 125 ExportedData.chunk125 (by decide)

theorem chunk126 :
    Exported.decodeRows ExportedData.chunk126.1 ExportedData.chunk126.2 =
      (List.range 128).map (fun i => byteRow (16128 + i)) := by
  exact Codes.certificate 126 ExportedData.chunk126 (by decide)

theorem chunk127 :
    Exported.decodeRows ExportedData.chunk127.1 ExportedData.chunk127.2 =
      (List.range 128).map (fun i => byteRow (16256 + i)) := by
  exact Codes.certificate 127 ExportedData.chunk127 (by decide)

theorem chunk128 :
    Exported.decodeRows ExportedData.chunk128.1 ExportedData.chunk128.2 =
      (List.range 128).map (fun i => byteRow (16384 + i)) := by
  exact Codes.certificate 128 ExportedData.chunk128 (by decide)

theorem chunk129 :
    Exported.decodeRows ExportedData.chunk129.1 ExportedData.chunk129.2 =
      (List.range 128).map (fun i => byteRow (16512 + i)) := by
  exact Codes.certificate 129 ExportedData.chunk129 (by decide)

theorem chunk130 :
    Exported.decodeRows ExportedData.chunk130.1 ExportedData.chunk130.2 =
      (List.range 128).map (fun i => byteRow (16640 + i)) := by
  exact Codes.certificate 130 ExportedData.chunk130 (by decide)

theorem chunk131 :
    Exported.decodeRows ExportedData.chunk131.1 ExportedData.chunk131.2 =
      (List.range 128).map (fun i => byteRow (16768 + i)) := by
  exact Codes.certificate 131 ExportedData.chunk131 (by decide)

theorem chunk132 :
    Exported.decodeRows ExportedData.chunk132.1 ExportedData.chunk132.2 =
      (List.range 128).map (fun i => byteRow (16896 + i)) := by
  exact Codes.certificate 132 ExportedData.chunk132 (by decide)

theorem chunk133 :
    Exported.decodeRows ExportedData.chunk133.1 ExportedData.chunk133.2 =
      (List.range 128).map (fun i => byteRow (17024 + i)) := by
  exact Codes.certificate 133 ExportedData.chunk133 (by decide)

theorem chunk134 :
    Exported.decodeRows ExportedData.chunk134.1 ExportedData.chunk134.2 =
      (List.range 128).map (fun i => byteRow (17152 + i)) := by
  exact Codes.certificate 134 ExportedData.chunk134 (by decide)

theorem chunk135 :
    Exported.decodeRows ExportedData.chunk135.1 ExportedData.chunk135.2 =
      (List.range 128).map (fun i => byteRow (17280 + i)) := by
  exact Codes.certificate 135 ExportedData.chunk135 (by decide)

theorem chunk136 :
    Exported.decodeRows ExportedData.chunk136.1 ExportedData.chunk136.2 =
      (List.range 128).map (fun i => byteRow (17408 + i)) := by
  exact Codes.certificate 136 ExportedData.chunk136 (by decide)

theorem chunk137 :
    Exported.decodeRows ExportedData.chunk137.1 ExportedData.chunk137.2 =
      (List.range 128).map (fun i => byteRow (17536 + i)) := by
  exact Codes.certificate 137 ExportedData.chunk137 (by decide)

theorem chunk138 :
    Exported.decodeRows ExportedData.chunk138.1 ExportedData.chunk138.2 =
      (List.range 128).map (fun i => byteRow (17664 + i)) := by
  exact Codes.certificate 138 ExportedData.chunk138 (by decide)

theorem chunk139 :
    Exported.decodeRows ExportedData.chunk139.1 ExportedData.chunk139.2 =
      (List.range 128).map (fun i => byteRow (17792 + i)) := by
  exact Codes.certificate 139 ExportedData.chunk139 (by decide)

theorem group06 (c : Fin 20) :
    let chunk := ExportedData.chunks[120 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(120 + c.val) + i)) := by
  fin_cases c
  · exact chunk120
  · exact chunk121
  · exact chunk122
  · exact chunk123
  · exact chunk124
  · exact chunk125
  · exact chunk126
  · exact chunk127
  · exact chunk128
  · exact chunk129
  · exact chunk130
  · exact chunk131
  · exact chunk132
  · exact chunk133
  · exact chunk134
  · exact chunk135
  · exact chunk136
  · exact chunk137
  · exact chunk138
  · exact chunk139

end CircuitCorrectness.ConcreteBytes
