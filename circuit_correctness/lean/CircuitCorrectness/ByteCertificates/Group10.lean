import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group06

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk200 :
    Exported.decodeRows ExportedData.chunk200.1 ExportedData.chunk200.2 =
      (List.range 128).map (fun i => byteRow (25600 + i)) := by
  exact Codes.certificate 200 ExportedData.chunk200 (by decide)

theorem chunk201 :
    Exported.decodeRows ExportedData.chunk201.1 ExportedData.chunk201.2 =
      (List.range 128).map (fun i => byteRow (25728 + i)) := by
  exact Codes.certificate 201 ExportedData.chunk201 (by decide)

theorem chunk202 :
    Exported.decodeRows ExportedData.chunk202.1 ExportedData.chunk202.2 =
      (List.range 128).map (fun i => byteRow (25856 + i)) := by
  exact Codes.certificate 202 ExportedData.chunk202 (by decide)

theorem chunk203 :
    Exported.decodeRows ExportedData.chunk203.1 ExportedData.chunk203.2 =
      (List.range 128).map (fun i => byteRow (25984 + i)) := by
  exact Codes.certificate 203 ExportedData.chunk203 (by decide)

theorem chunk204 :
    Exported.decodeRows ExportedData.chunk204.1 ExportedData.chunk204.2 =
      (List.range 128).map (fun i => byteRow (26112 + i)) := by
  exact Codes.certificate 204 ExportedData.chunk204 (by decide)

theorem chunk205 :
    Exported.decodeRows ExportedData.chunk205.1 ExportedData.chunk205.2 =
      (List.range 128).map (fun i => byteRow (26240 + i)) := by
  exact Codes.certificate 205 ExportedData.chunk205 (by decide)

theorem chunk206 :
    Exported.decodeRows ExportedData.chunk206.1 ExportedData.chunk206.2 =
      (List.range 128).map (fun i => byteRow (26368 + i)) := by
  exact Codes.certificate 206 ExportedData.chunk206 (by decide)

theorem chunk207 :
    Exported.decodeRows ExportedData.chunk207.1 ExportedData.chunk207.2 =
      (List.range 128).map (fun i => byteRow (26496 + i)) := by
  exact Codes.certificate 207 ExportedData.chunk207 (by decide)

theorem chunk208 :
    Exported.decodeRows ExportedData.chunk208.1 ExportedData.chunk208.2 =
      (List.range 128).map (fun i => byteRow (26624 + i)) := by
  exact Codes.certificate 208 ExportedData.chunk208 (by decide)

theorem chunk209 :
    Exported.decodeRows ExportedData.chunk209.1 ExportedData.chunk209.2 =
      (List.range 128).map (fun i => byteRow (26752 + i)) := by
  exact Codes.certificate 209 ExportedData.chunk209 (by decide)

theorem chunk210 :
    Exported.decodeRows ExportedData.chunk210.1 ExportedData.chunk210.2 =
      (List.range 128).map (fun i => byteRow (26880 + i)) := by
  exact Codes.certificate 210 ExportedData.chunk210 (by decide)

theorem chunk211 :
    Exported.decodeRows ExportedData.chunk211.1 ExportedData.chunk211.2 =
      (List.range 128).map (fun i => byteRow (27008 + i)) := by
  exact Codes.certificate 211 ExportedData.chunk211 (by decide)

theorem chunk212 :
    Exported.decodeRows ExportedData.chunk212.1 ExportedData.chunk212.2 =
      (List.range 128).map (fun i => byteRow (27136 + i)) := by
  exact Codes.certificate 212 ExportedData.chunk212 (by decide)

theorem chunk213 :
    Exported.decodeRows ExportedData.chunk213.1 ExportedData.chunk213.2 =
      (List.range 128).map (fun i => byteRow (27264 + i)) := by
  exact Codes.certificate 213 ExportedData.chunk213 (by decide)

theorem chunk214 :
    Exported.decodeRows ExportedData.chunk214.1 ExportedData.chunk214.2 =
      (List.range 128).map (fun i => byteRow (27392 + i)) := by
  exact Codes.certificate 214 ExportedData.chunk214 (by decide)

theorem chunk215 :
    Exported.decodeRows ExportedData.chunk215.1 ExportedData.chunk215.2 =
      (List.range 128).map (fun i => byteRow (27520 + i)) := by
  exact Codes.certificate 215 ExportedData.chunk215 (by decide)

theorem chunk216 :
    Exported.decodeRows ExportedData.chunk216.1 ExportedData.chunk216.2 =
      (List.range 128).map (fun i => byteRow (27648 + i)) := by
  exact Codes.certificate 216 ExportedData.chunk216 (by decide)

theorem chunk217 :
    Exported.decodeRows ExportedData.chunk217.1 ExportedData.chunk217.2 =
      (List.range 128).map (fun i => byteRow (27776 + i)) := by
  exact Codes.certificate 217 ExportedData.chunk217 (by decide)

theorem chunk218 :
    Exported.decodeRows ExportedData.chunk218.1 ExportedData.chunk218.2 =
      (List.range 128).map (fun i => byteRow (27904 + i)) := by
  exact Codes.certificate 218 ExportedData.chunk218 (by decide)

theorem chunk219 :
    Exported.decodeRows ExportedData.chunk219.1 ExportedData.chunk219.2 =
      (List.range 128).map (fun i => byteRow (28032 + i)) := by
  exact Codes.certificate 219 ExportedData.chunk219 (by decide)

theorem group10 (c : Fin 20) :
    let chunk := ExportedData.chunks[200 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(200 + c.val) + i)) := by
  fin_cases c
  · exact chunk200
  · exact chunk201
  · exact chunk202
  · exact chunk203
  · exact chunk204
  · exact chunk205
  · exact chunk206
  · exact chunk207
  · exact chunk208
  · exact chunk209
  · exact chunk210
  · exact chunk211
  · exact chunk212
  · exact chunk213
  · exact chunk214
  · exact chunk215
  · exact chunk216
  · exact chunk217
  · exact chunk218
  · exact chunk219

end CircuitCorrectness.ConcreteBytes
