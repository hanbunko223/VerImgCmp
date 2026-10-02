import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group07

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk220 :
    Exported.decodeRows ExportedData.chunk220.1 ExportedData.chunk220.2 =
      (List.range 128).map (fun i => byteRow (28160 + i)) := by
  exact Codes.certificate 220 ExportedData.chunk220 (by decide)

theorem chunk221 :
    Exported.decodeRows ExportedData.chunk221.1 ExportedData.chunk221.2 =
      (List.range 128).map (fun i => byteRow (28288 + i)) := by
  exact Codes.certificate 221 ExportedData.chunk221 (by decide)

theorem chunk222 :
    Exported.decodeRows ExportedData.chunk222.1 ExportedData.chunk222.2 =
      (List.range 128).map (fun i => byteRow (28416 + i)) := by
  exact Codes.certificate 222 ExportedData.chunk222 (by decide)

theorem chunk223 :
    Exported.decodeRows ExportedData.chunk223.1 ExportedData.chunk223.2 =
      (List.range 128).map (fun i => byteRow (28544 + i)) := by
  exact Codes.certificate 223 ExportedData.chunk223 (by decide)

theorem chunk224 :
    Exported.decodeRows ExportedData.chunk224.1 ExportedData.chunk224.2 =
      (List.range 128).map (fun i => byteRow (28672 + i)) := by
  exact Codes.certificate 224 ExportedData.chunk224 (by decide)

theorem chunk225 :
    Exported.decodeRows ExportedData.chunk225.1 ExportedData.chunk225.2 =
      (List.range 128).map (fun i => byteRow (28800 + i)) := by
  exact Codes.certificate 225 ExportedData.chunk225 (by decide)

theorem chunk226 :
    Exported.decodeRows ExportedData.chunk226.1 ExportedData.chunk226.2 =
      (List.range 128).map (fun i => byteRow (28928 + i)) := by
  exact Codes.certificate 226 ExportedData.chunk226 (by decide)

theorem chunk227 :
    Exported.decodeRows ExportedData.chunk227.1 ExportedData.chunk227.2 =
      (List.range 128).map (fun i => byteRow (29056 + i)) := by
  exact Codes.certificate 227 ExportedData.chunk227 (by decide)

theorem chunk228 :
    Exported.decodeRows ExportedData.chunk228.1 ExportedData.chunk228.2 =
      (List.range 128).map (fun i => byteRow (29184 + i)) := by
  exact Codes.certificate 228 ExportedData.chunk228 (by decide)

theorem chunk229 :
    Exported.decodeRows ExportedData.chunk229.1 ExportedData.chunk229.2 =
      (List.range 128).map (fun i => byteRow (29312 + i)) := by
  exact Codes.certificate 229 ExportedData.chunk229 (by decide)

theorem chunk230 :
    Exported.decodeRows ExportedData.chunk230.1 ExportedData.chunk230.2 =
      (List.range 128).map (fun i => byteRow (29440 + i)) := by
  exact Codes.certificate 230 ExportedData.chunk230 (by decide)

theorem chunk231 :
    Exported.decodeRows ExportedData.chunk231.1 ExportedData.chunk231.2 =
      (List.range 128).map (fun i => byteRow (29568 + i)) := by
  exact Codes.certificate 231 ExportedData.chunk231 (by decide)

theorem chunk232 :
    Exported.decodeRows ExportedData.chunk232.1 ExportedData.chunk232.2 =
      (List.range 128).map (fun i => byteRow (29696 + i)) := by
  exact Codes.certificate 232 ExportedData.chunk232 (by decide)

theorem chunk233 :
    Exported.decodeRows ExportedData.chunk233.1 ExportedData.chunk233.2 =
      (List.range 128).map (fun i => byteRow (29824 + i)) := by
  exact Codes.certificate 233 ExportedData.chunk233 (by decide)

theorem chunk234 :
    Exported.decodeRows ExportedData.chunk234.1 ExportedData.chunk234.2 =
      (List.range 128).map (fun i => byteRow (29952 + i)) := by
  exact Codes.certificate 234 ExportedData.chunk234 (by decide)

theorem chunk235 :
    Exported.decodeRows ExportedData.chunk235.1 ExportedData.chunk235.2 =
      (List.range 128).map (fun i => byteRow (30080 + i)) := by
  exact Codes.certificate 235 ExportedData.chunk235 (by decide)

theorem chunk236 :
    Exported.decodeRows ExportedData.chunk236.1 ExportedData.chunk236.2 =
      (List.range 128).map (fun i => byteRow (30208 + i)) := by
  exact Codes.certificate 236 ExportedData.chunk236 (by decide)

theorem chunk237 :
    Exported.decodeRows ExportedData.chunk237.1 ExportedData.chunk237.2 =
      (List.range 128).map (fun i => byteRow (30336 + i)) := by
  exact Codes.certificate 237 ExportedData.chunk237 (by decide)

theorem chunk238 :
    Exported.decodeRows ExportedData.chunk238.1 ExportedData.chunk238.2 =
      (List.range 128).map (fun i => byteRow (30464 + i)) := by
  exact Codes.certificate 238 ExportedData.chunk238 (by decide)

theorem chunk239 :
    Exported.decodeRows ExportedData.chunk239.1 ExportedData.chunk239.2 =
      (List.range 128).map (fun i => byteRow (30592 + i)) := by
  exact Codes.certificate 239 ExportedData.chunk239 (by decide)

theorem group11 (c : Fin 20) :
    let chunk := ExportedData.chunks[220 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(220 + c.val) + i)) := by
  fin_cases c
  · exact chunk220
  · exact chunk221
  · exact chunk222
  · exact chunk223
  · exact chunk224
  · exact chunk225
  · exact chunk226
  · exact chunk227
  · exact chunk228
  · exact chunk229
  · exact chunk230
  · exact chunk231
  · exact chunk232
  · exact chunk233
  · exact chunk234
  · exact chunk235
  · exact chunk236
  · exact chunk237
  · exact chunk238
  · exact chunk239

end CircuitCorrectness.ConcreteBytes
