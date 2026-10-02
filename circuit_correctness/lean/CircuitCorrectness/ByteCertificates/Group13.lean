import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group09

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk260 :
    Exported.decodeRows ExportedData.chunk260.1 ExportedData.chunk260.2 =
      (List.range 128).map (fun i => byteRow (33280 + i)) := by
  exact Codes.certificate 260 ExportedData.chunk260 (by decide)

theorem chunk261 :
    Exported.decodeRows ExportedData.chunk261.1 ExportedData.chunk261.2 =
      (List.range 128).map (fun i => byteRow (33408 + i)) := by
  exact Codes.certificate 261 ExportedData.chunk261 (by decide)

theorem chunk262 :
    Exported.decodeRows ExportedData.chunk262.1 ExportedData.chunk262.2 =
      (List.range 128).map (fun i => byteRow (33536 + i)) := by
  exact Codes.certificate 262 ExportedData.chunk262 (by decide)

theorem chunk263 :
    Exported.decodeRows ExportedData.chunk263.1 ExportedData.chunk263.2 =
      (List.range 128).map (fun i => byteRow (33664 + i)) := by
  exact Codes.certificate 263 ExportedData.chunk263 (by decide)

theorem chunk264 :
    Exported.decodeRows ExportedData.chunk264.1 ExportedData.chunk264.2 =
      (List.range 128).map (fun i => byteRow (33792 + i)) := by
  exact Codes.certificate 264 ExportedData.chunk264 (by decide)

theorem chunk265 :
    Exported.decodeRows ExportedData.chunk265.1 ExportedData.chunk265.2 =
      (List.range 128).map (fun i => byteRow (33920 + i)) := by
  exact Codes.certificate 265 ExportedData.chunk265 (by decide)

theorem chunk266 :
    Exported.decodeRows ExportedData.chunk266.1 ExportedData.chunk266.2 =
      (List.range 128).map (fun i => byteRow (34048 + i)) := by
  exact Codes.certificate 266 ExportedData.chunk266 (by decide)

theorem chunk267 :
    Exported.decodeRows ExportedData.chunk267.1 ExportedData.chunk267.2 =
      (List.range 128).map (fun i => byteRow (34176 + i)) := by
  exact Codes.certificate 267 ExportedData.chunk267 (by decide)

theorem chunk268 :
    Exported.decodeRows ExportedData.chunk268.1 ExportedData.chunk268.2 =
      (List.range 128).map (fun i => byteRow (34304 + i)) := by
  exact Codes.certificate 268 ExportedData.chunk268 (by decide)

theorem chunk269 :
    Exported.decodeRows ExportedData.chunk269.1 ExportedData.chunk269.2 =
      (List.range 128).map (fun i => byteRow (34432 + i)) := by
  exact Codes.certificate 269 ExportedData.chunk269 (by decide)

theorem chunk270 :
    Exported.decodeRows ExportedData.chunk270.1 ExportedData.chunk270.2 =
      (List.range 128).map (fun i => byteRow (34560 + i)) := by
  exact Codes.certificate 270 ExportedData.chunk270 (by decide)

theorem chunk271 :
    Exported.decodeRows ExportedData.chunk271.1 ExportedData.chunk271.2 =
      (List.range 128).map (fun i => byteRow (34688 + i)) := by
  exact Codes.certificate 271 ExportedData.chunk271 (by decide)

theorem chunk272 :
    Exported.decodeRows ExportedData.chunk272.1 ExportedData.chunk272.2 =
      (List.range 128).map (fun i => byteRow (34816 + i)) := by
  exact Codes.certificate 272 ExportedData.chunk272 (by decide)

theorem chunk273 :
    Exported.decodeRows ExportedData.chunk273.1 ExportedData.chunk273.2 =
      (List.range 128).map (fun i => byteRow (34944 + i)) := by
  exact Codes.certificate 273 ExportedData.chunk273 (by decide)

theorem chunk274 :
    Exported.decodeRows ExportedData.chunk274.1 ExportedData.chunk274.2 =
      (List.range 128).map (fun i => byteRow (35072 + i)) := by
  exact Codes.certificate 274 ExportedData.chunk274 (by decide)

theorem chunk275 :
    Exported.decodeRows ExportedData.chunk275.1 ExportedData.chunk275.2 =
      (List.range 128).map (fun i => byteRow (35200 + i)) := by
  exact Codes.certificate 275 ExportedData.chunk275 (by decide)

theorem chunk276 :
    Exported.decodeRows ExportedData.chunk276.1 ExportedData.chunk276.2 =
      (List.range 128).map (fun i => byteRow (35328 + i)) := by
  exact Codes.certificate 276 ExportedData.chunk276 (by decide)

theorem chunk277 :
    Exported.decodeRows ExportedData.chunk277.1 ExportedData.chunk277.2 =
      (List.range 128).map (fun i => byteRow (35456 + i)) := by
  exact Codes.certificate 277 ExportedData.chunk277 (by decide)

theorem chunk278 :
    Exported.decodeRows ExportedData.chunk278.1 ExportedData.chunk278.2 =
      (List.range 128).map (fun i => byteRow (35584 + i)) := by
  exact Codes.certificate 278 ExportedData.chunk278 (by decide)

theorem chunk279 :
    Exported.decodeRows ExportedData.chunk279.1 ExportedData.chunk279.2 =
      (List.range 128).map (fun i => byteRow (35712 + i)) := by
  exact Codes.certificate 279 ExportedData.chunk279 (by decide)

theorem group13 (c : Fin 20) :
    let chunk := ExportedData.chunks[260 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(260 + c.val) + i)) := by
  fin_cases c
  · exact chunk260
  · exact chunk261
  · exact chunk262
  · exact chunk263
  · exact chunk264
  · exact chunk265
  · exact chunk266
  · exact chunk267
  · exact chunk268
  · exact chunk269
  · exact chunk270
  · exact chunk271
  · exact chunk272
  · exact chunk273
  · exact chunk274
  · exact chunk275
  · exact chunk276
  · exact chunk277
  · exact chunk278
  · exact chunk279

end CircuitCorrectness.ConcreteBytes
