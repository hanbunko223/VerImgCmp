import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group10

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk280 :
    Exported.decodeRows ExportedData.chunk280.1 ExportedData.chunk280.2 =
      (List.range 128).map (fun i => byteRow (35840 + i)) := by
  exact Codes.certificate 280 ExportedData.chunk280 (by decide)

theorem chunk281 :
    Exported.decodeRows ExportedData.chunk281.1 ExportedData.chunk281.2 =
      (List.range 128).map (fun i => byteRow (35968 + i)) := by
  exact Codes.certificate 281 ExportedData.chunk281 (by decide)

theorem chunk282 :
    Exported.decodeRows ExportedData.chunk282.1 ExportedData.chunk282.2 =
      (List.range 128).map (fun i => byteRow (36096 + i)) := by
  exact Codes.certificate 282 ExportedData.chunk282 (by decide)

theorem chunk283 :
    Exported.decodeRows ExportedData.chunk283.1 ExportedData.chunk283.2 =
      (List.range 128).map (fun i => byteRow (36224 + i)) := by
  exact Codes.certificate 283 ExportedData.chunk283 (by decide)

theorem chunk284 :
    Exported.decodeRows ExportedData.chunk284.1 ExportedData.chunk284.2 =
      (List.range 128).map (fun i => byteRow (36352 + i)) := by
  exact Codes.certificate 284 ExportedData.chunk284 (by decide)

theorem chunk285 :
    Exported.decodeRows ExportedData.chunk285.1 ExportedData.chunk285.2 =
      (List.range 128).map (fun i => byteRow (36480 + i)) := by
  exact Codes.certificate 285 ExportedData.chunk285 (by decide)

theorem chunk286 :
    Exported.decodeRows ExportedData.chunk286.1 ExportedData.chunk286.2 =
      (List.range 128).map (fun i => byteRow (36608 + i)) := by
  exact Codes.certificate 286 ExportedData.chunk286 (by decide)

theorem chunk287 :
    Exported.decodeRows ExportedData.chunk287.1 ExportedData.chunk287.2 =
      (List.range 128).map (fun i => byteRow (36736 + i)) := by
  exact Codes.certificate 287 ExportedData.chunk287 (by decide)

theorem chunk288 :
    Exported.decodeRows ExportedData.chunk288.1 ExportedData.chunk288.2 =
      (List.range 128).map (fun i => byteRow (36864 + i)) := by
  exact Codes.certificate 288 ExportedData.chunk288 (by decide)

theorem chunk289 :
    Exported.decodeRows ExportedData.chunk289.1 ExportedData.chunk289.2 =
      (List.range 128).map (fun i => byteRow (36992 + i)) := by
  exact Codes.certificate 289 ExportedData.chunk289 (by decide)

theorem chunk290 :
    Exported.decodeRows ExportedData.chunk290.1 ExportedData.chunk290.2 =
      (List.range 128).map (fun i => byteRow (37120 + i)) := by
  exact Codes.certificate 290 ExportedData.chunk290 (by decide)

theorem chunk291 :
    Exported.decodeRows ExportedData.chunk291.1 ExportedData.chunk291.2 =
      (List.range 128).map (fun i => byteRow (37248 + i)) := by
  exact Codes.certificate 291 ExportedData.chunk291 (by decide)

theorem chunk292 :
    Exported.decodeRows ExportedData.chunk292.1 ExportedData.chunk292.2 =
      (List.range 128).map (fun i => byteRow (37376 + i)) := by
  exact Codes.certificate 292 ExportedData.chunk292 (by decide)

theorem chunk293 :
    Exported.decodeRows ExportedData.chunk293.1 ExportedData.chunk293.2 =
      (List.range 128).map (fun i => byteRow (37504 + i)) := by
  exact Codes.certificate 293 ExportedData.chunk293 (by decide)

theorem chunk294 :
    Exported.decodeRows ExportedData.chunk294.1 ExportedData.chunk294.2 =
      (List.range 128).map (fun i => byteRow (37632 + i)) := by
  exact Codes.certificate 294 ExportedData.chunk294 (by decide)

theorem chunk295 :
    Exported.decodeRows ExportedData.chunk295.1 ExportedData.chunk295.2 =
      (List.range 128).map (fun i => byteRow (37760 + i)) := by
  exact Codes.certificate 295 ExportedData.chunk295 (by decide)

theorem chunk296 :
    Exported.decodeRows ExportedData.chunk296.1 ExportedData.chunk296.2 =
      (List.range 128).map (fun i => byteRow (37888 + i)) := by
  exact Codes.certificate 296 ExportedData.chunk296 (by decide)

theorem chunk297 :
    Exported.decodeRows ExportedData.chunk297.1 ExportedData.chunk297.2 =
      (List.range 128).map (fun i => byteRow (38016 + i)) := by
  exact Codes.certificate 297 ExportedData.chunk297 (by decide)

theorem chunk298 :
    Exported.decodeRows ExportedData.chunk298.1 ExportedData.chunk298.2 =
      (List.range 128).map (fun i => byteRow (38144 + i)) := by
  exact Codes.certificate 298 ExportedData.chunk298 (by decide)

theorem chunk299 :
    Exported.decodeRows ExportedData.chunk299.1 ExportedData.chunk299.2 =
      (List.range 128).map (fun i => byteRow (38272 + i)) := by
  exact Codes.certificate 299 ExportedData.chunk299 (by decide)

theorem group14 (c : Fin 20) :
    let chunk := ExportedData.chunks[280 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(280 + c.val) + i)) := by
  fin_cases c
  · exact chunk280
  · exact chunk281
  · exact chunk282
  · exact chunk283
  · exact chunk284
  · exact chunk285
  · exact chunk286
  · exact chunk287
  · exact chunk288
  · exact chunk289
  · exact chunk290
  · exact chunk291
  · exact chunk292
  · exact chunk293
  · exact chunk294
  · exact chunk295
  · exact chunk296
  · exact chunk297
  · exact chunk298
  · exact chunk299

end CircuitCorrectness.ConcreteBytes
