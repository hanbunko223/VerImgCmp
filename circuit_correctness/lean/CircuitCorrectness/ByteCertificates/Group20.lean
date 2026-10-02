import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group16

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk400 :
    Exported.decodeRows ExportedData.chunk400.1 ExportedData.chunk400.2 =
      (List.range 128).map (fun i => byteRow (51200 + i)) := by
  exact Codes.certificate 400 ExportedData.chunk400 (by decide)

theorem chunk401 :
    Exported.decodeRows ExportedData.chunk401.1 ExportedData.chunk401.2 =
      (List.range 128).map (fun i => byteRow (51328 + i)) := by
  exact Codes.certificate 401 ExportedData.chunk401 (by decide)

theorem chunk402 :
    Exported.decodeRows ExportedData.chunk402.1 ExportedData.chunk402.2 =
      (List.range 128).map (fun i => byteRow (51456 + i)) := by
  exact Codes.certificate 402 ExportedData.chunk402 (by decide)

theorem chunk403 :
    Exported.decodeRows ExportedData.chunk403.1 ExportedData.chunk403.2 =
      (List.range 128).map (fun i => byteRow (51584 + i)) := by
  exact Codes.certificate 403 ExportedData.chunk403 (by decide)

theorem chunk404 :
    Exported.decodeRows ExportedData.chunk404.1 ExportedData.chunk404.2 =
      (List.range 128).map (fun i => byteRow (51712 + i)) := by
  exact Codes.certificate 404 ExportedData.chunk404 (by decide)

theorem chunk405 :
    Exported.decodeRows ExportedData.chunk405.1 ExportedData.chunk405.2 =
      (List.range 128).map (fun i => byteRow (51840 + i)) := by
  exact Codes.certificate 405 ExportedData.chunk405 (by decide)

theorem chunk406 :
    Exported.decodeRows ExportedData.chunk406.1 ExportedData.chunk406.2 =
      (List.range 128).map (fun i => byteRow (51968 + i)) := by
  exact Codes.certificate 406 ExportedData.chunk406 (by decide)

theorem chunk407 :
    Exported.decodeRows ExportedData.chunk407.1 ExportedData.chunk407.2 =
      (List.range 128).map (fun i => byteRow (52096 + i)) := by
  exact Codes.certificate 407 ExportedData.chunk407 (by decide)

theorem chunk408 :
    Exported.decodeRows ExportedData.chunk408.1 ExportedData.chunk408.2 =
      (List.range 128).map (fun i => byteRow (52224 + i)) := by
  exact Codes.certificate 408 ExportedData.chunk408 (by decide)

theorem chunk409 :
    Exported.decodeRows ExportedData.chunk409.1 ExportedData.chunk409.2 =
      (List.range 128).map (fun i => byteRow (52352 + i)) := by
  exact Codes.certificate 409 ExportedData.chunk409 (by decide)

theorem chunk410 :
    Exported.decodeRows ExportedData.chunk410.1 ExportedData.chunk410.2 =
      (List.range 128).map (fun i => byteRow (52480 + i)) := by
  exact Codes.certificate 410 ExportedData.chunk410 (by decide)

theorem chunk411 :
    Exported.decodeRows ExportedData.chunk411.1 ExportedData.chunk411.2 =
      (List.range 128).map (fun i => byteRow (52608 + i)) := by
  exact Codes.certificate 411 ExportedData.chunk411 (by decide)

theorem chunk412 :
    Exported.decodeRows ExportedData.chunk412.1 ExportedData.chunk412.2 =
      (List.range 128).map (fun i => byteRow (52736 + i)) := by
  exact Codes.certificate 412 ExportedData.chunk412 (by decide)

theorem chunk413 :
    Exported.decodeRows ExportedData.chunk413.1 ExportedData.chunk413.2 =
      (List.range 128).map (fun i => byteRow (52864 + i)) := by
  exact Codes.certificate 413 ExportedData.chunk413 (by decide)

theorem chunk414 :
    Exported.decodeRows ExportedData.chunk414.1 ExportedData.chunk414.2 =
      (List.range 128).map (fun i => byteRow (52992 + i)) := by
  exact Codes.certificate 414 ExportedData.chunk414 (by decide)

theorem chunk415 :
    Exported.decodeRows ExportedData.chunk415.1 ExportedData.chunk415.2 =
      (List.range 128).map (fun i => byteRow (53120 + i)) := by
  exact Codes.certificate 415 ExportedData.chunk415 (by decide)

theorem chunk416 :
    Exported.decodeRows ExportedData.chunk416.1 ExportedData.chunk416.2 =
      (List.range 128).map (fun i => byteRow (53248 + i)) := by
  exact Codes.certificate 416 ExportedData.chunk416 (by decide)

theorem chunk417 :
    Exported.decodeRows ExportedData.chunk417.1 ExportedData.chunk417.2 =
      (List.range 128).map (fun i => byteRow (53376 + i)) := by
  exact Codes.certificate 417 ExportedData.chunk417 (by decide)

theorem chunk418 :
    Exported.decodeRows ExportedData.chunk418.1 ExportedData.chunk418.2 =
      (List.range 128).map (fun i => byteRow (53504 + i)) := by
  exact Codes.certificate 418 ExportedData.chunk418 (by decide)

theorem chunk419 :
    Exported.decodeRows ExportedData.chunk419.1 ExportedData.chunk419.2 =
      (List.range 128).map (fun i => byteRow (53632 + i)) := by
  exact Codes.certificate 419 ExportedData.chunk419 (by decide)

theorem group20 (c : Fin 20) :
    let chunk := ExportedData.chunks[400 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(400 + c.val) + i)) := by
  fin_cases c
  · exact chunk400
  · exact chunk401
  · exact chunk402
  · exact chunk403
  · exact chunk404
  · exact chunk405
  · exact chunk406
  · exact chunk407
  · exact chunk408
  · exact chunk409
  · exact chunk410
  · exact chunk411
  · exact chunk412
  · exact chunk413
  · exact chunk414
  · exact chunk415
  · exact chunk416
  · exact chunk417
  · exact chunk418
  · exact chunk419

end CircuitCorrectness.ConcreteBytes
