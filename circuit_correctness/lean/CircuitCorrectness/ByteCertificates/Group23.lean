import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group19

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk460 :
    Exported.decodeRows ExportedData.chunk460.1 ExportedData.chunk460.2 =
      (List.range 128).map (fun i => byteRow (58880 + i)) := by
  exact Codes.certificate 460 ExportedData.chunk460 (by decide)

theorem chunk461 :
    Exported.decodeRows ExportedData.chunk461.1 ExportedData.chunk461.2 =
      (List.range 128).map (fun i => byteRow (59008 + i)) := by
  exact Codes.certificate 461 ExportedData.chunk461 (by decide)

theorem chunk462 :
    Exported.decodeRows ExportedData.chunk462.1 ExportedData.chunk462.2 =
      (List.range 128).map (fun i => byteRow (59136 + i)) := by
  exact Codes.certificate 462 ExportedData.chunk462 (by decide)

theorem chunk463 :
    Exported.decodeRows ExportedData.chunk463.1 ExportedData.chunk463.2 =
      (List.range 128).map (fun i => byteRow (59264 + i)) := by
  exact Codes.certificate 463 ExportedData.chunk463 (by decide)

theorem chunk464 :
    Exported.decodeRows ExportedData.chunk464.1 ExportedData.chunk464.2 =
      (List.range 128).map (fun i => byteRow (59392 + i)) := by
  exact Codes.certificate 464 ExportedData.chunk464 (by decide)

theorem chunk465 :
    Exported.decodeRows ExportedData.chunk465.1 ExportedData.chunk465.2 =
      (List.range 128).map (fun i => byteRow (59520 + i)) := by
  exact Codes.certificate 465 ExportedData.chunk465 (by decide)

theorem chunk466 :
    Exported.decodeRows ExportedData.chunk466.1 ExportedData.chunk466.2 =
      (List.range 128).map (fun i => byteRow (59648 + i)) := by
  exact Codes.certificate 466 ExportedData.chunk466 (by decide)

theorem chunk467 :
    Exported.decodeRows ExportedData.chunk467.1 ExportedData.chunk467.2 =
      (List.range 128).map (fun i => byteRow (59776 + i)) := by
  exact Codes.certificate 467 ExportedData.chunk467 (by decide)

theorem chunk468 :
    Exported.decodeRows ExportedData.chunk468.1 ExportedData.chunk468.2 =
      (List.range 128).map (fun i => byteRow (59904 + i)) := by
  exact Codes.certificate 468 ExportedData.chunk468 (by decide)

theorem chunk469 :
    Exported.decodeRows ExportedData.chunk469.1 ExportedData.chunk469.2 =
      (List.range 128).map (fun i => byteRow (60032 + i)) := by
  exact Codes.certificate 469 ExportedData.chunk469 (by decide)

theorem chunk470 :
    Exported.decodeRows ExportedData.chunk470.1 ExportedData.chunk470.2 =
      (List.range 128).map (fun i => byteRow (60160 + i)) := by
  exact Codes.certificate 470 ExportedData.chunk470 (by decide)

theorem chunk471 :
    Exported.decodeRows ExportedData.chunk471.1 ExportedData.chunk471.2 =
      (List.range 128).map (fun i => byteRow (60288 + i)) := by
  exact Codes.certificate 471 ExportedData.chunk471 (by decide)

theorem chunk472 :
    Exported.decodeRows ExportedData.chunk472.1 ExportedData.chunk472.2 =
      (List.range 128).map (fun i => byteRow (60416 + i)) := by
  exact Codes.certificate 472 ExportedData.chunk472 (by decide)

theorem chunk473 :
    Exported.decodeRows ExportedData.chunk473.1 ExportedData.chunk473.2 =
      (List.range 128).map (fun i => byteRow (60544 + i)) := by
  exact Codes.certificate 473 ExportedData.chunk473 (by decide)

theorem chunk474 :
    Exported.decodeRows ExportedData.chunk474.1 ExportedData.chunk474.2 =
      (List.range 128).map (fun i => byteRow (60672 + i)) := by
  exact Codes.certificate 474 ExportedData.chunk474 (by decide)

theorem chunk475 :
    Exported.decodeRows ExportedData.chunk475.1 ExportedData.chunk475.2 =
      (List.range 128).map (fun i => byteRow (60800 + i)) := by
  exact Codes.certificate 475 ExportedData.chunk475 (by decide)

theorem chunk476 :
    Exported.decodeRows ExportedData.chunk476.1 ExportedData.chunk476.2 =
      (List.range 128).map (fun i => byteRow (60928 + i)) := by
  exact Codes.certificate 476 ExportedData.chunk476 (by decide)

theorem chunk477 :
    Exported.decodeRows ExportedData.chunk477.1 ExportedData.chunk477.2 =
      (List.range 128).map (fun i => byteRow (61056 + i)) := by
  exact Codes.certificate 477 ExportedData.chunk477 (by decide)

theorem chunk478 :
    Exported.decodeRows ExportedData.chunk478.1 ExportedData.chunk478.2 =
      (List.range 128).map (fun i => byteRow (61184 + i)) := by
  exact Codes.certificate 478 ExportedData.chunk478 (by decide)

theorem chunk479 :
    Exported.decodeRows ExportedData.chunk479.1 ExportedData.chunk479.2 =
      (List.range 128).map (fun i => byteRow (61312 + i)) := by
  exact Codes.certificate 479 ExportedData.chunk479 (by decide)

theorem group23 (c : Fin 20) :
    let chunk := ExportedData.chunks[460 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(460 + c.val) + i)) := by
  fin_cases c
  · exact chunk460
  · exact chunk461
  · exact chunk462
  · exact chunk463
  · exact chunk464
  · exact chunk465
  · exact chunk466
  · exact chunk467
  · exact chunk468
  · exact chunk469
  · exact chunk470
  · exact chunk471
  · exact chunk472
  · exact chunk473
  · exact chunk474
  · exact chunk475
  · exact chunk476
  · exact chunk477
  · exact chunk478
  · exact chunk479

end CircuitCorrectness.ConcreteBytes
