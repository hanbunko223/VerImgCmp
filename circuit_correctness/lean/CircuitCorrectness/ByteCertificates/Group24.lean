import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group20

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk480 :
    Exported.decodeRows ExportedData.chunk480.1 ExportedData.chunk480.2 =
      (List.range 128).map (fun i => byteRow (61440 + i)) := by
  exact Codes.certificate 480 ExportedData.chunk480 (by decide)

theorem chunk481 :
    Exported.decodeRows ExportedData.chunk481.1 ExportedData.chunk481.2 =
      (List.range 128).map (fun i => byteRow (61568 + i)) := by
  exact Codes.certificate 481 ExportedData.chunk481 (by decide)

theorem chunk482 :
    Exported.decodeRows ExportedData.chunk482.1 ExportedData.chunk482.2 =
      (List.range 128).map (fun i => byteRow (61696 + i)) := by
  exact Codes.certificate 482 ExportedData.chunk482 (by decide)

theorem chunk483 :
    Exported.decodeRows ExportedData.chunk483.1 ExportedData.chunk483.2 =
      (List.range 128).map (fun i => byteRow (61824 + i)) := by
  exact Codes.certificate 483 ExportedData.chunk483 (by decide)

theorem chunk484 :
    Exported.decodeRows ExportedData.chunk484.1 ExportedData.chunk484.2 =
      (List.range 128).map (fun i => byteRow (61952 + i)) := by
  exact Codes.certificate 484 ExportedData.chunk484 (by decide)

theorem chunk485 :
    Exported.decodeRows ExportedData.chunk485.1 ExportedData.chunk485.2 =
      (List.range 128).map (fun i => byteRow (62080 + i)) := by
  exact Codes.certificate 485 ExportedData.chunk485 (by decide)

theorem chunk486 :
    Exported.decodeRows ExportedData.chunk486.1 ExportedData.chunk486.2 =
      (List.range 128).map (fun i => byteRow (62208 + i)) := by
  exact Codes.certificate 486 ExportedData.chunk486 (by decide)

theorem chunk487 :
    Exported.decodeRows ExportedData.chunk487.1 ExportedData.chunk487.2 =
      (List.range 128).map (fun i => byteRow (62336 + i)) := by
  exact Codes.certificate 487 ExportedData.chunk487 (by decide)

theorem chunk488 :
    Exported.decodeRows ExportedData.chunk488.1 ExportedData.chunk488.2 =
      (List.range 128).map (fun i => byteRow (62464 + i)) := by
  exact Codes.certificate 488 ExportedData.chunk488 (by decide)

theorem chunk489 :
    Exported.decodeRows ExportedData.chunk489.1 ExportedData.chunk489.2 =
      (List.range 128).map (fun i => byteRow (62592 + i)) := by
  exact Codes.certificate 489 ExportedData.chunk489 (by decide)

theorem chunk490 :
    Exported.decodeRows ExportedData.chunk490.1 ExportedData.chunk490.2 =
      (List.range 128).map (fun i => byteRow (62720 + i)) := by
  exact Codes.certificate 490 ExportedData.chunk490 (by decide)

theorem chunk491 :
    Exported.decodeRows ExportedData.chunk491.1 ExportedData.chunk491.2 =
      (List.range 128).map (fun i => byteRow (62848 + i)) := by
  exact Codes.certificate 491 ExportedData.chunk491 (by decide)

theorem chunk492 :
    Exported.decodeRows ExportedData.chunk492.1 ExportedData.chunk492.2 =
      (List.range 128).map (fun i => byteRow (62976 + i)) := by
  exact Codes.certificate 492 ExportedData.chunk492 (by decide)

theorem chunk493 :
    Exported.decodeRows ExportedData.chunk493.1 ExportedData.chunk493.2 =
      (List.range 128).map (fun i => byteRow (63104 + i)) := by
  exact Codes.certificate 493 ExportedData.chunk493 (by decide)

theorem chunk494 :
    Exported.decodeRows ExportedData.chunk494.1 ExportedData.chunk494.2 =
      (List.range 128).map (fun i => byteRow (63232 + i)) := by
  exact Codes.certificate 494 ExportedData.chunk494 (by decide)

theorem chunk495 :
    Exported.decodeRows ExportedData.chunk495.1 ExportedData.chunk495.2 =
      (List.range 128).map (fun i => byteRow (63360 + i)) := by
  exact Codes.certificate 495 ExportedData.chunk495 (by decide)

theorem chunk496 :
    Exported.decodeRows ExportedData.chunk496.1 ExportedData.chunk496.2 =
      (List.range 128).map (fun i => byteRow (63488 + i)) := by
  exact Codes.certificate 496 ExportedData.chunk496 (by decide)

theorem chunk497 :
    Exported.decodeRows ExportedData.chunk497.1 ExportedData.chunk497.2 =
      (List.range 128).map (fun i => byteRow (63616 + i)) := by
  exact Codes.certificate 497 ExportedData.chunk497 (by decide)

theorem chunk498 :
    Exported.decodeRows ExportedData.chunk498.1 ExportedData.chunk498.2 =
      (List.range 128).map (fun i => byteRow (63744 + i)) := by
  exact Codes.certificate 498 ExportedData.chunk498 (by decide)

theorem chunk499 :
    Exported.decodeRows ExportedData.chunk499.1 ExportedData.chunk499.2 =
      (List.range 128).map (fun i => byteRow (63872 + i)) := by
  exact Codes.certificate 499 ExportedData.chunk499 (by decide)

theorem group24 (c : Fin 20) :
    let chunk := ExportedData.chunks[480 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(480 + c.val) + i)) := by
  fin_cases c
  · exact chunk480
  · exact chunk481
  · exact chunk482
  · exact chunk483
  · exact chunk484
  · exact chunk485
  · exact chunk486
  · exact chunk487
  · exact chunk488
  · exact chunk489
  · exact chunk490
  · exact chunk491
  · exact chunk492
  · exact chunk493
  · exact chunk494
  · exact chunk495
  · exact chunk496
  · exact chunk497
  · exact chunk498
  · exact chunk499

end CircuitCorrectness.ConcreteBytes
