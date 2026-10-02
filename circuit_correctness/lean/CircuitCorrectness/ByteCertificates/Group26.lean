import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group22

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk520 :
    Exported.decodeRows ExportedData.chunk520.1 ExportedData.chunk520.2 =
      (List.range 128).map (fun i => byteRow (66560 + i)) := by
  exact Codes.certificate 520 ExportedData.chunk520 (by decide)

theorem chunk521 :
    Exported.decodeRows ExportedData.chunk521.1 ExportedData.chunk521.2 =
      (List.range 128).map (fun i => byteRow (66688 + i)) := by
  exact Codes.certificate 521 ExportedData.chunk521 (by decide)

theorem chunk522 :
    Exported.decodeRows ExportedData.chunk522.1 ExportedData.chunk522.2 =
      (List.range 128).map (fun i => byteRow (66816 + i)) := by
  exact Codes.certificate 522 ExportedData.chunk522 (by decide)

theorem chunk523 :
    Exported.decodeRows ExportedData.chunk523.1 ExportedData.chunk523.2 =
      (List.range 128).map (fun i => byteRow (66944 + i)) := by
  exact Codes.certificate 523 ExportedData.chunk523 (by decide)

theorem chunk524 :
    Exported.decodeRows ExportedData.chunk524.1 ExportedData.chunk524.2 =
      (List.range 128).map (fun i => byteRow (67072 + i)) := by
  exact Codes.certificate 524 ExportedData.chunk524 (by decide)

theorem chunk525 :
    Exported.decodeRows ExportedData.chunk525.1 ExportedData.chunk525.2 =
      (List.range 128).map (fun i => byteRow (67200 + i)) := by
  exact Codes.certificate 525 ExportedData.chunk525 (by decide)

theorem chunk526 :
    Exported.decodeRows ExportedData.chunk526.1 ExportedData.chunk526.2 =
      (List.range 128).map (fun i => byteRow (67328 + i)) := by
  exact Codes.certificate 526 ExportedData.chunk526 (by decide)

theorem chunk527 :
    Exported.decodeRows ExportedData.chunk527.1 ExportedData.chunk527.2 =
      (List.range 128).map (fun i => byteRow (67456 + i)) := by
  exact Codes.certificate 527 ExportedData.chunk527 (by decide)

theorem chunk528 :
    Exported.decodeRows ExportedData.chunk528.1 ExportedData.chunk528.2 =
      (List.range 128).map (fun i => byteRow (67584 + i)) := by
  exact Codes.certificate 528 ExportedData.chunk528 (by decide)

theorem chunk529 :
    Exported.decodeRows ExportedData.chunk529.1 ExportedData.chunk529.2 =
      (List.range 128).map (fun i => byteRow (67712 + i)) := by
  exact Codes.certificate 529 ExportedData.chunk529 (by decide)

theorem chunk530 :
    Exported.decodeRows ExportedData.chunk530.1 ExportedData.chunk530.2 =
      (List.range 128).map (fun i => byteRow (67840 + i)) := by
  exact Codes.certificate 530 ExportedData.chunk530 (by decide)

theorem chunk531 :
    Exported.decodeRows ExportedData.chunk531.1 ExportedData.chunk531.2 =
      (List.range 128).map (fun i => byteRow (67968 + i)) := by
  exact Codes.certificate 531 ExportedData.chunk531 (by decide)

theorem chunk532 :
    Exported.decodeRows ExportedData.chunk532.1 ExportedData.chunk532.2 =
      (List.range 128).map (fun i => byteRow (68096 + i)) := by
  exact Codes.certificate 532 ExportedData.chunk532 (by decide)

theorem chunk533 :
    Exported.decodeRows ExportedData.chunk533.1 ExportedData.chunk533.2 =
      (List.range 128).map (fun i => byteRow (68224 + i)) := by
  exact Codes.certificate 533 ExportedData.chunk533 (by decide)

theorem chunk534 :
    Exported.decodeRows ExportedData.chunk534.1 ExportedData.chunk534.2 =
      (List.range 128).map (fun i => byteRow (68352 + i)) := by
  exact Codes.certificate 534 ExportedData.chunk534 (by decide)

theorem chunk535 :
    Exported.decodeRows ExportedData.chunk535.1 ExportedData.chunk535.2 =
      (List.range 128).map (fun i => byteRow (68480 + i)) := by
  exact Codes.certificate 535 ExportedData.chunk535 (by decide)

theorem chunk536 :
    Exported.decodeRows ExportedData.chunk536.1 ExportedData.chunk536.2 =
      (List.range 128).map (fun i => byteRow (68608 + i)) := by
  exact Codes.certificate 536 ExportedData.chunk536 (by decide)

theorem chunk537 :
    Exported.decodeRows ExportedData.chunk537.1 ExportedData.chunk537.2 =
      (List.range 128).map (fun i => byteRow (68736 + i)) := by
  exact Codes.certificate 537 ExportedData.chunk537 (by decide)

theorem chunk538 :
    Exported.decodeRows ExportedData.chunk538.1 ExportedData.chunk538.2 =
      (List.range 128).map (fun i => byteRow (68864 + i)) := by
  exact Codes.certificate 538 ExportedData.chunk538 (by decide)

theorem chunk539 :
    Exported.decodeRows ExportedData.chunk539.1 ExportedData.chunk539.2 =
      (List.range 128).map (fun i => byteRow (68992 + i)) := by
  exact Codes.certificate 539 ExportedData.chunk539 (by decide)

theorem group26 (c : Fin 20) :
    let chunk := ExportedData.chunks[520 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(520 + c.val) + i)) := by
  fin_cases c
  · exact chunk520
  · exact chunk521
  · exact chunk522
  · exact chunk523
  · exact chunk524
  · exact chunk525
  · exact chunk526
  · exact chunk527
  · exact chunk528
  · exact chunk529
  · exact chunk530
  · exact chunk531
  · exact chunk532
  · exact chunk533
  · exact chunk534
  · exact chunk535
  · exact chunk536
  · exact chunk537
  · exact chunk538
  · exact chunk539

end CircuitCorrectness.ConcreteBytes
