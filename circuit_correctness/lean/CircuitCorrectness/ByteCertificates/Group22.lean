import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group18

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk440 :
    Exported.decodeRows ExportedData.chunk440.1 ExportedData.chunk440.2 =
      (List.range 128).map (fun i => byteRow (56320 + i)) := by
  exact Codes.certificate 440 ExportedData.chunk440 (by decide)

theorem chunk441 :
    Exported.decodeRows ExportedData.chunk441.1 ExportedData.chunk441.2 =
      (List.range 128).map (fun i => byteRow (56448 + i)) := by
  exact Codes.certificate 441 ExportedData.chunk441 (by decide)

theorem chunk442 :
    Exported.decodeRows ExportedData.chunk442.1 ExportedData.chunk442.2 =
      (List.range 128).map (fun i => byteRow (56576 + i)) := by
  exact Codes.certificate 442 ExportedData.chunk442 (by decide)

theorem chunk443 :
    Exported.decodeRows ExportedData.chunk443.1 ExportedData.chunk443.2 =
      (List.range 128).map (fun i => byteRow (56704 + i)) := by
  exact Codes.certificate 443 ExportedData.chunk443 (by decide)

theorem chunk444 :
    Exported.decodeRows ExportedData.chunk444.1 ExportedData.chunk444.2 =
      (List.range 128).map (fun i => byteRow (56832 + i)) := by
  exact Codes.certificate 444 ExportedData.chunk444 (by decide)

theorem chunk445 :
    Exported.decodeRows ExportedData.chunk445.1 ExportedData.chunk445.2 =
      (List.range 128).map (fun i => byteRow (56960 + i)) := by
  exact Codes.certificate 445 ExportedData.chunk445 (by decide)

theorem chunk446 :
    Exported.decodeRows ExportedData.chunk446.1 ExportedData.chunk446.2 =
      (List.range 128).map (fun i => byteRow (57088 + i)) := by
  exact Codes.certificate 446 ExportedData.chunk446 (by decide)

theorem chunk447 :
    Exported.decodeRows ExportedData.chunk447.1 ExportedData.chunk447.2 =
      (List.range 128).map (fun i => byteRow (57216 + i)) := by
  exact Codes.certificate 447 ExportedData.chunk447 (by decide)

theorem chunk448 :
    Exported.decodeRows ExportedData.chunk448.1 ExportedData.chunk448.2 =
      (List.range 128).map (fun i => byteRow (57344 + i)) := by
  exact Codes.certificate 448 ExportedData.chunk448 (by decide)

theorem chunk449 :
    Exported.decodeRows ExportedData.chunk449.1 ExportedData.chunk449.2 =
      (List.range 128).map (fun i => byteRow (57472 + i)) := by
  exact Codes.certificate 449 ExportedData.chunk449 (by decide)

theorem chunk450 :
    Exported.decodeRows ExportedData.chunk450.1 ExportedData.chunk450.2 =
      (List.range 128).map (fun i => byteRow (57600 + i)) := by
  exact Codes.certificate 450 ExportedData.chunk450 (by decide)

theorem chunk451 :
    Exported.decodeRows ExportedData.chunk451.1 ExportedData.chunk451.2 =
      (List.range 128).map (fun i => byteRow (57728 + i)) := by
  exact Codes.certificate 451 ExportedData.chunk451 (by decide)

theorem chunk452 :
    Exported.decodeRows ExportedData.chunk452.1 ExportedData.chunk452.2 =
      (List.range 128).map (fun i => byteRow (57856 + i)) := by
  exact Codes.certificate 452 ExportedData.chunk452 (by decide)

theorem chunk453 :
    Exported.decodeRows ExportedData.chunk453.1 ExportedData.chunk453.2 =
      (List.range 128).map (fun i => byteRow (57984 + i)) := by
  exact Codes.certificate 453 ExportedData.chunk453 (by decide)

theorem chunk454 :
    Exported.decodeRows ExportedData.chunk454.1 ExportedData.chunk454.2 =
      (List.range 128).map (fun i => byteRow (58112 + i)) := by
  exact Codes.certificate 454 ExportedData.chunk454 (by decide)

theorem chunk455 :
    Exported.decodeRows ExportedData.chunk455.1 ExportedData.chunk455.2 =
      (List.range 128).map (fun i => byteRow (58240 + i)) := by
  exact Codes.certificate 455 ExportedData.chunk455 (by decide)

theorem chunk456 :
    Exported.decodeRows ExportedData.chunk456.1 ExportedData.chunk456.2 =
      (List.range 128).map (fun i => byteRow (58368 + i)) := by
  exact Codes.certificate 456 ExportedData.chunk456 (by decide)

theorem chunk457 :
    Exported.decodeRows ExportedData.chunk457.1 ExportedData.chunk457.2 =
      (List.range 128).map (fun i => byteRow (58496 + i)) := by
  exact Codes.certificate 457 ExportedData.chunk457 (by decide)

theorem chunk458 :
    Exported.decodeRows ExportedData.chunk458.1 ExportedData.chunk458.2 =
      (List.range 128).map (fun i => byteRow (58624 + i)) := by
  exact Codes.certificate 458 ExportedData.chunk458 (by decide)

theorem chunk459 :
    Exported.decodeRows ExportedData.chunk459.1 ExportedData.chunk459.2 =
      (List.range 128).map (fun i => byteRow (58752 + i)) := by
  exact Codes.certificate 459 ExportedData.chunk459 (by decide)

theorem group22 (c : Fin 20) :
    let chunk := ExportedData.chunks[440 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(440 + c.val) + i)) := by
  fin_cases c
  · exact chunk440
  · exact chunk441
  · exact chunk442
  · exact chunk443
  · exact chunk444
  · exact chunk445
  · exact chunk446
  · exact chunk447
  · exact chunk448
  · exact chunk449
  · exact chunk450
  · exact chunk451
  · exact chunk452
  · exact chunk453
  · exact chunk454
  · exact chunk455
  · exact chunk456
  · exact chunk457
  · exact chunk458
  · exact chunk459

end CircuitCorrectness.ConcreteBytes
