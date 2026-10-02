import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group21

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk500 :
    Exported.decodeRows ExportedData.chunk500.1 ExportedData.chunk500.2 =
      (List.range 128).map (fun i => byteRow (64000 + i)) := by
  exact Codes.certificate 500 ExportedData.chunk500 (by decide)

theorem chunk501 :
    Exported.decodeRows ExportedData.chunk501.1 ExportedData.chunk501.2 =
      (List.range 128).map (fun i => byteRow (64128 + i)) := by
  exact Codes.certificate 501 ExportedData.chunk501 (by decide)

theorem chunk502 :
    Exported.decodeRows ExportedData.chunk502.1 ExportedData.chunk502.2 =
      (List.range 128).map (fun i => byteRow (64256 + i)) := by
  exact Codes.certificate 502 ExportedData.chunk502 (by decide)

theorem chunk503 :
    Exported.decodeRows ExportedData.chunk503.1 ExportedData.chunk503.2 =
      (List.range 128).map (fun i => byteRow (64384 + i)) := by
  exact Codes.certificate 503 ExportedData.chunk503 (by decide)

theorem chunk504 :
    Exported.decodeRows ExportedData.chunk504.1 ExportedData.chunk504.2 =
      (List.range 128).map (fun i => byteRow (64512 + i)) := by
  exact Codes.certificate 504 ExportedData.chunk504 (by decide)

theorem chunk505 :
    Exported.decodeRows ExportedData.chunk505.1 ExportedData.chunk505.2 =
      (List.range 128).map (fun i => byteRow (64640 + i)) := by
  exact Codes.certificate 505 ExportedData.chunk505 (by decide)

theorem chunk506 :
    Exported.decodeRows ExportedData.chunk506.1 ExportedData.chunk506.2 =
      (List.range 128).map (fun i => byteRow (64768 + i)) := by
  exact Codes.certificate 506 ExportedData.chunk506 (by decide)

theorem chunk507 :
    Exported.decodeRows ExportedData.chunk507.1 ExportedData.chunk507.2 =
      (List.range 128).map (fun i => byteRow (64896 + i)) := by
  exact Codes.certificate 507 ExportedData.chunk507 (by decide)

theorem chunk508 :
    Exported.decodeRows ExportedData.chunk508.1 ExportedData.chunk508.2 =
      (List.range 128).map (fun i => byteRow (65024 + i)) := by
  exact Codes.certificate 508 ExportedData.chunk508 (by decide)

theorem chunk509 :
    Exported.decodeRows ExportedData.chunk509.1 ExportedData.chunk509.2 =
      (List.range 128).map (fun i => byteRow (65152 + i)) := by
  exact Codes.certificate 509 ExportedData.chunk509 (by decide)

theorem chunk510 :
    Exported.decodeRows ExportedData.chunk510.1 ExportedData.chunk510.2 =
      (List.range 128).map (fun i => byteRow (65280 + i)) := by
  exact Codes.certificate 510 ExportedData.chunk510 (by decide)

theorem chunk511 :
    Exported.decodeRows ExportedData.chunk511.1 ExportedData.chunk511.2 =
      (List.range 128).map (fun i => byteRow (65408 + i)) := by
  exact Codes.certificate 511 ExportedData.chunk511 (by decide)

theorem chunk512 :
    Exported.decodeRows ExportedData.chunk512.1 ExportedData.chunk512.2 =
      (List.range 128).map (fun i => byteRow (65536 + i)) := by
  exact Codes.certificate 512 ExportedData.chunk512 (by decide)

theorem chunk513 :
    Exported.decodeRows ExportedData.chunk513.1 ExportedData.chunk513.2 =
      (List.range 128).map (fun i => byteRow (65664 + i)) := by
  exact Codes.certificate 513 ExportedData.chunk513 (by decide)

theorem chunk514 :
    Exported.decodeRows ExportedData.chunk514.1 ExportedData.chunk514.2 =
      (List.range 128).map (fun i => byteRow (65792 + i)) := by
  exact Codes.certificate 514 ExportedData.chunk514 (by decide)

theorem chunk515 :
    Exported.decodeRows ExportedData.chunk515.1 ExportedData.chunk515.2 =
      (List.range 128).map (fun i => byteRow (65920 + i)) := by
  exact Codes.certificate 515 ExportedData.chunk515 (by decide)

theorem chunk516 :
    Exported.decodeRows ExportedData.chunk516.1 ExportedData.chunk516.2 =
      (List.range 128).map (fun i => byteRow (66048 + i)) := by
  exact Codes.certificate 516 ExportedData.chunk516 (by decide)

theorem chunk517 :
    Exported.decodeRows ExportedData.chunk517.1 ExportedData.chunk517.2 =
      (List.range 128).map (fun i => byteRow (66176 + i)) := by
  exact Codes.certificate 517 ExportedData.chunk517 (by decide)

theorem chunk518 :
    Exported.decodeRows ExportedData.chunk518.1 ExportedData.chunk518.2 =
      (List.range 128).map (fun i => byteRow (66304 + i)) := by
  exact Codes.certificate 518 ExportedData.chunk518 (by decide)

theorem chunk519 :
    Exported.decodeRows ExportedData.chunk519.1 ExportedData.chunk519.2 =
      (List.range 128).map (fun i => byteRow (66432 + i)) := by
  exact Codes.certificate 519 ExportedData.chunk519 (by decide)

theorem group25 (c : Fin 20) :
    let chunk := ExportedData.chunks[500 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(500 + c.val) + i)) := by
  fin_cases c
  · exact chunk500
  · exact chunk501
  · exact chunk502
  · exact chunk503
  · exact chunk504
  · exact chunk505
  · exact chunk506
  · exact chunk507
  · exact chunk508
  · exact chunk509
  · exact chunk510
  · exact chunk511
  · exact chunk512
  · exact chunk513
  · exact chunk514
  · exact chunk515
  · exact chunk516
  · exact chunk517
  · exact chunk518
  · exact chunk519

end CircuitCorrectness.ConcreteBytes
