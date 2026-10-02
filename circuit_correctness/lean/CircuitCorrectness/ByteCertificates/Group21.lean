import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group17

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk420 :
    Exported.decodeRows ExportedData.chunk420.1 ExportedData.chunk420.2 =
      (List.range 128).map (fun i => byteRow (53760 + i)) := by
  exact Codes.certificate 420 ExportedData.chunk420 (by decide)

theorem chunk421 :
    Exported.decodeRows ExportedData.chunk421.1 ExportedData.chunk421.2 =
      (List.range 128).map (fun i => byteRow (53888 + i)) := by
  exact Codes.certificate 421 ExportedData.chunk421 (by decide)

theorem chunk422 :
    Exported.decodeRows ExportedData.chunk422.1 ExportedData.chunk422.2 =
      (List.range 128).map (fun i => byteRow (54016 + i)) := by
  exact Codes.certificate 422 ExportedData.chunk422 (by decide)

theorem chunk423 :
    Exported.decodeRows ExportedData.chunk423.1 ExportedData.chunk423.2 =
      (List.range 128).map (fun i => byteRow (54144 + i)) := by
  exact Codes.certificate 423 ExportedData.chunk423 (by decide)

theorem chunk424 :
    Exported.decodeRows ExportedData.chunk424.1 ExportedData.chunk424.2 =
      (List.range 128).map (fun i => byteRow (54272 + i)) := by
  exact Codes.certificate 424 ExportedData.chunk424 (by decide)

theorem chunk425 :
    Exported.decodeRows ExportedData.chunk425.1 ExportedData.chunk425.2 =
      (List.range 128).map (fun i => byteRow (54400 + i)) := by
  exact Codes.certificate 425 ExportedData.chunk425 (by decide)

theorem chunk426 :
    Exported.decodeRows ExportedData.chunk426.1 ExportedData.chunk426.2 =
      (List.range 128).map (fun i => byteRow (54528 + i)) := by
  exact Codes.certificate 426 ExportedData.chunk426 (by decide)

theorem chunk427 :
    Exported.decodeRows ExportedData.chunk427.1 ExportedData.chunk427.2 =
      (List.range 128).map (fun i => byteRow (54656 + i)) := by
  exact Codes.certificate 427 ExportedData.chunk427 (by decide)

theorem chunk428 :
    Exported.decodeRows ExportedData.chunk428.1 ExportedData.chunk428.2 =
      (List.range 128).map (fun i => byteRow (54784 + i)) := by
  exact Codes.certificate 428 ExportedData.chunk428 (by decide)

theorem chunk429 :
    Exported.decodeRows ExportedData.chunk429.1 ExportedData.chunk429.2 =
      (List.range 128).map (fun i => byteRow (54912 + i)) := by
  exact Codes.certificate 429 ExportedData.chunk429 (by decide)

theorem chunk430 :
    Exported.decodeRows ExportedData.chunk430.1 ExportedData.chunk430.2 =
      (List.range 128).map (fun i => byteRow (55040 + i)) := by
  exact Codes.certificate 430 ExportedData.chunk430 (by decide)

theorem chunk431 :
    Exported.decodeRows ExportedData.chunk431.1 ExportedData.chunk431.2 =
      (List.range 128).map (fun i => byteRow (55168 + i)) := by
  exact Codes.certificate 431 ExportedData.chunk431 (by decide)

theorem chunk432 :
    Exported.decodeRows ExportedData.chunk432.1 ExportedData.chunk432.2 =
      (List.range 128).map (fun i => byteRow (55296 + i)) := by
  exact Codes.certificate 432 ExportedData.chunk432 (by decide)

theorem chunk433 :
    Exported.decodeRows ExportedData.chunk433.1 ExportedData.chunk433.2 =
      (List.range 128).map (fun i => byteRow (55424 + i)) := by
  exact Codes.certificate 433 ExportedData.chunk433 (by decide)

theorem chunk434 :
    Exported.decodeRows ExportedData.chunk434.1 ExportedData.chunk434.2 =
      (List.range 128).map (fun i => byteRow (55552 + i)) := by
  exact Codes.certificate 434 ExportedData.chunk434 (by decide)

theorem chunk435 :
    Exported.decodeRows ExportedData.chunk435.1 ExportedData.chunk435.2 =
      (List.range 128).map (fun i => byteRow (55680 + i)) := by
  exact Codes.certificate 435 ExportedData.chunk435 (by decide)

theorem chunk436 :
    Exported.decodeRows ExportedData.chunk436.1 ExportedData.chunk436.2 =
      (List.range 128).map (fun i => byteRow (55808 + i)) := by
  exact Codes.certificate 436 ExportedData.chunk436 (by decide)

theorem chunk437 :
    Exported.decodeRows ExportedData.chunk437.1 ExportedData.chunk437.2 =
      (List.range 128).map (fun i => byteRow (55936 + i)) := by
  exact Codes.certificate 437 ExportedData.chunk437 (by decide)

theorem chunk438 :
    Exported.decodeRows ExportedData.chunk438.1 ExportedData.chunk438.2 =
      (List.range 128).map (fun i => byteRow (56064 + i)) := by
  exact Codes.certificate 438 ExportedData.chunk438 (by decide)

theorem chunk439 :
    Exported.decodeRows ExportedData.chunk439.1 ExportedData.chunk439.2 =
      (List.range 128).map (fun i => byteRow (56192 + i)) := by
  exact Codes.certificate 439 ExportedData.chunk439 (by decide)

theorem group21 (c : Fin 20) :
    let chunk := ExportedData.chunks[420 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(420 + c.val) + i)) := by
  fin_cases c
  · exact chunk420
  · exact chunk421
  · exact chunk422
  · exact chunk423
  · exact chunk424
  · exact chunk425
  · exact chunk426
  · exact chunk427
  · exact chunk428
  · exact chunk429
  · exact chunk430
  · exact chunk431
  · exact chunk432
  · exact chunk433
  · exact chunk434
  · exact chunk435
  · exact chunk436
  · exact chunk437
  · exact chunk438
  · exact chunk439

end CircuitCorrectness.ConcreteBytes
