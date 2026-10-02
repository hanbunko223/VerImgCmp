import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group14

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk360 :
    Exported.decodeRows ExportedData.chunk360.1 ExportedData.chunk360.2 =
      (List.range 128).map (fun i => byteRow (46080 + i)) := by
  exact Codes.certificate 360 ExportedData.chunk360 (by decide)

theorem chunk361 :
    Exported.decodeRows ExportedData.chunk361.1 ExportedData.chunk361.2 =
      (List.range 128).map (fun i => byteRow (46208 + i)) := by
  exact Codes.certificate 361 ExportedData.chunk361 (by decide)

theorem chunk362 :
    Exported.decodeRows ExportedData.chunk362.1 ExportedData.chunk362.2 =
      (List.range 128).map (fun i => byteRow (46336 + i)) := by
  exact Codes.certificate 362 ExportedData.chunk362 (by decide)

theorem chunk363 :
    Exported.decodeRows ExportedData.chunk363.1 ExportedData.chunk363.2 =
      (List.range 128).map (fun i => byteRow (46464 + i)) := by
  exact Codes.certificate 363 ExportedData.chunk363 (by decide)

theorem chunk364 :
    Exported.decodeRows ExportedData.chunk364.1 ExportedData.chunk364.2 =
      (List.range 128).map (fun i => byteRow (46592 + i)) := by
  exact Codes.certificate 364 ExportedData.chunk364 (by decide)

theorem chunk365 :
    Exported.decodeRows ExportedData.chunk365.1 ExportedData.chunk365.2 =
      (List.range 128).map (fun i => byteRow (46720 + i)) := by
  exact Codes.certificate 365 ExportedData.chunk365 (by decide)

theorem chunk366 :
    Exported.decodeRows ExportedData.chunk366.1 ExportedData.chunk366.2 =
      (List.range 128).map (fun i => byteRow (46848 + i)) := by
  exact Codes.certificate 366 ExportedData.chunk366 (by decide)

theorem chunk367 :
    Exported.decodeRows ExportedData.chunk367.1 ExportedData.chunk367.2 =
      (List.range 128).map (fun i => byteRow (46976 + i)) := by
  exact Codes.certificate 367 ExportedData.chunk367 (by decide)

theorem chunk368 :
    Exported.decodeRows ExportedData.chunk368.1 ExportedData.chunk368.2 =
      (List.range 128).map (fun i => byteRow (47104 + i)) := by
  exact Codes.certificate 368 ExportedData.chunk368 (by decide)

theorem chunk369 :
    Exported.decodeRows ExportedData.chunk369.1 ExportedData.chunk369.2 =
      (List.range 128).map (fun i => byteRow (47232 + i)) := by
  exact Codes.certificate 369 ExportedData.chunk369 (by decide)

theorem chunk370 :
    Exported.decodeRows ExportedData.chunk370.1 ExportedData.chunk370.2 =
      (List.range 128).map (fun i => byteRow (47360 + i)) := by
  exact Codes.certificate 370 ExportedData.chunk370 (by decide)

theorem chunk371 :
    Exported.decodeRows ExportedData.chunk371.1 ExportedData.chunk371.2 =
      (List.range 128).map (fun i => byteRow (47488 + i)) := by
  exact Codes.certificate 371 ExportedData.chunk371 (by decide)

theorem chunk372 :
    Exported.decodeRows ExportedData.chunk372.1 ExportedData.chunk372.2 =
      (List.range 128).map (fun i => byteRow (47616 + i)) := by
  exact Codes.certificate 372 ExportedData.chunk372 (by decide)

theorem chunk373 :
    Exported.decodeRows ExportedData.chunk373.1 ExportedData.chunk373.2 =
      (List.range 128).map (fun i => byteRow (47744 + i)) := by
  exact Codes.certificate 373 ExportedData.chunk373 (by decide)

theorem chunk374 :
    Exported.decodeRows ExportedData.chunk374.1 ExportedData.chunk374.2 =
      (List.range 128).map (fun i => byteRow (47872 + i)) := by
  exact Codes.certificate 374 ExportedData.chunk374 (by decide)

theorem chunk375 :
    Exported.decodeRows ExportedData.chunk375.1 ExportedData.chunk375.2 =
      (List.range 128).map (fun i => byteRow (48000 + i)) := by
  exact Codes.certificate 375 ExportedData.chunk375 (by decide)

theorem chunk376 :
    Exported.decodeRows ExportedData.chunk376.1 ExportedData.chunk376.2 =
      (List.range 128).map (fun i => byteRow (48128 + i)) := by
  exact Codes.certificate 376 ExportedData.chunk376 (by decide)

theorem chunk377 :
    Exported.decodeRows ExportedData.chunk377.1 ExportedData.chunk377.2 =
      (List.range 128).map (fun i => byteRow (48256 + i)) := by
  exact Codes.certificate 377 ExportedData.chunk377 (by decide)

theorem chunk378 :
    Exported.decodeRows ExportedData.chunk378.1 ExportedData.chunk378.2 =
      (List.range 128).map (fun i => byteRow (48384 + i)) := by
  exact Codes.certificate 378 ExportedData.chunk378 (by decide)

theorem chunk379 :
    Exported.decodeRows ExportedData.chunk379.1 ExportedData.chunk379.2 =
      (List.range 128).map (fun i => byteRow (48512 + i)) := by
  exact Codes.certificate 379 ExportedData.chunk379 (by decide)

theorem group18 (c : Fin 20) :
    let chunk := ExportedData.chunks[360 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(360 + c.val) + i)) := by
  fin_cases c
  · exact chunk360
  · exact chunk361
  · exact chunk362
  · exact chunk363
  · exact chunk364
  · exact chunk365
  · exact chunk366
  · exact chunk367
  · exact chunk368
  · exact chunk369
  · exact chunk370
  · exact chunk371
  · exact chunk372
  · exact chunk373
  · exact chunk374
  · exact chunk375
  · exact chunk376
  · exact chunk377
  · exact chunk378
  · exact chunk379

end CircuitCorrectness.ConcreteBytes
