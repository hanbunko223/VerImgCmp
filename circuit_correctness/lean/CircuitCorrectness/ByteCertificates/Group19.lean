import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group15

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk380 :
    Exported.decodeRows ExportedData.chunk380.1 ExportedData.chunk380.2 =
      (List.range 128).map (fun i => byteRow (48640 + i)) := by
  exact Codes.certificate 380 ExportedData.chunk380 (by decide)

theorem chunk381 :
    Exported.decodeRows ExportedData.chunk381.1 ExportedData.chunk381.2 =
      (List.range 128).map (fun i => byteRow (48768 + i)) := by
  exact Codes.certificate 381 ExportedData.chunk381 (by decide)

theorem chunk382 :
    Exported.decodeRows ExportedData.chunk382.1 ExportedData.chunk382.2 =
      (List.range 128).map (fun i => byteRow (48896 + i)) := by
  exact Codes.certificate 382 ExportedData.chunk382 (by decide)

theorem chunk383 :
    Exported.decodeRows ExportedData.chunk383.1 ExportedData.chunk383.2 =
      (List.range 128).map (fun i => byteRow (49024 + i)) := by
  exact Codes.certificate 383 ExportedData.chunk383 (by decide)

theorem chunk384 :
    Exported.decodeRows ExportedData.chunk384.1 ExportedData.chunk384.2 =
      (List.range 128).map (fun i => byteRow (49152 + i)) := by
  exact Codes.certificate 384 ExportedData.chunk384 (by decide)

theorem chunk385 :
    Exported.decodeRows ExportedData.chunk385.1 ExportedData.chunk385.2 =
      (List.range 128).map (fun i => byteRow (49280 + i)) := by
  exact Codes.certificate 385 ExportedData.chunk385 (by decide)

theorem chunk386 :
    Exported.decodeRows ExportedData.chunk386.1 ExportedData.chunk386.2 =
      (List.range 128).map (fun i => byteRow (49408 + i)) := by
  exact Codes.certificate 386 ExportedData.chunk386 (by decide)

theorem chunk387 :
    Exported.decodeRows ExportedData.chunk387.1 ExportedData.chunk387.2 =
      (List.range 128).map (fun i => byteRow (49536 + i)) := by
  exact Codes.certificate 387 ExportedData.chunk387 (by decide)

theorem chunk388 :
    Exported.decodeRows ExportedData.chunk388.1 ExportedData.chunk388.2 =
      (List.range 128).map (fun i => byteRow (49664 + i)) := by
  exact Codes.certificate 388 ExportedData.chunk388 (by decide)

theorem chunk389 :
    Exported.decodeRows ExportedData.chunk389.1 ExportedData.chunk389.2 =
      (List.range 128).map (fun i => byteRow (49792 + i)) := by
  exact Codes.certificate 389 ExportedData.chunk389 (by decide)

theorem chunk390 :
    Exported.decodeRows ExportedData.chunk390.1 ExportedData.chunk390.2 =
      (List.range 128).map (fun i => byteRow (49920 + i)) := by
  exact Codes.certificate 390 ExportedData.chunk390 (by decide)

theorem chunk391 :
    Exported.decodeRows ExportedData.chunk391.1 ExportedData.chunk391.2 =
      (List.range 128).map (fun i => byteRow (50048 + i)) := by
  exact Codes.certificate 391 ExportedData.chunk391 (by decide)

theorem chunk392 :
    Exported.decodeRows ExportedData.chunk392.1 ExportedData.chunk392.2 =
      (List.range 128).map (fun i => byteRow (50176 + i)) := by
  exact Codes.certificate 392 ExportedData.chunk392 (by decide)

theorem chunk393 :
    Exported.decodeRows ExportedData.chunk393.1 ExportedData.chunk393.2 =
      (List.range 128).map (fun i => byteRow (50304 + i)) := by
  exact Codes.certificate 393 ExportedData.chunk393 (by decide)

theorem chunk394 :
    Exported.decodeRows ExportedData.chunk394.1 ExportedData.chunk394.2 =
      (List.range 128).map (fun i => byteRow (50432 + i)) := by
  exact Codes.certificate 394 ExportedData.chunk394 (by decide)

theorem chunk395 :
    Exported.decodeRows ExportedData.chunk395.1 ExportedData.chunk395.2 =
      (List.range 128).map (fun i => byteRow (50560 + i)) := by
  exact Codes.certificate 395 ExportedData.chunk395 (by decide)

theorem chunk396 :
    Exported.decodeRows ExportedData.chunk396.1 ExportedData.chunk396.2 =
      (List.range 128).map (fun i => byteRow (50688 + i)) := by
  exact Codes.certificate 396 ExportedData.chunk396 (by decide)

theorem chunk397 :
    Exported.decodeRows ExportedData.chunk397.1 ExportedData.chunk397.2 =
      (List.range 128).map (fun i => byteRow (50816 + i)) := by
  exact Codes.certificate 397 ExportedData.chunk397 (by decide)

theorem chunk398 :
    Exported.decodeRows ExportedData.chunk398.1 ExportedData.chunk398.2 =
      (List.range 128).map (fun i => byteRow (50944 + i)) := by
  exact Codes.certificate 398 ExportedData.chunk398 (by decide)

theorem chunk399 :
    Exported.decodeRows ExportedData.chunk399.1 ExportedData.chunk399.2 =
      (List.range 128).map (fun i => byteRow (51072 + i)) := by
  exact Codes.certificate 399 ExportedData.chunk399 (by decide)

theorem group19 (c : Fin 20) :
    let chunk := ExportedData.chunks[380 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(380 + c.val) + i)) := by
  fin_cases c
  · exact chunk380
  · exact chunk381
  · exact chunk382
  · exact chunk383
  · exact chunk384
  · exact chunk385
  · exact chunk386
  · exact chunk387
  · exact chunk388
  · exact chunk389
  · exact chunk390
  · exact chunk391
  · exact chunk392
  · exact chunk393
  · exact chunk394
  · exact chunk395
  · exact chunk396
  · exact chunk397
  · exact chunk398
  · exact chunk399

end CircuitCorrectness.ConcreteBytes
