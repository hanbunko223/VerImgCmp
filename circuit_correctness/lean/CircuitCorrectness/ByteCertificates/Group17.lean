import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group13

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk340 :
    Exported.decodeRows ExportedData.chunk340.1 ExportedData.chunk340.2 =
      (List.range 128).map (fun i => byteRow (43520 + i)) := by
  exact Codes.certificate 340 ExportedData.chunk340 (by decide)

theorem chunk341 :
    Exported.decodeRows ExportedData.chunk341.1 ExportedData.chunk341.2 =
      (List.range 128).map (fun i => byteRow (43648 + i)) := by
  exact Codes.certificate 341 ExportedData.chunk341 (by decide)

theorem chunk342 :
    Exported.decodeRows ExportedData.chunk342.1 ExportedData.chunk342.2 =
      (List.range 128).map (fun i => byteRow (43776 + i)) := by
  exact Codes.certificate 342 ExportedData.chunk342 (by decide)

theorem chunk343 :
    Exported.decodeRows ExportedData.chunk343.1 ExportedData.chunk343.2 =
      (List.range 128).map (fun i => byteRow (43904 + i)) := by
  exact Codes.certificate 343 ExportedData.chunk343 (by decide)

theorem chunk344 :
    Exported.decodeRows ExportedData.chunk344.1 ExportedData.chunk344.2 =
      (List.range 128).map (fun i => byteRow (44032 + i)) := by
  exact Codes.certificate 344 ExportedData.chunk344 (by decide)

theorem chunk345 :
    Exported.decodeRows ExportedData.chunk345.1 ExportedData.chunk345.2 =
      (List.range 128).map (fun i => byteRow (44160 + i)) := by
  exact Codes.certificate 345 ExportedData.chunk345 (by decide)

theorem chunk346 :
    Exported.decodeRows ExportedData.chunk346.1 ExportedData.chunk346.2 =
      (List.range 128).map (fun i => byteRow (44288 + i)) := by
  exact Codes.certificate 346 ExportedData.chunk346 (by decide)

theorem chunk347 :
    Exported.decodeRows ExportedData.chunk347.1 ExportedData.chunk347.2 =
      (List.range 128).map (fun i => byteRow (44416 + i)) := by
  exact Codes.certificate 347 ExportedData.chunk347 (by decide)

theorem chunk348 :
    Exported.decodeRows ExportedData.chunk348.1 ExportedData.chunk348.2 =
      (List.range 128).map (fun i => byteRow (44544 + i)) := by
  exact Codes.certificate 348 ExportedData.chunk348 (by decide)

theorem chunk349 :
    Exported.decodeRows ExportedData.chunk349.1 ExportedData.chunk349.2 =
      (List.range 128).map (fun i => byteRow (44672 + i)) := by
  exact Codes.certificate 349 ExportedData.chunk349 (by decide)

theorem chunk350 :
    Exported.decodeRows ExportedData.chunk350.1 ExportedData.chunk350.2 =
      (List.range 128).map (fun i => byteRow (44800 + i)) := by
  exact Codes.certificate 350 ExportedData.chunk350 (by decide)

theorem chunk351 :
    Exported.decodeRows ExportedData.chunk351.1 ExportedData.chunk351.2 =
      (List.range 128).map (fun i => byteRow (44928 + i)) := by
  exact Codes.certificate 351 ExportedData.chunk351 (by decide)

theorem chunk352 :
    Exported.decodeRows ExportedData.chunk352.1 ExportedData.chunk352.2 =
      (List.range 128).map (fun i => byteRow (45056 + i)) := by
  exact Codes.certificate 352 ExportedData.chunk352 (by decide)

theorem chunk353 :
    Exported.decodeRows ExportedData.chunk353.1 ExportedData.chunk353.2 =
      (List.range 128).map (fun i => byteRow (45184 + i)) := by
  exact Codes.certificate 353 ExportedData.chunk353 (by decide)

theorem chunk354 :
    Exported.decodeRows ExportedData.chunk354.1 ExportedData.chunk354.2 =
      (List.range 128).map (fun i => byteRow (45312 + i)) := by
  exact Codes.certificate 354 ExportedData.chunk354 (by decide)

theorem chunk355 :
    Exported.decodeRows ExportedData.chunk355.1 ExportedData.chunk355.2 =
      (List.range 128).map (fun i => byteRow (45440 + i)) := by
  exact Codes.certificate 355 ExportedData.chunk355 (by decide)

theorem chunk356 :
    Exported.decodeRows ExportedData.chunk356.1 ExportedData.chunk356.2 =
      (List.range 128).map (fun i => byteRow (45568 + i)) := by
  exact Codes.certificate 356 ExportedData.chunk356 (by decide)

theorem chunk357 :
    Exported.decodeRows ExportedData.chunk357.1 ExportedData.chunk357.2 =
      (List.range 128).map (fun i => byteRow (45696 + i)) := by
  exact Codes.certificate 357 ExportedData.chunk357 (by decide)

theorem chunk358 :
    Exported.decodeRows ExportedData.chunk358.1 ExportedData.chunk358.2 =
      (List.range 128).map (fun i => byteRow (45824 + i)) := by
  exact Codes.certificate 358 ExportedData.chunk358 (by decide)

theorem chunk359 :
    Exported.decodeRows ExportedData.chunk359.1 ExportedData.chunk359.2 =
      (List.range 128).map (fun i => byteRow (45952 + i)) := by
  exact Codes.certificate 359 ExportedData.chunk359 (by decide)

theorem group17 (c : Fin 20) :
    let chunk := ExportedData.chunks[340 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(340 + c.val) + i)) := by
  fin_cases c
  · exact chunk340
  · exact chunk341
  · exact chunk342
  · exact chunk343
  · exact chunk344
  · exact chunk345
  · exact chunk346
  · exact chunk347
  · exact chunk348
  · exact chunk349
  · exact chunk350
  · exact chunk351
  · exact chunk352
  · exact chunk353
  · exact chunk354
  · exact chunk355
  · exact chunk356
  · exact chunk357
  · exact chunk358
  · exact chunk359

end CircuitCorrectness.ConcreteBytes
