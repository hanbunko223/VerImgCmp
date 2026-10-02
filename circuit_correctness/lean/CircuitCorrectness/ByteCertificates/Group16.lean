import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group12

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk320 :
    Exported.decodeRows ExportedData.chunk320.1 ExportedData.chunk320.2 =
      (List.range 128).map (fun i => byteRow (40960 + i)) := by
  exact Codes.certificate 320 ExportedData.chunk320 (by decide)

theorem chunk321 :
    Exported.decodeRows ExportedData.chunk321.1 ExportedData.chunk321.2 =
      (List.range 128).map (fun i => byteRow (41088 + i)) := by
  exact Codes.certificate 321 ExportedData.chunk321 (by decide)

theorem chunk322 :
    Exported.decodeRows ExportedData.chunk322.1 ExportedData.chunk322.2 =
      (List.range 128).map (fun i => byteRow (41216 + i)) := by
  exact Codes.certificate 322 ExportedData.chunk322 (by decide)

theorem chunk323 :
    Exported.decodeRows ExportedData.chunk323.1 ExportedData.chunk323.2 =
      (List.range 128).map (fun i => byteRow (41344 + i)) := by
  exact Codes.certificate 323 ExportedData.chunk323 (by decide)

theorem chunk324 :
    Exported.decodeRows ExportedData.chunk324.1 ExportedData.chunk324.2 =
      (List.range 128).map (fun i => byteRow (41472 + i)) := by
  exact Codes.certificate 324 ExportedData.chunk324 (by decide)

theorem chunk325 :
    Exported.decodeRows ExportedData.chunk325.1 ExportedData.chunk325.2 =
      (List.range 128).map (fun i => byteRow (41600 + i)) := by
  exact Codes.certificate 325 ExportedData.chunk325 (by decide)

theorem chunk326 :
    Exported.decodeRows ExportedData.chunk326.1 ExportedData.chunk326.2 =
      (List.range 128).map (fun i => byteRow (41728 + i)) := by
  exact Codes.certificate 326 ExportedData.chunk326 (by decide)

theorem chunk327 :
    Exported.decodeRows ExportedData.chunk327.1 ExportedData.chunk327.2 =
      (List.range 128).map (fun i => byteRow (41856 + i)) := by
  exact Codes.certificate 327 ExportedData.chunk327 (by decide)

theorem chunk328 :
    Exported.decodeRows ExportedData.chunk328.1 ExportedData.chunk328.2 =
      (List.range 128).map (fun i => byteRow (41984 + i)) := by
  exact Codes.certificate 328 ExportedData.chunk328 (by decide)

theorem chunk329 :
    Exported.decodeRows ExportedData.chunk329.1 ExportedData.chunk329.2 =
      (List.range 128).map (fun i => byteRow (42112 + i)) := by
  exact Codes.certificate 329 ExportedData.chunk329 (by decide)

theorem chunk330 :
    Exported.decodeRows ExportedData.chunk330.1 ExportedData.chunk330.2 =
      (List.range 128).map (fun i => byteRow (42240 + i)) := by
  exact Codes.certificate 330 ExportedData.chunk330 (by decide)

theorem chunk331 :
    Exported.decodeRows ExportedData.chunk331.1 ExportedData.chunk331.2 =
      (List.range 128).map (fun i => byteRow (42368 + i)) := by
  exact Codes.certificate 331 ExportedData.chunk331 (by decide)

theorem chunk332 :
    Exported.decodeRows ExportedData.chunk332.1 ExportedData.chunk332.2 =
      (List.range 128).map (fun i => byteRow (42496 + i)) := by
  exact Codes.certificate 332 ExportedData.chunk332 (by decide)

theorem chunk333 :
    Exported.decodeRows ExportedData.chunk333.1 ExportedData.chunk333.2 =
      (List.range 128).map (fun i => byteRow (42624 + i)) := by
  exact Codes.certificate 333 ExportedData.chunk333 (by decide)

theorem chunk334 :
    Exported.decodeRows ExportedData.chunk334.1 ExportedData.chunk334.2 =
      (List.range 128).map (fun i => byteRow (42752 + i)) := by
  exact Codes.certificate 334 ExportedData.chunk334 (by decide)

theorem chunk335 :
    Exported.decodeRows ExportedData.chunk335.1 ExportedData.chunk335.2 =
      (List.range 128).map (fun i => byteRow (42880 + i)) := by
  exact Codes.certificate 335 ExportedData.chunk335 (by decide)

theorem chunk336 :
    Exported.decodeRows ExportedData.chunk336.1 ExportedData.chunk336.2 =
      (List.range 128).map (fun i => byteRow (43008 + i)) := by
  exact Codes.certificate 336 ExportedData.chunk336 (by decide)

theorem chunk337 :
    Exported.decodeRows ExportedData.chunk337.1 ExportedData.chunk337.2 =
      (List.range 128).map (fun i => byteRow (43136 + i)) := by
  exact Codes.certificate 337 ExportedData.chunk337 (by decide)

theorem chunk338 :
    Exported.decodeRows ExportedData.chunk338.1 ExportedData.chunk338.2 =
      (List.range 128).map (fun i => byteRow (43264 + i)) := by
  exact Codes.certificate 338 ExportedData.chunk338 (by decide)

theorem chunk339 :
    Exported.decodeRows ExportedData.chunk339.1 ExportedData.chunk339.2 =
      (List.range 128).map (fun i => byteRow (43392 + i)) := by
  exact Codes.certificate 339 ExportedData.chunk339 (by decide)

theorem group16 (c : Fin 20) :
    let chunk := ExportedData.chunks[320 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(320 + c.val) + i)) := by
  fin_cases c
  · exact chunk320
  · exact chunk321
  · exact chunk322
  · exact chunk323
  · exact chunk324
  · exact chunk325
  · exact chunk326
  · exact chunk327
  · exact chunk328
  · exact chunk329
  · exact chunk330
  · exact chunk331
  · exact chunk332
  · exact chunk333
  · exact chunk334
  · exact chunk335
  · exact chunk336
  · exact chunk337
  · exact chunk338
  · exact chunk339

end CircuitCorrectness.ConcreteBytes
