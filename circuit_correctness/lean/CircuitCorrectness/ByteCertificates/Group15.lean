import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group11

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk300 :
    Exported.decodeRows ExportedData.chunk300.1 ExportedData.chunk300.2 =
      (List.range 128).map (fun i => byteRow (38400 + i)) := by
  exact Codes.certificate 300 ExportedData.chunk300 (by decide)

theorem chunk301 :
    Exported.decodeRows ExportedData.chunk301.1 ExportedData.chunk301.2 =
      (List.range 128).map (fun i => byteRow (38528 + i)) := by
  exact Codes.certificate 301 ExportedData.chunk301 (by decide)

theorem chunk302 :
    Exported.decodeRows ExportedData.chunk302.1 ExportedData.chunk302.2 =
      (List.range 128).map (fun i => byteRow (38656 + i)) := by
  exact Codes.certificate 302 ExportedData.chunk302 (by decide)

theorem chunk303 :
    Exported.decodeRows ExportedData.chunk303.1 ExportedData.chunk303.2 =
      (List.range 128).map (fun i => byteRow (38784 + i)) := by
  exact Codes.certificate 303 ExportedData.chunk303 (by decide)

theorem chunk304 :
    Exported.decodeRows ExportedData.chunk304.1 ExportedData.chunk304.2 =
      (List.range 128).map (fun i => byteRow (38912 + i)) := by
  exact Codes.certificate 304 ExportedData.chunk304 (by decide)

theorem chunk305 :
    Exported.decodeRows ExportedData.chunk305.1 ExportedData.chunk305.2 =
      (List.range 128).map (fun i => byteRow (39040 + i)) := by
  exact Codes.certificate 305 ExportedData.chunk305 (by decide)

theorem chunk306 :
    Exported.decodeRows ExportedData.chunk306.1 ExportedData.chunk306.2 =
      (List.range 128).map (fun i => byteRow (39168 + i)) := by
  exact Codes.certificate 306 ExportedData.chunk306 (by decide)

theorem chunk307 :
    Exported.decodeRows ExportedData.chunk307.1 ExportedData.chunk307.2 =
      (List.range 128).map (fun i => byteRow (39296 + i)) := by
  exact Codes.certificate 307 ExportedData.chunk307 (by decide)

theorem chunk308 :
    Exported.decodeRows ExportedData.chunk308.1 ExportedData.chunk308.2 =
      (List.range 128).map (fun i => byteRow (39424 + i)) := by
  exact Codes.certificate 308 ExportedData.chunk308 (by decide)

theorem chunk309 :
    Exported.decodeRows ExportedData.chunk309.1 ExportedData.chunk309.2 =
      (List.range 128).map (fun i => byteRow (39552 + i)) := by
  exact Codes.certificate 309 ExportedData.chunk309 (by decide)

theorem chunk310 :
    Exported.decodeRows ExportedData.chunk310.1 ExportedData.chunk310.2 =
      (List.range 128).map (fun i => byteRow (39680 + i)) := by
  exact Codes.certificate 310 ExportedData.chunk310 (by decide)

theorem chunk311 :
    Exported.decodeRows ExportedData.chunk311.1 ExportedData.chunk311.2 =
      (List.range 128).map (fun i => byteRow (39808 + i)) := by
  exact Codes.certificate 311 ExportedData.chunk311 (by decide)

theorem chunk312 :
    Exported.decodeRows ExportedData.chunk312.1 ExportedData.chunk312.2 =
      (List.range 128).map (fun i => byteRow (39936 + i)) := by
  exact Codes.certificate 312 ExportedData.chunk312 (by decide)

theorem chunk313 :
    Exported.decodeRows ExportedData.chunk313.1 ExportedData.chunk313.2 =
      (List.range 128).map (fun i => byteRow (40064 + i)) := by
  exact Codes.certificate 313 ExportedData.chunk313 (by decide)

theorem chunk314 :
    Exported.decodeRows ExportedData.chunk314.1 ExportedData.chunk314.2 =
      (List.range 128).map (fun i => byteRow (40192 + i)) := by
  exact Codes.certificate 314 ExportedData.chunk314 (by decide)

theorem chunk315 :
    Exported.decodeRows ExportedData.chunk315.1 ExportedData.chunk315.2 =
      (List.range 128).map (fun i => byteRow (40320 + i)) := by
  exact Codes.certificate 315 ExportedData.chunk315 (by decide)

theorem chunk316 :
    Exported.decodeRows ExportedData.chunk316.1 ExportedData.chunk316.2 =
      (List.range 128).map (fun i => byteRow (40448 + i)) := by
  exact Codes.certificate 316 ExportedData.chunk316 (by decide)

theorem chunk317 :
    Exported.decodeRows ExportedData.chunk317.1 ExportedData.chunk317.2 =
      (List.range 128).map (fun i => byteRow (40576 + i)) := by
  exact Codes.certificate 317 ExportedData.chunk317 (by decide)

theorem chunk318 :
    Exported.decodeRows ExportedData.chunk318.1 ExportedData.chunk318.2 =
      (List.range 128).map (fun i => byteRow (40704 + i)) := by
  exact Codes.certificate 318 ExportedData.chunk318 (by decide)

theorem chunk319 :
    Exported.decodeRows ExportedData.chunk319.1 ExportedData.chunk319.2 =
      (List.range 128).map (fun i => byteRow (40832 + i)) := by
  exact Codes.certificate 319 ExportedData.chunk319 (by decide)

theorem group15 (c : Fin 20) :
    let chunk := ExportedData.chunks[300 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(300 + c.val) + i)) := by
  fin_cases c
  · exact chunk300
  · exact chunk301
  · exact chunk302
  · exact chunk303
  · exact chunk304
  · exact chunk305
  · exact chunk306
  · exact chunk307
  · exact chunk308
  · exact chunk309
  · exact chunk310
  · exact chunk311
  · exact chunk312
  · exact chunk313
  · exact chunk314
  · exact chunk315
  · exact chunk316
  · exact chunk317
  · exact chunk318
  · exact chunk319

end CircuitCorrectness.ConcreteBytes
