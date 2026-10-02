import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group08

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk240 :
    Exported.decodeRows ExportedData.chunk240.1 ExportedData.chunk240.2 =
      (List.range 128).map (fun i => byteRow (30720 + i)) := by
  exact Codes.certificate 240 ExportedData.chunk240 (by decide)

theorem chunk241 :
    Exported.decodeRows ExportedData.chunk241.1 ExportedData.chunk241.2 =
      (List.range 128).map (fun i => byteRow (30848 + i)) := by
  exact Codes.certificate 241 ExportedData.chunk241 (by decide)

theorem chunk242 :
    Exported.decodeRows ExportedData.chunk242.1 ExportedData.chunk242.2 =
      (List.range 128).map (fun i => byteRow (30976 + i)) := by
  exact Codes.certificate 242 ExportedData.chunk242 (by decide)

theorem chunk243 :
    Exported.decodeRows ExportedData.chunk243.1 ExportedData.chunk243.2 =
      (List.range 128).map (fun i => byteRow (31104 + i)) := by
  exact Codes.certificate 243 ExportedData.chunk243 (by decide)

theorem chunk244 :
    Exported.decodeRows ExportedData.chunk244.1 ExportedData.chunk244.2 =
      (List.range 128).map (fun i => byteRow (31232 + i)) := by
  exact Codes.certificate 244 ExportedData.chunk244 (by decide)

theorem chunk245 :
    Exported.decodeRows ExportedData.chunk245.1 ExportedData.chunk245.2 =
      (List.range 128).map (fun i => byteRow (31360 + i)) := by
  exact Codes.certificate 245 ExportedData.chunk245 (by decide)

theorem chunk246 :
    Exported.decodeRows ExportedData.chunk246.1 ExportedData.chunk246.2 =
      (List.range 128).map (fun i => byteRow (31488 + i)) := by
  exact Codes.certificate 246 ExportedData.chunk246 (by decide)

theorem chunk247 :
    Exported.decodeRows ExportedData.chunk247.1 ExportedData.chunk247.2 =
      (List.range 128).map (fun i => byteRow (31616 + i)) := by
  exact Codes.certificate 247 ExportedData.chunk247 (by decide)

theorem chunk248 :
    Exported.decodeRows ExportedData.chunk248.1 ExportedData.chunk248.2 =
      (List.range 128).map (fun i => byteRow (31744 + i)) := by
  exact Codes.certificate 248 ExportedData.chunk248 (by decide)

theorem chunk249 :
    Exported.decodeRows ExportedData.chunk249.1 ExportedData.chunk249.2 =
      (List.range 128).map (fun i => byteRow (31872 + i)) := by
  exact Codes.certificate 249 ExportedData.chunk249 (by decide)

theorem chunk250 :
    Exported.decodeRows ExportedData.chunk250.1 ExportedData.chunk250.2 =
      (List.range 128).map (fun i => byteRow (32000 + i)) := by
  exact Codes.certificate 250 ExportedData.chunk250 (by decide)

theorem chunk251 :
    Exported.decodeRows ExportedData.chunk251.1 ExportedData.chunk251.2 =
      (List.range 128).map (fun i => byteRow (32128 + i)) := by
  exact Codes.certificate 251 ExportedData.chunk251 (by decide)

theorem chunk252 :
    Exported.decodeRows ExportedData.chunk252.1 ExportedData.chunk252.2 =
      (List.range 128).map (fun i => byteRow (32256 + i)) := by
  exact Codes.certificate 252 ExportedData.chunk252 (by decide)

theorem chunk253 :
    Exported.decodeRows ExportedData.chunk253.1 ExportedData.chunk253.2 =
      (List.range 128).map (fun i => byteRow (32384 + i)) := by
  exact Codes.certificate 253 ExportedData.chunk253 (by decide)

theorem chunk254 :
    Exported.decodeRows ExportedData.chunk254.1 ExportedData.chunk254.2 =
      (List.range 128).map (fun i => byteRow (32512 + i)) := by
  exact Codes.certificate 254 ExportedData.chunk254 (by decide)

theorem chunk255 :
    Exported.decodeRows ExportedData.chunk255.1 ExportedData.chunk255.2 =
      (List.range 128).map (fun i => byteRow (32640 + i)) := by
  exact Codes.certificate 255 ExportedData.chunk255 (by decide)

theorem chunk256 :
    Exported.decodeRows ExportedData.chunk256.1 ExportedData.chunk256.2 =
      (List.range 128).map (fun i => byteRow (32768 + i)) := by
  exact Codes.certificate 256 ExportedData.chunk256 (by decide)

theorem chunk257 :
    Exported.decodeRows ExportedData.chunk257.1 ExportedData.chunk257.2 =
      (List.range 128).map (fun i => byteRow (32896 + i)) := by
  exact Codes.certificate 257 ExportedData.chunk257 (by decide)

theorem chunk258 :
    Exported.decodeRows ExportedData.chunk258.1 ExportedData.chunk258.2 =
      (List.range 128).map (fun i => byteRow (33024 + i)) := by
  exact Codes.certificate 258 ExportedData.chunk258 (by decide)

theorem chunk259 :
    Exported.decodeRows ExportedData.chunk259.1 ExportedData.chunk259.2 =
      (List.range 128).map (fun i => byteRow (33152 + i)) := by
  exact Codes.certificate 259 ExportedData.chunk259 (by decide)

theorem group12 (c : Fin 20) :
    let chunk := ExportedData.chunks[240 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(240 + c.val) + i)) := by
  fin_cases c
  · exact chunk240
  · exact chunk241
  · exact chunk242
  · exact chunk243
  · exact chunk244
  · exact chunk245
  · exact chunk246
  · exact chunk247
  · exact chunk248
  · exact chunk249
  · exact chunk250
  · exact chunk251
  · exact chunk252
  · exact chunk253
  · exact chunk254
  · exact chunk255
  · exact chunk256
  · exact chunk257
  · exact chunk258
  · exact chunk259

end CircuitCorrectness.ConcreteBytes
