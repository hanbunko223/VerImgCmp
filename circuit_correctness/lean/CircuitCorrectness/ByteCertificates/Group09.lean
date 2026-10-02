import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group05

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk180 :
    Exported.decodeRows ExportedData.chunk180.1 ExportedData.chunk180.2 =
      (List.range 128).map (fun i => byteRow (23040 + i)) := by
  exact Codes.certificate 180 ExportedData.chunk180 (by decide)

theorem chunk181 :
    Exported.decodeRows ExportedData.chunk181.1 ExportedData.chunk181.2 =
      (List.range 128).map (fun i => byteRow (23168 + i)) := by
  exact Codes.certificate 181 ExportedData.chunk181 (by decide)

theorem chunk182 :
    Exported.decodeRows ExportedData.chunk182.1 ExportedData.chunk182.2 =
      (List.range 128).map (fun i => byteRow (23296 + i)) := by
  exact Codes.certificate 182 ExportedData.chunk182 (by decide)

theorem chunk183 :
    Exported.decodeRows ExportedData.chunk183.1 ExportedData.chunk183.2 =
      (List.range 128).map (fun i => byteRow (23424 + i)) := by
  exact Codes.certificate 183 ExportedData.chunk183 (by decide)

theorem chunk184 :
    Exported.decodeRows ExportedData.chunk184.1 ExportedData.chunk184.2 =
      (List.range 128).map (fun i => byteRow (23552 + i)) := by
  exact Codes.certificate 184 ExportedData.chunk184 (by decide)

theorem chunk185 :
    Exported.decodeRows ExportedData.chunk185.1 ExportedData.chunk185.2 =
      (List.range 128).map (fun i => byteRow (23680 + i)) := by
  exact Codes.certificate 185 ExportedData.chunk185 (by decide)

theorem chunk186 :
    Exported.decodeRows ExportedData.chunk186.1 ExportedData.chunk186.2 =
      (List.range 128).map (fun i => byteRow (23808 + i)) := by
  exact Codes.certificate 186 ExportedData.chunk186 (by decide)

theorem chunk187 :
    Exported.decodeRows ExportedData.chunk187.1 ExportedData.chunk187.2 =
      (List.range 128).map (fun i => byteRow (23936 + i)) := by
  exact Codes.certificate 187 ExportedData.chunk187 (by decide)

theorem chunk188 :
    Exported.decodeRows ExportedData.chunk188.1 ExportedData.chunk188.2 =
      (List.range 128).map (fun i => byteRow (24064 + i)) := by
  exact Codes.certificate 188 ExportedData.chunk188 (by decide)

theorem chunk189 :
    Exported.decodeRows ExportedData.chunk189.1 ExportedData.chunk189.2 =
      (List.range 128).map (fun i => byteRow (24192 + i)) := by
  exact Codes.certificate 189 ExportedData.chunk189 (by decide)

theorem chunk190 :
    Exported.decodeRows ExportedData.chunk190.1 ExportedData.chunk190.2 =
      (List.range 128).map (fun i => byteRow (24320 + i)) := by
  exact Codes.certificate 190 ExportedData.chunk190 (by decide)

theorem chunk191 :
    Exported.decodeRows ExportedData.chunk191.1 ExportedData.chunk191.2 =
      (List.range 128).map (fun i => byteRow (24448 + i)) := by
  exact Codes.certificate 191 ExportedData.chunk191 (by decide)

theorem chunk192 :
    Exported.decodeRows ExportedData.chunk192.1 ExportedData.chunk192.2 =
      (List.range 128).map (fun i => byteRow (24576 + i)) := by
  exact Codes.certificate 192 ExportedData.chunk192 (by decide)

theorem chunk193 :
    Exported.decodeRows ExportedData.chunk193.1 ExportedData.chunk193.2 =
      (List.range 128).map (fun i => byteRow (24704 + i)) := by
  exact Codes.certificate 193 ExportedData.chunk193 (by decide)

theorem chunk194 :
    Exported.decodeRows ExportedData.chunk194.1 ExportedData.chunk194.2 =
      (List.range 128).map (fun i => byteRow (24832 + i)) := by
  exact Codes.certificate 194 ExportedData.chunk194 (by decide)

theorem chunk195 :
    Exported.decodeRows ExportedData.chunk195.1 ExportedData.chunk195.2 =
      (List.range 128).map (fun i => byteRow (24960 + i)) := by
  exact Codes.certificate 195 ExportedData.chunk195 (by decide)

theorem chunk196 :
    Exported.decodeRows ExportedData.chunk196.1 ExportedData.chunk196.2 =
      (List.range 128).map (fun i => byteRow (25088 + i)) := by
  exact Codes.certificate 196 ExportedData.chunk196 (by decide)

theorem chunk197 :
    Exported.decodeRows ExportedData.chunk197.1 ExportedData.chunk197.2 =
      (List.range 128).map (fun i => byteRow (25216 + i)) := by
  exact Codes.certificate 197 ExportedData.chunk197 (by decide)

theorem chunk198 :
    Exported.decodeRows ExportedData.chunk198.1 ExportedData.chunk198.2 =
      (List.range 128).map (fun i => byteRow (25344 + i)) := by
  exact Codes.certificate 198 ExportedData.chunk198 (by decide)

theorem chunk199 :
    Exported.decodeRows ExportedData.chunk199.1 ExportedData.chunk199.2 =
      (List.range 128).map (fun i => byteRow (25472 + i)) := by
  exact Codes.certificate 199 ExportedData.chunk199 (by decide)

theorem group09 (c : Fin 20) :
    let chunk := ExportedData.chunks[180 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(180 + c.val) + i)) := by
  fin_cases c
  · exact chunk180
  · exact chunk181
  · exact chunk182
  · exact chunk183
  · exact chunk184
  · exact chunk185
  · exact chunk186
  · exact chunk187
  · exact chunk188
  · exact chunk189
  · exact chunk190
  · exact chunk191
  · exact chunk192
  · exact chunk193
  · exact chunk194
  · exact chunk195
  · exact chunk196
  · exact chunk197
  · exact chunk198
  · exact chunk199

end CircuitCorrectness.ConcreteBytes
