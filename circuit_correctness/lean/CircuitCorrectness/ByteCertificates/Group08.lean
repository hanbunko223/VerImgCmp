import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group04

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk160 :
    Exported.decodeRows ExportedData.chunk160.1 ExportedData.chunk160.2 =
      (List.range 128).map (fun i => byteRow (20480 + i)) := by
  exact Codes.certificate 160 ExportedData.chunk160 (by decide)

theorem chunk161 :
    Exported.decodeRows ExportedData.chunk161.1 ExportedData.chunk161.2 =
      (List.range 128).map (fun i => byteRow (20608 + i)) := by
  exact Codes.certificate 161 ExportedData.chunk161 (by decide)

theorem chunk162 :
    Exported.decodeRows ExportedData.chunk162.1 ExportedData.chunk162.2 =
      (List.range 128).map (fun i => byteRow (20736 + i)) := by
  exact Codes.certificate 162 ExportedData.chunk162 (by decide)

theorem chunk163 :
    Exported.decodeRows ExportedData.chunk163.1 ExportedData.chunk163.2 =
      (List.range 128).map (fun i => byteRow (20864 + i)) := by
  exact Codes.certificate 163 ExportedData.chunk163 (by decide)

theorem chunk164 :
    Exported.decodeRows ExportedData.chunk164.1 ExportedData.chunk164.2 =
      (List.range 128).map (fun i => byteRow (20992 + i)) := by
  exact Codes.certificate 164 ExportedData.chunk164 (by decide)

theorem chunk165 :
    Exported.decodeRows ExportedData.chunk165.1 ExportedData.chunk165.2 =
      (List.range 128).map (fun i => byteRow (21120 + i)) := by
  exact Codes.certificate 165 ExportedData.chunk165 (by decide)

theorem chunk166 :
    Exported.decodeRows ExportedData.chunk166.1 ExportedData.chunk166.2 =
      (List.range 128).map (fun i => byteRow (21248 + i)) := by
  exact Codes.certificate 166 ExportedData.chunk166 (by decide)

theorem chunk167 :
    Exported.decodeRows ExportedData.chunk167.1 ExportedData.chunk167.2 =
      (List.range 128).map (fun i => byteRow (21376 + i)) := by
  exact Codes.certificate 167 ExportedData.chunk167 (by decide)

theorem chunk168 :
    Exported.decodeRows ExportedData.chunk168.1 ExportedData.chunk168.2 =
      (List.range 128).map (fun i => byteRow (21504 + i)) := by
  exact Codes.certificate 168 ExportedData.chunk168 (by decide)

theorem chunk169 :
    Exported.decodeRows ExportedData.chunk169.1 ExportedData.chunk169.2 =
      (List.range 128).map (fun i => byteRow (21632 + i)) := by
  exact Codes.certificate 169 ExportedData.chunk169 (by decide)

theorem chunk170 :
    Exported.decodeRows ExportedData.chunk170.1 ExportedData.chunk170.2 =
      (List.range 128).map (fun i => byteRow (21760 + i)) := by
  exact Codes.certificate 170 ExportedData.chunk170 (by decide)

theorem chunk171 :
    Exported.decodeRows ExportedData.chunk171.1 ExportedData.chunk171.2 =
      (List.range 128).map (fun i => byteRow (21888 + i)) := by
  exact Codes.certificate 171 ExportedData.chunk171 (by decide)

theorem chunk172 :
    Exported.decodeRows ExportedData.chunk172.1 ExportedData.chunk172.2 =
      (List.range 128).map (fun i => byteRow (22016 + i)) := by
  exact Codes.certificate 172 ExportedData.chunk172 (by decide)

theorem chunk173 :
    Exported.decodeRows ExportedData.chunk173.1 ExportedData.chunk173.2 =
      (List.range 128).map (fun i => byteRow (22144 + i)) := by
  exact Codes.certificate 173 ExportedData.chunk173 (by decide)

theorem chunk174 :
    Exported.decodeRows ExportedData.chunk174.1 ExportedData.chunk174.2 =
      (List.range 128).map (fun i => byteRow (22272 + i)) := by
  exact Codes.certificate 174 ExportedData.chunk174 (by decide)

theorem chunk175 :
    Exported.decodeRows ExportedData.chunk175.1 ExportedData.chunk175.2 =
      (List.range 128).map (fun i => byteRow (22400 + i)) := by
  exact Codes.certificate 175 ExportedData.chunk175 (by decide)

theorem chunk176 :
    Exported.decodeRows ExportedData.chunk176.1 ExportedData.chunk176.2 =
      (List.range 128).map (fun i => byteRow (22528 + i)) := by
  exact Codes.certificate 176 ExportedData.chunk176 (by decide)

theorem chunk177 :
    Exported.decodeRows ExportedData.chunk177.1 ExportedData.chunk177.2 =
      (List.range 128).map (fun i => byteRow (22656 + i)) := by
  exact Codes.certificate 177 ExportedData.chunk177 (by decide)

theorem chunk178 :
    Exported.decodeRows ExportedData.chunk178.1 ExportedData.chunk178.2 =
      (List.range 128).map (fun i => byteRow (22784 + i)) := by
  exact Codes.certificate 178 ExportedData.chunk178 (by decide)

theorem chunk179 :
    Exported.decodeRows ExportedData.chunk179.1 ExportedData.chunk179.2 =
      (List.range 128).map (fun i => byteRow (22912 + i)) := by
  exact Codes.certificate 179 ExportedData.chunk179 (by decide)

theorem group08 (c : Fin 20) :
    let chunk := ExportedData.chunks[160 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(160 + c.val) + i)) := by
  fin_cases c
  · exact chunk160
  · exact chunk161
  · exact chunk162
  · exact chunk163
  · exact chunk164
  · exact chunk165
  · exact chunk166
  · exact chunk167
  · exact chunk168
  · exact chunk169
  · exact chunk170
  · exact chunk171
  · exact chunk172
  · exact chunk173
  · exact chunk174
  · exact chunk175
  · exact chunk176
  · exact chunk177
  · exact chunk178
  · exact chunk179

end CircuitCorrectness.ConcreteBytes
