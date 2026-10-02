import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group00

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk080 :
    Exported.decodeRows ExportedData.chunk80.1 ExportedData.chunk80.2 =
      (List.range 128).map (fun i => byteRow (10240 + i)) := by
  exact Codes.certificate 80 ExportedData.chunk80 (by decide)

theorem chunk081 :
    Exported.decodeRows ExportedData.chunk81.1 ExportedData.chunk81.2 =
      (List.range 128).map (fun i => byteRow (10368 + i)) := by
  exact Codes.certificate 81 ExportedData.chunk81 (by decide)

theorem chunk082 :
    Exported.decodeRows ExportedData.chunk82.1 ExportedData.chunk82.2 =
      (List.range 128).map (fun i => byteRow (10496 + i)) := by
  exact Codes.certificate 82 ExportedData.chunk82 (by decide)

theorem chunk083 :
    Exported.decodeRows ExportedData.chunk83.1 ExportedData.chunk83.2 =
      (List.range 128).map (fun i => byteRow (10624 + i)) := by
  exact Codes.certificate 83 ExportedData.chunk83 (by decide)

theorem chunk084 :
    Exported.decodeRows ExportedData.chunk84.1 ExportedData.chunk84.2 =
      (List.range 128).map (fun i => byteRow (10752 + i)) := by
  exact Codes.certificate 84 ExportedData.chunk84 (by decide)

theorem chunk085 :
    Exported.decodeRows ExportedData.chunk85.1 ExportedData.chunk85.2 =
      (List.range 128).map (fun i => byteRow (10880 + i)) := by
  exact Codes.certificate 85 ExportedData.chunk85 (by decide)

theorem chunk086 :
    Exported.decodeRows ExportedData.chunk86.1 ExportedData.chunk86.2 =
      (List.range 128).map (fun i => byteRow (11008 + i)) := by
  exact Codes.certificate 86 ExportedData.chunk86 (by decide)

theorem chunk087 :
    Exported.decodeRows ExportedData.chunk87.1 ExportedData.chunk87.2 =
      (List.range 128).map (fun i => byteRow (11136 + i)) := by
  exact Codes.certificate 87 ExportedData.chunk87 (by decide)

theorem chunk088 :
    Exported.decodeRows ExportedData.chunk88.1 ExportedData.chunk88.2 =
      (List.range 128).map (fun i => byteRow (11264 + i)) := by
  exact Codes.certificate 88 ExportedData.chunk88 (by decide)

theorem chunk089 :
    Exported.decodeRows ExportedData.chunk89.1 ExportedData.chunk89.2 =
      (List.range 128).map (fun i => byteRow (11392 + i)) := by
  exact Codes.certificate 89 ExportedData.chunk89 (by decide)

theorem chunk090 :
    Exported.decodeRows ExportedData.chunk90.1 ExportedData.chunk90.2 =
      (List.range 128).map (fun i => byteRow (11520 + i)) := by
  exact Codes.certificate 90 ExportedData.chunk90 (by decide)

theorem chunk091 :
    Exported.decodeRows ExportedData.chunk91.1 ExportedData.chunk91.2 =
      (List.range 128).map (fun i => byteRow (11648 + i)) := by
  exact Codes.certificate 91 ExportedData.chunk91 (by decide)

theorem chunk092 :
    Exported.decodeRows ExportedData.chunk92.1 ExportedData.chunk92.2 =
      (List.range 128).map (fun i => byteRow (11776 + i)) := by
  exact Codes.certificate 92 ExportedData.chunk92 (by decide)

theorem chunk093 :
    Exported.decodeRows ExportedData.chunk93.1 ExportedData.chunk93.2 =
      (List.range 128).map (fun i => byteRow (11904 + i)) := by
  exact Codes.certificate 93 ExportedData.chunk93 (by decide)

theorem chunk094 :
    Exported.decodeRows ExportedData.chunk94.1 ExportedData.chunk94.2 =
      (List.range 128).map (fun i => byteRow (12032 + i)) := by
  exact Codes.certificate 94 ExportedData.chunk94 (by decide)

theorem chunk095 :
    Exported.decodeRows ExportedData.chunk95.1 ExportedData.chunk95.2 =
      (List.range 128).map (fun i => byteRow (12160 + i)) := by
  exact Codes.certificate 95 ExportedData.chunk95 (by decide)

theorem chunk096 :
    Exported.decodeRows ExportedData.chunk96.1 ExportedData.chunk96.2 =
      (List.range 128).map (fun i => byteRow (12288 + i)) := by
  exact Codes.certificate 96 ExportedData.chunk96 (by decide)

theorem chunk097 :
    Exported.decodeRows ExportedData.chunk97.1 ExportedData.chunk97.2 =
      (List.range 128).map (fun i => byteRow (12416 + i)) := by
  exact Codes.certificate 97 ExportedData.chunk97 (by decide)

theorem chunk098 :
    Exported.decodeRows ExportedData.chunk98.1 ExportedData.chunk98.2 =
      (List.range 128).map (fun i => byteRow (12544 + i)) := by
  exact Codes.certificate 98 ExportedData.chunk98 (by decide)

theorem chunk099 :
    Exported.decodeRows ExportedData.chunk99.1 ExportedData.chunk99.2 =
      (List.range 128).map (fun i => byteRow (12672 + i)) := by
  exact Codes.certificate 99 ExportedData.chunk99 (by decide)

theorem group04 (c : Fin 20) :
    let chunk := ExportedData.chunks[80 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(80 + c.val) + i)) := by
  fin_cases c
  · exact chunk080
  · exact chunk081
  · exact chunk082
  · exact chunk083
  · exact chunk084
  · exact chunk085
  · exact chunk086
  · exact chunk087
  · exact chunk088
  · exact chunk089
  · exact chunk090
  · exact chunk091
  · exact chunk092
  · exact chunk093
  · exact chunk094
  · exact chunk095
  · exact chunk096
  · exact chunk097
  · exact chunk098
  · exact chunk099

end CircuitCorrectness.ConcreteBytes
