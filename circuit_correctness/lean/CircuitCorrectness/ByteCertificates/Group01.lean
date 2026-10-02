import CircuitCorrectness.ByteCertificates.Decode

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk020 :
    Exported.decodeRows ExportedData.chunk20.1 ExportedData.chunk20.2 =
      (List.range 128).map (fun i => byteRow (2560 + i)) := by
  exact Codes.certificate 20 ExportedData.chunk20 (by decide)

theorem chunk021 :
    Exported.decodeRows ExportedData.chunk21.1 ExportedData.chunk21.2 =
      (List.range 128).map (fun i => byteRow (2688 + i)) := by
  exact Codes.certificate 21 ExportedData.chunk21 (by decide)

theorem chunk022 :
    Exported.decodeRows ExportedData.chunk22.1 ExportedData.chunk22.2 =
      (List.range 128).map (fun i => byteRow (2816 + i)) := by
  exact Codes.certificate 22 ExportedData.chunk22 (by decide)

theorem chunk023 :
    Exported.decodeRows ExportedData.chunk23.1 ExportedData.chunk23.2 =
      (List.range 128).map (fun i => byteRow (2944 + i)) := by
  exact Codes.certificate 23 ExportedData.chunk23 (by decide)

theorem chunk024 :
    Exported.decodeRows ExportedData.chunk24.1 ExportedData.chunk24.2 =
      (List.range 128).map (fun i => byteRow (3072 + i)) := by
  exact Codes.certificate 24 ExportedData.chunk24 (by decide)

theorem chunk025 :
    Exported.decodeRows ExportedData.chunk25.1 ExportedData.chunk25.2 =
      (List.range 128).map (fun i => byteRow (3200 + i)) := by
  exact Codes.certificate 25 ExportedData.chunk25 (by decide)

theorem chunk026 :
    Exported.decodeRows ExportedData.chunk26.1 ExportedData.chunk26.2 =
      (List.range 128).map (fun i => byteRow (3328 + i)) := by
  exact Codes.certificate 26 ExportedData.chunk26 (by decide)

theorem chunk027 :
    Exported.decodeRows ExportedData.chunk27.1 ExportedData.chunk27.2 =
      (List.range 128).map (fun i => byteRow (3456 + i)) := by
  exact Codes.certificate 27 ExportedData.chunk27 (by decide)

theorem chunk028 :
    Exported.decodeRows ExportedData.chunk28.1 ExportedData.chunk28.2 =
      (List.range 128).map (fun i => byteRow (3584 + i)) := by
  exact Codes.certificate 28 ExportedData.chunk28 (by decide)

theorem chunk029 :
    Exported.decodeRows ExportedData.chunk29.1 ExportedData.chunk29.2 =
      (List.range 128).map (fun i => byteRow (3712 + i)) := by
  exact Codes.certificate 29 ExportedData.chunk29 (by decide)

theorem chunk030 :
    Exported.decodeRows ExportedData.chunk30.1 ExportedData.chunk30.2 =
      (List.range 128).map (fun i => byteRow (3840 + i)) := by
  exact Codes.certificate 30 ExportedData.chunk30 (by decide)

theorem chunk031 :
    Exported.decodeRows ExportedData.chunk31.1 ExportedData.chunk31.2 =
      (List.range 128).map (fun i => byteRow (3968 + i)) := by
  exact Codes.certificate 31 ExportedData.chunk31 (by decide)

theorem chunk032 :
    Exported.decodeRows ExportedData.chunk32.1 ExportedData.chunk32.2 =
      (List.range 128).map (fun i => byteRow (4096 + i)) := by
  exact Codes.certificate 32 ExportedData.chunk32 (by decide)

theorem chunk033 :
    Exported.decodeRows ExportedData.chunk33.1 ExportedData.chunk33.2 =
      (List.range 128).map (fun i => byteRow (4224 + i)) := by
  exact Codes.certificate 33 ExportedData.chunk33 (by decide)

theorem chunk034 :
    Exported.decodeRows ExportedData.chunk34.1 ExportedData.chunk34.2 =
      (List.range 128).map (fun i => byteRow (4352 + i)) := by
  exact Codes.certificate 34 ExportedData.chunk34 (by decide)

theorem chunk035 :
    Exported.decodeRows ExportedData.chunk35.1 ExportedData.chunk35.2 =
      (List.range 128).map (fun i => byteRow (4480 + i)) := by
  exact Codes.certificate 35 ExportedData.chunk35 (by decide)

theorem chunk036 :
    Exported.decodeRows ExportedData.chunk36.1 ExportedData.chunk36.2 =
      (List.range 128).map (fun i => byteRow (4608 + i)) := by
  exact Codes.certificate 36 ExportedData.chunk36 (by decide)

theorem chunk037 :
    Exported.decodeRows ExportedData.chunk37.1 ExportedData.chunk37.2 =
      (List.range 128).map (fun i => byteRow (4736 + i)) := by
  exact Codes.certificate 37 ExportedData.chunk37 (by decide)

theorem chunk038 :
    Exported.decodeRows ExportedData.chunk38.1 ExportedData.chunk38.2 =
      (List.range 128).map (fun i => byteRow (4864 + i)) := by
  exact Codes.certificate 38 ExportedData.chunk38 (by decide)

theorem chunk039 :
    Exported.decodeRows ExportedData.chunk39.1 ExportedData.chunk39.2 =
      (List.range 128).map (fun i => byteRow (4992 + i)) := by
  exact Codes.certificate 39 ExportedData.chunk39 (by decide)

theorem group01 (c : Fin 20) :
    let chunk := ExportedData.chunks[20 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(20 + c.val) + i)) := by
  fin_cases c
  · exact chunk020
  · exact chunk021
  · exact chunk022
  · exact chunk023
  · exact chunk024
  · exact chunk025
  · exact chunk026
  · exact chunk027
  · exact chunk028
  · exact chunk029
  · exact chunk030
  · exact chunk031
  · exact chunk032
  · exact chunk033
  · exact chunk034
  · exact chunk035
  · exact chunk036
  · exact chunk037
  · exact chunk038
  · exact chunk039

end CircuitCorrectness.ConcreteBytes
