import CircuitCorrectness.ByteCertificates.Decode

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk060 :
    Exported.decodeRows ExportedData.chunk60.1 ExportedData.chunk60.2 =
      (List.range 128).map (fun i => byteRow (7680 + i)) := by
  exact Codes.certificate 60 ExportedData.chunk60 (by decide)

theorem chunk061 :
    Exported.decodeRows ExportedData.chunk61.1 ExportedData.chunk61.2 =
      (List.range 128).map (fun i => byteRow (7808 + i)) := by
  exact Codes.certificate 61 ExportedData.chunk61 (by decide)

theorem chunk062 :
    Exported.decodeRows ExportedData.chunk62.1 ExportedData.chunk62.2 =
      (List.range 128).map (fun i => byteRow (7936 + i)) := by
  exact Codes.certificate 62 ExportedData.chunk62 (by decide)

theorem chunk063 :
    Exported.decodeRows ExportedData.chunk63.1 ExportedData.chunk63.2 =
      (List.range 128).map (fun i => byteRow (8064 + i)) := by
  exact Codes.certificate 63 ExportedData.chunk63 (by decide)

theorem chunk064 :
    Exported.decodeRows ExportedData.chunk64.1 ExportedData.chunk64.2 =
      (List.range 128).map (fun i => byteRow (8192 + i)) := by
  exact Codes.certificate 64 ExportedData.chunk64 (by decide)

theorem chunk065 :
    Exported.decodeRows ExportedData.chunk65.1 ExportedData.chunk65.2 =
      (List.range 128).map (fun i => byteRow (8320 + i)) := by
  exact Codes.certificate 65 ExportedData.chunk65 (by decide)

theorem chunk066 :
    Exported.decodeRows ExportedData.chunk66.1 ExportedData.chunk66.2 =
      (List.range 128).map (fun i => byteRow (8448 + i)) := by
  exact Codes.certificate 66 ExportedData.chunk66 (by decide)

theorem chunk067 :
    Exported.decodeRows ExportedData.chunk67.1 ExportedData.chunk67.2 =
      (List.range 128).map (fun i => byteRow (8576 + i)) := by
  exact Codes.certificate 67 ExportedData.chunk67 (by decide)

theorem chunk068 :
    Exported.decodeRows ExportedData.chunk68.1 ExportedData.chunk68.2 =
      (List.range 128).map (fun i => byteRow (8704 + i)) := by
  exact Codes.certificate 68 ExportedData.chunk68 (by decide)

theorem chunk069 :
    Exported.decodeRows ExportedData.chunk69.1 ExportedData.chunk69.2 =
      (List.range 128).map (fun i => byteRow (8832 + i)) := by
  exact Codes.certificate 69 ExportedData.chunk69 (by decide)

theorem chunk070 :
    Exported.decodeRows ExportedData.chunk70.1 ExportedData.chunk70.2 =
      (List.range 128).map (fun i => byteRow (8960 + i)) := by
  exact Codes.certificate 70 ExportedData.chunk70 (by decide)

theorem chunk071 :
    Exported.decodeRows ExportedData.chunk71.1 ExportedData.chunk71.2 =
      (List.range 128).map (fun i => byteRow (9088 + i)) := by
  exact Codes.certificate 71 ExportedData.chunk71 (by decide)

theorem chunk072 :
    Exported.decodeRows ExportedData.chunk72.1 ExportedData.chunk72.2 =
      (List.range 128).map (fun i => byteRow (9216 + i)) := by
  exact Codes.certificate 72 ExportedData.chunk72 (by decide)

theorem chunk073 :
    Exported.decodeRows ExportedData.chunk73.1 ExportedData.chunk73.2 =
      (List.range 128).map (fun i => byteRow (9344 + i)) := by
  exact Codes.certificate 73 ExportedData.chunk73 (by decide)

theorem chunk074 :
    Exported.decodeRows ExportedData.chunk74.1 ExportedData.chunk74.2 =
      (List.range 128).map (fun i => byteRow (9472 + i)) := by
  exact Codes.certificate 74 ExportedData.chunk74 (by decide)

theorem chunk075 :
    Exported.decodeRows ExportedData.chunk75.1 ExportedData.chunk75.2 =
      (List.range 128).map (fun i => byteRow (9600 + i)) := by
  exact Codes.certificate 75 ExportedData.chunk75 (by decide)

theorem chunk076 :
    Exported.decodeRows ExportedData.chunk76.1 ExportedData.chunk76.2 =
      (List.range 128).map (fun i => byteRow (9728 + i)) := by
  exact Codes.certificate 76 ExportedData.chunk76 (by decide)

theorem chunk077 :
    Exported.decodeRows ExportedData.chunk77.1 ExportedData.chunk77.2 =
      (List.range 128).map (fun i => byteRow (9856 + i)) := by
  exact Codes.certificate 77 ExportedData.chunk77 (by decide)

theorem chunk078 :
    Exported.decodeRows ExportedData.chunk78.1 ExportedData.chunk78.2 =
      (List.range 128).map (fun i => byteRow (9984 + i)) := by
  exact Codes.certificate 78 ExportedData.chunk78 (by decide)

theorem chunk079 :
    Exported.decodeRows ExportedData.chunk79.1 ExportedData.chunk79.2 =
      (List.range 128).map (fun i => byteRow (10112 + i)) := by
  exact Codes.certificate 79 ExportedData.chunk79 (by decide)

theorem group03 (c : Fin 20) :
    let chunk := ExportedData.chunks[60 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(60 + c.val) + i)) := by
  fin_cases c
  · exact chunk060
  · exact chunk061
  · exact chunk062
  · exact chunk063
  · exact chunk064
  · exact chunk065
  · exact chunk066
  · exact chunk067
  · exact chunk068
  · exact chunk069
  · exact chunk070
  · exact chunk071
  · exact chunk072
  · exact chunk073
  · exact chunk074
  · exact chunk075
  · exact chunk076
  · exact chunk077
  · exact chunk078
  · exact chunk079

end CircuitCorrectness.ConcreteBytes
