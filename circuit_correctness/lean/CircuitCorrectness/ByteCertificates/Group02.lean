import CircuitCorrectness.ByteCertificates.Decode

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk040 :
    Exported.decodeRows ExportedData.chunk40.1 ExportedData.chunk40.2 =
      (List.range 128).map (fun i => byteRow (5120 + i)) := by
  exact Codes.certificate 40 ExportedData.chunk40 (by decide)

theorem chunk041 :
    Exported.decodeRows ExportedData.chunk41.1 ExportedData.chunk41.2 =
      (List.range 128).map (fun i => byteRow (5248 + i)) := by
  exact Codes.certificate 41 ExportedData.chunk41 (by decide)

theorem chunk042 :
    Exported.decodeRows ExportedData.chunk42.1 ExportedData.chunk42.2 =
      (List.range 128).map (fun i => byteRow (5376 + i)) := by
  exact Codes.certificate 42 ExportedData.chunk42 (by decide)

theorem chunk043 :
    Exported.decodeRows ExportedData.chunk43.1 ExportedData.chunk43.2 =
      (List.range 128).map (fun i => byteRow (5504 + i)) := by
  exact Codes.certificate 43 ExportedData.chunk43 (by decide)

theorem chunk044 :
    Exported.decodeRows ExportedData.chunk44.1 ExportedData.chunk44.2 =
      (List.range 128).map (fun i => byteRow (5632 + i)) := by
  exact Codes.certificate 44 ExportedData.chunk44 (by decide)

theorem chunk045 :
    Exported.decodeRows ExportedData.chunk45.1 ExportedData.chunk45.2 =
      (List.range 128).map (fun i => byteRow (5760 + i)) := by
  exact Codes.certificate 45 ExportedData.chunk45 (by decide)

theorem chunk046 :
    Exported.decodeRows ExportedData.chunk46.1 ExportedData.chunk46.2 =
      (List.range 128).map (fun i => byteRow (5888 + i)) := by
  exact Codes.certificate 46 ExportedData.chunk46 (by decide)

theorem chunk047 :
    Exported.decodeRows ExportedData.chunk47.1 ExportedData.chunk47.2 =
      (List.range 128).map (fun i => byteRow (6016 + i)) := by
  exact Codes.certificate 47 ExportedData.chunk47 (by decide)

theorem chunk048 :
    Exported.decodeRows ExportedData.chunk48.1 ExportedData.chunk48.2 =
      (List.range 128).map (fun i => byteRow (6144 + i)) := by
  exact Codes.certificate 48 ExportedData.chunk48 (by decide)

theorem chunk049 :
    Exported.decodeRows ExportedData.chunk49.1 ExportedData.chunk49.2 =
      (List.range 128).map (fun i => byteRow (6272 + i)) := by
  exact Codes.certificate 49 ExportedData.chunk49 (by decide)

theorem chunk050 :
    Exported.decodeRows ExportedData.chunk50.1 ExportedData.chunk50.2 =
      (List.range 128).map (fun i => byteRow (6400 + i)) := by
  exact Codes.certificate 50 ExportedData.chunk50 (by decide)

theorem chunk051 :
    Exported.decodeRows ExportedData.chunk51.1 ExportedData.chunk51.2 =
      (List.range 128).map (fun i => byteRow (6528 + i)) := by
  exact Codes.certificate 51 ExportedData.chunk51 (by decide)

theorem chunk052 :
    Exported.decodeRows ExportedData.chunk52.1 ExportedData.chunk52.2 =
      (List.range 128).map (fun i => byteRow (6656 + i)) := by
  exact Codes.certificate 52 ExportedData.chunk52 (by decide)

theorem chunk053 :
    Exported.decodeRows ExportedData.chunk53.1 ExportedData.chunk53.2 =
      (List.range 128).map (fun i => byteRow (6784 + i)) := by
  exact Codes.certificate 53 ExportedData.chunk53 (by decide)

theorem chunk054 :
    Exported.decodeRows ExportedData.chunk54.1 ExportedData.chunk54.2 =
      (List.range 128).map (fun i => byteRow (6912 + i)) := by
  exact Codes.certificate 54 ExportedData.chunk54 (by decide)

theorem chunk055 :
    Exported.decodeRows ExportedData.chunk55.1 ExportedData.chunk55.2 =
      (List.range 128).map (fun i => byteRow (7040 + i)) := by
  exact Codes.certificate 55 ExportedData.chunk55 (by decide)

theorem chunk056 :
    Exported.decodeRows ExportedData.chunk56.1 ExportedData.chunk56.2 =
      (List.range 128).map (fun i => byteRow (7168 + i)) := by
  exact Codes.certificate 56 ExportedData.chunk56 (by decide)

theorem chunk057 :
    Exported.decodeRows ExportedData.chunk57.1 ExportedData.chunk57.2 =
      (List.range 128).map (fun i => byteRow (7296 + i)) := by
  exact Codes.certificate 57 ExportedData.chunk57 (by decide)

theorem chunk058 :
    Exported.decodeRows ExportedData.chunk58.1 ExportedData.chunk58.2 =
      (List.range 128).map (fun i => byteRow (7424 + i)) := by
  exact Codes.certificate 58 ExportedData.chunk58 (by decide)

theorem chunk059 :
    Exported.decodeRows ExportedData.chunk59.1 ExportedData.chunk59.2 =
      (List.range 128).map (fun i => byteRow (7552 + i)) := by
  exact Codes.certificate 59 ExportedData.chunk59 (by decide)

theorem group02 (c : Fin 20) :
    let chunk := ExportedData.chunks[40 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(40 + c.val) + i)) := by
  fin_cases c
  · exact chunk040
  · exact chunk041
  · exact chunk042
  · exact chunk043
  · exact chunk044
  · exact chunk045
  · exact chunk046
  · exact chunk047
  · exact chunk048
  · exact chunk049
  · exact chunk050
  · exact chunk051
  · exact chunk052
  · exact chunk053
  · exact chunk054
  · exact chunk055
  · exact chunk056
  · exact chunk057
  · exact chunk058
  · exact chunk059

end CircuitCorrectness.ConcreteBytes
