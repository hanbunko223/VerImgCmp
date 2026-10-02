import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group01

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk100 :
    Exported.decodeRows ExportedData.chunk100.1 ExportedData.chunk100.2 =
      (List.range 128).map (fun i => byteRow (12800 + i)) := by
  exact Codes.certificate 100 ExportedData.chunk100 (by decide)

theorem chunk101 :
    Exported.decodeRows ExportedData.chunk101.1 ExportedData.chunk101.2 =
      (List.range 128).map (fun i => byteRow (12928 + i)) := by
  exact Codes.certificate 101 ExportedData.chunk101 (by decide)

theorem chunk102 :
    Exported.decodeRows ExportedData.chunk102.1 ExportedData.chunk102.2 =
      (List.range 128).map (fun i => byteRow (13056 + i)) := by
  exact Codes.certificate 102 ExportedData.chunk102 (by decide)

theorem chunk103 :
    Exported.decodeRows ExportedData.chunk103.1 ExportedData.chunk103.2 =
      (List.range 128).map (fun i => byteRow (13184 + i)) := by
  exact Codes.certificate 103 ExportedData.chunk103 (by decide)

theorem chunk104 :
    Exported.decodeRows ExportedData.chunk104.1 ExportedData.chunk104.2 =
      (List.range 128).map (fun i => byteRow (13312 + i)) := by
  exact Codes.certificate 104 ExportedData.chunk104 (by decide)

theorem chunk105 :
    Exported.decodeRows ExportedData.chunk105.1 ExportedData.chunk105.2 =
      (List.range 128).map (fun i => byteRow (13440 + i)) := by
  exact Codes.certificate 105 ExportedData.chunk105 (by decide)

theorem chunk106 :
    Exported.decodeRows ExportedData.chunk106.1 ExportedData.chunk106.2 =
      (List.range 128).map (fun i => byteRow (13568 + i)) := by
  exact Codes.certificate 106 ExportedData.chunk106 (by decide)

theorem chunk107 :
    Exported.decodeRows ExportedData.chunk107.1 ExportedData.chunk107.2 =
      (List.range 128).map (fun i => byteRow (13696 + i)) := by
  exact Codes.certificate 107 ExportedData.chunk107 (by decide)

theorem chunk108 :
    Exported.decodeRows ExportedData.chunk108.1 ExportedData.chunk108.2 =
      (List.range 128).map (fun i => byteRow (13824 + i)) := by
  exact Codes.certificate 108 ExportedData.chunk108 (by decide)

theorem chunk109 :
    Exported.decodeRows ExportedData.chunk109.1 ExportedData.chunk109.2 =
      (List.range 128).map (fun i => byteRow (13952 + i)) := by
  exact Codes.certificate 109 ExportedData.chunk109 (by decide)

theorem chunk110 :
    Exported.decodeRows ExportedData.chunk110.1 ExportedData.chunk110.2 =
      (List.range 128).map (fun i => byteRow (14080 + i)) := by
  exact Codes.certificate 110 ExportedData.chunk110 (by decide)

theorem chunk111 :
    Exported.decodeRows ExportedData.chunk111.1 ExportedData.chunk111.2 =
      (List.range 128).map (fun i => byteRow (14208 + i)) := by
  exact Codes.certificate 111 ExportedData.chunk111 (by decide)

theorem chunk112 :
    Exported.decodeRows ExportedData.chunk112.1 ExportedData.chunk112.2 =
      (List.range 128).map (fun i => byteRow (14336 + i)) := by
  exact Codes.certificate 112 ExportedData.chunk112 (by decide)

theorem chunk113 :
    Exported.decodeRows ExportedData.chunk113.1 ExportedData.chunk113.2 =
      (List.range 128).map (fun i => byteRow (14464 + i)) := by
  exact Codes.certificate 113 ExportedData.chunk113 (by decide)

theorem chunk114 :
    Exported.decodeRows ExportedData.chunk114.1 ExportedData.chunk114.2 =
      (List.range 128).map (fun i => byteRow (14592 + i)) := by
  exact Codes.certificate 114 ExportedData.chunk114 (by decide)

theorem chunk115 :
    Exported.decodeRows ExportedData.chunk115.1 ExportedData.chunk115.2 =
      (List.range 128).map (fun i => byteRow (14720 + i)) := by
  exact Codes.certificate 115 ExportedData.chunk115 (by decide)

theorem chunk116 :
    Exported.decodeRows ExportedData.chunk116.1 ExportedData.chunk116.2 =
      (List.range 128).map (fun i => byteRow (14848 + i)) := by
  exact Codes.certificate 116 ExportedData.chunk116 (by decide)

theorem chunk117 :
    Exported.decodeRows ExportedData.chunk117.1 ExportedData.chunk117.2 =
      (List.range 128).map (fun i => byteRow (14976 + i)) := by
  exact Codes.certificate 117 ExportedData.chunk117 (by decide)

theorem chunk118 :
    Exported.decodeRows ExportedData.chunk118.1 ExportedData.chunk118.2 =
      (List.range 128).map (fun i => byteRow (15104 + i)) := by
  exact Codes.certificate 118 ExportedData.chunk118 (by decide)

theorem chunk119 :
    Exported.decodeRows ExportedData.chunk119.1 ExportedData.chunk119.2 =
      (List.range 128).map (fun i => byteRow (15232 + i)) := by
  exact Codes.certificate 119 ExportedData.chunk119 (by decide)

theorem group05 (c : Fin 20) :
    let chunk := ExportedData.chunks[100 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(100 + c.val) + i)) := by
  fin_cases c
  · exact chunk100
  · exact chunk101
  · exact chunk102
  · exact chunk103
  · exact chunk104
  · exact chunk105
  · exact chunk106
  · exact chunk107
  · exact chunk108
  · exact chunk109
  · exact chunk110
  · exact chunk111
  · exact chunk112
  · exact chunk113
  · exact chunk114
  · exact chunk115
  · exact chunk116
  · exact chunk117
  · exact chunk118
  · exact chunk119

end CircuitCorrectness.ConcreteBytes
