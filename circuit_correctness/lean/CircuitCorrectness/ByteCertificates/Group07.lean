import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.ByteCertificates.Group03

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk140 :
    Exported.decodeRows ExportedData.chunk140.1 ExportedData.chunk140.2 =
      (List.range 128).map (fun i => byteRow (17920 + i)) := by
  exact Codes.certificate 140 ExportedData.chunk140 (by decide)

theorem chunk141 :
    Exported.decodeRows ExportedData.chunk141.1 ExportedData.chunk141.2 =
      (List.range 128).map (fun i => byteRow (18048 + i)) := by
  exact Codes.certificate 141 ExportedData.chunk141 (by decide)

theorem chunk142 :
    Exported.decodeRows ExportedData.chunk142.1 ExportedData.chunk142.2 =
      (List.range 128).map (fun i => byteRow (18176 + i)) := by
  exact Codes.certificate 142 ExportedData.chunk142 (by decide)

theorem chunk143 :
    Exported.decodeRows ExportedData.chunk143.1 ExportedData.chunk143.2 =
      (List.range 128).map (fun i => byteRow (18304 + i)) := by
  exact Codes.certificate 143 ExportedData.chunk143 (by decide)

theorem chunk144 :
    Exported.decodeRows ExportedData.chunk144.1 ExportedData.chunk144.2 =
      (List.range 128).map (fun i => byteRow (18432 + i)) := by
  exact Codes.certificate 144 ExportedData.chunk144 (by decide)

theorem chunk145 :
    Exported.decodeRows ExportedData.chunk145.1 ExportedData.chunk145.2 =
      (List.range 128).map (fun i => byteRow (18560 + i)) := by
  exact Codes.certificate 145 ExportedData.chunk145 (by decide)

theorem chunk146 :
    Exported.decodeRows ExportedData.chunk146.1 ExportedData.chunk146.2 =
      (List.range 128).map (fun i => byteRow (18688 + i)) := by
  exact Codes.certificate 146 ExportedData.chunk146 (by decide)

theorem chunk147 :
    Exported.decodeRows ExportedData.chunk147.1 ExportedData.chunk147.2 =
      (List.range 128).map (fun i => byteRow (18816 + i)) := by
  exact Codes.certificate 147 ExportedData.chunk147 (by decide)

theorem chunk148 :
    Exported.decodeRows ExportedData.chunk148.1 ExportedData.chunk148.2 =
      (List.range 128).map (fun i => byteRow (18944 + i)) := by
  exact Codes.certificate 148 ExportedData.chunk148 (by decide)

theorem chunk149 :
    Exported.decodeRows ExportedData.chunk149.1 ExportedData.chunk149.2 =
      (List.range 128).map (fun i => byteRow (19072 + i)) := by
  exact Codes.certificate 149 ExportedData.chunk149 (by decide)

theorem chunk150 :
    Exported.decodeRows ExportedData.chunk150.1 ExportedData.chunk150.2 =
      (List.range 128).map (fun i => byteRow (19200 + i)) := by
  exact Codes.certificate 150 ExportedData.chunk150 (by decide)

theorem chunk151 :
    Exported.decodeRows ExportedData.chunk151.1 ExportedData.chunk151.2 =
      (List.range 128).map (fun i => byteRow (19328 + i)) := by
  exact Codes.certificate 151 ExportedData.chunk151 (by decide)

theorem chunk152 :
    Exported.decodeRows ExportedData.chunk152.1 ExportedData.chunk152.2 =
      (List.range 128).map (fun i => byteRow (19456 + i)) := by
  exact Codes.certificate 152 ExportedData.chunk152 (by decide)

theorem chunk153 :
    Exported.decodeRows ExportedData.chunk153.1 ExportedData.chunk153.2 =
      (List.range 128).map (fun i => byteRow (19584 + i)) := by
  exact Codes.certificate 153 ExportedData.chunk153 (by decide)

theorem chunk154 :
    Exported.decodeRows ExportedData.chunk154.1 ExportedData.chunk154.2 =
      (List.range 128).map (fun i => byteRow (19712 + i)) := by
  exact Codes.certificate 154 ExportedData.chunk154 (by decide)

theorem chunk155 :
    Exported.decodeRows ExportedData.chunk155.1 ExportedData.chunk155.2 =
      (List.range 128).map (fun i => byteRow (19840 + i)) := by
  exact Codes.certificate 155 ExportedData.chunk155 (by decide)

theorem chunk156 :
    Exported.decodeRows ExportedData.chunk156.1 ExportedData.chunk156.2 =
      (List.range 128).map (fun i => byteRow (19968 + i)) := by
  exact Codes.certificate 156 ExportedData.chunk156 (by decide)

theorem chunk157 :
    Exported.decodeRows ExportedData.chunk157.1 ExportedData.chunk157.2 =
      (List.range 128).map (fun i => byteRow (20096 + i)) := by
  exact Codes.certificate 157 ExportedData.chunk157 (by decide)

theorem chunk158 :
    Exported.decodeRows ExportedData.chunk158.1 ExportedData.chunk158.2 =
      (List.range 128).map (fun i => byteRow (20224 + i)) := by
  exact Codes.certificate 158 ExportedData.chunk158 (by decide)

theorem chunk159 :
    Exported.decodeRows ExportedData.chunk159.1 ExportedData.chunk159.2 =
      (List.range 128).map (fun i => byteRow (20352 + i)) := by
  exact Codes.certificate 159 ExportedData.chunk159 (by decide)

theorem group07 (c : Fin 20) :
    let chunk := ExportedData.chunks[140 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(140 + c.val) + i)) := by
  fin_cases c
  · exact chunk140
  · exact chunk141
  · exact chunk142
  · exact chunk143
  · exact chunk144
  · exact chunk145
  · exact chunk146
  · exact chunk147
  · exact chunk148
  · exact chunk149
  · exact chunk150
  · exact chunk151
  · exact chunk152
  · exact chunk153
  · exact chunk154
  · exact chunk155
  · exact chunk156
  · exact chunk157
  · exact chunk158
  · exact chunk159

end CircuitCorrectness.ConcreteBytes
