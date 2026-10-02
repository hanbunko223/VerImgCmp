import CircuitCorrectness.ByteCertificates.Decode

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes

theorem chunk000 :
    Exported.decodeRows ExportedData.chunk0.1 ExportedData.chunk0.2 =
      (List.range 128).map (fun i => byteRow (0 + i)) := by
  exact Codes.certificate 0 ExportedData.chunk0 (by decide)

theorem chunk001 :
    Exported.decodeRows ExportedData.chunk1.1 ExportedData.chunk1.2 =
      (List.range 128).map (fun i => byteRow (128 + i)) := by
  exact Codes.certificate 1 ExportedData.chunk1 (by decide)

theorem chunk002 :
    Exported.decodeRows ExportedData.chunk2.1 ExportedData.chunk2.2 =
      (List.range 128).map (fun i => byteRow (256 + i)) := by
  exact Codes.certificate 2 ExportedData.chunk2 (by decide)

theorem chunk003 :
    Exported.decodeRows ExportedData.chunk3.1 ExportedData.chunk3.2 =
      (List.range 128).map (fun i => byteRow (384 + i)) := by
  exact Codes.certificate 3 ExportedData.chunk3 (by decide)

theorem chunk004 :
    Exported.decodeRows ExportedData.chunk4.1 ExportedData.chunk4.2 =
      (List.range 128).map (fun i => byteRow (512 + i)) := by
  exact Codes.certificate 4 ExportedData.chunk4 (by decide)

theorem chunk005 :
    Exported.decodeRows ExportedData.chunk5.1 ExportedData.chunk5.2 =
      (List.range 128).map (fun i => byteRow (640 + i)) := by
  exact Codes.certificate 5 ExportedData.chunk5 (by decide)

theorem chunk006 :
    Exported.decodeRows ExportedData.chunk6.1 ExportedData.chunk6.2 =
      (List.range 128).map (fun i => byteRow (768 + i)) := by
  exact Codes.certificate 6 ExportedData.chunk6 (by decide)

theorem chunk007 :
    Exported.decodeRows ExportedData.chunk7.1 ExportedData.chunk7.2 =
      (List.range 128).map (fun i => byteRow (896 + i)) := by
  exact Codes.certificate 7 ExportedData.chunk7 (by decide)

theorem chunk008 :
    Exported.decodeRows ExportedData.chunk8.1 ExportedData.chunk8.2 =
      (List.range 128).map (fun i => byteRow (1024 + i)) := by
  exact Codes.certificate 8 ExportedData.chunk8 (by decide)

theorem chunk009 :
    Exported.decodeRows ExportedData.chunk9.1 ExportedData.chunk9.2 =
      (List.range 128).map (fun i => byteRow (1152 + i)) := by
  exact Codes.certificate 9 ExportedData.chunk9 (by decide)

theorem chunk010 :
    Exported.decodeRows ExportedData.chunk10.1 ExportedData.chunk10.2 =
      (List.range 128).map (fun i => byteRow (1280 + i)) := by
  exact Codes.certificate 10 ExportedData.chunk10 (by decide)

theorem chunk011 :
    Exported.decodeRows ExportedData.chunk11.1 ExportedData.chunk11.2 =
      (List.range 128).map (fun i => byteRow (1408 + i)) := by
  exact Codes.certificate 11 ExportedData.chunk11 (by decide)

theorem chunk012 :
    Exported.decodeRows ExportedData.chunk12.1 ExportedData.chunk12.2 =
      (List.range 128).map (fun i => byteRow (1536 + i)) := by
  exact Codes.certificate 12 ExportedData.chunk12 (by decide)

theorem chunk013 :
    Exported.decodeRows ExportedData.chunk13.1 ExportedData.chunk13.2 =
      (List.range 128).map (fun i => byteRow (1664 + i)) := by
  exact Codes.certificate 13 ExportedData.chunk13 (by decide)

theorem chunk014 :
    Exported.decodeRows ExportedData.chunk14.1 ExportedData.chunk14.2 =
      (List.range 128).map (fun i => byteRow (1792 + i)) := by
  exact Codes.certificate 14 ExportedData.chunk14 (by decide)

theorem chunk015 :
    Exported.decodeRows ExportedData.chunk15.1 ExportedData.chunk15.2 =
      (List.range 128).map (fun i => byteRow (1920 + i)) := by
  exact Codes.certificate 15 ExportedData.chunk15 (by decide)

theorem chunk016 :
    Exported.decodeRows ExportedData.chunk16.1 ExportedData.chunk16.2 =
      (List.range 128).map (fun i => byteRow (2048 + i)) := by
  exact Codes.certificate 16 ExportedData.chunk16 (by decide)

theorem chunk017 :
    Exported.decodeRows ExportedData.chunk17.1 ExportedData.chunk17.2 =
      (List.range 128).map (fun i => byteRow (2176 + i)) := by
  exact Codes.certificate 17 ExportedData.chunk17 (by decide)

theorem chunk018 :
    Exported.decodeRows ExportedData.chunk18.1 ExportedData.chunk18.2 =
      (List.range 128).map (fun i => byteRow (2304 + i)) := by
  exact Codes.certificate 18 ExportedData.chunk18 (by decide)

theorem chunk019 :
    Exported.decodeRows ExportedData.chunk19.1 ExportedData.chunk19.2 =
      (List.range 128).map (fun i => byteRow (2432 + i)) := by
  exact Codes.certificate 19 ExportedData.chunk19 (by decide)

theorem group00 (c : Fin 20) :
    let chunk := ExportedData.chunks[0 + c.val]!
    Exported.decodeRows chunk.1 chunk.2 =
      (List.range 128).map (fun i => byteRow (128*(0 + c.val) + i)) := by
  fin_cases c
  · exact chunk000
  · exact chunk001
  · exact chunk002
  · exact chunk003
  · exact chunk004
  · exact chunk005
  · exact chunk006
  · exact chunk007
  · exact chunk008
  · exact chunk009
  · exact chunk010
  · exact chunk011
  · exact chunk012
  · exact chunk013
  · exact chunk014
  · exact chunk015
  · exact chunk016
  · exact chunk017
  · exact chunk018
  · exact chunk019

end CircuitCorrectness.ConcreteBytes
