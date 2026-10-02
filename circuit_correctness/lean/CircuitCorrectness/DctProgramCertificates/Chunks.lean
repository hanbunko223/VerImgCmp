import CircuitCorrectness.DctProgramCertificates.Group09
import CircuitCorrectness.DctProgramCertificates.Group10
import CircuitCorrectness.DctProgramCertificates.Group11
import CircuitCorrectness.DctProgramCertificates.Group12
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram ConcreteBytes.Codes
theorem first_chunk (c : Fin 40) :
    let chunk := ExportedData.chunks[540+c.val]!
    rows chunk.1 chunk.2 = (firstRows.drop (128*c.val)).take 128 := by
  fin_cases c
  · exact chunk540
  · exact chunk541
  · exact chunk542
  · exact chunk543
  · exact chunk544
  · exact chunk545
  · exact chunk546
  · exact chunk547
  · exact chunk548
  · exact chunk549
  · exact chunk550
  · exact chunk551
  · exact chunk552
  · exact chunk553
  · exact chunk554
  · exact chunk555
  · exact chunk556
  · exact chunk557
  · exact chunk558
  · exact chunk559
  · exact chunk560
  · exact chunk561
  · exact chunk562
  · exact chunk563
  · exact chunk564
  · exact chunk565
  · exact chunk566
  · exact chunk567
  · exact chunk568
  · exact chunk569
  · exact chunk570
  · exact chunk571
  · exact chunk572
  · exact chunk573
  · exact chunk574
  · exact chunk575
  · exact chunk576
  · exact chunk577
  · exact chunk578
  · exact chunk579
theorem horner_chunk (c : Fin 25) (row : Row) :
    row ∈ (hornerRows.drop (128*c.val)).take 128 →
    let chunk := ExportedData.chunks[580+c.val]!
    row ∈ rows chunk.1 chunk.2 := by
  fin_cases c
  · change row ∈ (hornerRows.drop 0).take 128 → row ∈ rows ExportedData.chunk580.1 ExportedData.chunk580.2
    rw [← chunk580]; exact id
  · change row ∈ (hornerRows.drop 128).take 128 → row ∈ rows ExportedData.chunk581.1 ExportedData.chunk581.2
    rw [← chunk581]; exact id
  · change row ∈ (hornerRows.drop 256).take 128 → row ∈ rows ExportedData.chunk582.1 ExportedData.chunk582.2
    rw [← chunk582]; exact id
  · change row ∈ (hornerRows.drop 384).take 128 → row ∈ rows ExportedData.chunk583.1 ExportedData.chunk583.2
    rw [← chunk583]; exact id
  · change row ∈ (hornerRows.drop 512).take 128 → row ∈ rows ExportedData.chunk584.1 ExportedData.chunk584.2
    rw [← chunk584]; exact id
  · change row ∈ (hornerRows.drop 640).take 128 → row ∈ rows ExportedData.chunk585.1 ExportedData.chunk585.2
    rw [← chunk585]; exact id
  · change row ∈ (hornerRows.drop 768).take 128 → row ∈ rows ExportedData.chunk586.1 ExportedData.chunk586.2
    rw [← chunk586]; exact id
  · change row ∈ (hornerRows.drop 896).take 128 → row ∈ rows ExportedData.chunk587.1 ExportedData.chunk587.2
    rw [← chunk587]; exact id
  · change row ∈ (hornerRows.drop 1024).take 128 → row ∈ rows ExportedData.chunk588.1 ExportedData.chunk588.2
    rw [← chunk588]; exact id
  · change row ∈ (hornerRows.drop 1152).take 128 → row ∈ rows ExportedData.chunk589.1 ExportedData.chunk589.2
    rw [← chunk589]; exact id
  · change row ∈ (hornerRows.drop 1280).take 128 → row ∈ rows ExportedData.chunk590.1 ExportedData.chunk590.2
    rw [← chunk590]; exact id
  · change row ∈ (hornerRows.drop 1408).take 128 → row ∈ rows ExportedData.chunk591.1 ExportedData.chunk591.2
    rw [← chunk591]; exact id
  · change row ∈ (hornerRows.drop 1536).take 128 → row ∈ rows ExportedData.chunk592.1 ExportedData.chunk592.2
    rw [← chunk592]; exact id
  · change row ∈ (hornerRows.drop 1664).take 128 → row ∈ rows ExportedData.chunk593.1 ExportedData.chunk593.2
    rw [← chunk593]; exact id
  · change row ∈ (hornerRows.drop 1792).take 128 → row ∈ rows ExportedData.chunk594.1 ExportedData.chunk594.2
    rw [← chunk594]; exact id
  · change row ∈ (hornerRows.drop 1920).take 128 → row ∈ rows ExportedData.chunk595.1 ExportedData.chunk595.2
    rw [← chunk595]; exact id
  · change row ∈ (hornerRows.drop 2048).take 128 → row ∈ rows ExportedData.chunk596.1 ExportedData.chunk596.2
    rw [← chunk596]; exact id
  · change row ∈ (hornerRows.drop 2176).take 128 → row ∈ rows ExportedData.chunk597.1 ExportedData.chunk597.2
    rw [← chunk597]; exact id
  · change row ∈ (hornerRows.drop 2304).take 128 → row ∈ rows ExportedData.chunk598.1 ExportedData.chunk598.2
    rw [← chunk598]; exact id
  · change row ∈ (hornerRows.drop 2432).take 128 → row ∈ rows ExportedData.chunk599.1 ExportedData.chunk599.2
    rw [← chunk599]; exact id
  · change row ∈ (hornerRows.drop 2560).take 128 → row ∈ rows ExportedData.chunk600.1 ExportedData.chunk600.2
    rw [← chunk600]; exact id
  · change row ∈ (hornerRows.drop 2688).take 128 → row ∈ rows ExportedData.chunk601.1 ExportedData.chunk601.2
    rw [← chunk601]; exact id
  · change row ∈ (hornerRows.drop 2816).take 128 → row ∈ rows ExportedData.chunk602.1 ExportedData.chunk602.2
    rw [← chunk602]; exact id
  · change row ∈ (hornerRows.drop 2944).take 128 → row ∈ rows ExportedData.chunk603.1 ExportedData.chunk603.2
    rw [← chunk603]; exact id
  · change row ∈ (hornerRows.drop 3072).take 128 → row ∈ rows ExportedData.chunk604.1 ExportedData.chunk604.2
    rw [← chunk604]; exact List.mem_of_mem_take
end CircuitCorrectness.DctProgramCertificates
