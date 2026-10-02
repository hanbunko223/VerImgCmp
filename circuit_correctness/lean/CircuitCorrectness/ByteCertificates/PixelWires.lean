import CircuitCorrectness.ConcreteBytes
import CircuitCorrectness.ByteCertificates.PixelGroup11
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ConcreteBytes
theorem pixel_wires : ExportedData.pixelWires.toList =
    (List.range 7680).map valueWire := by
  unfold ExportedData.pixelWires
  simp only [pixel_chunk_0, pixel_chunk_1, pixel_chunk_2, pixel_chunk_3, pixel_chunk_4, pixel_chunk_5, pixel_chunk_6, pixel_chunk_7, pixel_chunk_8, pixel_chunk_9, pixel_chunk_10, pixel_chunk_11, pixel_chunk_12, pixel_chunk_13, pixel_chunk_14, pixel_chunk_15, pixel_chunk_16, pixel_chunk_17, pixel_chunk_18, pixel_chunk_19, pixel_chunk_20, pixel_chunk_21, pixel_chunk_22, pixel_chunk_23, pixel_chunk_24, pixel_chunk_25, pixel_chunk_26, pixel_chunk_27, pixel_chunk_28, pixel_chunk_29, pixel_chunk_30, pixel_chunk_31, pixel_chunk_32, pixel_chunk_33, pixel_chunk_34, pixel_chunk_35, pixel_chunk_36, pixel_chunk_37, pixel_chunk_38, pixel_chunk_39, pixel_chunk_40, pixel_chunk_41, pixel_chunk_42, pixel_chunk_43, pixel_chunk_44, pixel_chunk_45, pixel_chunk_46, pixel_chunk_47, pixel_chunk_48, pixel_chunk_49, pixel_chunk_50, pixel_chunk_51, pixel_chunk_52, pixel_chunk_53, pixel_chunk_54, pixel_chunk_55, pixel_chunk_56, pixel_chunk_57, pixel_chunk_58, pixel_chunk_59]
  change (((List.range 60).map expectedPixelChunk).toArray.flatten).toList = _
  simp only [Array.toList_flatten, List.map_map]
  simp only [← List.flatMap_def]
  exact flatMap_blocks valueWire 128 60


theorem array_index_of_list (a : Array Nat) (f : Nat → Nat) (m : Nat)
    (he : a.toList = (List.range m).map f) (n : Nat) (hn : n < m) : a[n]! = f n := by
  have h := congrArg (fun xs : List Nat => xs[n]?) he
  simp only [Array.getElem?_toList, List.getElem?_map,
    List.getElem?_range hn, Option.map_some] at h
  simp only [getElem!_def, h]

theorem pixel_wire (n : Nat) (hn : n < 7680) :
    ExportedData.pixelWires[n]! = valueWire n :=
  array_index_of_list ExportedData.pixelWires valueWire 7680 pixel_wires n hn

end CircuitCorrectness.ConcreteBytes
