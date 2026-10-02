import CircuitCorrectness.HashNodeSound
import CircuitCorrectness.PackingCertificates
import CircuitCorrectness.HashTree
import CircuitCorrectness.InputBridge
namespace CircuitCorrectness.ActualHash

/-- The hash component follows from actual exported constraints. Prepared hash
wires are not assumed correct; their equality rows are covered by completeness. -/
theorem hash_transition (w : Assignment) (x : Spec.Image)
    (hm : HashProgram.Matches w x) (hs : Exported.circuit.Sat w) :
    w 97632 = Spec.hash2 (w 0) (Spec.stepDigest x) := by
  have hrows : ∀ r<16, w (HashProgram.rowStart r+1189) = Spec.rowHash x r := by
    intro r hr
    have hp := PackingCertificates.satisfied w hs r hr
    have pixels : ∀ c<160, w (HashProgram.rowStart r+c) = (Spec.packedPixel x r c : F) := by
      intro c hc
      apply HashProgram.pixel_sound w x r c hr hc hm hs.1
      exact hp.1 _ (List.mem_map.mpr ⟨c,List.mem_range.mpr hc,rfl⟩)
    have chunks : ∀ c<16, w (HashProgram.rowStart r+160+c) = (Spec.packedChunk x r c : F) := by
      intro c hc
      apply HashProgram.chunk_sound w x r c hc pixels hs.1
      exact hp.2 _ (List.mem_map.mpr ⟨c,List.mem_range.mpr hc,rfl⟩)
    exact HashTree.row_sound w x r chunks (HashWiring.left w hs r hr)
      (HashWiring.right w hs r hr) (HashWiring.combine w hs r hr)
  exact HashTree.chain_sound w x hrows (HashWiring.digest_left w hs)
    (HashWiring.digest_right w hs) (HashWiring.digest_combine_value hs) (HashWiring.chain_value hs)
end CircuitCorrectness.ActualHash
