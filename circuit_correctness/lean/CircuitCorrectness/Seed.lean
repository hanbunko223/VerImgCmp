import CircuitCorrectness.ConcreteBytes
import CircuitCorrectness.ByteCertificates.PixelWires
import CircuitCorrectness.Target
import CircuitCorrectness.StraightLine

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000

namespace CircuitCorrectness.Seed
open ConcreteBytes

/-- Row-major RGB enumeration of the original, uncentered input bytes. -/
def pixel (x : Spec.Image) (n : Nat) : Nat := x (n / 480) (n / 3 % 160) (n % 3)

theorem pixel_lt (x : Spec.Image) (hx : Spec.ValidImage x) (n : Nat) (hn : n < 7680) :
    pixel x n < 256 := by
  apply hx
  · omega
  · exact Nat.mod_lt _ (by decide)
  · exact Nat.mod_lt _ (by decide)

theorem pixel_index (x : Spec.Image) (r c ch : Nat)
    (_hr : r < 16) (hc : c < 160) (hch : ch < 3) :
    pixel x ((r * 160 + c) * 3 + ch) = x r c ch := by
  unfold pixel
  congr 1 <;> omega

/-- The actual exported image mapping is precisely the row-major byte layout. -/
theorem matches_iff (w : Assignment) (x : Spec.Image) :
    Target.matchesImage w x ↔ ∀ n < 7680, w (valueWire n) = (pixel x n : F) := by
  constructor
  · intro hm n hn
    have hr : n / 480 < 16 := by omega
    have hc : n / 3 % 160 < 160 := Nat.mod_lt _ (by decide)
    have hch : n % 3 < 3 := Nat.mod_lt _ (by decide)
    have he : (n / 480 * 160 + n / 3 % 160) * 3 + n % 3 = n := by omega
    have h := hm (n / 480) hr (n / 3 % 160) hc (n % 3) hch
    rw [he, pixel_wire n hn] at h
    exact h
  · intro hm r hr c hc ch hch
    have hn : (r * 160 + c) * 3 + ch < 7680 := by omega
    rw [pixel_wire _ hn, hm _ hn, pixel_index x r c ch hr hc hch]

/-- Initial values: state, bytes and their eight bits, plus the constant-one wire.
Subsequent arithmetic wires may start at zero because execution overwrites them. -/
def assignment (x : Spec.Image) (s : Spec.State) : Assignment := fun j =>
  if j = 97634 then 1 else
  if j = 0 then s.h else if j = 1 then s.a else
  if j = 2 then s.r else if j = 3 then s.t else
  if j < 69124 then
    if (j - 4) % 9 = 0 then (pixel x ((j - 4) / 9) : F)
    else (((pixel x ((j - 4) / 9)).testBit ((j - 4) % 9 - 1)).toNat : F)
  else 0

theorem one (x : Spec.Image) (s : Spec.State) : assignment x s 97634 = 1 := by
  simp [assignment]

theorem value (x : Spec.Image) (s : Spec.State) (n : Nat) (hn : n < 7680) :
    assignment x s (valueWire n) = (pixel x n : F) := by
  have hb : valueWire n < 69124 := by unfold valueWire; omega
  have h4 : 4 ≤ valueWire n := by unfold valueWire; omega
  have he : valueWire n - 4 = 9 * n := by unfold valueWire; omega
  simp [assignment, show valueWire n ≠ 97634 by omega,
    show valueWire n ≠ 0 by omega, show valueWire n ≠ 1 by omega,
    show valueWire n ≠ 2 by omega, show valueWire n ≠ 3 by omega, hb, he]

theorem bit (x : Spec.Image) (s : Spec.State) (n : Nat) (hn : n < 7680) (i : Fin 8) :
    assignment x s (bitWire n i) = (((pixel x n).testBit i.val).toNat : F) := by
  have hb : bitWire n i < 69124 := by unfold bitWire valueWire; omega
  have h4 : 4 ≤ bitWire n i := by unfold bitWire valueWire; omega
  have hd : (bitWire n i - 4) / 9 = n := by unfold bitWire valueWire; omega
  have hm : (bitWire n i - 4) % 9 = i.val + 1 := by unfold bitWire valueWire; omega
  simp [assignment, show bitWire n i ≠ 97634 by omega,
    show bitWire n i ≠ 0 by omega, show bitWire n i ≠ 1 by omega,
    show bitWire n i ≠ 2 by omega, show bitWire n i ≠ 3 by omega, hb, hd, hm]

theorem incoming (x : Spec.Image) (s : Spec.State) :
    Target.incomingState (assignment x s) = s := by
  change (⟨assignment x s 0, assignment x s 1, assignment x s 2,
    assignment x s 3⟩ : Spec.State) = s
  simp [assignment]

theorem matchesImage (x : Spec.Image) (s : Spec.State) : Target.matchesImage (assignment x s) x := by
  intro r hr c hc ch hch
  have hn : (r * 160 + c) * 3 + ch < 7680 := by omega
  rw [pixel_wire _ hn, value x s _ hn, pixel_index x r c ch hr hc hch]

theorem prefix_sat (x : Spec.Image) (s : Spec.State) (hx : Spec.ValidImage x) :
    ∀ row ∈ Exported.rows.take 69120, row.Sat (assignment x s) :=
  complete_of_values (assignment x s) (pixel x) (one x s)
    (pixel_lt x hx) (value x s) (bit x s)

/-- All initial wires that the actual arithmetic program must preserve. -/
abbrev Initial := StraightLine.Frontier 69124 97634

theorem preserved_one (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (ha : StraightLine.Agree Initial w (assignment x s)) : w 97634 = 1 :=
  (ha 97634 (Or.inr rfl)).trans (one x s)

theorem preserved_value (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (ha : StraightLine.Agree Initial w (assignment x s)) (n : Nat) (hn : n < 7680) :
    w (valueWire n) = (pixel x n : F) := by
  exact (ha _ (Or.inl (by unfold valueWire; omega))).trans (value x s n hn)

theorem preserved_bit (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (ha : StraightLine.Agree Initial w (assignment x s)) (n : Nat) (hn : n < 7680)
    (i : Fin 8) : w (bitWire n i) = (((pixel x n).testBit i.val).toNat : F) := by
  exact (ha _ (Or.inl (by unfold bitWire valueWire; omega))).trans (bit x s n hn i)

theorem preserved_incoming (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (ha : StraightLine.Agree Initial w (assignment x s)) : Target.incomingState w = s := by
  rw [← incoming x s]
  change (⟨w 0, w 1, w 2, w 3⟩ : Spec.State) =
    ⟨assignment x s 0, assignment x s 1, assignment x s 2, assignment x s 3⟩
  congr 1 <;> exact ha _ (Or.inl (by decide))

theorem preserved_matches (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (ha : StraightLine.Agree Initial w (assignment x s)) : Target.matchesImage w x := by
  intro r hr c hc ch hch
  have hn : (r * 160 + c) * 3 + ch < 7680 := by omega
  rw [pixel_wire _ hn, preserved_value x s w ha _ hn, pixel_index x r c ch hr hc hch]

theorem preserved_prefix_sat (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (hx : Spec.ValidImage x) (ha : StraightLine.Agree Initial w (assignment x s)) :
    ∀ row ∈ Exported.rows.take 69120, row.Sat w :=
  complete_of_values w (pixel x) (preserved_one x s w ha)
    (pixel_lt x hx) (preserved_value x s w ha) (preserved_bit x s w ha)

/-- Completeness interface: executing a well-formed suffix preserves all byte
constraints and the exact exported image/state mappings. -/
theorem preserved (x : Spec.Image) (s : Spec.State) (w : Assignment)
    (hx : Spec.ValidImage x) (ha : StraightLine.Agree Initial w (assignment x s)) :
    w 97634 = 1 ∧ Target.matchesImage w x ∧ Target.incomingState w = s ∧
      ∀ row ∈ Exported.rows.take 69120, row.Sat w :=
  ⟨preserved_one x s w ha, preserved_matches x s w ha,
    preserved_incoming x s w ha, preserved_prefix_sat x s w hx ha⟩

end CircuitCorrectness.Seed
