import CircuitCorrectness.PoseidonProgram
import CircuitCorrectness.DctSpec

namespace CircuitCorrectness.HashProgram
open StraightLine
open PoseidonProgram (evalState compileHash)

def one : Nat := 97634
def pixelWire (r c ch : Nat) : Nat := 4+9*((r*160+c)*3+ch)
def rowStart (r : Nat) : Nat := 77324+1191*r

def linear (dst : Nat) (x : LinearCombination) : Instruction :=
  ⟨dst,x,Affine.constant one 1,[]⟩

def pixelOp (r c : Nat) : Instruction := linear (rowStart r+c)
  (Affine.sum ((List.range 3).map fun ch =>
    Affine.scale (2^(8*ch)) (Affine.wire (pixelWire r c ch))))

def chunkOp (r c : Nat) : Instruction := linear (rowStart r+160+c)
  (Affine.sum ((List.range 10).map fun j =>
    Affine.scale (2^(24*j)) (Affine.wire (rowStart r+10*c+j))))

def inputs (base n : Nat) : Array LinearCombination :=
  (List.range n).toArray.map fun i => Affine.wire (base+i)

def rowLeft (r : Nat) := compileHash Spec.params8 one (rowStart r+176) 0x48415348
  (inputs (rowStart r+160) 8)
def rowRight (r : Nat) := compileHash Spec.params8 one (rowStart r+564) 0x48415348
  (inputs (rowStart r+168) 8)
def rowCombine (r : Nat) := compileHash Spec.params2 one (rowStart r+952) 0x50414952
  #[Affine.wire (rowStart r+563), Affine.wire (rowStart r+951)]

def rowProgram (r : Nat) : Program :=
  (List.range 160).map (pixelOp r) ++ (List.range 16).map (chunkOp r) ++
    (rowLeft r).program ++ (rowRight r).program ++ (rowCombine r).program ++
    [linear (rowStart r+1190) (Affine.wire (rowStart r+1189))]

def Matches (w : Assignment) (x : Spec.Image) : Prop :=
  ∀ r < 16, ∀ c < 160, ∀ ch < 3, w (pixelWire r c ch) = (x r c ch : F)

theorem linear_sound (w : Assignment) (dst : Nat) (x : LinearCombination)
    (h1 : w one=1) (hs : (linear dst x).Sat w) : w dst = evalLC w x := by
  simpa [linear, Instruction.Sat, Instruction.value, h1] using hs

theorem pixel_sound (w : Assignment) (x : Spec.Image) (r c : Nat)
    (hr : r<16) (hc : c<160) (hm : Matches w x) (h1 : w one=1)
    (hs : (pixelOp r c).Sat w) : w (rowStart r+c) = (Spec.packedPixel x r c : F) := by
  have hl := linear_sound w _ _ h1 hs
  rw [hl]
  simp only [Affine.eval_sum, List.map_map, Function.comp_def, Affine.eval_scale,
    Affine.eval_variable]
  simp [List.range_succ, hm r hr c hc, Spec.packedPixel, Nat.cast_add, Nat.cast_mul]
  ring

theorem chunk_sound (w : Assignment) (x : Spec.Image) (r c : Nat)
    (hc : c<16) (hp : ∀ i<160, w (rowStart r+i) = (Spec.packedPixel x r i : F))
    (h1 : w one=1) (hs : (chunkOp r c).Sat w) :
    w (rowStart r+160+c) = (Spec.packedChunk x r c : F) := by
  have hl := linear_sound w _ _ h1 hs
  rw [hl]
  simp only [Affine.eval_sum, List.map_map, Function.comp_def, Affine.eval_scale,
    Affine.eval_variable]
  rw [DctSpec.cast_packedChunk]
  apply congrArg List.sum
  apply List.map_congr_left
  intro j hj
  have hj' : j<10 := List.mem_range.mp hj
  have hi : 10*c+j<160 := by omega
  rw [show rowStart r+10*c+j = rowStart r+(10*c+j) by omega, hp _ hi]
  simp [Nat.cast_pow, mul_comm]

theorem first_half {α : Type} (f : Nat → α) :
    ((List.range 16).toArray.map f).extract 0 8 = (List.range 8).toArray.map f := by
  apply Array.ext
  · simp
  · intro i hi hi'
    simp

theorem second_half {α : Type} (f : Nat → α) :
    ((List.range 16).toArray.map f).extract 8 16 =
      (List.range 8).toArray.map (fun i => f (8+i)) := by
  apply Array.ext
  · simp
  · intro i hi hi'
    simp

theorem eval_inputs (w : Assignment) (base n : Nat) :
    evalState w (inputs base n) = (List.range n).toArray.map (fun i => w (base+i)) := by
  simp [inputs, evalState, Array.map_map, Function.comp_def]

theorem eval_pair (w : Assignment) (i j : Nat) :
    evalState w #[Affine.wire i, Affine.wire j] = #[w i,w j] := by
  apply Array.ext
  · simp [evalState]
  · intro k hk hk'
    have hk2 : k<2 := by simpa using hk'
    interval_cases k <;> simp [evalState]

theorem rowProgram_sound (w : Assignment) (x : Spec.Image) (r : Nat)
    (hr : r<16) (hm : Matches w x) (h1 : w one=1)
    (hs : Satisfies (rowProgram r) w) :
    w (rowStart r+1189) = Spec.rowHash x r := by
  simp only [rowProgram, satisfies_append] at hs
  rcases hs with ⟨⟨⟨⟨⟨hps,hcs⟩,hls⟩,hrs⟩,hhs⟩,_⟩
  have hp : ∀ i<160, w (rowStart r+i) = (Spec.packedPixel x r i : F) := by
    intro i hi
    exact pixel_sound w x r i hr hi hm h1 (hps _ (List.mem_map.mpr ⟨i,List.mem_range.mpr hi,rfl⟩))
  have hc : ∀ i<16, w (rowStart r+160+i) = (Spec.packedChunk x r i : F) := by
    intro i hi
    exact chunk_sound w x r i hi hp h1 (hcs _ (List.mem_map.mpr ⟨i,List.mem_range.mpr hi,rfl⟩))
  have hl := PoseidonProgram.hash_sound w Spec.params8 one (rowStart r+176)
    0x48415348 (inputs (rowStart r+160) 8) h1 hls
  have hh := PoseidonProgram.hash_sound w Spec.params8 one (rowStart r+564)
    0x48415348 (inputs (rowStart r+168) 8) h1 hrs
  have hf := PoseidonProgram.hash_sound w Spec.params2 one (rowStart r+952)
    0x50414952 #[Affine.wire (rowStart r+563), Affine.wire (rowStart r+951)] h1 hhs
  simp only [PoseidonProgram.hash8_output, eval_inputs] at hl hh
  rw [show rowStart r+176+387 = rowStart r+563 by omega] at hl
  rw [show rowStart r+564+387 = rowStart r+951 by omega] at hh
  simp only [PoseidonProgram.hash2_output] at hf
  rw [show rowStart r+952+237 = rowStart r+1189 by omega] at hf
  have hleft : (List.range 8).toArray.map (fun i => w (rowStart r+160+i)) =
      (List.range 8).toArray.map (fun i => (Spec.packedChunk x r i : F)) := by
    apply Array.map_congr_left
    intro i hi
    apply hc
    have hi' : i<8 := by simpa using hi
    omega
  have hright : (List.range 8).toArray.map (fun i => w (rowStart r+168+i)) =
      (List.range 8).toArray.map (fun i => (Spec.packedChunk x r (8+i) : F)) := by
    apply Array.map_congr_left
    intro i hi
    have hi' : i<8 := by simpa using hi
    have hi'' : 8+i<16 := by omega
    rw [show rowStart r+168+i = rowStart r+160+(8+i) by omega]
    exact hc (8+i) hi'' 
  rw [hleft] at hl
  rw [hright] at hh
  rw [eval_pair] at hf
  rw [hl, hh] at hf
  simpa only [Spec.rowHash, first_half, second_half, Spec.hash2, Spec.hash8] using hf

def digestInputs (base : Nat) : Array LinearCombination :=
  (List.range 8).toArray.map fun i => Affine.wire (rowStart (base+i)+1189)
def digestLeft := compileHash Spec.params8 one 96380 0x48415348 (digestInputs 0)
def digestRight := compileHash Spec.params8 one 96768 0x48415348 (digestInputs 8)
def digestCombine := compileHash Spec.params2 one 97156 0x50414952
  #[Affine.wire 96767,Affine.wire 97155]
def chain := compileHash Spec.params2 one 97395 0x50414952
  #[Affine.wire 0,Affine.wire 97393]

def program : Program :=
  (List.range 16).flatMap rowProgram ++ digestLeft.program ++ digestRight.program ++
    digestCombine.program ++ [linear 97394 (Affine.wire 97393)] ++ chain.program

theorem digest_inputs_sound (w : Assignment) (x : Spec.Image) (base : Nat)
    (hh : ∀ i<8, w (rowStart (base+i)+1189) = Spec.rowHash x (base+i)) :
    evalState w (digestInputs base) =
      (List.range 8).toArray.map (fun i => Spec.rowHash x (base+i)) := by
  simp only [digestInputs, evalState, Array.map_map]
  apply Array.map_congr_left
  intro i hi
  simpa only [Function.comp_def, Affine.eval_variable] using hh i (by simpa using hi)

theorem program_sound (w : Assignment) (x : Spec.Image) (hm : Matches w x)
    (h1 : w one=1) (hs : Satisfies program w) :
    w 97632 = Spec.hash2 (w 0) (Spec.stepDigest x) := by
  simp only [program, satisfies_append] at hs
  rcases hs with ⟨⟨⟨⟨⟨hrs, hls⟩,hrrs⟩,hcs⟩,_⟩,hchain⟩
  have hr : ∀ r<16, w (rowStart r+1189) = Spec.rowHash x r := by
    intro r hri
    apply rowProgram_sound w x r hri hm h1
    intro op hop
    exact hrs op (List.mem_flatMap.mpr ⟨r,List.mem_range.mpr hri,hop⟩)
  have hl := PoseidonProgram.hash_sound w Spec.params8 one 96380 0x48415348
    (digestInputs 0) h1 hls
  have hh := PoseidonProgram.hash_sound w Spec.params8 one 96768 0x48415348
    (digestInputs 8) h1 hrrs
  have hc := PoseidonProgram.hash_sound w Spec.params2 one 97156 0x50414952
    #[Affine.wire 96767,Affine.wire 97155] h1 hcs
  have hf := PoseidonProgram.hash_sound w Spec.params2 one 97395 0x50414952
    #[Affine.wire 0,Affine.wire 97393] h1 hchain
  simp only [PoseidonProgram.hash8_output] at hl hh
  simp only [PoseidonProgram.hash2_output, eval_pair] at hc hf
  have hil := digest_inputs_sound w x 0 (by
    intro i hi
    exact hr (0+i) (by omega))
  have hir := digest_inputs_sound w x 8 (by
    intro i hi
    exact hr (8+i) (by omega))
  rw [hil] at hl
  rw [hir] at hh
  rw [hl, hh] at hc
  rw [hc] at hf
  simpa only [Spec.stepDigest, Spec.hash2, Spec.hash8, first_half, second_half,
    Nat.zero_add] using hf

end CircuitCorrectness.HashProgram
