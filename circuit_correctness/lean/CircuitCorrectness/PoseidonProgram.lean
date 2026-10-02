import CircuitCorrectness.Affine
import CircuitCorrectness.StraightLine
import CircuitCorrectness.Spec

namespace CircuitCorrectness.PoseidonProgram
open StraightLine
abbrev LC := LinearCombination

def evalState (w : Assignment) (xs : Array LC) : Array F := xs.map (evalLC w)

/-- Three multiplication assignments implement one quintic, including the
optimized constants before and after the fifth power. -/
def sbox (one next : Nat) (x : LC) (pre post : Nat) : Program :=
  let base := Affine.add x (Affine.constant one pre)
  [⟨next, base, base, []⟩,
   ⟨next+1, Affine.wire next, Affine.wire next, []⟩,
   ⟨next+2, base, Affine.wire (next+1), Affine.constant one post⟩]

theorem sbox_sound (w : Assignment) (one next : Nat) (x : LC) (pre post : Nat)
    (h1 : w one = 1) (hs : Satisfies (sbox one next x pre post) w) :
    w (next+2) = (evalLC w x + (pre : F))^5 + (post : F) := by
  simp only [Satisfies, sbox, List.mem_cons, List.not_mem_nil, or_false,
    forall_eq_or_imp] at hs
  rcases hs with ⟨h2, h4, h5⟩
  change w next = _ at h2
  change w (next+1) = _ at h4
  have h5 := h5 _ rfl
  change w (next+2) = _ at h5
  simp only [Instruction.value, Affine.eval_add, Affine.eval_constant,
    Affine.eval_variable, Affine.eval_nil, h1, mul_one, add_zero] at h2 h4 h5
  rw [h5, h4, h2]
  ring

def denseMix (matrix : Array (Array Nat)) (xs : Array LC) : Array LC :=
  (List.range xs.size).toArray.map fun j =>
    Affine.sum ((List.range xs.size).map fun i => Affine.scale (matrix[i]!)[j]! xs[i]!)

def sparseMix (one : Nat) (vW vV : Array Nat) (xs : Array LC) : Array LC :=
  (List.range xs.size).toArray.map fun j =>
    if j = 0 then Affine.sum ((List.range xs.size).map fun i => Affine.scale vW[i]! xs[i]!)
    else Affine.add xs[j]! (Affine.scale vV[j-1]! xs[0]!)


@[simp] theorem evalState_size (w : Assignment) (xs : Array LC) :
    (evalState w xs).size = xs.size := by simp [evalState]

@[simp] theorem evalState_get (w : Assignment) (xs : Array LC) (i : Nat) :
    (evalState w xs)[i]! = evalLC w xs[i]! := by
  by_cases h : i < xs.size
  · simp [evalState, getElem!_pos, h]
  · simp only [evalState, Array.size_map, getElem!_neg, h, not_false_eq_true]
    rfl

@[simp] theorem evalMap_get (w : Assignment) (xs : Array LC) (i : Nat) :
    (xs.map (evalLC w))[i]! = evalLC w xs[i]! := evalState_get w xs i

theorem eval_denseMix (w : Assignment) (m : Array (Array Nat)) (xs : Array LC) :
    evalState w (denseMix m xs) = Spec.denseMix m (evalState w xs) := by
  simp [evalState, denseMix, Spec.denseMix, List.map_map, Array.map_map,
    Function.comp_def, mul_comm]

theorem eval_sparseMix (w : Assignment) (one : Nat) (a b : Array Nat) (xs : Array LC) :
    evalState w (sparseMix one a b xs) = Spec.sparseMix a b (evalState w xs) := by
  simp only [evalState, sparseMix, Spec.sparseMix, Array.size_map, Array.map_map]
  congr 1
  funext j
  dsimp only [Function.comp_def]
  split <;> simp [List.map_map, Function.comp_def, mul_comm]

structure State where
  values : Array LC
  offset : Nat
  next : Nat
  deriving DecidableEq, Repr

def isFull (p : Spec.PoseidonParameters) (r : Nat) : Prop :=
  r < p.fullRounds/2 ∨ p.fullRounds/2 + p.partialRounds ≤ r
instance (p : Spec.PoseidonParameters) (r : Nat) : Decidable (isFull p r) :=
  inferInstanceAs (Decidable (_ ∨ _))
def isTerminal (p : Spec.PoseidonParameters) (r : Nat) : Bool :=
  r+1 == p.fullRounds+p.partialRounds
def keyStart (p : Spec.PoseidonParameters) (st : State) (r : Nat) : Nat :=
  if r=0 then st.offset+p.arity+1 else st.offset

def preKey (p : Spec.PoseidonParameters) (st : State) (r i : Nat) : Nat :=
  if r=0 then p.keys[st.offset+i]! else 0
def postKey (p : Spec.PoseidonParameters) (st : State) (r i : Nat) : Nat :=
  if isTerminal p r then 0 else p.keys[keyStart p st r+i]!

def roundValues (p : Spec.PoseidonParameters) (st : State) (r : Nat) : Array LC :=
  (List.range (p.arity+1)).toArray.map fun i =>
    if isFull p r ∨ i=0 then Affine.wire (st.next+3*i+2) else st.values[i]!

def roundProgram (p : Spec.PoseidonParameters) (one : Nat) (st : State) (r : Nat) : Program :=
  (List.range (p.arity+1)).flatMap fun i =>
    if isFull p r ∨ i=0 then
      sbox one (st.next+3*i) st.values[i]! (preKey p st r i) (postKey p st r i)
    else []

/-- Explicit wire-level round; Spec.poseidonRound remains independent. -/
def round (p : Spec.PoseidonParameters) (one : Nat) (st : State) (r : Nat) :
    State × Program :=
  let half := p.fullRounds / 2
  let transformed := roundValues p st r
  let mixed := if r = half - 1 then denseMix p.preSparse transformed
    else if half - 1 < r ∧ r < half + p.partialRounds then
      sparseMix one p.sparseW[r-half]! p.sparseV[r-half]! transformed
    else denseMix p.mds transformed
  (⟨mixed, if isTerminal p r then keyStart p st r else
      keyStart p st r + (if isFull p r then p.arity+1 else 1),
      st.next + 3*(if isFull p r then p.arity+1 else 1)⟩,
    roundProgram p one st r)

theorem roundValues_sound (w : Assignment) (p : Spec.PoseidonParameters)
    (one : Nat) (st : State) (r : Nat) (h1 : w one=1)
    (hs : Satisfies (roundProgram p one st r) w) :
    evalState w (roundValues p st r) =
      (List.range (p.arity+1)).toArray.map (fun i =>
        if isFull p r ∨ i=0 then
          ((evalState w st.values)[i]! + (preKey p st r i : F))^5 +
            (postKey p st r i : F)
        else (evalState w st.values)[i]!) := by
  simp only [roundValues, evalState, Array.map_map]
  apply Array.map_congr_left
  intro i hi
  have hi' : i < p.arity+1 := by simpa using hi
  by_cases hc : isFull p r ∨ i=0
  · have hs' : Satisfies (sbox one (st.next+3*i) st.values[i]!
        (preKey p st r i) (postKey p st r i)) w := by
      intro op ho
      apply hs op
      apply List.mem_flatMap.mpr
      exact ⟨i, List.mem_range.mpr hi', by simpa [hc] using ho⟩
    simpa [hc, Function.comp_def] using
      sbox_sound w one (st.next+3*i) st.values[i]!
        (preKey p st r i) (postKey p st r i) h1 hs'
  · simp [hc, Function.comp_def]

theorem round_sound (w : Assignment) (p : Spec.PoseidonParameters)
    (one : Nat) (st : State) (r : Nat) (h1 : w one=1)
    (hs : Satisfies (round p one st r).2 w) :
    (evalState w (round p one st r).1.values, (round p one st r).1.offset) =
      Spec.poseidonRound p (evalState w st.values, st.offset) r := by
  have ht := roundValues_sound w p one st r h1 hs
  simp only [round, Spec.poseidonRound]
  simp only [apply_ite (evalState w), eval_denseMix, eval_sparseMix, ht]
  simp [preKey, postKey, keyStart, isFull, isTerminal, Nat.add_assoc]


def rounds (p : Spec.PoseidonParameters) (one : Nat) :
    List Nat → State → State × Program
  | [], st => (st, [])
  | r::rs, st =>
    let first := round p one st r
    let rest := rounds p one rs first.1
    (rest.1, first.2 ++ rest.2)

theorem rounds_sound (w : Assignment) (p : Spec.PoseidonParameters)
    (one : Nat) (rs : List Nat) (st : State) (h1 : w one=1)
    (hs : Satisfies (rounds p one rs st).2 w) :
    (evalState w (rounds p one rs st).1.values, (rounds p one rs st).1.offset) =
      rs.foldl (Spec.poseidonRound p) (evalState w st.values, st.offset) := by
  induction rs generalizing st with
  | nil => rfl
  | cons r rs ih =>
    have hh := (satisfies_append _ _ w).mp hs
    have hf := round_sound w p one st r h1 hh.1
    have ht := ih (round p one st r).1 hh.2
    simpa only [rounds, List.foldl_cons, hf] using ht

structure CompiledHash where
  output : Nat
  program : Program

def compileHash (p : Spec.PoseidonParameters) (one next domain : Nat)
    (input : Array LC) : CompiledHash :=
  let initial : State := ⟨#[Affine.constant one (Spec.domainTag p.arity domain)] ++ input, 0, next⟩
  let result := rounds p one (List.range (p.fullRounds+p.partialRounds)) initial
  ⟨result.1.next, result.2 ++
    [⟨result.1.next, result.1.values[1]!, Affine.constant one 1, []⟩]⟩

theorem hash_sound (w : Assignment) (p : Spec.PoseidonParameters)
    (one next domain : Nat) (input : Array LC) (h1 : w one=1)
    (hs : Satisfies (compileHash p one next domain input).program w) :
    w (compileHash p one next domain input).output =
      Spec.hash p domain (evalState w input) := by
  let initial : State := ⟨#[Affine.constant one (Spec.domainTag p.arity domain)] ++ input, 0, next⟩
  let result := rounds p one (List.range (p.fullRounds+p.partialRounds)) initial
  change Satisfies (result.2 ++ [⟨result.1.next, result.1.values[1]!,
    Affine.constant one 1, []⟩]) w at hs
  have hh := (satisfies_append _ _ w).mp hs
  have ho := hh.2 _ (List.mem_cons_self)
  change w result.1.next = _ at ho
  simp only [Instruction.value, Affine.eval_constant, Nat.cast_one, one_mul,
    h1, mul_one, Affine.eval_nil, add_zero] at ho
  have hr := rounds_sound w p one (List.range (p.fullRounds+p.partialRounds)) initial h1 hh.1
  have hv := congrArg (fun t : Array F × Nat => t.1[1]!) hr
  dsimp only at hv
  rw [evalState_get] at hv
  change w result.1.next = _
  rw [ho, hv]
  simp [Spec.hash, initial, evalState, Array.map_append, h1]

def sboxCount (p : Spec.PoseidonParameters) (rs : List Nat) : Nat :=
  (rs.map fun r => if isFull p r then p.arity+1 else 1).sum

theorem rounds_next (p : Spec.PoseidonParameters) (one : Nat) (rs : List Nat) (st : State) :
    (rounds p one rs st).1.next = st.next + 3*sboxCount p rs := by
  induction rs generalizing st with
  | nil => simp [rounds, sboxCount]
  | cons r rs ih =>
    simp only [rounds, ih, round, sboxCount, List.map_cons, List.sum_cons]
    omega

theorem hash_output (p : Spec.PoseidonParameters) (one next domain : Nat) (input : Array LC) :
    (compileHash p one next domain input).output =
      next+3*sboxCount p (List.range (p.fullRounds+p.partialRounds)) :=
  rounds_next p one _ _

@[simp] theorem hash8_output (one next domain : Nat) (input : Array LC) :
    (compileHash Spec.params8 one next domain input).output = next+387 := by
  rw [hash_output]
  have hc : sboxCount Spec.params8 (List.range (8+57)) = 129 := by decide
  change next+3*sboxCount Spec.params8 (List.range (8+57)) = _
  rw [hc]

@[simp] theorem hash2_output (one next domain : Nat) (input : Array LC) :
    (compileHash Spec.params2 one next domain input).output = next+237 := by
  rw [hash_output]
  have hc : sboxCount Spec.params2 (List.range (8+55)) = 79 := by decide
  change next+3*sboxCount Spec.params2 (List.range (8+55)) = _
  rw [hc]

end CircuitCorrectness.PoseidonProgram
