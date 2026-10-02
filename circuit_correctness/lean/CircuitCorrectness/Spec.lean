import CircuitCorrectness.Parameters
import CircuitCorrectness.Gadgets

set_option maxRecDepth 1024
set_option maxHeartbeats 200000
namespace CircuitCorrectness.Spec

structure PoseidonParameters where
  arity : Nat
  fullRounds : Nat
  partialRounds : Nat
  keys : Array Nat
  mds : Array (Array Nat)
  preSparse : Array (Array Nat)
  sparseW : Array (Array Nat)
  sparseV : Array (Array Nat)

def params2 : PoseidonParameters :=
  ⟨2, 8, 55, Parameters.keys2, Parameters.mds2, Parameters.preSparse2,
   Parameters.sparseW2, Parameters.sparseV2⟩
def params8 : PoseidonParameters :=
  ⟨8, 8, 57, Parameters.keys8, Parameters.mds8, Parameters.preSparse8,
   Parameters.sparseW8, Parameters.sparseV8⟩

def domainTag (arity domain : Nat) : Nat :=
  let b := 2^128 - 159
  ((2^31 + arity) * b + b^2 + domain * b^3) % 2^128

def denseMix (matrix : Array (Array Nat)) (xs : Array F) : Array F :=
  (List.range xs.size).toArray.map fun j =>
    ((List.range xs.size).map fun i => xs[i]! * (((matrix[i]!)[j]! : Nat) : F)).sum

def sparseMix (w v : Array Nat) (xs : Array F) : Array F :=
  (List.range xs.size).toArray.map fun j =>
    if j = 0 then ((List.range xs.size).map fun i => xs[i]! * ((w[i]! : Nat) : F)).sum
    else xs[j]! + xs[0]! * ((v[j-1]! : Nat) : F)

/-- Functional optimized round schedule. Constants are ordinary fixed field values. -/
def poseidonRound (p : PoseidonParameters) (state : Array F × Nat) (round : Nat) :
    Array F × Nat :=
  let (xs, offset) := state
  let half := p.fullRounds / 2
  let full := round < half ∨ half + p.partialRounds ≤ round
  let terminal := round + 1 = p.fullRounds + p.partialRounds
  let width := p.arity + 1
  let start := if round = 0 then offset + width else offset
  let transformed := (List.range width).toArray.map fun i =>
    if full ∨ i = 0 then
      let pre : F := if round = 0 then ((p.keys[offset+i]! : Nat) : F) else 0
      let post : F := if terminal then 0 else ((p.keys[start+i]! : Nat) : F)
      (xs[i]! + pre)^5 + post
    else xs[i]!
  let nextOffset := if terminal then start else start + (if full then width else 1)
  let mixed := if round = half - 1 then denseMix p.preSparse transformed
    else if half - 1 < round ∧ round < half + p.partialRounds then
      sparseMix p.sparseW[round-half]! p.sparseV[round-half]! transformed
    else denseMix p.mds transformed
  (mixed, nextOffset)

/-- One full-rate absorb followed by one squeeze; there is no message padding round. -/
def hash (p : PoseidonParameters) (domain : Nat) (input : Array F) : F :=
  let initial := #[(domainTag p.arity domain : F)] ++ input
  let final := (List.range (p.fullRounds+p.partialRounds)).foldl
    (poseidonRound p) (initial, 0)
  final.1[1]!

def hash2 (a b : F) : F := hash params2 0x50414952 #[a,b]
def hash8 (xs : Array F) : F := hash params8 0x48415348 xs

/-- Only indices r<16, c<160, ch<3 are read. -/
abbrev Image := Nat → Nat → Nat → Nat
def ValidImage (x : Image) : Prop :=
  ∀ r < 16, ∀ c < 160, ∀ ch < 3, x r c ch < 256

def matrix (r c : Nat) : Int := (Parameters.dct[r]!)[c]!
def multiplier (ch r c : Nat) : Nat := ((Parameters.multipliers[ch]!)[r]!)[c]!
def retained (ch r c : Nat) : Bool := multiplier ch r c != 0

/-- The specification is the direct double sum, with the right-hand transpose explicit. -/
def coefficient (x : Image) (r c ch : Nat) : Int :=
  let br := r / 8 * 8
  let bc := c / 8 * 8
  (multiplier ch (r%8) (c%8) : Int) *
    ((List.range 8).map fun i =>
      ((List.range 8).map fun j =>
        matrix (r%8) i * ((x (br+i) (bc+j) ch : Int)-128) * matrix (c%8) j).sum).sum

def retainedRow (r : Nat) : List (Nat × Nat) :=
  (List.range 8).flatMap fun c =>
    ((List.range 3).filter fun ch => retained ch r c).map fun ch => (c,ch)

def retainedCoordinates : List (Nat × Nat × Nat) :=
  (List.range 16).flatMap fun r =>
    (List.range 20).flatMap fun b =>
      (retainedRow (r%8)).map fun (c,ch) => (r,8*b+c,ch)

def coefficients (x : Image) : List F :=
  retainedCoordinates.map fun (r,c,ch) => (coefficient x r c ch : F)

def packedPixel (x : Image) (r c : Nat) : Nat :=
  x r c 0 + 2^8 * x r c 1 + 2^16 * x r c 2

def packedChunk (x : Image) (r chunk : Nat) : Nat :=
  ((List.range 10).map fun j => packedPixel x r (10*chunk+j) * 2^(24*j)).sum

def rowHash (x : Image) (r : Nat) : F :=
  let chunks := (List.range 16).toArray.map fun c => (packedChunk x r c : F)
  hash2 (hash8 (chunks.extract 0 8)) (hash8 (chunks.extract 8 16))

def stepDigest (x : Image) : F :=
  let rows := (List.range 16).toArray.map (rowHash x)
  hash2 (hash8 (rows.extract 0 8)) (hash8 (rows.extract 8 16))

structure State where
  h : F
  a : F
  r : F
  t : F

def step (x : Image) (s : State) : State :=
  ⟨hash2 s.h (stepDigest x), Gadgets.horner s.a s.r (coefficients x), s.r, s.t+1⟩


end CircuitCorrectness.Spec
