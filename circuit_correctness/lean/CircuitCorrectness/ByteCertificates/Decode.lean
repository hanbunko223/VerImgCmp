import CircuitCorrectness.ByteCertificates.Base
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace CircuitCorrectness.ConcreteBytes
namespace Codes

def terms : Nat → Nat → LinearCombination × Nat
  | 0, data => ([], data)
  | n+1, data =>
    let word := data % Exported.radix
    let term := (word % 97635, word / 97635)
    let tail := terms n (data / Exported.radix)
    (term :: tail.1, tail.2)

def lc (data : Nat) : LinearCombination × Nat :=
  terms (data % Exported.radix) (data / Exported.radix)
def row (data : Nat) : Row × Nat :=
  let a := lc data
  let b := lc a.2
  let c := lc b.2
  (⟨a.1,b.1,c.1⟩,c.2)
def rows : Nat → Nat → List Row
  | 0, _ => []
  | n+1, data =>
    let r := row data
    r.1 :: rows n r.2

def expandLC (xs : LinearCombination) : LinearCombination :=
  xs.map fun (i,c) => (i,ExportedData.coefficientPool[c]!)
def expandRow (r : Row) : Row := ⟨expandLC r.a,expandLC r.b,expandLC r.c⟩

theorem terms_expand (n data : Nat) :
    Exported.decodeTerms n data = (expandLC (terms n data).1, (terms n data).2) := by
  induction n generalizing data with
  | zero => rfl
  | succ n ih => simp only [Exported.decodeTerms, terms, ih, expandLC, List.map_cons]

theorem lc_expand (data : Nat) :
    Exported.decodeLC data = (expandLC (lc data).1, (lc data).2) := terms_expand ..

theorem row_expand (data : Nat) :
    Exported.decodeRow data = (expandRow (row data).1, (row data).2) := by
  simp only [Exported.decodeRow, lc_expand, row, expandRow]

theorem rows_expand (n data : Nat) :
    Exported.decodeRows n data = (rows n data).map expandRow := by
  induction n generalizing data with
  | zero => rfl
  | succ n ih => simp only [Exported.decodeRows, row_expand, rows, ih, List.map_cons]

def powerCode (i : Fin 8) : Nat := ![0,1,2,3,5,7,13,14] i

theorem powerCode_expand : ∀ i : Fin 8,
    ExportedData.coefficientPool[powerCode i]! = 2 ^ i.val := by decide

theorem one_expand : ExportedData.coefficientPool[0]! = 1 := by decide
theorem neg_one_expand : ExportedData.coefficientPool[1218]! = modulus - 1 := by decide

def byte (j : Nat) : Row :=
  let n := j / 9
  let k := j % 9
  let bit := valueWire n + 1 + k
  if k < 8 then
    ⟨[(bit, 1218), (97634, 0)], [(bit, 0)], []⟩
  else
    ⟨(valueWire n, 1218) :: List.ofFn (fun i : Fin 8 => (bitWire n i, powerCode i)),
      [(97634, 0)], []⟩

theorem byte_expand (j : Nat) : expandRow (byte j) = byteRow j := by
  unfold byte byteRow
  dsimp only
  split <;> simp only [expandRow, expandLC, List.map_cons, List.map_nil,
    one_expand, neg_one_expand, List.map_ofFn, Function.comp_def, powerCode_expand]

theorem certificate (c : Nat) (chunk : Nat × Nat)
    (h : rows chunk.1 chunk.2 = (List.range 128).map (fun i => byte (128*c+i))) :
    Exported.decodeRows chunk.1 chunk.2 = (List.range 128).map (fun i => byteRow (128*c+i)) := by
  rw [rows_expand, h, List.map_map]
  simp only [Function.comp_def, byte_expand]
end Codes
end CircuitCorrectness.ConcreteBytes
