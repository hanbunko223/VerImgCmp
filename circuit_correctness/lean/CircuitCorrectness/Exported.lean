import CircuitCorrectness.ExportedData

set_option maxRecDepth 10000
namespace CircuitCorrectness.Exported

def radix : Nat := 2^32

def decodeTerms : Nat → Nat → LinearCombination × Nat
  | 0, data => ([], data)
  | n+1, data =>
    let word := data % radix
    let term := (word % 97635, ExportedData.coefficientPool[word / 97635]!)
    let tail := decodeTerms n (data / radix)
    (term :: tail.1, tail.2)

def decodeLC (data : Nat) : LinearCombination × Nat :=
  decodeTerms (data % radix) (data / radix)

def decodeRow (data : Nat) : Row × Nat :=
  let a := decodeLC data
  let b := decodeLC a.2
  let c := decodeLC b.2
  (⟨a.1,b.1,c.1⟩,c.2)

def decodeRows : Nat → Nat → List Row
  | 0, _ => []
  | n+1, data =>
    let row := decodeRow data
    row.1 :: decodeRows n row.2

def rows : List Row := ExportedData.chunks.toList.flatMap fun (n,data) => decodeRows n data
def circuit : R1CS := ⟨97635,97634,rows⟩

theorem decodeRows_length (n data : Nat) : (decodeRows n data).length = n := by
  induction n generalizing data with
  | zero => rfl
  | succ n ih => simp only [decodeRows, List.length_cons, ih]

/-- The literal data contains all 97,630 rows, including constraints unused by an output. -/
theorem row_count : rows.length = 97630 := by
  simp only [rows, List.length_flatMap, decodeRows_length]
  decide

/-- Challenge preservation is an alias in the actual exported interface. -/
theorem challenge_alias : ExportedData.outgoing[2]! = ExportedData.incoming[2]! := by decide

end CircuitCorrectness.Exported
