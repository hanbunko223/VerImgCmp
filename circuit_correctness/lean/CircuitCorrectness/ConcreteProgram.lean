import CircuitCorrectness.ByteCertificates.Decode
import CircuitCorrectness.Exported
import CircuitCorrectness.StraightLine
import CircuitCorrectness.Affine
import Mathlib.Tactic.LinearCombination

namespace CircuitCorrectness.ConcreteProgram
open StraightLine

/-- Separates all occurrences of a wire. Coefficients are summed as naturals,
then interpreted in the field by evalLC. -/
def splitWire (dst : Nat) : LinearCombination → Nat × LinearCombination
  | [] => (0, [])
  | (i,c)::tail =>
    let (n, rest) := splitWire dst tail
    if i = dst then (c+n,rest) else (n,(i,c)::rest)

def negateLC (lc : LinearCombination) : LinearCombination :=
  Affine.scale (modulus-1) lc

theorem eval_splitWire (dst : Nat) (lc : LinearCombination) (w : Assignment) :
    evalLC w lc = ((splitWire dst lc).1 : F) * w dst + evalLC w (splitWire dst lc).2 := by
  induction lc with
  | nil => simp [splitWire, evalLC]
  | cons t tail ih =>
    rcases t with ⟨i,c⟩
    by_cases h : i = dst
    · subst i
      simpa [splitWire, evalLC, Nat.cast_add, add_mul, add_assoc] using
        congrArg (fun x => (c:F)*w dst+x) ih
    · simp only [splitWire, h, ↓reduceIte]
      change (c:F)*w i + evalLC w tail = _
      rw [ih]
      simp only [evalLC, List.map_cons, List.sum_cons]
      ring

theorem cast_neg_one : ((modulus-1 : Nat) : F) = -1 := by
  have hm : 1 ≤ modulus := by decide
  rw [Nat.cast_sub hm]
  simp [F]

theorem eval_negateLC (lc : LinearCombination) (w : Assignment) :
    evalLC w (negateLC lc) = -evalLC w lc := by
  simp [negateLC, Affine.eval_scale, cast_neg_one]

/-- No symbolic semantics are inferred from allocation names. This parser
recognizes algebraic row forms and verifies the destination coefficient. -/
def extract (one dst : Nat) (row : Row) : Option Instruction :=
  if row.b = [(one,1)] ∧ row.c = [] ∧ (splitWire dst row.a).1 = modulus-1 then
    some ⟨dst,(splitWire dst row.a).2,[(one,1)],[]⟩
  else if (splitWire dst row.c).1 = 1 then
    some ⟨dst,row.a,row.b,negateLC (splitWire dst row.c).2⟩
  else none

theorem extract_correct {one dst : Nat} {row : Row} {op : Instruction}
    (he : extract one dst row = some op) {w : Assignment} (h1 : w one = 1) :
    row.Sat w ↔ op.Sat w := by
  unfold extract at he
  split at he
  · rename_i h
    cases Option.some.inj he
    rcases h with ⟨hb,hc,hd⟩
    unfold Row.Sat Instruction.Sat Instruction.value
    rw [hb,hc,eval_splitWire dst row.a w,hd,cast_neg_one]
    simp only [evalLC, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      Nat.cast_one, h1, add_zero, mul_one]
    constructor <;> intro h <;> linear_combination -h
  · split at he
    · rename_i hc
      cases Option.some.inj he
      unfold Row.Sat Instruction.Sat Instruction.value
      rw [eval_splitWire dst row.c w,hc,eval_negateLC]
      simp only [Nat.cast_one, one_mul]
      constructor <;> intro h <;> linear_combination -h
    · contradiction

def extractRows (one : Nat) : Nat → List Row → Option Program
  | _, [] => some []
  | dst, row::rows =>
    match extract one dst row, extractRows one (dst+1) rows with
    | some op, some tail => some (op::tail)
    | _, _ => none

theorem extractRows_correct {one dst : Nat} {rows : List Row} {p : Program}
    (he : extractRows one dst rows = some p) {w : Assignment} (h1 : w one = 1) :
    (∀ row ∈ rows, row.Sat w) ↔ Satisfies p w := by
  induction rows generalizing dst p with
  | nil =>
    simp only [extractRows, Option.some.injEq] at he
    subst p
    simp [Satisfies]
  | cons row rows ih =>
    simp only [extractRows] at he
    cases ho : extract one dst row with
    | none => simp [ho] at he
    | some op =>
      cases ht : extractRows one (dst+1) rows with
      | none => simp [ho,ht] at he
      | some tail =>
        simp only [ho,ht,Option.some.injEq] at he
        subst p
        simp only [List.mem_cons, forall_eq_or_imp, Satisfies]
        rw [extract_correct ho h1]
        exact and_congr_right (fun _ => ih ht)

def chunkProgram (one dst : Nat) (rows : List Row) : Program :=
  (extractRows one dst rows).getD []

def chunkCheck (one dst : Nat) (rows : List Row) : Bool :=
  match extractRows one dst rows with
  | none => false
  | some p => orderedCheck one dst p

theorem chunkCheck_extract {one dst : Nat} {rows : List Row}
    (h : chunkCheck one dst rows = true) :
    extractRows one dst rows = some (chunkProgram one dst rows) := by
  unfold chunkCheck at h
  cases he : extractRows one dst rows with
  | none => simp [he] at h
  | some p => simp [chunkProgram, he]

theorem chunkCheck_ordered {one dst : Nat} {rows : List Row}
    (h : chunkCheck one dst rows = true) :
    Ordered one dst (chunkProgram one dst rows) := by
  have he := chunkCheck_extract h
  unfold chunkCheck at h
  rw [he] at h
  exact (orderedCheck_correct _ _ _).mp h

theorem chunkCheck_correct {one dst : Nat} {rows : List Row}
    (h : chunkCheck one dst rows = true) {w : Assignment} (h1 : w one = 1) :
    (∀ row ∈ rows, row.Sat w) ↔ Satisfies (chunkProgram one dst rows) w :=
  extractRows_correct (chunkCheck_extract h) h1

theorem extractRows_length {one dst : Nat} {rows : List Row} {p : Program}
    (h : extractRows one dst rows = some p) : p.length = rows.length := by
  induction rows generalizing dst p with
  | nil => simp [extractRows] at h; subst p; rfl
  | cons row rows ih =>
    unfold extractRows at h
    cases ho : extract one dst row with
    | none => simp [ho] at h
    | some op =>
      cases ht : extractRows one (dst+1) rows with
      | none => simp [ho,ht] at h
      | some tail =>
        simp only [ho,ht,Option.some.injEq] at h
        subst p
        simp only [List.length_cons, ih ht]

theorem chunkCheck_length {one dst : Nat} {rows : List Row}
    (h : chunkCheck one dst rows = true) :
    (chunkProgram one dst rows).length = rows.length :=
  extractRows_length (chunkCheck_extract h)

/-- Pairwise verified chunks compose without expanding their field operations. -/
theorem flatten_correct {rowChunks : List (List Row)} {programChunks : List Program}
    (h : List.Forall₂ (fun rs ps => ∀ w : Assignment, w 97634 = 1 →
      ((∀ row ∈ rs, row.Sat w) ↔ Satisfies ps w)) rowChunks programChunks)
    {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rowChunks.flatten, row.Sat w) ↔ Satisfies programChunks.flatten w := by
  induction h with
  | nil => simp [Satisfies]
  | cons hr _ ih =>
    simp only [List.flatten_cons, List.mem_append, or_imp, forall_and,
      satisfies_append, hr w h1, ih]

namespace Coded
open ConcreteBytes.Codes

theorem codes_length (n data : Nat) : (rows n data).length = n := by
  have h := Exported.decodeRows_length n data
  rw [rows_expand, List.length_map] at h
  exact h

def removeWire (dst : Nat) (lc : LinearCombination) := lc.filter fun t => t.1 != dst
def atWire (dst : Nat) (lc : LinearCombination) := lc.filter fun t => t.1 == dst

def linear (one dst : Nat) (row : Row) : Bool :=
  decide (row.b = [(one,0)] ∧ row.c = [] ∧ atWire dst row.a = [(dst,1218)])

def instruction (one dst : Nat) (row : Row) : Instruction :=
  if linear one dst row then
    ⟨dst,expandLC (removeWire dst row.a),[(one,1)],[]⟩
  else ⟨dst,expandLC row.a,expandLC row.b,negateLC (expandLC (removeWire dst row.c))⟩

def check (one dst : Nat) (row : Row) : Bool :=
  decide (dst ≠ one) &&
  if linear one dst row then readsCheck dst one (removeWire dst row.a)
  else decide (atWire dst row.c = [(dst,0)]) && readsCheck dst one row.a &&
    readsCheck dst one row.b && readsCheck dst one (removeWire dst row.c)

theorem eval_partition (w : Assignment) (dst : Nat) (lc : LinearCombination) :
    evalLC w (expandLC lc) = evalLC w (expandLC (atWire dst lc)) +
      evalLC w (expandLC (removeWire dst lc)) := by
  induction lc with
  | nil => simp [atWire,removeWire,expandLC,evalLC]
  | cons t tail ih =>
    rcases t with ⟨i,c⟩
    by_cases h : i=dst
    · subst i
      simp only [atWire,removeWire,List.filter_cons,beq_self_eq_true,bne_self_eq_false,
        Bool.false_eq_true, ↓reduceIte,expandLC,List.map_cons,evalLC,List.sum_cons] at *
      rw [ih]
      ring
    · simp only [atWire,removeWire,List.filter_cons,beq_iff_eq,ne_eq,h,not_false_eq_true,
        bne_iff_ne, ↓reduceIte,expandLC,List.map_cons,evalLC,List.sum_cons] at *
      rw [ih]
      ring

theorem reads_expand {K : Known} {lc : LinearCombination} (h : Reads K lc) :
    Reads K (expandLC lc) := by
  intro t ht
  obtain ⟨u,hu,rfl⟩ := List.mem_map.mp ht
  exact h u hu

theorem reads_negate {K : Known} {lc : LinearCombination} (h : Reads K lc) :
    Reads K (negateLC lc) := by
  intro t ht
  obtain ⟨hu,_⟩ := List.mem_filter.mp ht
  obtain ⟨v,hv,rfl⟩ := List.mem_map.mp hu
  exact h v hv

theorem check_ready {one dst : Nat} {row : Row} (h : check one dst row = true) :
    (instruction one dst row).Ready (Frontier dst one) := by
  unfold check at h
  simp only [Bool.and_eq_true,decide_eq_true_eq] at h
  have hn : ¬Frontier dst one dst := by simpa [Frontier] using h.1
  unfold instruction
  split
  · rename_i hl
    have hr := h.2
    simp only [hl, ↓reduceIte] at hr
    refine ⟨hn, reads_expand ((readsCheck_correct _ _ _).mp hr), ?_, ?_⟩
    · intro t ht
      exact Or.inr (congrArg Prod.fst (List.mem_singleton.mp ht))
    · simp [Reads]
  · rename_i hl
    have hr := h.2
    simp only [hl, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq, readsCheck_correct] at hr
    exact ⟨hn,reads_expand hr.1.1.2,reads_expand hr.1.2,
      reads_negate (reads_expand hr.2)⟩

theorem check_correct {one dst : Nat} {row : Row} (h : check one dst row = true)
    {w : Assignment} (h1 : w one = 1) :
    (expandRow row).Sat w ↔ (instruction one dst row).Sat w := by
  unfold check at h
  simp only [Bool.and_eq_true,decide_eq_true_eq] at h
  unfold instruction
  split
  · rename_i hl
    have hs : row.b = [(one,0)] ∧ row.c = [] ∧ atWire dst row.a = [(dst,1218)] :=
      of_decide_eq_true hl
    unfold Row.Sat Instruction.Sat Instruction.value expandRow
    rw [eval_partition w dst row.a,hs.1,hs.2.1,hs.2.2]
    simp only [expandLC,List.map_cons,List.map_nil,evalLC,List.sum_cons,List.sum_nil,
      one_expand,neg_one_expand,cast_neg_one,Nat.cast_one,h1,one_mul,add_zero,mul_one]
    constructor <;> intro h <;> linear_combination -h
  · rename_i hl
    have hc := h.2
    simp only [hl, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at hc
    unfold Row.Sat Instruction.Sat Instruction.value expandRow
    rw [eval_partition w dst row.c,hc.1.1.1,eval_negateLC]
    simp only [expandLC,List.map_cons,List.map_nil,evalLC,List.sum_cons,List.sum_nil,
      one_expand,Nat.cast_one,one_mul,add_zero]
    constructor <;> intro h <;> linear_combination -h

def program (one : Nat) : Nat → List Row → Program
  | _, [] => []
  | dst, row::rows => instruction one dst row :: program one (dst+1) rows

def checkRows (one : Nat) : Nat → List Row → Bool
  | _, [] => true
  | dst, row::rows => check one dst row && checkRows one (dst+1) rows

theorem program_length (one dst : Nat) (rows : List Row) :
    (program one dst rows).length = rows.length := by
  induction rows generalizing dst with
  | nil => rfl
  | cons row rows ih => simp [program,ih]

theorem checked_ordered {one dst : Nat} {rows : List Row}
    (h : checkRows one dst rows = true) : Ordered one dst (program one dst rows) := by
  induction rows generalizing dst with
  | nil => trivial
  | cons row rows ih =>
    simp only [checkRows,Bool.and_eq_true] at h
    have hr := check_ready h.1
    have hd : (instruction one dst row).dst = dst := by
      unfold instruction; split <;> rfl
    refine ⟨hd,?_,hr.2.1,hr.2.2.1,hr.2.2.2,ih h.2⟩
    intro he
    exact hr.1 (Or.inr (hd.trans he))

theorem checked_correct {one dst : Nat} {rows : List Row}
    (h : checkRows one dst rows = true) {w : Assignment} (h1 : w one = 1) :
    (∀ row ∈ rows.map expandRow, row.Sat w) ↔ Satisfies (program one dst rows) w := by
  induction rows generalizing dst with
  | nil => simp [program,Satisfies]
  | cons row rows ih =>
    simp only [checkRows,Bool.and_eq_true] at h
    simp only [List.map_cons,program,List.mem_cons,forall_eq_or_imp,Satisfies]
    rw [check_correct h.1 h1]
    exact and_congr_right (fun _ => ih h.2)

end Coded
end CircuitCorrectness.ConcreteProgram
