import CircuitCorrectness.R1CS

namespace CircuitCorrectness.StraightLine

/-- Deterministic field assignment; later certificates connect these to actual R1CS rows. -/
structure Instruction where
  dst : Nat
  lhs : LinearCombination
  rhs : LinearCombination
  add : LinearCombination
  deriving DecidableEq, Repr

abbrev Known := Nat → Prop
def Extend (K : Known) (dst : Nat) : Known := fun i => i = dst ∨ K i
def Agree (K : Known) (w v : Assignment) : Prop := ∀ i, K i → w i = v i
def Reads (K : Known) (lc : LinearCombination) : Prop := ∀ t ∈ lc, K t.1

def Instruction.Ready (op : Instruction) (K : Known) : Prop :=
  ¬K op.dst ∧ Reads K op.lhs ∧ Reads K op.rhs ∧ Reads K op.add

def Instruction.value (op : Instruction) (w : Assignment) : F :=
  evalLC w op.lhs * evalLC w op.rhs + evalLC w op.add

def Instruction.Sat (op : Instruction) (w : Assignment) : Prop :=
  w op.dst = op.value w

def Instruction.exec (op : Instruction) (w : Assignment) : Assignment :=
  Function.update w op.dst (op.value w)

abbrev Program := List Instruction

def WellFormed : Known → Program → Prop
  | _, [] => True
  | K, op :: tail => op.Ready K ∧ WellFormed (Extend K op.dst) tail

def KnownAfter : Known → Program → Known
  | K, [] => K
  | K, op :: tail => KnownAfter (Extend K op.dst) tail

def run : Program → Assignment → Assignment
  | [], w => w
  | op :: tail, w => run tail (op.exec w)

def Satisfies (p : Program) (w : Assignment) : Prop := ∀ op ∈ p, op.Sat w

theorem agree_refl (K : Known) (w : Assignment) : Agree K w w := by
  intro i _
  rfl

theorem agree_symm {K : Known} {w v : Assignment} (h : Agree K w v) : Agree K v w :=
  fun i hi => (h i hi).symm

theorem agree_trans {K : Known} {u v w : Assignment}
    (h₁ : Agree K u v) (h₂ : Agree K v w) : Agree K u w :=
  fun i hi => (h₁ i hi).trans (h₂ i hi)

theorem lc_agree {K : Known} {w v : Assignment} {lc : LinearCombination}
    (h : Agree K w v) (hr : Reads K lc) : evalLC w lc = evalLC v lc :=
  evalLC_congr lc (fun t ht => h t.1 (hr t ht))

theorem value_agree {K : Known} {w v : Assignment} {op : Instruction}
    (h : Agree K w v) (hr : op.Ready K) : op.value w = op.value v := by
  unfold Instruction.value
  rw [lc_agree h hr.2.1, lc_agree h hr.2.2.1, lc_agree h hr.2.2.2]

theorem exec_preserves {K : Known} {w : Assignment} {op : Instruction}
    (hr : op.Ready K) : Agree K (op.exec w) w := by
  intro i hi
  have hn : i ≠ op.dst := fun he => hr.1 (he ▸ hi)
  simp [Instruction.exec, Function.update_of_ne hn]

theorem exec_satisfies {K : Known} {w : Assignment} {op : Instruction}
    (hr : op.Ready K) : op.Sat (op.exec w) := by
  unfold Instruction.Sat
  rw [value_agree (exec_preserves hr) hr]
  simp [Instruction.exec]

theorem knownAfter_contains (K : Known) (p : Program) :
    ∀ i, K i → KnownAfter K p i := by
  induction p generalizing K with
  | nil => exact fun _ hi => hi
  | cons op tail ih =>
    intro i hi
    exact ih (Extend K op.dst) i (Or.inr hi)

theorem run_preserves_known {p : Program} {K : Known} {w : Assignment}
    (hf : WellFormed K p) : Agree K (run p w) w := by
  induction p generalizing K w with
  | nil => exact agree_refl K w
  | cons op tail ih =>
    intro i hi
    exact (ih hf.2 i (Or.inr hi)).trans (exec_preserves hf.1 i hi)

theorem sat_congr {op : Instruction} {K : Known} {w v : Assignment}
    (hr : op.Ready K) (h : Agree (Extend K op.dst) w v) : op.Sat w ↔ op.Sat v := by
  have hd := h op.dst (Or.inl rfl)
  have hv := value_agree (K := K) (fun i hi => h i (Or.inr hi)) hr
  unfold Instruction.Sat
  rw [hd, hv]

/-- Construction satisfies every instruction, not just a selected output equation. -/
theorem run_satisfies {p : Program} {K : Known} {w : Assignment}
    (hf : WellFormed K p) : Satisfies p (run p w) := by
  induction p generalizing K w with
  | nil => simp [Satisfies]
  | cons op tail ih =>
    intro other hm
    rcases List.mem_cons.mp hm with he | ht
    · subst other
      exact (sat_congr hf.1 (run_preserves_known hf.2)).mpr (exec_satisfies hf.1)
    · exact ih hf.2 other ht

/-- Every satisfying assignment agrees with execution on all determined wires. -/
theorem satisfying_agrees_run {p : Program} {K : Known} {w seed : Assignment}
    (hf : WellFormed K p) (hs : Satisfies p w) (hi : Agree K w seed) :
    Agree (KnownAfter K p) w (run p seed) := by
  induction p generalizing K seed with
  | nil => exact hi
  | cons op tail ih =>
    apply ih hf.2
    · intro other hm
      exact hs other (List.mem_cons_of_mem op hm)
    · intro i hk
      rcases hk with he | hk
      · subst i
        rw [hs op (List.mem_cons_self)]
        rw [value_agree hi hf.1]
        simp [Instruction.exec]
      · exact (hi i hk).trans (exec_preserves hf.1 i hk).symm

theorem satisfying_determinism {p : Program} {K : Known} {w v : Assignment}
    (hf : WellFormed K p) (hw : Satisfies p w) (hv : Satisfies p v)
    (hi : Agree K w v) : Agree (KnownAfter K p) w v := by
  exact agree_trans (satisfying_agrees_run hf hw hi)
    (agree_symm (satisfying_agrees_run hf hv (agree_refl K v)))

theorem exists_satisfying_extension {p : Program} {K : Known} (seed : Assignment)
    (hf : WellFormed K p) : ∃ w, Agree K w seed ∧ Satisfies p w :=
  ⟨run p seed, run_preserves_known hf, run_satisfies hf⟩

/-- Splitting a certificate into chunks changes neither execution nor known wires. -/
theorem run_append (p q : Program) (w : Assignment) :
    run (p ++ q) w = run q (run p w) := by
  induction p generalizing w with
  | nil => rfl
  | cons op tail ih => exact ih (op.exec w)

theorem knownAfter_append (K : Known) (p q : Program) :
    KnownAfter K (p ++ q) = KnownAfter (KnownAfter K p) q := by
  induction p generalizing K with
  | nil => rfl
  | cons op tail ih => exact ih (Extend K op.dst)

theorem wellFormed_append (K : Known) (p q : Program) :
    WellFormed K (p ++ q) ↔ WellFormed K p ∧ WellFormed (KnownAfter K p) q := by
  induction p generalizing K with
  | nil => simp [WellFormed, KnownAfter]
  | cons op tail ih =>
    simp only [List.cons_append, WellFormed, KnownAfter, ih]
    exact and_assoc.symm

theorem satisfies_append (p q : Program) (w : Assignment) :
    Satisfies (p ++ q) w ↔ Satisfies p w ∧ Satisfies q w := by
  simp only [Satisfies, List.mem_append, or_imp, forall_and]

/-- Efficient numerical frontier for a contiguous suffix of auxiliary assignments.
The fixed constant-one wire can lie above the entire auxiliary range. -/
def Frontier (next one : Nat) : Known := fun i => i < next ∨ i = one

theorem frontier_extend (next one : Nat) :
    Extend (Frontier next one) next = Frontier (next + 1) one := by
  funext i
  apply propext
  simp only [Extend, Frontier]
  omega

/-- A decidable certificate predicate that avoids growing sets of known wires. -/
def Ordered (one : Nat) : Nat → Program → Prop
  | _, [] => True
  | next, op :: tail =>
    op.dst = next ∧ next ≠ one ∧ Reads (Frontier next one) op.lhs ∧
      Reads (Frontier next one) op.rhs ∧ Reads (Frontier next one) op.add ∧
      Ordered one (next + 1) tail

theorem ordered_append (one next : Nat) (p q : Program) :
    Ordered one next (p ++ q) ↔ Ordered one next p ∧ Ordered one (next+p.length) q := by
  induction p generalizing next with
  | nil => simp [Ordered]
  | cons op tail ih =>
    simp only [List.cons_append, Ordered, List.length_cons, ih]
    have he : next + (tail.length+1) = next+1+tail.length := by omega
    rw [he]
    tauto

theorem ordered_wellFormed {one next : Nat} {p : Program}
    (h : Ordered one next p) : WellFormed (Frontier next one) p := by
  induction p generalizing next with
  | nil => trivial
  | cons op tail ih =>
    rcases h with ⟨hd, hn, hl, hr, ha, ht⟩
    refine ⟨⟨?_, hl, hr, ha⟩, ?_⟩
    · simp only [hd, Frontier, lt_self_iff_false, false_or]
      exact hn
    · rw [hd, frontier_extend]
      exact ih ht

theorem ordered_knownAfter {one next : Nat} {p : Program}
    (h : Ordered one next p) :
    KnownAfter (Frontier next one) p = Frontier (next + p.length) one := by
  induction p generalizing next with
  | nil => rfl
  | cons op tail ih =>
    rcases h with ⟨hd, _, _, _, _, ht⟩
    simp only [KnownAfter, hd, frontier_extend, ih ht, List.length_cons]
    congr 1; omega

def readsCheck (next one : Nat) (lc : LinearCombination) : Bool :=
  lc.all fun t => decide (t.1 < next ∨ t.1 = one)

def orderedCheck (one : Nat) : Nat → Program → Bool
  | _, [] => true
  | next, op :: tail =>
    (op.dst == next) && decide (next ≠ one) && readsCheck next one op.lhs &&
      readsCheck next one op.rhs && readsCheck next one op.add &&
      orderedCheck one (next + 1) tail

theorem readsCheck_correct (next one : Nat) (lc : LinearCombination) :
    readsCheck next one lc = true ↔ Reads (Frontier next one) lc := by
  simp [readsCheck, List.all_eq_true, Reads, Frontier]

theorem orderedCheck_correct (one next : Nat) (p : Program) :
    orderedCheck one next p = true ↔ Ordered one next p := by
  induction p generalizing next with
  | nil => simp [orderedCheck, Ordered]
  | cons op tail ih =>
    simp only [orderedCheck, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      readsCheck_correct, ih, Ordered]
    tauto

theorem checked_wellFormed {one next : Nat} {p : Program}
    (h : orderedCheck one next p = true) : WellFormed (Frontier next one) p :=
  ordered_wellFormed ((orderedCheck_correct one next p).mp h)

end CircuitCorrectness.StraightLine
