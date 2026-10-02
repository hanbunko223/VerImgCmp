import CircuitCorrectness.PoseidonProgram
namespace CircuitCorrectness.HashTrace
open PoseidonProgram StraightLine

/-- Checkpoints avoid expanding a complete symbolic hash in the kernel at once.
Each local equality remains a kernel-checked compiler equation. -/
theorem checkpoints_sound (w : Assignment) (p : Spec.PoseidonParameters) (one count : Nat)
    (states : Nat → PoseidonProgram.State) (ops : Nat → Program)
    (h1 : w one=1)
    (hc : ∀ r<count, round p one (states r) r = (states (r+1),ops r))
    (hs : ∀ r<count, Satisfies (ops r) w) :
    (evalState w (states count).values,(states count).offset) =
      (List.range count).foldl (Spec.poseidonRound p)
        (evalState w (states 0).values,(states 0).offset) := by
  have aux : ∀ n≤count,
      (evalState w (states n).values,(states n).offset) =
        (List.range n).foldl (Spec.poseidonRound p)
          (evalState w (states 0).values,(states 0).offset) := by
    intro n
    induction n with
    | zero => intro _; rfl
    | succ n ih =>
      intro hn
      have hn' : n<count := by omega
      have hlocal : Satisfies (round p one (states n) n).2 w := by
        rw [hc n hn']; exact hs n hn'
      have hround := round_sound w p one (states n) n h1 hlocal
      rw [hc n hn'] at hround
      simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil,
        ← ih (by omega)]
      exact hround
  exact aux count (Nat.le_refl count)

theorem hash_sound (w : Assignment) (p : Spec.PoseidonParameters) (one domain output : Nat)
    (input : Array F) (states : Nat → PoseidonProgram.State) (ops : Nat → Program)
    (h1 : w one=1)
    (hc : ∀ r<p.fullRounds+p.partialRounds,
      round p one (states r) r = (states (r+1),ops r))
    (hs : ∀ r<p.fullRounds+p.partialRounds, Satisfies (ops r) w)
    (hi : (evalState w (states 0).values,(states 0).offset) =
      (#[(Spec.domainTag p.arity domain : F)] ++ input,0))
    (ho : w output = evalLC w (states (p.fullRounds+p.partialRounds)).values[1]!) :
    w output = Spec.hash p domain input := by
  have hh := checkpoints_sound w p one (p.fullRounds+p.partialRounds) states ops h1 hc hs
  rw [hi] at hh
  have hv := congrArg (fun t : Array F × Nat => t.1[1]!) hh
  dsimp only at hv
  rw [evalState_get] at hv
  exact ho.trans hv
end CircuitCorrectness.HashTrace
