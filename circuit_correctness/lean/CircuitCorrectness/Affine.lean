import CircuitCorrectness.R1CS
import Mathlib.Tactic.Ring

namespace CircuitCorrectness.Affine

@[simp] theorem cast_mod (n : Nat) : ((n % modulus : Nat) : F) = (n : F) :=
  ZMod.natCast_mod n modulus

/-- Insertion merges coefficients of an existing wire; semantics does not
require the argument to be sorted. The builders supply sorted lists. -/
def insert (i c : Nat) : LinearCombination → LinearCombination
  | [] => [(i,c % modulus)]
  | (j,d)::xs =>
    if i < j then (i,c % modulus)::(j,d)::xs
    else if i = j then (j,(c+d) % modulus)::xs
    else (j,d)::insert i c xs

def clean (xs : LinearCombination) : LinearCombination :=
  xs.filter fun t => t.2 != 0

def add (xs ys : LinearCombination) : LinearCombination :=
  clean (ys.foldl (fun acc t => insert t.1 t.2 acc) xs)

def scale (c : Nat) (xs : LinearCombination) : LinearCombination :=
  clean (xs.map fun (i,d) => (i,(c*d)%modulus))

def constant (one c : Nat) : LinearCombination :=
  if c % modulus = 0 then [] else [(one,c%modulus)]

def wire (i : Nat) : LinearCombination := [(i,1)]

def weighted (xs : List LinearCombination) (cs : List Nat) : LinearCombination :=
  (xs.zip cs).foldl (fun acc (x,c) => add acc (scale c x)) []

@[simp] theorem eval_nil (w : Assignment) : evalLC w [] = 0 := rfl
@[simp] theorem eval_cons (w : Assignment) (i c : Nat) (xs : LinearCombination) :
    evalLC w ((i,c)::xs) = (c : F) * w i + evalLC w xs := rfl

theorem eval_insert (w : Assignment) (i c : Nat) (xs : LinearCombination) :
    evalLC w (insert i c xs) = (c : F) * w i + evalLC w xs := by
  induction xs with
  | nil => simp [insert]
  | cons t xs ih =>
    rcases t with ⟨j,d⟩
    simp only [insert]
    split
    · simp
    · split
      · rename_i heq
        subst j
        simp [Nat.cast_add, add_mul, add_assoc]
      · simp [ih, add_left_comm, add_assoc]

@[simp] theorem eval_clean (w : Assignment) (xs : LinearCombination) :
    evalLC w (clean xs) = evalLC w xs := by
  induction xs with
  | nil => rfl
  | cons t xs ih =>
    rcases t with ⟨i,c⟩
    by_cases h : c = 0
    · simpa [clean, h] using ih
    · simpa [clean, h] using congrArg (fun z : F => (c : F) * w i + z) ih

theorem eval_foldInsert (w : Assignment) (ys xs : LinearCombination) :
    evalLC w (ys.foldl (fun acc t => insert t.1 t.2 acc) xs) =
      evalLC w xs + evalLC w ys := by
  induction ys generalizing xs with
  | nil => simp
  | cons t ys ih =>
    rcases t with ⟨i,c⟩
    simp [ih, eval_insert, add_assoc, add_left_comm]

@[simp] theorem eval_add (w : Assignment) (xs ys : LinearCombination) :
    evalLC w (add xs ys) = evalLC w xs + evalLC w ys := by
  simp [add, eval_foldInsert]

@[simp] theorem eval_scale (w : Assignment) (c : Nat) (xs : LinearCombination) :
    evalLC w (scale c xs) = (c : F) * evalLC w xs := by
  simp only [scale, eval_clean]
  induction xs with
  | nil => simp
  | cons t xs ih =>
    rcases t with ⟨i,d⟩
    simp [ih, Nat.cast_mul, mul_add, mul_assoc]

@[simp] theorem eval_constant (w : Assignment) (one c : Nat) :
    evalLC w (constant one c) = (c : F) * w one := by
  unfold constant
  split
  · rename_i h
    have hc : (c : F) = 0 := by rw [← cast_mod c, h]; rfl
    simp [hc]
  · simp

@[simp] theorem eval_variable (w : Assignment) (i : Nat) :
    evalLC w (wire i) = w i := by simp [wire]

theorem eval_weighted_fold (w : Assignment) (zs : List (LinearCombination × Nat))
    (acc : LinearCombination) :
    evalLC w (zs.foldl (fun acc (x,c) => add acc (scale c x)) acc) =
      evalLC w acc + (zs.map fun (x,c) => (c : F) * evalLC w x).sum := by
  induction zs generalizing acc with
  | nil => simp
  | cons t zs ih =>
    rcases t with ⟨x,c⟩
    simp [ih, add_assoc]

@[simp] theorem eval_weighted (w : Assignment) (xs : List LinearCombination) (cs : List Nat) :
    evalLC w (weighted xs cs) =
      ((xs.zip cs).map fun (x,c) => (c : F) * evalLC w x).sum := by
  simp [weighted, eval_weighted_fold]

def sum (xs : List LinearCombination) : LinearCombination := xs.foldl add []

theorem eval_sum_fold (w : Assignment) (xs : List LinearCombination)
    (acc : LinearCombination) :
    evalLC w (xs.foldl add acc) = evalLC w acc + (xs.map (evalLC w)).sum := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih => simp [ih, add_assoc]

@[simp] theorem eval_sum (w : Assignment) (xs : List LinearCombination) :
    evalLC w (sum xs) = (xs.map (evalLC w)).sum := by simp [sum, eval_sum_fold]

end CircuitCorrectness.Affine
