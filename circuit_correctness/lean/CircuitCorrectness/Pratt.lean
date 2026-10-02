/-
Copyright (c) 2020 Bolton Bailey. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Bolton Bailey
-/
module

public import Mathlib.Tactic.ReduceModChar
public import Mathlib.Tactic.NormNum
public import Mathlib.NumberTheory.LucasPrimality

/-!
# The Lucas test for primes.

This file implements the Lucas test for primes (not to be confused with the Lucas-Lehmer test for
Mersenne primes). A number `a` witnesses that `n` is prime if `a` has order `n-1` in the
multiplicative group of integers mod `n`. This is checked by verifying that `a^(n-1) = 1 (mod n)`
and `a^d ≠ 1 (mod n)` for any divisor `d | n - 1`. This test is the basis of the Pratt primality
certificate.

## TODO

- Bonus: Show the reverse implication i.e. if a number is prime then it has a Lucas witness.
  Use `Units.IsCyclic` from `RingTheory/IntegralDomain` to show the group is cyclic.
- Write a tactic that uses this theorem to generate Pratt primality certificates
- Integrate Pratt primality certificates into the norm_num primality verifier

## Implementation notes

Note that the proof for `lucas_primality` relies on analyzing the multiplicative group
modulo `p`. Despite this, the theorem still holds vacuously for `p = 0` and `p = 1`: In these
cases, we can take `q` to be any prime and see that `hd` does not hold, since `a^((p-1)/q)` reduces
to `1`.
-/

@[expose] public section

section New

-- TODO: port to `Mathlib`?
lemma Nat.Prime.dvd_mul_list {p : ℕ} {l : List ℕ} (h : p.Prime) :
    p ∣ l.prod ↔ ∃ r ∈ l, p ∣ r := by
  constructor
  · intro hdiv
    induction l with
    | nil =>
      simp at *
      rw [hdiv] at h
      aesop
    | cons hd tl ih =>
      rw [List.prod_cons] at hdiv
      rcases h.dvd_mul.mp hdiv with (hdiv|hdiv)
      · use hd
        simp only [List.mem_cons, true_or, true_and]
        exact hdiv
      · rcases ih hdiv with ⟨r, hr, hdiv⟩
        use r
        simp at *
        exact ⟨Or.inr hr, hdiv⟩
  · intro h
    rcases h with ⟨r, hr, hdiv⟩
    rw [←List.prod_erase hr]
    exact h.dvd_mul.mpr (Or.inl hdiv)

/-- Recursive form of a Pratt certificate for `p`, which may take in Pratt certificates
  for a list of prime powers `r` less than `p` whose product is equal to `p - 1`. -/
inductive PrattPartList : (p : ℕ) → (a : ZMod p) → ℕ → Prop
  | prime : {p : ℕ} → {a : ZMod p} → (n k nk : ℕ) → n.Prime →
      a ^ ((p - 1) / n) ≠ 1 → n ^ k = nk → PrattPartList p a nk
  | split : {p : ℕ} → {a : ZMod p} → {n : ℕ} → (list : List ℕ) →
      (∀ r ∈ list, PrattPartList p a r) → list.prod = n → PrattPartList p a n

/-- Alternative form of a Pratt certificate for `p`, which may take in Pratt certificates
  for a list of prime powers `r` less than `p` whose product is equal to `p - 1`. -/
structure PrattCertificate' (p : ℕ) : Type where
  a : ZMod p
  pow_eq_one : a ^ (p - 1) = 1
  part : PrattPartList p a (p - 1)

theorem PrattPartList.out {p : ℕ} {a : ZMod p} {n : ℕ} (h : PrattPartList p a n) :
    ∀ q : ℕ, q.Prime → q ∣ n → a ^ ((p - 1) / q) ≠ 1 := by
  induction h with
  | prime n' k nk hprime hpow hnk =>
      subst hnk
      intro q hq hdiv
      cases (Nat.prime_dvd_prime_iff_eq hq hprime).mp (hq.dvd_of_dvd_pow hdiv)
      exact hpow
  | split list hprev hprod ih =>
    rw [←hprod]
    intro q hq hdiv
    rcases hq.dvd_mul_list.mp hdiv with ⟨r, hr, hdiv⟩
    · exact ih r hr q hq hdiv

theorem PrattCertificate'.out {p : ℕ} (c : PrattCertificate' p) : p.Prime :=
  lucas_primality p c.a c.pow_eq_one c.part.out

end New
