#!/usr/bin/env python3
import json
from reference import fixture,ROOT
v=json.loads((ROOT/'results/rust_validation.json').read_text())
r=fixture(4);values=[x for row in r for pix in row for x in pix]
s='''import CircuitCorrectness.Spec
open CircuitCorrectness
namespace CircuitCorrectness.Differential
'''

for j in range(0,len(values),128):
    s+=f'def randomChunk{j//128} : Array Nat := #['+', '.join(map(str,values[j:j+128]))+']\n'
s+='def randomPixels : Array Nat := (#['+', '.join(f'randomChunk{j//128}' for j in range(0,len(values),128))+'] : Array (Array Nat)).flatten\n'
s+='def expected : Array (Array Nat) := #[\n'+',\n'.join('#['+', '.join(c['outgoing'])+']' for c in v['cases'])+']\n'
s+='''def image (kind : Nat) : Spec.Image := fun r c ch =>
  if kind = 0 then 0
  else if kind = 1 then 255
  else if kind = 2 then if (r+c+ch)%2 = 0 then 0 else 255
  else if kind = 3 then if r = 3 ∧ c = 5 ∧ ch = 1 then 255 else 0
  else randomPixels[(r*160+c)*3+ch]!

-- These evaluations are regression tests, not kernel-checked universal proofs.
def run : IO Unit := do
  for k in [0:5] do
    let state : Spec.State := ⟨-1,11,(k : F),-2⟩
    let out := Spec.step (image k) state
    let actual := #[out.h.val,out.a.val,out.r.val,out.t.val]
    if actual != expected[k]! then
      throw (IO.userError s!"Lean/Rust mismatch for fixture {k}: {actual}")
    IO.println s!"Lean/Rust fixture {k}: pass"
end CircuitCorrectness.Differential

def main : IO Unit := CircuitCorrectness.Differential.run
'''
(ROOT/'lean/Differential.lean').write_text(s)
