#!/usr/bin/env python3
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--check",action="store_true")
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
for n,total,start,ins,out,domain in [(8,65,77500,list(range(77484,77492)),77887,0x48415348),(2,63,78276,[77887,78275],78513,0x50414952)]:
    ns=f'CircuitCorrectness.HashTrace{n}'
    content=f'''import {ns}.Output
import CircuitCorrectness.HashTrace
import CircuitCorrectness.HashWiringTemplates
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace {ns}
open PoseidonProgram StraightLine
'''
    for name,typ,last,prefix in [('states','PoseidonProgram.State',total,'state'),('ops','Program',total-1,'ops'),('codes','List Row',total-1,'codes')]:
        content+=f'def {name} : Nat → {typ}\n'
        content+=''.join(f'  | {i} => {prefix}{i:02}\n' for i in range(last+1))
        content+=f'  | _ => {prefix}{last:02}\n'
    content+=f'''def allCodes : List Row := (List.range {total}).flatMap codes ++ outputCodes

theorem actual_codes : HashWiring.template{n}Codes = allCodes := by
  rw [HashWiring.template{n}_literal]
  decide +kernel

theorem checkpoints : ∀ r<{total},
    round Spec.params{n} 97634 (states r) r = (states (r+1),ops r) := by
  intro r hr
  interval_cases r
'''
    content+=''.join(f'  · exact round_exact{i:02}\n' for i in range(total))
    content+=f'''theorem extracted : ∀ r<{total}, ConcreteProgram.extractRows 97634 (states r).next
    ((codes r).map ConcreteBytes.Codes.expandRow) = some (ops r) := by
  intro r hr
  interval_cases r
'''
    content+=''.join(f'  · exact rows_exact{i:02}\n' for i in range(total))
    inputs='#['+','.join(f'Affine.wire {i}' for i in ins)+']'
    content+=f'''def input : Array LinearCombination := {inputs}

theorem initial : states 0 =
    ⟨#[Affine.constant 97634 (Spec.domainTag Spec.params{n}.arity {domain})] ++ input,0,{start}⟩ := by decide +kernel

theorem output_extracted : ConcreteProgram.extractRows 97634 {out}
    (outputCodes.map ConcreteBytes.Codes.expandRow) =
    some [⟨{out},(states {total}).values[1]!,Affine.constant 97634 1,[]⟩] := by decide +kernel

theorem template_sound (w : Assignment) (h1 : w 97634=1)
    (hs : ∀ row∈HashWiring.template{n},row.Sat w) :
    w {out} = Spec.hash Spec.params{n} {domain} (evalState w input) := by
  simp only [HashWiring.template{n},actual_codes] at hs
  have hlocal : ∀ r<{total}, Satisfies (ops r) w := by
    intro r hr
    apply (ConcreteProgram.extractRows_correct (extracted r hr) h1).mp
    intro row hm
    rcases List.mem_map.mp hm with ⟨code,hcode,rfl⟩
    apply hs _
    apply List.mem_map.mpr
    refine ⟨code,List.mem_append_left _ ?_,rfl⟩
    exact List.mem_flatMap.mpr ⟨r,List.mem_range.mpr hr,hcode⟩
  have hout : Satisfies
      [⟨{out},(states {total}).values[1]!,Affine.constant 97634 1,[]⟩] w := by
    apply (ConcreteProgram.extractRows_correct output_extracted h1).mp
    intro row hm
    rcases List.mem_map.mp hm with ⟨code,hcode,rfl⟩
    exact hs _ (List.mem_map.mpr ⟨code,List.mem_append_right _ hcode,rfl⟩)
  have ho := hout _ (List.mem_cons_self)
  simp only [Instruction.Sat,Instruction.value,Affine.eval_constant,Nat.cast_one,
    h1,mul_one,Affine.eval_nil,add_zero] at ho
  apply HashTrace.hash_sound w Spec.params{n} 97634 {domain} {out}
    (evalState w input) states ops h1 checkpoints hlocal ?_ ho
  rw [initial]
  simp [evalState,h1,Array.map_append]

#print axioms template_sound
end {ns}
'''
    path=ROOT/f'lean/CircuitCorrectness/HashTrace{n}/All.lean'
    if args.check:
        if not path.exists() or path.read_text()!=content:
            raise SystemExit(f'Generated hash composition drift: {path}')
    else: path.write_text(content)
print('Generated composition for exact hash template rows; build required.')
