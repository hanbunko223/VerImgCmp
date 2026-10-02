#!/usr/bin/env python3
"""Generate kernel certificates for all actual hash wiring.

Each reflection obligation decodes at most one original 128-row chunk. Literal
coefficient-ID templates are themselves proved equal to the pinned export.
Python supplies proof candidates only; Lean's kernel verifies every equality.
"""
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
CHECK = parser.parse_args().check
def emit(path, content):
    if CHECK:
        assert path.read_text() == content, f"Generated certificate differs: {path}"
    else:
        path.write_text(content)
from pathlib import Path
from structured_check import Builder, reconstruct
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'lean/CircuitCorrectness/HashWiringCertificates'
if not CHECK:
    OUT.mkdir(exist_ok=True)
raw=json.loads((ROOT/'artifacts/step.json').read_text())['rows']
ids={x:i for i,x in enumerate(sorted({int(c) for r in raw for lc in r for _,c in lc}))}
def row_text(row):
    return '⟨'+','.join('['+','.join(f'({i},{ids[int(c)]})' for i,c in lc)+']' for lc in row)+'⟩'

def pieces(start,count):
    out=[];offset=0
    while offset<count:
        pos=start+offset;c=pos//128;skip=pos%128
        length=min(128-skip,count-offset)
        out.append((c,skip,count-offset,offset,length));offset+=length
    return out

def code_expr(c,skip,count):
    return f'((ConcreteBytes.Codes.rows ExportedData.chunk{c}.1 ExportedData.chunk{c}.2).drop {skip}).take {count}'

def checks(name,n,start,count,rename='id'):
    lines=[];ps=pieces(start,count)
    for j,(c,skip,left,offset,length) in enumerate(ps):
        lines += [f'def {name}Piece{j} := {code_expr(c,skip,left)}',
          f'theorem {name}Piece{j}_checked : {name}Piece{j} =',
          f'    ((literalCodes{n}.drop {offset}).take {length}).map ({rename}) := by decide','']
    return lines

lines=['import CircuitCorrectness.HashWiring','','set_option maxRecDepth 100000',
'set_option maxHeartbeats 10000000','namespace CircuitCorrectness.HashWiring','',
'def sliced : List (Nat × Nat) → Nat → Nat → List Row',
'  | [], _, _ => []',
'  | chunk::tail, skip, count =>',
'    ((ConcreteBytes.Codes.rows chunk.1 chunk.2).drop skip).take count ++',
'      sliced tail (skip-chunk.1) (count-(chunk.1-skip))','',
'''theorem sliced_eq (chunks : List (Nat × Nat)) (skip count : Nat) :
    ((chunks.flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop skip).take count =
      sliced chunks skip count := by
  induction chunks generalizing skip count with
  | nil => simp [sliced]
  | cons chunk tail ih =>
    simp only [List.flatMap_cons,List.drop_append,List.take_append,List.length_drop,
      ConcreteProgram.Coded.codes_length,ih,sliced]
''']
for n,start,count in [(8,77496,388),(2,78272,238)]:
    lines += [f'def literalCodes{n} : List Row := [', ',\n'.join(row_text(r) for r in raw[start:start+count]),']','']
    lines += checks(f'literal{n}',n,start,count)
    ps=pieces(start,count)
    lines += [f'theorem template{n}_literal : template{n}Codes = literalCodes{n} := by',
        f'  unfold template{n}Codes', '  rw [sliced_eq]',
        '  change '+ ' ++ '.join(f'literal{n}Piece{j}' for j in range(len(ps)))+' ++ [] = _',
        '  rw ['+', '.join(f'literal{n}Piece{j}_checked' for j in range(len(ps)))+']',
        '  simp only [List.map_id, List.append_nil]', '  rfl','']
lines += ['end CircuitCorrectness.HashWiring','']
emit(ROOT/'lean/CircuitCorrectness/HashWiringTemplates.lean','\n'.join(lines))

nodes=[]; original_hash=Builder.hash
def recorded_hash(self,xs):
    start=self.next;inputs=[next(iter(x)) for x in xs]
    assert all(x=={i:1} for x,i in zip(xs,inputs))
    result=original_hash(self,xs);nodes.append((len(xs),start,inputs));return result
Builder.hash=recorded_hash
reconstruct();assert len(nodes)==52
for g in range(13):
    imports=['import CircuitCorrectness.HashWiringTemplates']
    if g>=2:imports += [f'import CircuitCorrectness.HashWiringCertificates.Group{g-2:02d}']
    lines=imports+['','set_option maxRecDepth 100000','set_option maxHeartbeats 10000000',
      'namespace CircuitCorrectness.HashWiring','']
    for k in range(4*g,4*g+4):
        n,start,inputs=nodes[k];count=388 if n==8 else 238;ps=pieces(start-4,count)
        chunks=', '.join(f'ExportedData.chunk{c}' for c,_,_,_,_ in ps);skip=(start-4)%128
        lines += [f'def node{k:02d} : Node := ⟨{start}, {inputs}⟩',
          f'def codes{k:02d} : List Row :=',
          f'  (([{chunks}].flatMap (fun c => ConcreteBytes.Codes.rows c.1 c.2)).drop {skip}).take {count}',
          f'theorem window{k:02d} : codedWindow (node{k:02d}.start-4) {count} = codes{k:02d} := by rfl','']
        lines += checks(f'node{k:02d}',n,start-4,count,f'renameRow (rename{n} node{k:02d})')
        lines += [f'theorem checked{k:02d} : codes{k:02d} =',
          f'    template{n}Codes.map (renameRow (rename{n} node{k:02d})) := by',
          f'  rw [template{n}_literal]',f'  unfold codes{k:02d}', '  rw [sliced_eq]',
          '  change '+' ++ '.join(f'node{k:02d}Piece{j}' for j in range(len(ps)))+' ++ [] = _',
          '  rw ['+', '.join(f'node{k:02d}Piece{j}_checked' for j in range(len(ps)))+']',
          '  simp only [List.append_nil, ← List.map_append]', '  congr 1',
          f'theorem sat{k:02d} {{w : Assignment}} (hs : Exported.circuit.Sat w) :',
          f'    ∀ row ∈ template{n}, row.Sat (w ∘ rename{n} node{k:02d}) :=',
          f'  transport{n} node{k:02d} (window{k:02d}.trans checked{k:02d}) hs','']
    lines += ['end CircuitCorrectness.HashWiring',''];emit(OUT/f'Group{g:02d}.lean','\n'.join(lines))
lines=[f'import CircuitCorrectness.HashWiringCertificates.Group{g:02d}' for g in [11,12]]
lines += ['','namespace CircuitCorrectness.HashWiring','']
for label,n,ks in [('left',8,list(range(0,48,3))),('right',8,list(range(1,48,3))),('combine',2,list(range(2,48,3)))]:
    offset={'left':176,'right':564,'combine':952}[label]
    inputs={'left':'(List.range 8).map (fun i => 77324+1191*r+160+i)',
            'right':'(List.range 8).map (fun i => 77324+1191*r+168+i)',
            'combine':'[77324+1191*r+563,77324+1191*r+951]'}[label]
    lines += [f'def {label}Node (r : Nat) : Node := ⟨77324+1191*r+{offset}, {inputs}⟩',
      f'theorem {label}_sat (r : Fin 16) {{w : Assignment}} (hs : Exported.circuit.Sat w) :',
      f'    ∀ row ∈ template{n}, row.Sat (w ∘ rename{n} ({label}Node r.val)) := by','  fin_cases r']
    for k in ks:lines += [f'  · exact sat{k:02d} hs']
    lines += ['']
lines += ['end CircuitCorrectness.HashWiring',''];emit(OUT/'All.lean','\n'.join(lines))
print(('Checked' if CHECK else 'Generated') + ' templates and 52 exact hash wiring certificates with at most128 source rows per check.')
