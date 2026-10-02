#!/usr/bin/env python3
"""Generate kernel-checked identities for all original RGB/chunk packing rows.

The exported coefficient pool supplies only codes. Semantic expansion and row
meaning are proved independently in PackingCertificates/Base.lean.
"""
import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'lean/CircuitCorrectness/PackingCertificates'
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
CHECK = parser.parse_args().check
if not CHECK:
    OUT.mkdir(exist_ok=True)
def write(name, text):
    path = OUT / name
    if CHECK:
        assert path.read_text() == text, f'Packing certificate differs: {path}'
    else:
        path.write_text(text)
artifact = json.loads((ROOT / 'artifacts/step.json').read_text())
pool = sorted({int(k) for row in artifact['rows'] for lc in row for _, k in lc})
ids = {k: i for i, k in enumerate(pool)}
pixel = [ids[2**(8*i)] for i in range(3)]
chunk = [ids[2**(24*i)] for i in range(10)]
write('Data.lean', '''import CircuitCorrectness.HashProgram
import CircuitCorrectness.ByteCertificates.Decode
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.PackingCertificates
'''+f'def pixelCodes : Array Nat := #[{", ".join(map(str,pixel))}]\n'+
 f'def chunkCodes : Array Nat := #[{", ".join(map(str,chunk))}]\n'+'''
theorem pixelCode_expand : ∀ i : Fin 3,
    ExportedData.coefficientPool[pixelCodes[i.val]!]! = 2^(8*i.val) := by
  intro i; fin_cases i <;> rfl
theorem chunkCode_expand : ∀ i : Fin 10,
    ExportedData.coefficientPool[chunkCodes[i.val]!]! = 2^(24*i.val) := by
  intro i; fin_cases i <;> rfl
end CircuitCorrectness.PackingCertificates
''')
for r in range(16):
    start=77320+1191*r
    left=176
    pos=start
    segments=[]
    while left:
        c,offset=divmod(pos,128)
        take=min(left,128-offset)
        segments.append((c,offset,take))
        pos+=take
        left-=take
    imports=['import CircuitCorrectness.PackingCertificates.Base']
    if r:
        imports += [f'import CircuitCorrectness.PackingCertificates.Row{r-1:02}']
    lines=imports+['set_option maxRecDepth 100000','set_option maxHeartbeats 10000000',
      'namespace CircuitCorrectness.PackingCertificates','open ConcreteBytes.Codes',
      f'theorem packing_rows_{r:02} : packingRows {r} =']
    exprs=[f'((rows ExportedData.chunk{c}.1 ExportedData.chunk{c}.2).drop {offset}).take {take}' for c,offset,take in segments]
    expr=exprs[-1]
    for head in reversed(exprs[:-1]):
        expr=head+' ++\n    ('+expr+')'
    lines += ['    '+expr+' := by decide',
       f'theorem packing_mem_{r:02} (row : Row) (hr : row ∈ packingRows {r}) :',
       '    expandRow row ∈ Exported.rows := by',f'  rw [packing_rows_{r:02}] at hr']
    for k,(c,offset,take) in enumerate(segments):
        if k<len(segments)-1:
            # Parentheses above fix the right-associated slice decomposition
            lines+=['  rcases List.mem_append.mp hr with hs | hr',
              f'  · exact coded_row_mem {c} (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hs))']
        else:
            lines += [f'  exact coded_row_mem {c} (by decide) row (List.mem_of_mem_drop (List.mem_of_mem_take hr))']
    lines += ['end CircuitCorrectness.PackingCertificates','']
    write(f'Row{r:02}.lean','\n'.join(lines))
print('Checked' if CHECK else 'Generated', 'all 16 RGB/chunk packing row certificates.')
