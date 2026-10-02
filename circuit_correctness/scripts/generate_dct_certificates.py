#!/usr/bin/env python3
"""Generate DCT code tables and kernel-checked exported-row certificates."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'lean/CircuitCorrectness/DctProgramCertificates'
parser=argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
CHECK=parser.parse_args().check
if not CHECK: OUT.mkdir(exist_ok=True)
def write(path, text):
    if CHECK:
        assert path.read_text() == text, f'DCT certificate differs: {path}'
    else:
        path.write_text(text)
a=json.loads((ROOT/'artifacts/step.json').read_text())
p=json.loads((ROOT/'artifacts/parameters.json').read_text())
q=int(a['modulus'],16)
pool=sorted({int(k) for r in a['rows'] for lc in r for _,k in lc})
ids={k:i for i,k in enumerate(pool)}
mat=p['dct']; mult=p['multipliers']
mc=[ids[k%q] for row in mat for k in row]
fc=[ids.get((-mult[ch][r][c]*mat[c][k])%q,0) for ch in range(3) for r in range(8) for c in range(8) for k in range(8)]
data='''import CircuitCorrectness.DctProgram
import CircuitCorrectness.ByteCertificates.Decode
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace CircuitCorrectness.DctProgramCertificates
open DctProgram ConcreteBytes.Codes
'''
data+=f'def matrixCodes : Array Nat := #[{", ".join(map(str,mc))}]\n'
data+='def fusedCodes : Array (Array Nat) := #['+ ', '.join('#['+', '.join(map(str,fc[i:i+64]))+']' for i in range(0,len(fc),64))+']\n'
data+='''def matrixCode (r i : Nat) : Nat := matrixCodes[8*r+i]!
def fusedCode (ch r c k : Nat) : Nat := (fusedCodes[ch*8+r]!)[c*8+k]!
theorem matrixCode_expand : ∀ r i : Fin 8,
    ExportedData.coefficientPool[matrixCode r.val i.val]! = encodeInt (Spec.matrix r.val i.val) := by decide

'''
data+=f'def centerCode : Nat := {ids[(-46080)%q]}\n'
data+='''theorem centerCode_expand : ExportedData.coefficientPool[centerCode]! = encodeInt (-46080) := by decide

theorem matrix_row_sum : ∀ r : Fin 8,
    ((List.range 8).map fun i => Spec.matrix r.val i).sum = if r.val = 0 then 360 else 0 := by decide

end CircuitCorrectness.DctProgramCertificates
'''
write(OUT/'Data.lean', data)
for ch in range(3):
    lines=['import CircuitCorrectness.DctProgramCertificates.'+('Data' if ch==0 else f'Fused{ch-1}'),
           'set_option maxRecDepth 100000', 'set_option maxHeartbeats 0',
           'namespace CircuitCorrectness.DctProgramCertificates', 'open DctProgram']
    for r in range(8):
        lines += [f'theorem fusedCode_expand_{ch}_{r} : ∀ c k : Fin 8,',
          f'    Spec.retained {ch} {r} c.val = true →',
          f'    ExportedData.coefficientPool[fusedCode {ch} {r} c.val k.val]! =',
          f'      encodeInt (-((Spec.multiplier {ch} {r} c.val : Int) * Spec.matrix c.val k.val)) := by decide']
    lines+=['end CircuitCorrectness.DctProgramCertificates']
    write(OUT/f'Fused{ch}.lean', '\n'.join(lines)+'\n')
print('Generated DCT coefficient-code tables and expansion certificates.')
coords=[(r,8*b+c,ch) for r in range(16) for b in range(20) for c in range(8) for ch in range(3) if mult[ch][r%8][c]!=0]
coordtext='import CircuitCorrectness.DctProgramCertificates.Base\nset_option maxRecDepth 100000\nset_option maxHeartbeats 0\nnamespace CircuitCorrectness.DctProgramCertificates\n'
for r in range(16):
    coordtext+=f'def retainedLiteral{r} : List (Nat × Nat × Nat) := ['+', '.join(f'({rr},{c},{ch})' for rr,c,ch in coords if rr==r)+']\n'
coordtext+='def retainedLiteral : List (Nat × Nat × Nat) := '+ ' ++ '.join(f'retainedLiteral{r}' for r in range(16))+'\n'
for r in range(16):
    coordtext+=f'theorem retainedLiteral{r}_eq : retainedLiteral{r} = (List.range 20).flatMap (fun b => (Spec.retainedRow ({r}%8)).map (fun (c,ch) => ({r},8*b+c,ch))) := by decide\n'
coordtext+='theorem retainedLiteral_eq : retainedLiteral = Spec.retainedCoordinates := by\n  unfold retainedLiteral\n  rw ['+', '.join(f'retainedLiteral{r}_eq' for r in range(16))+']\n  rfl\nend CircuitCorrectness.DctProgramCertificates\n'
write(OUT/'Coordinates.lean',coordtext)
for g in range(13):
    lines=['import CircuitCorrectness.DctProgramCertificates.'+('Coordinates' if g >= 8 else 'Base')]
    if g >= 2: lines += [f'import CircuitCorrectness.DctProgramCertificates.Group{g-2:02d}']
    lines += ['set_option maxRecDepth 100000', 'set_option maxHeartbeats 0',
              'namespace CircuitCorrectness.DctProgramCertificates', 'open ConcreteBytes.Codes']
    for c in range(540+5*g,540+5*(g+1)):
        stage='first' if c<580 else 'horner'
        off=128*(c-(540 if c<580 else 580))
        count=128
        lhs=f'rows ExportedData.chunk{c}.1 ExportedData.chunk{c}.2'
        if c==604: lhs=f'({lhs}).take 8'
        lines += [f'theorem chunk{c} : {lhs} =',
            f'    ({stage}Rows.drop {off}).take {count} := by'+('\n  rw [hornerRows, ← retainedLiteral_eq]\n  decide' if stage=='horner' else ' decide')]
    lines += ['end CircuitCorrectness.DctProgramCertificates']
    write(OUT/f'Group{g:02d}.lean', '\n'.join(lines)+'\n')
print('Generated all 65 first-stage and Horner concrete chunk certificates.')
# The final eight DCT rows share the export chunk with later hash constraints.
lines=[f'import CircuitCorrectness.DctProgramCertificates.Group{g:02d}' for g in range(9,13)]
lines+=['set_option maxRecDepth 100000', 'set_option maxHeartbeats 4000000',
        'namespace CircuitCorrectness.DctProgramCertificates', 'open DctProgram ConcreteBytes.Codes',
'''theorem first_chunk (c : Fin 40) :
    let chunk := ExportedData.chunks[540+c.val]!
    rows chunk.1 chunk.2 = (firstRows.drop (128*c.val)).take 128 := by
  fin_cases c''']
lines += [f'  · exact chunk{540+c}' for c in range(40)]
lines += ['''theorem horner_chunk (c : Fin 25) (row : Row) :
    row ∈ (hornerRows.drop (128*c.val)).take 128 →
    let chunk := ExportedData.chunks[580+c.val]!
    row ∈ rows chunk.1 chunk.2 := by
  fin_cases c''']
lines += [f'  · change row ∈ (hornerRows.drop {128*c}).take 128 → row ∈ rows ExportedData.chunk{580+c}.1 ExportedData.chunk{580+c}.2\n    rw [← chunk{580+c}]; exact id' for c in range(24)]
lines += ['  · change row ∈ (hornerRows.drop 3072).take 128 → row ∈ rows ExportedData.chunk604.1 ExportedData.chunk604.2\n    rw [← chunk604]; exact List.mem_of_mem_take', 'end CircuitCorrectness.DctProgramCertificates']
write(OUT/'Chunks.lean', '\n'.join(lines)+'\n')
