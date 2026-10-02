#!/usr/bin/env python3
"""Generate kernel-checked deterministic-program certificates for all suffix rows."""
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
CHECK = parser.parse_args().check
def emit(path, content):
    if CHECK:
        assert path.read_text() == content, f"Generated certificate differs: {path}"
    else:
        path.write_text(content)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'lean/CircuitCorrectness/ProgramCertificates'
if not CHECK:
    OUT.mkdir(exist_ok=True)
for g in range(12):
    cs = list(range(540+20*g, min(540+20*(g+1),763)))
    imports = ['import CircuitCorrectness.ConcreteProgram']
    if g >= 4:
        imports.append(f'import CircuitCorrectness.ProgramCertificates.Group{g-4:02d}')
    lines = imports + ['', 'set_option maxRecDepth 100000', 'set_option maxHeartbeats 10000000',
        'namespace CircuitCorrectness.ProgramCertificates', 'open ConcreteProgram StraightLine ConcreteBytes', '']
    for c in cs:
        dst=128*c+4
        n=94 if c==762 else 128
        lines += [f'def rows{c} := Exported.decodeRows ExportedData.chunk{c}.1 ExportedData.chunk{c}.2',
            f'def codes{c} := Codes.rows ExportedData.chunk{c}.1 ExportedData.chunk{c}.2',
            f'def program{c} := Coded.program 97634 {dst} codes{c}',
            f'theorem checked{c} : Coded.checkRows 97634 {dst} codes{c} = true := by decide',
            f'theorem ordered{c} : Ordered 97634 {dst} program{c} := Coded.checked_ordered checked{c}',
            f'theorem length{c} : program{c}.length = {n} := by',
            f'  rw [program{c}, Coded.program_length]',
            f'  exact Coded.codes_length _ _',
            f'theorem correct{c} {{w : Assignment}} (h1 : w 97634 = 1) :',
            f'    (∀ row ∈ rows{c}, row.Sat w) ↔ Satisfies program{c} w := by',
            f'  rw [rows{c}, Codes.rows_expand]',
            f'  exact Coded.checked_correct checked{c} h1', '']
    lines += ['end CircuitCorrectness.ProgramCertificates','']
    emit(OUT/f'Group{g:02d}.lean', '\n'.join(lines))
print(('Checked' if CHECK else 'Wrote') + ' 223 suffix chunk certificates in 12 grouped Lean modules.')
cs=list(range(540,763))
lines=[f'import CircuitCorrectness.ProgramCertificates.Group{g:02d}' for g in range(8,12)]
lines += ['', 'set_option maxRecDepth 100000', 'set_option maxHeartbeats 10000000',
    'namespace CircuitCorrectness.ProgramCertificates', 'open ConcreteProgram StraightLine', '',
    'def rowChunks : List (List Row) := ['+', '.join(f'rows{c}' for c in cs)+']',
    'def programChunks : List Program := ['+', '.join(f'program{c}' for c in cs)+']',
    'def program : Program := programChunks.flatten', '',
    'theorem rowChunks_eq : rowChunks =',
    '    (ExportedData.chunks.toList.drop 540).map',
    '      (fun chunk => Exported.decodeRows chunk.1 chunk.2) := by rfl', '',
    'theorem actual_suffix : Exported.rows.drop 69120 = rowChunks.flatten := by',
    '  rw [rowChunks_eq, ← List.flatMap_def]',
    '  have hl : ((ExportedData.chunks.toList.take 540).flatMap',
    '      (fun chunk => Exported.decodeRows chunk.1 chunk.2)).length = 69120 := by',
    '    simp only [List.length_flatMap, Exported.decodeRows_length]',
    '    decide',
    '  have hs := List.take_append_drop 540 ExportedData.chunks.toList',
    '  unfold Exported.rows',
    '  conv_lhs => rw [← hs, List.flatMap_append]',
    "  exact List.drop_left' hl", '',
    'theorem chunks_correct : List.Forall₂',
    '    (fun rs ps => ∀ w : Assignment, w 97634 = 1 →',
    '      ((∀ row ∈ rs, row.Sat w) ↔ Satisfies ps w)) rowChunks programChunks := by',
    '  unfold rowChunks programChunks']
for c in cs:
    lines += [f'  refine .cons (fun w h1 => correct{c} h1) ?_']
lines += ['  exact .nil', '',
    'theorem correct {w : Assignment} (h1 : w 97634 = 1) :',
    '    (∀ row ∈ Exported.rows.drop 69120, row.Sat w) ↔ Satisfies program w := by',
    '  rw [actual_suffix]',
    '  exact flatten_correct chunks_correct h1', '',
    'theorem length : program.length = 28510 := by',
    '  simp only [program, programChunks, List.length_flatten, List.map_cons,',
    '    List.map_nil, List.sum_cons, List.sum_nil,',
    '    '+', '.join(f'length{c}' for c in cs)+']', '  rfl', '',
    'theorem ordered : Ordered 97634 69124 program := by',
    '  unfold program programChunks',
    '  simp only [List.flatten_cons, List.flatten_nil]',
]
for c in cs:
    lines += ['  apply (ordered_append _ _ _ _).mpr', f'  refine ⟨ordered{c}, ?_⟩', f'  rw [length{c}]', '  change Ordered 97634 '+str(min((c+1)*128+4,97634))+' _']
lines += ['  trivial', '',
    'theorem wellFormed : WellFormed (Frontier 69124 97634) program :=',
    '  ordered_wellFormed ordered', '',
    'theorem knownAfter : KnownAfter (Frontier 69124 97634) program = Frontier 97634 97634 := by',
    '  rw [ordered_knownAfter ordered, length]', '',
    'end CircuitCorrectness.ProgramCertificates', '']
p=OUT/'All.lean'
s='\n'.join(lines).replace('end CircuitCorrectness.ProgramCertificates', '''/-- A mathematical assignment satisfying every post-byte exported row. -/
theorem execution_satisfies (seed : Assignment) (h1 : seed 97634 = 1) :
    (∀ row ∈ Exported.rows.drop 69120, row.Sat (run program seed)) := by
  apply (correct ?_).mpr (run_satisfies wellFormed)
  exact (run_preserves_known wellFormed 97634 (Or.inr rfl)).trans h1

/-- Every already allocated wire, including the constant-one wire, is preserved. -/
theorem execution_preserves (seed : Assignment) :
    Agree (Frontier 69124 97634) (run program seed) seed :=
  run_preserves_known wellFormed

end CircuitCorrectness.ProgramCertificates''')
emit(p,s)
