#!/usr/bin/env python3
"""Encode pinned public constants as Lean numerals, without evaluating constraints."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'artifacts/parameters.json').read_text())
q=int(json.loads((ROOT/'artifacts/step.json').read_text())['modulus'],16)
def scalar(s):
    n=int.from_bytes(bytes.fromhex(s),'little')
    assert n<q
    return n
def arr(x):
    if isinstance(x,list): return '#['+', '.join(arr(v) for v in x)+']'
    return str(x)
lines=['import CircuitCorrectness.Field','namespace CircuitCorrectness.Parameters']
for key,name in [('dct','dct'),('divisors','divisors'),('multipliers','multipliers')]:
    ty='Array (Array Int)' if key=='dct' else 'Array (Array (Array Nat))'
    lines.append(f'def {name} : {ty} := {arr(p[key])}')
for arity in (2,8):
    c=p[f'poseidon{arity}']
    def cv(x): return [cv(v) for v in x] if isinstance(x,list) else scalar(x)
    for src,dest in [(c['crc'],'keys'),(c['mds']['m'],'mds'),(c['psm'],'preSparse'),([s['w_hat'] for s in c['sm']],'sparseW'),([s['v_rest'] for s in c['sm']],'sparseV')]:
        ty='Array Nat' if dest=='keys' else 'Array (Array Nat)'
        lines.append(f'def {dest}{arity} : {ty} := {arr(cv(src))}')
lines.append('end CircuitCorrectness.Parameters')
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
dest=ROOT/'lean/CircuitCorrectness/Parameters.lean';data='\n\n'.join(lines)+'\n'
if args.check: assert dest.read_text()==data, 'Lean parameter data differs'
else: dest.write_text(data)
