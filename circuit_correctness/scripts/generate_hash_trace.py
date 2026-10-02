#!/usr/bin/env python3
"""Generate literal round checkpoints. Every arithmetic/row identity is checked in Lean."""
import json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--check",action="store_true")
args=parser.parse_args()
def emit(path,text):
    if args.check:
        if not path.exists() or path.read_text()!=text:
            raise SystemExit(f"Generated hash trace drift: {path}")
    else: path.write_text(text)
from pathlib import Path
from structured_check import Builder, plus, const, weighted, scale, ONE
from reference import PARAM, ROOT, Q
raw=json.loads((ROOT/'artifacts/step.json').read_text())
# Read canonical pool directly from the generated Lean data.
import re
src=(ROOT/'lean/CircuitCorrectness/ExportedData.lean').read_text()
pooltext=src.split('def coefficientPool : Array Nat := #[',1)[1].split(']',1)[0]
poolvalues=[int(x.strip(),0) for x in pooltext.split(',') if x.strip()]
codes={v:i for i,v in enumerate(poolvalues)}
def lc(x):return '['+','.join(f'({i},{v})' for i,v in sorted(x.items()))+']'
def codeLC(x):return '['+','.join(f'({i},{codes[v]})' for i,v in sorted(x.items()))+']'
def prog(ops):return '['+','.join('⟨'+str(i)+','+lc(a)+','+lc(b)+','+lc(c)+'⟩' for i,a,b,c in ops)+']'
def crows(rows):return '['+','.join('⟨'+','.join(codeLC({i:int(v) for i,v in z}) for z in row)+'⟩' for row in rows)+']'
def state(xs,offset,n):return '⟨#['+','.join(lc(x) for x in xs)+f'],{offset},{n}⟩'
for n,start,inputs,domain in [(8,77500,list(range(77484,77492)),0x48415348),(2,78276,[77887,78275],0x50414952)]:
    root=ROOT/f'lean/CircuitCorrectness/HashTrace{n}';root.mkdir(exist_ok=True)
    p=PARAM[n];width=n+1;half=p['rf']//2;b=Builder();b.next=start
    bb=2**128-159
    xs=[const(((2**31+n)*bb+bb*bb+domain*bb**3)%2**128)]+[{i:1} for i in inputs]
    offset=0
    emit(root/'Initial.lean','import CircuitCorrectness.PoseidonProgram\nimport CircuitCorrectness.ConcreteProgram\nnamespace CircuitCorrectness.HashTrace'+str(n)+'\nopen PoseidonProgram StraightLine\ndef state00 : PoseidonProgram.State := '+state(xs,offset,b.next)+'\nend CircuitCorrectness.HashTrace'+str(n)+'\n')
    for r in range(p['rf']+p['rp']):
        full=r<half or r>=half+p['rp'];last=r+1==p['rf']+p['rp'];begin=offset+width if r==0 else offset
        rowstart=len(b.rows);ops=[]
        def mul(a,bb,post=0):
            dst=b.next;ops.append((dst,a,bb,const(post)));return b.mul(a,bb,post)
        for i in range(width if full else 1):
            x=plus(xs[i],const(p['crc'][offset+i] if r==0 else 0))
            sq=mul(x,x);quad=mul(sq,sq);xs[i]=mul(x,quad,0 if last else p['crc'][begin+i])
        offset=begin if last else begin+(width if full else 1)
        if r==half-1:mat=p['psm']
        elif half-1<r<half+p['rp']:
            sm=p['sm'][r-half]
            xs=[weighted(xs,sm['w_hat'])]+[plus(xs[j],scale(xs[0],sm['v_rest'][j-1])) for j in range(1,width)]
            mat=None
        else:mat=p['mds']['m']
        if mat is not None:xs=[weighted(xs,[mat[i][j] for i in range(width)]) for j in range(width)]
        assert b.rows[rowstart:]==raw['rows'][start-4+rowstart:start-4+len(b.rows)]
        prev='Initial' if r==0 else f'Round{r-1:02}'
        content=f'''import CircuitCorrectness.HashTrace{n}.{prev}
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace CircuitCorrectness.HashTrace{n}
open PoseidonProgram StraightLine
def state{r+1:02} : PoseidonProgram.State := {state(xs,offset,b.next)}
def ops{r:02} : Program := {prog(ops)}
def codes{r:02} : List Row := {crows(b.rows[rowstart:])}
theorem round_exact{r:02} : PoseidonProgram.round Spec.params{n} 97634 state{r:02} {r} =
    (state{r+1:02},ops{r:02}) := by decide +kernel
theorem rows_exact{r:02} : ConcreteProgram.extractRows 97634 state{r:02}.next
    (codes{r:02}.map ConcreteBytes.Codes.expandRow) = some ops{r:02} := by decide +kernel
end CircuitCorrectness.HashTrace{n}
'''
        emit(root/f'Round{r:02}.lean',content)
    out=b.linear(xs[1],preserve=True)
    emit(root/'Output.lean',f'import CircuitCorrectness.HashTrace{n}.Round{r:02}\nnamespace CircuitCorrectness.HashTrace{n}\ndef outputCodes : List Row := {crows(b.rows[-1:])}\nend CircuitCorrectness.HashTrace{n}\n')
print('Generated bounded hash-round checkpoints for both arities; Lean checking still required.')
