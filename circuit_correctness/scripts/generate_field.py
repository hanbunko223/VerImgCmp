#!/usr/bin/env python3
"""Generate Pratt proof terms; Python's computations are rechecked by Lean.

Large-factor data: CompElliptic/Fields/Pasta.lean (Daira-Emma Hopwood),
MIT/Apache-2.0. The generated proofs use only the local Pratt checker and mathlib.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'lean/CircuitCorrectness'
DEST.mkdir(exist_ok=True)
Q = int('40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001',16)
KNOWN = {
 Q: {2:32,3:2,1709:1,24859:1,1690502597179744445941507:1,10427374428728808478656897599072717:1},
 1690502597179744445941507: {2:1,3:1,13:1,4129989133:1,5247740253619:1},
 4129989133:{2:2,3:1,359:1,958679:1},
 5247740253619:{2:1,3:3,17:1,71:1,80513981:1},
 10427374428728808478656897599072717:{2:2,294793:1,4229279:1,399082391:1,5239247429827:1},
 5239247429827:{2:1,3:2,757:1,12149:1,31649:1}}

def factors(p):
    if p in KNOWN: return KNOWN[p]
    n=p-1; out={}; d=2
    while d*d<=n:
        while n%d==0: out[d]=out.get(d,0)+1; n//=d
        d+=1
    if n>1: out[n]=1
    return out

def generate():
    checker=(ROOT/'references/PrattCertificate.lean').read_text().split('end New')[0]+'end New\n'
    checker=checker.replace('public import Mathlib.Tactic.ReduceModChar','public import Mathlib.Tactic.ReduceModChar\npublic import Mathlib.Tactic.NormNum')
    (DEST/'Pratt.lean').write_text(checker)
    declarations=[]; done=set()
    def node(p):
        if p in done:return
        done.add(p)
        if p==2:
            declarations.append('theorem prime_2 : Nat.Prime 2 := Nat.prime_two\n');return
        fs=factors(p)
        for n in fs:node(n)
        from math import prod
        assert prod(n**k for n,k in fs.items())==p-1
        a=next(a for a in range(2,1000) if pow(a,p-1,p)==1 and all(pow(a,(p-1)//n,p)!=1 for n in fs))
        exprs=[f'{n} ^ {k}' for n,k in fs.items()]
        lines=[f'theorem prime_{p} : Nat.Prime {p} := by',
          f"  refine PrattCertificate'.out ⟨{a}, (by reduce_mod_char), ?_⟩",
          '  refine .split ['+', '.join(exprs)+'] (fun r hr => ?_) (by norm_num)',
          '  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr',
          '  rcases hr with '+' | '.join('rfl' for i in range(len(fs)))]
        for i,(n,k) in enumerate(fs.items()):
            lines+= [f'  · exact .prime {n} {k} _ prime_{n} (by reduce_mod_char; decide) (by norm_num)']
        declarations.append('\n'.join(lines)+'\n')
    node(Q)
    text='import CircuitCorrectness.Pratt\n\nset_option maxRecDepth 4096\nset_option maxHeartbeats 4000000\nnamespace CircuitCorrectness\nnamespace Primality\n\n'+'\n'.join(declarations)
    text+='\nend Primality\n\ndef modulus : Nat := '+str(Q)+'\n'
    text+=f'theorem modulus_prime : Nat.Prime modulus := Primality.prime_{Q}\n'
    text+='instance : Fact modulus.Prime := ⟨modulus_prime⟩\nabbrev F := ZMod modulus\n'
    text+='theorem chunk_fits : 2 ^ 240 < modulus := by decide\n'
    text+='theorem signed_coefficients_fit : 2 * 1554357600 < modulus := by decide\n'
    text+='end CircuitCorrectness\n'
    (DEST/'Field.lean').write_text(text)
if __name__=='__main__':generate()
