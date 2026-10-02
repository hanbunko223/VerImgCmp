#!/usr/bin/env python3
"""Independent executable mathematical reference; regression evidence, not a proof."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'artifacts/parameters.json').read_text())
Q=int(json.loads((ROOT/'artifacts/step.json').read_text())['modulus'],16)
def cv(x):
    if isinstance(x,str): return int.from_bytes(bytes.fromhex(x),'little')
    if isinstance(x,list): return [cv(v) for v in x]
    if isinstance(x,dict): return {k:cv(v) for k,v in x.items() if k not in ('s','ht')}
    return x
PARAM={n:cv(P[f'poseidon{n}']) for n in (2,8)}
def h(xs):
    n=len(xs); p=PARAM[n]; width=n+1; half=p['rf']//2
    domain={2:0x50414952,8:0x48415348}[n]; b=2**128-159
    tag=((2**31+n)*b+b*b+domain*b**3)%2**128
    s=[tag]+list(xs); offset=0
    for r in range(p['rf']+p['rp']):
        full=r<half or r>=half+p['rp']; last=r==p['rf']+p['rp']-1
        start=offset+width if r==0 else offset
        s=[(pow((v+(p['crc'][offset+i] if r==0 else 0))%Q,5,Q)+(0 if last else p['crc'][start+i]))%Q if full or i==0 else v for i,v in enumerate(s)]
        offset=start if last else start+(width if full else 1)
        if r==half-1: matrix=p['psm']
        elif half-1<r<half+p['rp']:
            sm=p['sm'][r-half]
            s=[sum(a*b for a,b in zip(s,sm['w_hat']))%Q]+[(s[j]+s[0]*sm['v_rest'][j-1])%Q for j in range(1,width)]
            continue
        else: matrix=p['mds']['m']
        s=[sum(s[i]*matrix[i][j] for i in range(width))%Q for j in range(width)]
    assert offset==len(p['crc'])
    return s[1]
def fixture(kind):
    state=0x97630123456789ab; x=[]
    for r in range(16):
        row=[]
        for c in range(160):
            pix=[]
            for ch in range(3):
                state^=(state<<13)&(2**64-1);state^=state>>7;state^=(state<<17)&(2**64-1)
                pix.append(0 if kind==0 else 255 if kind==1 else (255 if (r+c+ch)%2 else 0) if kind==2 else (255 if (r,c,ch)==(3,5,1) else 0) if kind==3 else state&255)
            row.append(pix)
        x.append(row)
    return x
def evaluate(x,z):
    row_hash=[]
    for row in x:
        pixels=[r+256*g+65536*b for r,g,b in row]
        chunks=[sum(pixels[10*c+j]*2**(24*j) for j in range(10)) for c in range(16)]
        row_hash.append(h([h(chunks[:8]),h(chunks[8:])]))
    digest=h([h(row_hash[:8]),h(row_hash[8:])]); out_hash=h([z[0],digest])
    A=P['dct']; M=P['multipliers']; acc=z[1]
    for r in range(16):
        for c in range(160):
            for ch in range(3):
                m=M[ch][r%8][c%8]
                if m:
                    val=m*sum(A[r%8][i]*(x[r//8*8+i][c//8*8+j][ch]-128)*A[c%8][j] for i in range(8) for j in range(8))
                    acc=(acc*z[2]+val)%Q
    return [out_hash,acc,z[2],(z[3]+1)%Q]
if __name__=='__main__':
    validation=json.loads((ROOT/'results/rust_validation.json').read_text())
    results=[]
    for case in validation['cases']:
        actual=evaluate(fixture(case['fixture']),list(map(int,case['incoming'])))
        assert actual==list(map(int,case['outgoing'])), (case['fixture'],actual,case['outgoing'])
        results.append({'fixture':case['fixture'],'independent_spec_matches_rust':True})
    (ROOT/'results/reference_validation.json').write_text(json.dumps({'status':'pass','cases':results},indent=2)+'\n')
    print(json.dumps({'status':'pass','cases':len(results)}))
