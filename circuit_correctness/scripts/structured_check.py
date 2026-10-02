#!/usr/bin/env python3
"""Reconstruct all rows from a structured, independently written circuit template.

This check is executable regression evidence. It is NOT the missing Lean theorem
that this template implements Spec.step, nor a kernel-checked row certificate.
"""
import json
from pathlib import Path
from reference import P, PARAM, Q, ROOT
ONE=97634

def lc(terms):
    out={}
    for i,k in terms: out[i]=(out.get(i,0)+k)%Q
    return {i:k for i,k in out.items() if k}
def plus(*xs): return lc(t for x in xs for t in x.items())
def scale(x,k): return lc((i,k*v) for i,v in x.items())
def const(k): return lc([(ONE,k)])
def weighted(xs,coeffs): return plus(*(scale(x,k) for x,k in zip(xs,coeffs)))
class Builder:
    def __init__(self): self.next=0;self.rows=[];self.groups=[]
    def alloc(self):
        i=self.next;self.next+=1
        return {i:1}
    def enforce(self,a,b,c):
        self.rows.append([[[i,str(k)] for i,k in sorted(x.items())] for x in (a,b,c)])
    def linear(self,x,preserve=False):
        y=self.alloc()
        self.enforce(x if preserve else plus(x,scale(y,-1)),const(1),y if preserve else {})
        return y
    def mul(self,a,b,post=0):
        y=self.alloc();self.enforce(a,b,plus(y,const(-post)));return y
    def equal(self,a,b): self.enforce(plus(a,scale(b,-1)),const(1),{})
    def hash(self,xs):
        start=len(self.rows);n=len(xs);p=PARAM[n];width=n+1;half=p['rf']//2
        domain={2:0x50414952,8:0x48415348}[n];b=2**128-159
        s=[const(((2**31+n)*b+b*b+domain*b**3)%2**128)]+xs
        offset=0
        for r in range(p['rf']+p['rp']):
            full=r<half or r>=half+p['rp'];last=r==p['rf']+p['rp']-1
            begin=offset+width if r==0 else offset
            for i in range(width if full else 1):
                x=plus(s[i],const(p['crc'][offset+i] if r==0 else 0))
                l2=self.mul(x,x);l4=self.mul(l2,l2)
                s[i]=self.mul(x,l4,0 if last else p['crc'][begin+i])
            offset=begin if last else begin+(width if full else 1)
            if r==half-1: matrix=p['psm']
            elif half-1<r<half+p['rp']:
                sm=p['sm'][r-half]
                s=[weighted(s,sm['w_hat'])]+[plus(s[j],scale(s[0],sm['v_rest'][j-1])) for j in range(1,width)]
                continue
            else: matrix=p['mds']['m']
            s=[weighted(s,[matrix[i][j] for i in range(width)]) for j in range(width)]
        y=self.linear(s[1],preserve=True)
        self.groups.append({'kind':f'poseidon{n}','start':start,'count':len(self.rows)-start,'output':next(iter(y))})
        return y

def reconstruct(parameters=P):
    b=Builder();state=[b.alloc() for _ in range(4)];pixels={};a=parameters['dct'];m=parameters['multipliers']
    for r in range(16):
        for c in range(160):
            for ch in range(3):
                x=b.alloc();bits=[b.alloc() for _ in range(8)];pixels[r,c,ch]=x
                for bit in bits:b.enforce(plus(const(1),scale(bit,-1)),bit,{})
                b.equal(weighted(bits,[2**i for i in range(8)]),x)
    left={}
    for ch in range(3):
        for br in range(2):
            for bc in range(20):
                for r in range(8):
                    if not any(m[ch][r]): continue
                    for c in range(8):
                        inputs=[pixels[br*8+k,bc*8+c,ch] for k in range(8)]
                        left[ch,br*8+r,bc*8+c]=b.linear(plus(weighted(inputs,a[r]),const(-128*sum(a[r]))))
    acc=state[1]
    for r in range(16):
        for c in range(160):
            for ch in range(3):
                k=m[ch][r%8][c%8]
                if not k:continue
                output=b.alloc()
                coeff=weighted([left[ch,r,c//8*8+j] for j in range(8)],[k*v for v in a[c%8]])
                b.enforce(acc,state[2],plus(output,scale(coeff,-1)));acc=output
    hashes=[]
    for r in range(16):
        packed=[b.linear(weighted([pixels[r,c,ch] for ch in range(3)],[1,256,65536])) for c in range(160)]
        chunks=[b.linear(weighted(packed[10*c:10*c+10],[2**(24*j) for j in range(10)])) for c in range(16)]
        rh=b.hash([b.hash(chunks[:8]),b.hash(chunks[8:])]);hashes.append(rh)
        b.equal(rh,b.alloc())
    digest=b.hash([b.hash(hashes[:8]),b.hash(hashes[8:])]);b.equal(digest,b.alloc())
    h=b.hash([state[0],digest]);t=b.linear(plus(state[3],const(1)))
    outputs=[next(iter(x)) for x in (h,acc,state[2],t)]
    return b,outputs,pixels

def check():
    raw=json.loads((ROOT/'artifacts/step.json').read_text());b,outputs,pixels=reconstruct()
    assert b.next==raw['auxiliary_count']==97634
    assert len(b.rows)==raw['constraint_count']==97630
    for i,(x,y) in enumerate(zip(b.rows,raw['rows'])):
        assert x==y, f'structured row mismatch at {i}'
    assert outputs==raw['outgoing_state']
    assert [next(iter(pixels[p['row'],p['column'],p['channel']])) for p in raw['pixels']]==[p['wire'] for p in raw['pixels']]
    result={'status':'pass','evidence':'Python structural comparison, not formal certification',
            'rows_checked':len(b.rows),'outputs':outputs,'poseidon_gadgets':b.groups}
    (ROOT/'results/structured_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='poseidon_gadgets'}))
    return result
if __name__=='__main__':check()
