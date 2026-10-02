#!/usr/bin/env python3
"""Lossless finite data encoding. No matrix rows are deleted or algebraically changed."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def generated():
    a=json.loads((ROOT/'artifacts/step.json').read_text())
    q=int(a['modulus'],16); n=a['variable_count']
    assert (q,n,a['one_wire'],a['constraint_count'])==(int('40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001',16),97635,97634,97630)
    assert len(a['rows'])==97630
    coefs=sorted({int(k) for r in a['rows'] for lc in r for _,k in lc})
    assert all(0<k<q for k in coefs)
    ids={k:i for i,k in enumerate(coefs)}
    chunks=[]
    for start in range(0,len(a['rows']),128):
        rs=a['rows'][start:start+128];words=[]
        for r in rs:
            assert len(r)==3
            for lc in r:
                assert [i for i,_ in lc]==sorted({i for i,_ in lc})
                assert all(0<=i<n for i,_ in lc)
                words.append(len(lc));words.extend(i+n*ids[int(k)] for i,k in lc)
        assert all(0<=x<2**32 for x in words)
        data=int.from_bytes(b''.join(w.to_bytes(4,'little') for w in words),'little')
        recovered=int.to_bytes(data,4*len(words),'little')
        assert [int.from_bytes(recovered[j:j+4],'little') for j in range(0,len(recovered),4)]==words
        chunks.append((len(rs),data))
    out=['import CircuitCorrectness.R1CS','set_option maxRecDepth 20000','namespace CircuitCorrectness.ExportedData',
      'def coefficientPool : Array Nat := #['+', '.join(map(str,coefs))+']']
    for i,(count,data) in enumerate(chunks):
        out.append(f'def chunk{i} : Nat × Nat := ({count}, 0x{data:x})')
    out.append('def chunks : Array (Nat × Nat) := #['+', '.join(f'chunk{i}' for i in range(len(chunks)))+']')
    
    pixel_names=[]
    for j in range(0,len(a['pixels']),128):
        name=f'pixelChunk{j//128}';pixel_names.append(name)
        out.append(f'def {name} : Array Nat := #['+', '.join(str(x['wire']) for x in a['pixels'][j:j+128])+']')
    out.append('def pixelWires : Array Nat := (#['+', '.join(pixel_names)+'] : Array (Array Nat)).flatten')
    out.append('def incoming : Array Nat := #['+', '.join(map(str,a['incoming_state']))+']')
    out.append('def outgoing : Array Nat := #['+', '.join(map(str,a['outgoing_state']))+']')
    out.append('end CircuitCorrectness.ExportedData')
    return '\n\n'.join(out)+'\n'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    dest=ROOT/'lean/CircuitCorrectness/ExportedData.lean';data=generated()
    if args.check: assert dest.read_text()==data, 'Lean row data differs from artifact'
    else:dest.write_text(data)
    print(json.dumps({'status':'pass','bytes':len(data),'rows':97630}))
