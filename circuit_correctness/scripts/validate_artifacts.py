#!/usr/bin/env python3
"""Strict artifact validation plus targeted malformed-artifact regression tests."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
Q=int('40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001',16)
def validate(a):
    assert a['format']=='poseidon97-step-r1cs-v1'
    assert int(a['modulus'],16)==Q
    assert (a['variable_count'],a['auxiliary_count'],a['constraint_count'],a['one_wire'])==(97635,97634,97630,97634)
    assert a['incoming_state']==[0,1,2,3]
    assert a['outgoing_state']==[97632,77323,2,97633]
    assert len(a['rows'])==97630 and len(a['pixels'])==7680
    assert a['pixels']==[{'row':r,'column':c,'channel':ch,'wire':4+9*((r*160+c)*3+ch)} for r in range(16) for c in range(160) for ch in range(3)]
    for row in a['rows']:
        assert len(row)==3
        for lc in row:
            prev=-1
            for i,k in lc:
                assert type(i) is int and prev<i<97635;prev=i
                assert isinstance(k,str) and k==str(int(k)) and 0<int(k)<Q
    digest=hashlib.sha256(json.dumps(a['rows'],separators=(',',':')).encode()).hexdigest()
    assert digest==a['rows_sha256']=='120153f7a3043425cc47718bbd175ac4f9c777cca833b4691f4f64d444393dd3'
    return True
if __name__=='__main__':
    a=json.loads((ROOT/'artifacts/step.json').read_text());validate(a)
    mutations={
      'output_wire':lambda x:x['outgoing_state'].__setitem__(0,0),
      'pixel_mapping':lambda x:x['pixels'][0].__setitem__('wire',5),
      'noncanonical_coefficient':lambda x:x['rows'][0][0][0].__setitem__(1,str(Q)),
      'out_of_range_index':lambda x:x['rows'][0][0][0].__setitem__(0,97635),
      'matrix_entry':lambda x:x['rows'][69120][0][0].__setitem__(1,'46'),
      'centering_constant':lambda x:x['rows'][69120][0][-1].__setitem__(1,'1'),
      'horner_order':lambda x:x['rows'][74241].__setitem__(0,[[1,'1']]),
      'poseidon_round_constant':lambda x:x['rows'][77496][0][0].__setitem__(1,'1'),
      'missing_row':lambda x:x['rows'].pop(),
      'one_wire':lambda x:x.__setitem__('one_wire',0)}
    results=[]
    for name,f in mutations.items():
        b=copy.deepcopy(a);f(b)
        try:validate(b)
        except (AssertionError,ValueError,KeyError,IndexError):results.append({'mutation':name,'rejected':True})
        else:raise RuntimeError('accepted malformed artifact '+name)
    out={'status':'pass','rows':97630,'terms':832541,'malformed_artifact_tests':results,
       'evidence':'validation and fixed-artifact identity checks, not formal semantic proofs'}
    (ROOT/'results/artifact_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'pass','malformed_cases_rejected':len(results)}))
