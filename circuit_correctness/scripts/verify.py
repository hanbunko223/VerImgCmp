#!/usr/bin/env python3
"""Reproduction driver. Default exits nonzero until FULL certification is present."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from sources import check as source_check
from audit import CORE_MODULES
ROOT=Path(__file__).resolve().parents[1]
from toolchain import lake
LAKE=lake()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development',action='store_true',help='Run foundation checks; never report full certification')
    parser.add_argument('--artifact-only',action='store_true',help='Skip Rust/source checks; use committed constraint data')
    args=parser.parse_args();records=[];result={'status':'running','certified':False,'checks':records}
    (ROOT/'results').mkdir(exist_ok=True)
    dest=ROOT/'results/verification.json'
    result['mode']='development' if args.development else 'full'
    result['artifact_only']=args.artifact_only
    def save():dest.write_text(json.dumps(result,indent=2)+'\n')
    def run(name,cmd,cwd=ROOT,expected=0):
        started=time.perf_counter()
        log=ROOT/'results'/f'{name}.log'
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(ROOT/'target')
        with log.open('w') as f:
            p=subprocess.run(['python3',str(ROOT/'scripts/measure.py'),str(ROOT/'results'/f'{name}.resources.json'),*map(str,cmd)],cwd=cwd,env=env,stdout=f,stderr=f)
        resources=json.loads((ROOT/'results'/f'{name}.resources.json').read_text())
        row={'name':name,'exit_code':p.returncode,'seconds':time.perf_counter()-started,
            'maximum_child_rss_bytes':resources['maximum_child_rss_bytes'],'log':str(log.relative_to(ROOT))}
        records.append(row);save();print(json.dumps(row),flush=True)
        if p.returncode!=expected:raise RuntimeError(f'{name} failed; see {log}')
        return row
    save()
    try:
        if not args.artifact_only:
            result['source_before']=source_check()
            run('rust_export',['cargo','run','--offline','--locked','--release','--manifest-path',ROOT/'exporter/Cargo.toml','--','check-export'])
        run('artifact_validation',['python3',ROOT/'scripts/validate_artifacts.py'])
        run('parameter_data_check',['python3',ROOT/'scripts/generate_parameters.py','--check'])
        run('row_data_check',['python3',ROOT/'scripts/generate_rows.py','--check'])
        for generator in ['byte_certificates','program_certificates','dct_certificates',
                          'packing_certificates','hash_trace','hash_trace_composition','hash_wiring']:
            run(generator+'_data_check',['python3',ROOT/'scripts'/('generate_'+generator+'.py'),'--check'])
        run('structured_comparison',['python3',ROOT/'scripts/structured_check.py'])
        if not args.artifact_only:run('python_reference',['python3',ROOT/'scripts/reference.py'])
        run('lean_foundation',[LAKE,'build','CircuitCorrectness',*CORE_MODULES],ROOT/'lean')
        run('lean_certificate_rejection',[LAKE,'build',
            'CircuitCorrectness.ProgramCertificates.Rejection',
            'CircuitCorrectness.DctProgramCertificates.Tests',
            'CircuitCorrectness.HashWiringCertificates.Rejection',
            'CircuitCorrectness.PackingCertificates.Rejection'],ROOT/'lean')
        run('lean_differential',[LAKE,'env','lean','--run','Differential.lean'],ROOT/'lean')
        run('axiom_audit',['python3',ROOT/'scripts/audit.py'])
        if not args.artifact_only:result['source_after']=source_check()
        result['foundation_checks_passed']=True
        if args.development:
            result['status']='incomplete';result['reason']='Development mode does not execute the full theorem and transitive axiom acceptance gates.'
            save();return 0
        run('full_certification',[LAKE,'build','Certification'],ROOT/'lean')
        run('full_axiom_audit',['python3',ROOT/'scripts/audit.py','--full'])
        if not args.artifact_only:result['source_after']=source_check()
        result['status']='pass';result['certified']=True
        save();return 0
    except (Exception,SystemExit) as e:
        result['status']='incomplete';result['reason']=str(e)
        if not args.artifact_only:
            try:result['source_after']=source_check()
            except (Exception,SystemExit) as source_error:result['source_after']={'status':'fail','reason':str(source_error)}
        save();print(str(e),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
