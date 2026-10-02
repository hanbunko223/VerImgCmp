#!/usr/bin/env python3
"""Fresh-process macOS HD benchmark; no modification of historical results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from resources import measured

P = Path(__file__).resolve().parents[1]
ROOT = P.parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    out = P / 'benchmarks/runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    binary = P / 'target/release/poseidon_97'
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
    def command(verb, *rest):
        return [str(binary), verb, '--resolution', 'HD', *map(str, rest)]
    def plain(cmd):
        subprocess.run(list(map(str, cmd)), check=True, env=env)
    plain([sys.executable, ROOT / 'create_input.py', 'HD', out / 'image.json'])
    plain(command('digest', '--input', out / 'image.json', '--output', out / 'candidate-digest.json'))
    plain(command('coefficients', '--input', out / 'image.json', '--output', out / 'coefficients.json'))
    common = ['--threads', str(args.threads), '--segment-steps', '360',
              '--input-digest', out / 'candidate-digest.json', '--coefficients', out / 'coefficients.json']
    records = []
    def measure(name, cmd, metric):
        row = measured(cmd, out / (name + '.log'), env)
        row['name'] = name
        if metric.exists():
            row['metrics'] = json.loads(metric.read_text())
        records.append(row)
        (out / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        if row['exit_code'] or row['resource_failure']:
            raise RuntimeError(f'{name} failed; see {out}')
    for label in ('warmup', 'run1', 'run2', 'run3'):
        folder = out / label
        folder.mkdir()
        measure(label, command('bench', *common, '--input', out / 'image.json',
                               '--output', folder / 'proof.json', '--spartan-compress',
                               '--metrics', folder / 'prove.json'), folder / 'prove.json')
        for kind, filename in [('recursive', 'proof.json'), ('compressed', 'proof.spartan.json')]:
            metric = folder / f'verify-{kind}.json'
            measure(f'{label}-verify-{kind}', command('verify', *common, '--proof', folder / filename,
                                                     '--metrics', metric), metric)
    groups = {'prove': ['run1', 'run2', 'run3'],
              'verify_recursive': [f'run{i}-verify-recursive' for i in range(1, 4)],
              'verify_compressed': [f'run{i}-verify-compressed' for i in range(1, 4)]}
    summary = {}
    for group, names in groups.items():
        rows = [r for r in records if r['name'] in names]
        values = [{**{k:v for k,v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                   **r.get('metrics', {})} for r in rows]
        summary[group] = {key: {'median': statistics.median(nums), 'min': min(nums), 'max': max(nums)}
                          for key in set.intersection(*(set(v) for v in values))
                          if all(isinstance(v[key], (int, float)) and not isinstance(v[key], bool) for v in values)
                          for nums in [[v[key] for v in values]]}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    paths = [P/'Cargo.lock', binary, ROOT/'samples/HD.png', out/'image.json', out/'coefficients.json']
    (out / 'provenance.json').write_text(json.dumps({'platform': platform.platform(), 'python': sys.version,
        'threads': args.threads, 'steps': 360, 'proof_count': 1,
        'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
        'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'flags': {'RUSTFLAGS': '-C target-cpu=native', 'LTO': 'thin', 'codegen_units': 1}}, indent=2)+'\n')
    print(out)

if __name__ == '__main__':
    main()
