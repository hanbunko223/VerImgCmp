#!/usr/bin/env python3
"""Fingerprint protected sources. Initialization is explicit and never part of verify."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MANIFEST = ROOT / 'artifacts/source_manifest.json'

def inventory():
    paths = []
    for project in ('poseidon_97', 'nova60/Nova'):
        # Manifest names are immutable certified identities; only their location changed.
        base = WORKSPACE / ('SNARKPEG_Poseidon' if project == 'poseidon_97' else project)
        paths.extend(p for p in (base / 'src').rglob('*') if p.is_file())
        paths.extend(p for p in (base / 'Cargo.toml', base / 'Cargo.lock', base / 'build.rs') if p.is_file())
    return {str(p.relative_to(WORKSPACE)).replace('SNARKPEG_Poseidon/', 'poseidon_97/', 1): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}

def check():
    expected = json.loads(MANIFEST.read_text())['files']
    actual = inventory()
    changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    if changed:
        raise SystemExit('Protected source mismatch:\n' + '\n'.join(changed))
    return {'status': 'pass', 'files_checked': len(actual)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('initialize', 'check-source'))
    args = parser.parse_args()
    if args.command == 'initialize':
        if MANIFEST.exists():
            raise SystemExit('Refusing to replace the source manifest')
        MANIFEST.write_text(json.dumps({'format': 'poseidon97-source-manifest-v1', 'files': inventory()}, indent=2) + '\n')
    print(json.dumps(check()))
