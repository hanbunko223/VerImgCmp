"""Resolve Lake; lean/lean-toolchain remains the authoritative version pin."""
import os, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def lake():
    local = ROOT / '.elan/toolchains/leanprover--lean4---v4.28.0/bin/lake'
    override = os.environ.get('CIRCUIT_CORRECTNESS_LAKE')
    if override:
        return Path(override)
    if local.is_file():
        return local
    found = shutil.which('lake')
    if found:
        return Path(found)
    raise RuntimeError('Install elan and Lean v4.28.0; see README.md')
