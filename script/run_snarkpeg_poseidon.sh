#!/bin/sh
# Demo only: production must independently authorize the input digest.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
resolution=${1:-HD}
case "$resolution" in SD|HD|FHD|QHD|4K) ;; *) echo 'Usage: run_snarkpeg_poseidon.sh SD|HD|FHD|QHD|4K' >&2; exit 2;; esac
threads=${RAYON_NUM_THREADS:-8}
OUT="$ROOT/output/poseidon97/$resolution"
mkdir -p "$OUT"
"$ROOT/SNARKPEG_Poseidon/build.sh"
B="$ROOT/SNARKPEG_Poseidon/target/release/poseidon_97"
python3 "$ROOT/create_input.py" "$resolution" "$OUT/input.json"
"$B" digest --resolution "$resolution" --input "$OUT/input.json" --output "$OUT/candidate-digest.json"
"$B" coefficients --resolution "$resolution" --input "$OUT/input.json" --output "$OUT/coefficients.json"
echo 'DEMO ONLY: the self-generated digest does not establish external authorization.'
"$B" prove --resolution "$resolution" --threads "$threads" --input "$OUT/input.json" --input-digest "$OUT/candidate-digest.json" --coefficients "$OUT/coefficients.json" --output "$OUT/proof.json" --spartan-compress --metrics "$OUT/prove-metrics.json"
"$B" verify --resolution "$resolution" --threads "$threads" --input-digest "$OUT/candidate-digest.json" --coefficients "$OUT/coefficients.json" --proof "$OUT/proof.json" --metrics "$OUT/verify-recursive.json"
"$B" verify --resolution "$resolution" --threads "$threads" --input-digest "$OUT/candidate-digest.json" --coefficients "$OUT/coefficients.json" --proof "$OUT/proof.spartan.json" --metrics "$OUT/verify-compressed.json"
