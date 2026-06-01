#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
pixels_per_step=2560
packed_rows_per_step=16

usage() {
  cat <<'EOF'
Usage:
  ./script/run_snarkpeg_poseidon.sh <SD|HD|FHD|QHD|4K>

Current fixed schedule:
  SD   -> 120 steps
  HD   -> 360 steps
  FHD  -> 810 steps
  QHD  -> 1440 steps
  4K   -> 3240 steps

The local step geometry is fixed at 16 packed rows x 160 pixels = 2560 pixels/step.
EOF
}

expected_steps_for_resolution() {
  case "$1" in
    SD) echo 120 ;;
    HD) echo 360 ;;
    FHD) echo 810 ;;
    QHD) echo 1440 ;;
    4K) echo 3240 ;;
    *)
      echo "unsupported resolution: $1" >&2
      return 1
      ;;
  esac
}

for cmd in cargo python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

resolution="${1:-HD}"
threads="${RAYON_NUM_THREADS:-8}"
expected_steps=$(expected_steps_for_resolution "$resolution")

case "$resolution" in
  SD|HD|FHD|QHD|4K) ;;
  *)
    echo "Unknown resolution: $resolution" >&2
    echo "Expected one of: SD, HD, FHD, QHD, 4K" >&2
    exit 1
    ;;
  esac

mkdir -p \
  "$repo_root/output/poseidon" \
  "$repo_root/log/poseidon"

input_json="/tmp/snarkpeg_poseidon_${resolution}.json"
proof_json="$repo_root/output/poseidon/${resolution}-proof-spartan.json"
log_file="$repo_root/log/poseidon/${resolution}-spartan-run.log"

cmd=(
  "$repo_root/SNARKPEG_Poseidon/target/release/SNARKPEG_Poseidon"
  --function dctq
  --resolution "$resolution"
  --input "$input_json"
  --output "$proof_json"
  --rayon-threads "$threads"
  --spartan-compress
)

echo "======================================================================"
echo "SNARKPEG_Poseidon runner"
echo "======================================================================"
echo "resolution:           $resolution"
echo "rayon threads:        $threads"
echo "expected steps:       $expected_steps"
echo "pixels per step:      $pixels_per_step"
echo "packed rows per step: $packed_rows_per_step"
echo "input json:           $input_json"
echo "proof output:         $proof_json"
echo "log file:             $log_file"

python3 "$repo_root/create_input.py" "$resolution" "$input_json"

cargo build --release --manifest-path "$repo_root/SNARKPEG_Poseidon/Cargo.toml"

RAYON_NUM_THREADS="$threads" "${cmd[@]}" 2>&1 | tee "$log_file"
