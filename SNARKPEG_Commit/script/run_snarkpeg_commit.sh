#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")

for cmd in cmake go python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

"$script_dir/build.sh"
cmake --build "$repo_root/cmake-build-release" --target demo_dctq_run -- -j 6

mkdir -p "$repo_root/.go-build-cache"
(
  cd "$repo_root/kzh_gnark"
  GOCACHE="$repo_root/.go-build-cache" go build -o "$repo_root/kzh_gnark/zkcnn_kzh_cli" .
)

run_file="$repo_root/cmake-build-release/VICsrc/demo_dctq_run"
kzh_bin="$repo_root/kzh_gnark/zkcnn_kzh_cli"

mkdir -p "$repo_root/data/dctq" "$repo_root/output/dctq" "$repo_root/log/dctq" "$repo_root/output/kzh_srs"

resolution="${1:-HD}"
qd_file="$repo_root/qD.txt"
qr_file="$repo_root/qR.txt"
out_file="${2:-$repo_root/output/dctq/dctq-${resolution}-output.txt}"
log_file="$repo_root/log/dctq/dctq-${resolution}-run.log"

case "$resolution" in
  SD) input_file="$repo_root/data/dctq/SD.txt" ;;
  HD) input_file="$repo_root/data/dctq/HD.txt" ;;
  FHD) input_file="$repo_root/data/dctq/FHD.txt" ;;
  QHD) input_file="$repo_root/data/dctq/QHD.txt" ;;
  4k) input_file="$repo_root/data/dctq/4k.txt" ;;
  *)
    echo "Unknown resolution: $resolution" >&2
    echo "Expected one of: SD, HD, FHD, QHD, 4k" >&2
    exit 1
    ;;
esac

if [ ! -f "$qd_file" ] || [ ! -f "$qr_file" ]; then
  echo "Missing qD.txt or qR.txt at repo root." >&2
  exit 1
fi

REPO_ROOT="$repo_root" python3 - <<'PY'
import os
import random

random.seed(7)
repo_root = os.environ["REPO_ROOT"]
base = os.path.join(repo_root, "data", "dctq")
sizes = {
    "SD": (640, 480),
    "HD": (1280, 720),
    "FHD": (1920, 1080),
    "QHD": (2560, 1440),
    "4k": (3840, 2160),
}

os.makedirs(base, exist_ok=True)
for name, (width, height) in sizes.items():
    path = os.path.join(base, f"{name}.txt")
    if os.path.exists(path):
        continue
    with open(path, "w", encoding="ascii") as f:
        for _ in range(height):
            row = [str(random.randint(0, 255)) for _ in range(width)]
            f.write(" ".join(row) + "\n")
PY

ZKCNN_KZH_BIN="$kzh_bin" \
ZKCNN_KZH_SRS_DIR="$repo_root/output/kzh_srs" \
"$run_file" "$input_file" "$qd_file" "$qr_file" "$out_file" > "$log_file" 2>&1
