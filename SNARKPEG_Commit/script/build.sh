#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
build_dir="$repo_root/cmake-build-release"

mkdir -p "$build_dir"
cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
