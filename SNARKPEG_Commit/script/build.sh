#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
build_dir="$repo_root/cmake-build-release"

if [ -f "$build_dir/CMakeCache.txt" ]; then
    cached_source=$(
        awk -F= '/^CMAKE_HOME_DIRECTORY:INTERNAL=/{print $2}' "$build_dir/CMakeCache.txt" | tail -n 1
    )
    if [ -n "${cached_source:-}" ] && [ "$cached_source" != "$repo_root" ]; then
        echo "Removing stale CMake cache from $build_dir" >&2
        echo "Cached source: $cached_source" >&2
        echo "Current source: $repo_root" >&2
        rm -rf "$build_dir"
    fi
fi

mkdir -p "$build_dir"
cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
