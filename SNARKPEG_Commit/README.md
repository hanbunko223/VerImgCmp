# SNARKPEG_Commit

Standalone `VICsrc + KZH4 (Go/gnark-crypto)` package.

## What this directory contains

- `VICsrc/`: the DCTQ proving and verification code
- `kzh_gnark/`: the Go KZH4 sidecar used by `VICsrc`
- `3rd/hyrax-bls12-381/`: bundled `mcl` dependency used by `VICsrc`
- `script/run_snarkpeg_commit.sh`: the main entrypoint

## Prerequisites

You need these tools installed before running the script:

- `cmake`
- a C++ compiler supported by CMake
- `go`
- `python3`
- GMP development headers (`gmpxx.h`)

### macOS (Apple Silicon)

Recommended:

```bash
brew install cmake go gmp
```

### Ubuntu / Debian

Recommended:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake golang python3 libgmp-dev
```

## Run

From inside this directory:

```bash
./script/run_snarkpeg_commit.sh HD
```

Supported resolutions:

- `SD`
- `HD`
- `FHD`
- `QHD`
- `4k`

Examples:

```bash
./script/run_snarkpeg_commit.sh SD
./script/run_snarkpeg_commit.sh HD
./script/run_snarkpeg_commit.sh FHD
./script/run_snarkpeg_commit.sh QHD
./script/run_snarkpeg_commit.sh 4k
```

## Outputs

- output matrix:
  - `output/dctq/dctq-<resolution>-output.txt`
- run log:
  - `log/dctq/dctq-<resolution>-run.log`
- cached KZH SRS:
  - `output/kzh_srs/`

## Notes

- The script builds the C++ binary and the Go sidecar automatically.
- Input images under `data/dctq/` are generated automatically if they are missing.
- The first run for a new input size is slower because the KZH SRS is generated and cached.
