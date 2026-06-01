# SNARKPEG

We have two modes for SNARKPEG:

- `SNARKPEG_Poseidon/`
  - our NeutronNova-based prover
  - proves range checks, packing, Poseidon hashing, and Lossify
- `SNARKPEG_Commit/`
  - our GKR prover
  - use GKR with KZH4 (in gnark)

## Samples

`samples/` contains five resulution images

- `SD.png`
- `HD.png`
- `FHD.png`
- `QHD.png`
- `4K.png`

## Run `SNARKPEG_Poseidon`

Prerequisites:

- Rust toolchain with `cargo`
- Python 3
- `Pillow` and `numpy`

run it with script

```bash
cd /Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp
./script/run_snarkpeg_poseidon.sh resolution
```
e.g.
```bash
cd /Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp
./script/run_snarkpeg_poseidon.sh HD
```

Supported resolutions:

- `SD`
- `HD`
- `FHD`
- `QHD`
- `4K`

Manual flow:

Create an input JSON:

```bash
cd /Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp
python3 create_input.py HD /tmp/snarkpeg_poseidon_hd_input.json
```

Build:

```bash
cargo build --release --manifest-path /Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp/SNARKPEG_Poseidon/Cargo.toml
```

Run:

```bash
/Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp/SNARKPEG_Poseidon/target/release/SNARKPEG_Poseidon \
  --function dctq \
  --resolution HD \
  --input /tmp/snarkpeg_poseidon_hd_input.json \
  --output /tmp/snarkpeg_poseidon_hd_proof.json \
  --rayon-threads 8
```

## Run `SNARKPEG_Commit`

run it with script
```bash
cd /Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp/SNARKPEG_Commit
./script/run_snarkpeg_commit.sh resolution
```
e.g.
```bash
cd /Users/hanbunko/Downloads/vcvc/vimz/VerImgCmp/SNARKPEG_Commit
./script/run_snarkpeg_commit.sh HD
```

Supported resolutions:

- `SD`
- `HD`
- `FHD`
- `QHD`
- `4k`
