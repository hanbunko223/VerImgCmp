# S(NARK)PEG

We have two modes for S(NARK)PEG:

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
./script/run_snarkpeg_poseidon.sh resolution
```
e.g.
```bash
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

```bash\
python3 create_input.py HD /tmp/snarkpeg_poseidon_hd_input.json
```

Build:

```bash
cargo build --release --manifest-path ./VerImgCmp/SNARKPEG_Poseidon/Cargo.toml
```

Run:

```bash
VerImgCmp/SNARKPEG_Poseidon/target/release/SNARKPEG_Poseidon \
  --function dctq \
  --resolution HD \
  --input /tmp/snarkpeg_poseidon_hd_input.json \
  --output /tmp/snarkpeg_poseidon_hd_proof.json \
  --rayon-threads 8
```

## Run `SNARKPEG_Commit`

run it with script
```bash
cd VerImgCmp/SNARKPEG_Commit
./script/run_snarkpeg_commit.sh resolution
```
e.g.
```bash
cd VerImgCmp/SNARKPEG_Commit
./script/run_snarkpeg_commit.sh HD
```

Supported resolutions:

- `SD`
- `HD`
- `FHD`
- `QHD`
- `4k`


## Verify the Lean proof

Lean/mathlib are pinned to v4.28.0; exact dependency revisions are in
`circuit_correctness/lean/lake-manifest.json`. Install elan, then:

```sh
cd circuit_correctness
elan toolchain install leanprover/lean4:v4.28.0
cd lean
lake exe cache get
lake build Certification
cd ..
cargo fetch --locked --manifest-path exporter/Cargo.toml
python3 scripts/verify.py
```

The last command checks source identity, regenerates and compares the R1CS,
checks the independent specification, builds the complete theorems, and audits
their transitive axioms. The source-path mapping is explicit; the original
fingerprint manifest is unchanged. See the [formal verification guide](circuit_correctness/README.md).

The proof certifies that the correctness exported step circuit indeed implements the function in Specification: the function deterministically lead the input witness to specific coefficients. The witness is binding with Hash, the output is deterministically computed and checked via polynomial opening at the challenge point. Together with the security from hash collision resistance, Fiat–Shamir, Nova, Spartan, file parsing, lead to our Protocol's security.
