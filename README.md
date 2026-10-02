# Verifiable Image Compression

| Directory | Purpose |
|---|---|
| [SNARKPEG_Poseidon](SNARKPEG_Poseidon) | Current `poseidon_97`: Poseidon image commitment and integer DCT-Q bound by a Fiat–Shamir polynomial fingerprint, using Nova and Spartan. |
| [circuit_correctness](circuit_correctness) | Lean 4/mathlib soundness, completeness, and output-determinism proofs for the pinned application-step R1CS. |
| [SNARKPEG_Commit](SNARKPEG_Commit) | Existing GKR/KZH4 backend, unchanged by this publication. |
| [nova60/Nova](nova60/Nova) | Pinned local Nova dependency. |

`SNARKPEG_Poseidon` now contains the **16×160 RGB / 360-step HD** implementation,
not `poseidon_97_180`. The Rust package and executable retain their certified
name `poseidon_97`. All Rust sources and Cargo files are byte-identical to the
original source fingerprints in the Lean project.

## Prover quick start

Install Rust/Cargo, Python 3, and Pillow (`python3 -m pip install -r requirements.txt`,
preferably inside a virtual environment). From the repository root:

```sh
./script/run_snarkpeg_poseidon.sh HD
```

This converts `samples/HD.png`, builds/tests the prover, generates public
coefficients and a candidate digest, proves, and verifies saved recursive and
compressed artifacts in separate processes. The generated digest is **not**
an authorized digest: production must supply an independently authenticated
anchor. See the [manual commands and protocol](SNARKPEG_Poseidon/README.md).

Sample resolutions: SD, HD, FHD, QHD, and 4K. Pixels are flattened in raster/RGB
order into packed 160-pixel rows; these blocks are not ordinary spatial blocks
of the original image. Larger images use connected bundles when the step count
exceeds the current CLI's 360-step segment limit.

The relation is centered, unnormalized integer DCT-Q with a 128-scale matrix
and fixed threshold-97 reciprocal multipliers. Outputs are large signed
coefficients, not normalized JPEG bytes. No conventional JPEG encoding or
normalization operation is certified.

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

The proof certifies the exported application circuit, not the Rust compiler,
exporter, hash collision resistance, Fiat–Shamir, Nova, Spartan, file parsing,
or the surrounding proof verifier.

## Measurements

The [prover report](SNARKPEG_Poseidon/REPORT.md) and
[historical records](SNARKPEG_Poseidon/benchmarks/historical) preserve the original
HD experiment. These are historical measurements, not new performance claims
from this publication. Build caches and large generated witness/proof files
are excluded. See [benchmark reproduction](SNARKPEG_Poseidon/benchmarks/README.md).

## Existing Commit backend

Its entry point remains unchanged:

```sh
cd SNARKPEG_Commit
./script/run_snarkpeg_commit.sh HD
```

Consult that project's files for its prerequisites. The Poseidon/Lean update
does not change the Commit implementation or results.
