# SNARKPEG_Poseidon (certified poseidon_97)

Published from the original `poseidon_97` source, with its Rust files and Cargo
lockfile unchanged. The repository directory is `SNARKPEG_Poseidon`; the
executable remains `poseidon_97` to preserve the certified package identity.
The Lean proof in `../circuit_correctness` certifies this application's exported
step constraints. It does not certify the surrounding cryptographic protocol.

Independent copy of `SPEG_Poseidon_hh`, using a centered 128-scale DCT, fixed channel-specific q≤97 masks, and integer reciprocal quantization. The implementation proves **unnormalized integer coefficients**:

\[
C_{ch} = (A(X_{ch}-128)A^T)\odot M_{ch},\quad
A=\operatorname{round}(128D),\quad
M_{ch,r,c}=\begin{cases}\lfloor1024/q_{ch,r,c}+1/2\rfloor&q_{ch,r,c}\le97\\0&q_{ch,r,c}>97.\end{cases}
\]

Red uses the existing JPEG luminance divisor table; green and blue use the chrominance table. These are protocol constants, not prover inputs. There is **no division, normalization, quotient/remainder range check, or output bit decomposition in this circuit**. Pixels remain range-checked bytes. Centering applies only to DCT; the original pixel bytes are hashed exactly as in HH.

The reference Python q97 compression experiment rounds `(C + 2^23) // 2^24` before entropy coding. That rounding is **not proved here**: this proof binds the large exact integer C. See [REPORT.md](REPORT.md) for measured performance, precise scope, and validation.

## Build and test

From `VerImgCmp`:

```sh
./SNARKPEG_Poseidon/build.sh
```

Requires Rust/Cargo and the included `nova60/Nova` dependency. The build uses
locked dependencies, `target-cpu=native`, thin LTO, and one code-generation unit.
It runs the Rust tests and builds both executables. The first build downloads
registry dependencies; subsequent runs can use the cache.

## Commands

```sh
B=./SNARKPEG_Poseidon/target/release/poseidon_97
F=./output/poseidon97/HD
mkdir -p "$F"
python3 create_input.py HD "$F/HD.json"
$B inspect
$B digest --resolution HD --input "$F/HD.json" --output "$F/candidate-digest.json"
$B coefficients --resolution HD --input "$F/HD.json" --output "$F/HD-coefficients.json"
$B prove --resolution HD --threads 8 --segment-steps 360 \
  --input "$F/HD.json" --input-digest "$F/HD-digest.json" \
  --coefficients "$F/HD-coefficients.json" --output "$F/proof.json" \
  --spartan-compress --metrics "$F/prove-metrics.json"
$B verify --resolution HD --threads 8 \
  --input-digest "$F/HD-digest.json" --coefficients "$F/HD-coefficients.json" \
  --proof "$F/proof.spartan.json" --metrics "$F/verify-metrics.json"
```

`bench` has the same behavior as `prove`; the campaign invokes it in fresh processes. `verify` also accepts `proof.json` for recursive verification. It never needs the private image. `digest` computes a candidate anchor; it does not authorize it. The application must independently authenticate `HD-digest.json`.

For HD, the full-image proof uses 360 Nova steps and one final Spartan compression. Optional `--segment-steps` keeps the inherited connected-bundle interface; use the same schedule when verifying. Only the full 360-step configuration is the benchmark headline.

## Public formats and verifier obligations

Input digests retain `speg-poseidon-hh-input-digest-v1`. New coefficient files use `poseidon-97-centered-dct128-recip1024-coefficients-v1` and the existing `[packed row][column][RGB]` layout. Each step contains 16×160 pixels. Each packed 8×8 block stays within these rows, rather than being interpreted as a spatial block of the 1280-pixel-wide image.

The full JSON includes omitted positions as zero. The loader checks dimensions, signed integer values against tight per-coordinate attainable bounds, and zero at every omitted position. Only retained entries enter Horner, in step/row/column/channel order. All entries remain in the canonical SHA-256 digest, encoded as five-byte little-endian `C + 2^39` with a new domain prefix.

The verifier reloads authoritative public files, derives the SHA coefficient digest and public Poseidon challenge, and reconstructs both the initial and expected final states. It checks `(input hash, coefficient evaluation, challenge, step count)`. The new challenge uses domain `0x50393731` (`P971`) and binds the digest, coefficient digest, resolution, active length, and step count. Artifacts cannot redefine these values. The public challenge is known before proving; coefficient binding is probabilistic under the stated Fiat–Shamir/random-oracle assumptions, not deterministic vector equality. Host-side exact coefficient comparison is an early rejection check only.

## Reproduce measurements

See [benchmarks/README.md](benchmarks/README.md) and run `benchmarks/run.py`
after building. This measures the published implementation with one warm-up
and three fresh-process HD runs, saving and verifying both proof types.
Historical comparison results remain in [REPORT.md](REPORT.md) and
`benchmarks/historical/`; the historical HH control is not part of the current
release. Those measurements are not silently replaced by publication checks.

## Release identity

The original 106-file source manifest remains unchanged. Its logical
`poseidon_97/` prefix maps to this directory via the exporter's source checker.
The mapping is recorded in
`../circuit_correctness/artifacts/publication_mapping.json`. Protocol domains,
coefficient formats, challenge derivation, and all circuit source files retain
their original identities. Old pre-Horner artifacts are incompatible.

The coefficient command generates the correct public file; the pixel converter
does not calculate an independent or legacy approximation. For real proving,
provide the independently authorized `HD-digest.json` in the commands above.
