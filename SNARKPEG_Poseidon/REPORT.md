> Publication note: historical measurements from the original `poseidon_97`
> campaign. The implementation now lives in `SNARKPEG_Poseidon/` with identical
> certified Rust source bytes. The historical HH control and construction-only
> scripts are not shipped here; see `benchmarks/README.md` for reproducing the
> current prover measurements. This publication does not rerun the old comparison.

# poseidon_97 implementation and performance

This experiment proves the exact **centered, unnormalized** relation `C = A(X−128)Aᵀ ⊙ M`, with `A = round(128D)` and `M = round(1024/q)` for q≤97 and zero otherwise. Red uses the luminance divisor table, green/blue the chrominance table. There is no final normalization, reciprocal division, or rounding gadget. This is not the normalized public-byte relation of the previous pow2 project.

## Fresh full-HD benchmark

Apple M1 Pro (8 CPU cores: 6 performance + 2 efficiency), 16 GB memory; 8 Rayon threads; HD 1280×720 RGB; 360 original steps in **one recursive proof**, followed by **one Spartan compression**. One warm-up and three measured fresh-process runs per implementation. Both use matching release flags: `target-cpu=native`, thin LTO, one code-generation unit, the same Nova checkout and dependency versions. Runs are sequential, alternating the two implementations. Median [minimum, maximum]; seconds unless otherwise specified. Each total is computed per run before taking its median, so medians of individual phases need not sum to the median total.

The original HH control preserves the original uncentered 1024-scale transform, shared 51-position reciprocal table, circuit, coefficient format and Fiat–Shamir challenge. The two rows therefore prove intentionally different transform definitions, not identical output vectors. Its copied CLI adds timing and saved-proof verification. These are fresh matched measurements, not the historical approximately 29-second result.

| Measurement | Original Poseidon HH | poseidon_97 |
|---|---:|---:|
| Input/statement preparation inside recursive creation | 0.744 [0.741, 0.786] | 0.464 [0.462, 0.593] |
| Recursive proving, including preparation | 27.101 [25.476, 27.162] | 25.277 [23.930, 26.556] |
| Recursive backend only | 26.357 [24.735, 26.376] | 24.815 [23.466, 25.962] |
| Spartan compression | 11.683 [11.059, 12.418] | 11.391 [11.102, 11.691] |
| Total prover: recursive creation + Spartan | 38.784 [36.534, 39.580] | 36.967 [35.032, 37.946] |
| Public parameter setup | 2.482 [2.477, 2.575] | 2.475 [2.457, 2.658] |
| Spartan key setup | 0.311 [0.303, 0.312] | 0.302 [0.292, 0.304] |
| Initial public statement reconstruction | 0.333 [0.326, 0.347] | 0.257 [0.253, 0.343] |
| Recursive serialization | 0.111 [0.105, 0.112] | 0.104 [0.103, 0.107] |
| Compressed serialization | 0.001 [0.001, 0.010] | 0.001 [0.001, 0.001] |
| Complete proving application (includes checks/setup/I/O) | 42.444 [40.169, 43.388] | 40.518 [38.535, 41.764] |
| Recursive verification, separate process core | 0.160 [0.142, 0.177] | 0.135 [0.135, 0.197] |
| Compressed verification, separate process core | 0.263 [0.253, 0.313] | 0.257 [0.245, 0.287] |
| Standalone recursive verification including setup/loading | 3.120 [3.009, 3.627] | 2.978 [2.917, 3.123] |
| Standalone compressed verification including setup/loading | 3.458 [3.355, 3.534] | 3.259 [3.217, 3.433] |
| Recursive proof bytes | 17616557 [17616557, 17616557] | 17616557 [17616557, 17616557] |
| Compressed proof bytes | 27719 [27719, 27719] | 27719 [27719, 27719] |
| Recursive artifact bytes | 17617088 [17617088, 17617088] | 17617121 [17617121, 17617121] |
| Compressed artifact bytes | 28248 [28248, 28248] | 28281 [28281, 28281] |
| Peak proving RSS (GiB) | 0.507 [0.506, 0.640] | 0.493 [0.491, 0.609] |
| Peak macOS physical footprint (GiB) | 0.268 [0.220, 0.273] | 0.257 [0.227, 0.262] |

Recursive creation improves by **6.7%** on the median. Setup and compression are reported separately: the total prover row includes Spartan, but excludes reusable setup, serialization, and verifier checks. The complete application row includes all of them. Recursive creation includes native preflight preparation, in-circuit witness generation, commitments, and folding; the backend row excludes only the separately timed native preparation. Internal circuit witness generation is part of synthesis and has not been falsely reported as a separately isolated total.

All runs fit the 10 GiB process-tree RSS/footprint guard; no segmentation fallback or failed full-image run was used. Peak RSS is macOS `getrusage`/`time -l`; physical footprint is the separate macOS metric, not interchangeable with RSS. Actual UTF-8 JSON serialized proof bytes are measured. Public coefficient files, image data and reusable public parameters are excluded from proof size; artifact bytes include statement metadata. The full coefficient file is still needed by the verifier.

The public digest is generated compatibly with original HH and checked against the cached authorized fixture. The application, rather than this experiment, decides which input digest is authorized.

## Circuit changes

| Quantity per step | Original HH | poseidon_97 |
|---|---:|---:|
| Retained positions per RGB block | 51+51+51 = 153 | 51+13+13 = 77 |
| First-stage rows per channel | 8,8,8 | 8,4,4 |
| First-stage allocated values | 7,680 | 5,120 |
| Fused second-stage/Q/Horner constraints | 6,120 | 3,080 |
| Raw application constraints | 103,230 | 97,630 |
| Raw variables including the four initial state variables | 103,234 | 97,634 |
| Padded application dimension | 131,072 | 131,072 |
| HD active coefficients | 2,203,200 | 1,108,800 |

The exact saving is 2,560 first-stage linear constraints plus 3,040 Horner multiplication constraints per step. Smaller integer constants alone do not eliminate R1CS constraints. Both designs still have the same padded dimension; the unchanged pixel range checks, input hashing, and commitment/folding work limit the improvement.

The matrix is pinned in `src/dctq.rs` (also recorded with both original divisor tables and reciprocal multipliers in `benchmarks/transform.json`):

```text
 45  45  45  45  45  45  45  45
 63  53  36  12 -12 -36 -53 -63
 59  24 -24 -59 -59 -24  24  59
 53 -12 -63 -36  36  63  12 -53
 45 -45 -45  45  45 -45 -45  45
 36 -63  12  53 -53 -12  63 -36
 24 -59  59 -24 -24  59 -59  24
 12 -36  53 -63  63 -53  36 -12
```

At each retained first-stage row, the constraint directly enforces `L = Σ A·pixel − 128Σ A`; centering is a constant linear-combination offset. No extra centering witness is allocated. Unused first-stage rows and omitted second-stage outputs are not allocated. For each retained output, `a·r = a_next − Σ (M·A)·L` simultaneously enforces the right-hand transpose, reciprocal multiplication, and Horner update. Matrix/mask decisions happen during circuit construction from compiled constants, without in-circuit comparisons to 97 or multiplication-by-zero checks. Native preparation uses the same pruning.

For each RGB block, red retains 51 coordinates and green/blue 13 each. There are 40 RGB blocks per step. Coefficients are visited in row, column, RGB order, with omission checked separately per channel. The accumulator persists across all 360 steps. State is `(input_hash, coefficient_evaluation, challenge, step_counter)`. The input hash branch sees uncentered RGB bytes and is otherwise unchanged.

The tight global post-Q envelopes are red [−1,554,357,600, 1,554,357,600] and green/blue [−995,328,000, 987,552,000]. The loader validates tighter per-position bounds derived by summing the endpoints of each integer linear weight over pixels in [0,255]. All arithmetic magnitudes are far below the scalar modulus, so the input byte bounds and linear constraints uniquely determine the intended signed integer outputs without coefficient range decomposition.

## Public statement and saved verification

New coefficient format: `poseidon-97-centered-dct128-recip1024-coefficients-v1`. New coefficient SHA domain: `POSEIDON_97_CENTERED_DCT128_RECIP1024_COEFFICIENTS_V1\0`. Each full-vector coefficient is encoded as little-endian u40 `C + 2^39`, preceded by the domain, resolution tag, step count and vector length. Omitted positions remain in JSON and SHA encoding as checked zeros; they are absent from Horner.

The Poseidon challenge domain is `P971 = 0x50393731`; the eight inputs are the original-pixel digest, four u64 limbs of the coefficient SHA digest, resolution tag, total active length and expected step count. These are recomputed by the verifier from authoritative public files. This fixed protocol identity binds the centered matrix, channel quantizers, rounding of reciprocals, mask, ordering, and lack of normalization. Old coefficient/artifact formats are rejected.

The verifier independently constructs `(0,0,r,0)` and expects `(d_I,P(r),r,360)`. Both recursive and Spartan proofs are serialized and verified in fresh processes with no private image argument. Claimed metadata cannot override the trusted statement. SHA/Poseidon collision resistance, Nova/Spartan soundness, and Fiat–Shamir assumptions still apply. The challenge is public before proving; polynomial equality has negligible soundness error, not literal deterministic vector equality. Exact host coefficient preflight is not the cryptographic output-binding mechanism. This is integration and adversarial validation, not a cryptographic audit or Lean formal proof.

## Validation

- 33 Rust tests pass, including native/gadget Poseidon agreement, byte range rejection, challenge/context rebinding, full-step satisfiability, and two-step recursive proving.
- 175 independent NumPy vectors cover constants 0/128/255, an asymmetric impulse, alternating patterns, random inputs and both extrema of every retained coordinate. All expected signed outputs agree. The normalized Python q97 reference also equals `(C + 2^23) // 2^24` for these same blocks.
- Every one of the 2,764,800 full-HD coefficient positions matches independent Python computation. Flattened input RGB bytes match `samples/HD.png` exactly.
- Direct circuit tests bypass native preflight, alter a pixel bit, first-stage DCT value, first and final Horner values, and recommit each malicious witness before checking R1CS rejection. Additional recursive tests reject incorrect initial hash/accumulator/challenge/counter and step count.
- Public-loader tests check both attainable endpoints and immediately out-of-range values at all positions, omitted zeros and legacy formats.
- Standalone adversarial verification rejects altered public coefficients/digest, omitted nonzeros, bounds violations, legacy formats, altered challenge/step metadata, missing proof fields and a modified scalar inside a syntactically valid cryptographic proof. Results are in `benchmarks/historical/adversarial/results.json`.
- All eight full-HD proving runs (including warm-ups) independently verify both saved proof kinds. Public-only verification uses no image.
- Protected source hashes are checked against the pre-edit snapshot; no original project or Nova source changed. Other existing workspace changes predate this experiment and were not reset.

The proof layout remains packed 160-pixel rows. Earlier spatial compression/SSIM measurements concern the same transform on different blocks and are not claimed as quality measurements of this proof fixture. No new compression-quality campaign is part of this task.

## Internal recursive timing

These overlapping library counters help explain cost; do not sum them as independent phases.

| Counter (seconds) | Original HH | poseidon_97 |
|---|---:|---:|
| augmented_synthesize_s | 3.222 [3.210, 3.224] | 3.202 [3.167, 3.251] |
| commit_e_s | 1.454 [1.359, 1.505] | 1.471 [1.372, 1.486] |
| commit_w_s | 12.711 [12.013, 12.786] | 11.530 [10.949, 12.052] |
| fold_witness_s | 0.390 [0.301, 0.409] | 0.359 [0.306, 0.389] |
| multiply_vec_z1_s | 3.923 [3.671, 3.965] | 3.695 [3.494, 4.055] |
| multiply_vec_z2_s | 3.916 [3.655, 3.975] | 3.695 [3.479, 4.042] |
| prove_helper_s | 3.185 [2.924, 3.251] | 3.231 [2.959, 3.350] |

## Reproduction and evidence

See `README.md`, `build.sh`, and `benchmarks/README.md` for current reproduction. Historical `benchmarks/historical/runs.json`, `summary.json`, and `provenance.json` retain the measured runs and machine identity. Large proof files and construction-only scripts are excluded from Git. The original comparison included an HH control that is not shipped in this release; the current benchmark runner measures the certified implementation only.
