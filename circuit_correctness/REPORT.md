# Certified `poseidon_97` application-step R1CS

The full exported step has universal Lean proofs of soundness, completeness,
output determinism, and connected iteration. The strict acceptance command
`python3 scripts/verify.py` has passed with `certified: true`. This report
concerns correspondence between the pinned R1CS and its independent mathematical
specification. It does not certify the surrounding cryptographic protocol.

## Certified statement

The target is the original `poseidon_97` application step: 16×160 RGB pixels,
used 360 times in the HD configuration. `poseidon_97_180` and Nova's augmented
circuit are excluded. The field is `ZMod q`, where

```text
q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
```

`Field.lean` supplies a kernel-checked recursive primality certificate. No
primality or circuit-correctness axiom is introduced.

With `x` a valid image, `s` an incoming field state, and `w` an arbitrary wire
assignment, `Target.step_soundness` proves exactly:

```text
∀ x s w,
  Spec.ValidImage x →
  Target.matchesImage w x →
  Target.incomingState w = s →
  Exported.circuit.Sat w →
  Target.outgoingState w = Spec.step x s
```

`Target.step_completeness` proves:

```text
∀ x s, Spec.ValidImage x →
  ∃ w, Target.matchesImage w x ∧
       Target.incomingState w = s ∧
       Exported.circuit.Sat w ∧
       Target.outgoingState w = Spec.step x s
```

`Target.step_determinism` proves equal outgoing states for any two satisfying
assignments matching the same valid image and incoming state. All names have
the prefix `CircuitCorrectness.`. Their exact types are checked in
`lean/Certification.lean`, and their proofs are in
`lean/CircuitCorrectness/Complete.lean`.

`Target.connected_steps` and `Target.connected360` additionally prove that a
sequence whose adjacent states are explicitly connected computes repeated
`Spec.step`. They assume those connections; they make no claim that Nova enforces
them. The counter increment is field arithmetic, including possible wraparound.

These are universal statements, not claims limited to regression fixtures.
Completeness constructs mathematical auxiliaries and does not assume universal
correctness of the Rust witness generator.

## Independent computation specification

`Spec.lean` defines an image as a function of row, column, and RGB channel, with
`ValidImage` bounding each used channel to `0..255`. The DCT-Q definition is the
direct integer double sum

\[
C_{r,c,ch}=M_{ch,r\bmod8,c\bmod8}
\sum_{i,j=0}^{7} A_{r\bmod8,i}
(X_{8\lfloor r/8\rfloor+i,8\lfloor c/8\rfloor+j,ch}-128)
A_{c\bmod8,j}.
\]

The pinned matrix is:

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

For each fixed channel-specific original divisor `q`, the multiplier is zero
when `q>97`; otherwise it is the integer `(2048+q)/(2*q)`. This is the pinned
nearest reciprocal-1024 construction. There is no runtime output normalization,
rounding, or comparison against the threshold. The retained counts are 51 red,
13 green, and 13 blue per RGB block. Omitted coefficients are mathematically zero.
The 40 RGB blocks in a step give 3,080 retained coefficients, evaluated in the
actual row/column/RGB order with the incoming Horner accumulator and challenge.

Input hashing packs original, uncentered bytes into 24-bit RGB pixels and ten
pixels into each 240-bit chunk. Each 160-pixel row gives 16 chunks, reduced by
two arity-8 hashes and one arity-2 hash. The 16 row hashes undergo the same
reduction, then an arity-2 hash chains the step digest with the incoming hash.

Poseidon is a functional definition independent of R1CS satisfaction. It uses
the pinned compressed round keys, dense/pre-sparse/sparse matrices, full and
partial rounds, domain initialization, full-rate absorption, and one squeeze.
Arity 2 uses 8 full and 55 partial rounds; arity 8 uses 8 full and 57 partial
rounds. The domain identifiers are `0x50414952` and `0x48415348`, respectively.

The resulting state specification is:

```text
(hash2 incoming.h (stepDigest image),
 Horner incoming.a incoming.r (coefficients image),
 incoming.r,
 incoming.t + 1)
```

`DctSpec.lean` separately proves staged/direct-transform equality, transpose
indexing, pruning, integer casts, per-position coefficient bounds, and the
conservative global interval `[-1,554,357,600, 1,554,357,600]`. The embedding of
that interval into the field is injective. Packing theorems establish
`pixel<2^24`, `chunk<2^240<q`, and preservation of the chunk's integer value under
field conversion. These facts do not assert injectivity of the Horner
fingerprint or the image hash.

## Actual constraint coverage

| Item | Value |
|---|---:|
| Constraints | 97,630 |
| Auxiliary variables, including incoming state | 97,634 |
| Constant-one wire, separately recorded | 97,634 |
| Total wire slots | 97,635 |
| Incoming state wires `(h,a,r,t)` | `(0,1,2,3)` |
| Outgoing state wires | `(97632,77323,2,97633)` |
| Pixel byte wires | 7,680 |
| Sparse A/B/C terms | 832,541 |
| Horner updates | 3,080 |

| Component | Constraints |
|---|---:|
| Byte Booleanity and recomposition | 69,120 |
| Retained first-stage DCT rows | 5,120 |
| Fused second-stage transpose/Q/Horner | 3,080 |
| Pixel and chunk packing | 2,816 |
| 34 arity-8 Poseidon gadgets, 388 rows each | 13,192 |
| 18 arity-2 Poseidon gadgets, 238 rows each | 4,284 |
| Prepared row/step hash equalities | 17 |
| Counter increment | 1 |
| **Total** | **97,630** |

The proof uses the concrete exported relation, not a replacement circuit:

| Proof layer | Concrete connection and semantic result |
|---|---|
| `ConcreteBytes`, `Byte`, `Seed` | Checked equality of all 69,120 prefix rows with byte gadgets; canonical pixel-wire map; universal valid input/state seed |
| `ConcreteProgram`, `ProgramCertificates.All`, `StraightLine` | Checked extraction and well-formedness of all 28,510 suffix rows into an executable mathematical program; satisfaction and preservation |
| `DctSpec`, `DctProgram`, `DctProgramCertificates.All` | Actual first-stage and fused Horner rows implement the independent integer transform, retained order, and casts |
| `PackingCertificates` | All 16 actual row-packing segments equal the proved pixel/chunk operations |
| `Affine`, `PoseidonProgram`, `HashTrace` | Verified affine arithmetic and symbolic Poseidon compiler; round checkpoint composition |
| `HashTrace2.All`, `HashTrace8.All` | Kernel-checked actual template rows and compiler checkpoints establish both complete hash arities |
| `HashWiringCertificates`, `HashWiringTemplates`, `HashNodeSound` | Actual wire renamings connect all 52 hash nodes to the two verified templates |
| `HashTree`, `ActualHash`, `InputBridge` | Packing and node results compose to the independent image hash transition |
| `Counter`, `Target` | Actual exported counter row and unchanged challenge alias |
| `SatisfyingWitness`, `Complete`, `Composition` | Full-row completeness, full-step soundness/determinism, and connected iteration |

The completeness proof splits `Exported.rows` at 69,120 and uses
`List.take_append_drop`: the proved byte seed covers the prefix and the proved
program execution covers the entire suffix. Thus every actual exported row,
including the 17 prepared-hash equality constraints, is satisfied. Prepared
hashes are auxiliary wires, never assumed correct external values.

Finite certificates use kernel reduction and reusable mathematical lemmas.
Chunking and round checkpoints avoid repeatedly expanding the entire circuit.
The generators produce data; their success is not a semantic correctness axiom.

## Validation and proof-integrity evidence

The full acceptance driver regenerates and compares the ShapeCS artifacts,
Lean row data, parameters, and seven certificate families. Plain and traced
ShapeCS exports agree. Source fingerprints are checked before and after.

Executable regression tests cover five complete-step fixtures: zero, maximum,
alternating, asymmetric impulse, and seeded random. They use varied incoming
field states, including negative representatives. Actual Rust witnesses satisfy
all exported equations; native outputs agree with the independent Python and
executable Lean specifications. Shape agreement across fixtures is regression
evidence, not a proof of arbitrary Rust synthesis behavior.

Witness mutations alter pixel bits, first-stage values, prepared hashes, and
outgoing hash/accumulator/counter values after witness generation, bypassing
preflight comparisons. Artifact validation rejects changed mappings, invalid or
noncanonical entries, constants, ordering, and missing rows. Additional
kernel-checked certificate tests cover bad operation coefficients/destinations,
future-wire reads, ordering, RGB/Horner mapping, swapped hash inputs, and
packing constants/wiring. The small tutorial proves a false output becomes
possible when its output row is omitted.

`results/verification.json` records the strict full run. The axiom reports are
`results/axiom_audit.json` and `results/full_axiom_audit.json`. The permitted
transitive axioms are exactly:

```text
propext
Classical.choice
Quot.sound
```

These are Lean's standard logical axioms. The certification chain contains no
`sorryAx`, custom correctness axiom, `native_decide`, or `Lean.ofReduceBool`.
The full audit names all five principal theorems; the supporting audit also
covers the separately required bounded-integer and packing interpretations.
Executable evaluation in differential tests is separate from this chain.

## Pinned identities and extraction trust

The complete list of 106 protected files is in
`artifacts/source_manifest.json`. Selected SHA-256 identities:

| File | SHA-256 |
|---|---|
| `poseidon_97/src/circuit.rs` | `6979e5f558394248d74f8589d6c54ace1d2ebeab778c54853fe19ff83122808e` |
| `poseidon_97/src/dctq.rs` | `7f296979214a0157daf0609833c77444af60ebecbd572796bfef8c9d97b340bf` |
| `poseidon_97/src/poseidon.rs` | `a3a29cb0c7ea07c64e3e48bec31d7199d17d4167e303e024edde224675de591a` |
| `poseidon_97/Cargo.lock` | `17311849e0c7e9401cc144d61b85a72428d45e859cc81fd41d0eea790efbad5f` |
| `artifacts/step.json` | `b224af7062f6d689449f1689bb91d8845de9fc77d3230919f99b6f22eda594d7` |
| `artifacts/parameters.json` | `4bd0b820ce5f9dfa38b42b2bb9ccf6a4587b458fb929e977afb3de45a5b11a51` |
| `artifacts/source_manifest.json` | `bf357f11e8b1e8879e87218f232f0da4eb77990b4b53771d9efcca6ee9df051b` |

The canonical sparse-row digest, distinct from the JSON file digest, is
`120153f7a3043425cc47718bbd175ac4f9c777cca833b4691f4f64d444393dd3`.

Environment: macOS 15.6.1 arm64; Rust/cargo 1.90.0; Lean 4.28.0 commit
`7e01a1bf5c70fc6167d49c345d3bf80596e9a79b`; mathlib revision
`8f9d9cff6bd728b17a24e163c9402775d9e6a365`. The lockfile pins transitive
mathlib dependencies. `results/environment.json` records the environment and
artifact identities.

Accepted extraction trust includes the original Rust sources/dependencies,
Rust compiler and build environment, ShapeCS exporter, source-to-wire mapping,
serialization, and conversion to Lean literals. Namespace annotations and
fingerprints identify provenance; they are not mathematical hypotheses.
Concrete row-to-specification correspondence is proved within Lean.

Excluded claims: compiler/exporter formal correctness, universal Rust witness
generation correctness, collision resistance or unique images per digest,
polynomial binding, Fiat–Shamir, public-file parsing, Nova folding, Spartan,
proof-verifier correctness, and compression/image-quality guarantees. No
conventional JPEG equivalence is asserted.

## Reproduction and measurement

From `circuit_correctness`:

```sh
python3 scripts/verify.py
python3 scripts/clean_build.py --clean --jobs 2 --sample-tree-rss
.elan/toolchains/leanprover--lean4---v4.28.0/bin/lake -d lean build Certification
python3 scripts/audit.py --full
```

The clean runner deletes only this project's `lean/.lake/build`. It preserves
external mathlib dependency caches and the pinned toolchain. It builds the 280
local modules in the `Certification` dependency closure with at most two local
module builds concurrently. Measurements therefore include project compilation
and repeated Lake invocation overhead, but exclude dependency installation and
mathlib compilation. They are formal-verification build costs, not Nova proving
or proof-verification performance.

Measured clean build and standalone verification both passed:

| Measurement | Result |
|---|---:|
| Clean runner total wall time, including cache preflight | 1596.05 s |
| Local build stage wall time | 1594.33 s |
| Successfully rebuilt local modules | 280 / 280 |
| Maximum accounted child high-water RSS | 7.859 GiB |
| Peak sampled process-tree RSS | 10.402 GiB |
| Process-tree samples | 5541 |
| Fresh-process standalone Lean elaboration, built imports cached | 4.55 s |
| Standalone maximum child RSS | 2.003 GiB |

The clean build checked all 280 local modules and the exact `Certification`
target. All module exit codes were zero. Sources and build configuration were
fingerprinted before and after the run. See
`results/clean_certification.resources.json` and its per-module logs.

After that rebuild, a separate process ran `lake env lean Certification.lean`
from `lean`, without Rust synthesis, an image file, or a Nova prover. It checked
the exact target propositions and printed the standard-axiom dependencies.
This 4.55-second standalone check reuses the freshly built imports; it is not
another clean build. Evidence: `results/standalone_lean.resources.json` and
`results/standalone_lean.log`.

Memory metrics are deliberately separate: `RUSAGE_CHILDREN` reports the maximum
accounted child high-water RSS, not a sum of simultaneous processes. Sampled
process-tree RSS sums the runner and live descendants at 0.25-second intervals,
excluding the sampling process; it can miss short peaks and double-count shared
pages. Neither metric is macOS memory footprint.

Earlier incomplete foundation runs and the stopped monolithic hash experiment
remain available as historical records. They are not evidence of current
failure and are not substituted for the completed full-proof measurement.

## Final acceptance record

The post-clean-build `python3 scripts/verify.py` run exited **0**. All 19 phases
passed and `results/verification.json` records `status: pass`, `mode: full`,
`artifact_only: false`, and `certified: true`. The supporting axiom audit covers
79 declarations, and the principal audit covers all five required theorems.
The summed measured phase wall times were 95.62 seconds; this is a phase sum,
not a separately instrumented end-to-end wall-time measurement.

The final source check passed for all 106 pinned production/Nova files. A
requirement-by-requirement evidence record is in `results/completion_audit.json`.
No mathematical proof obligations remain within the stated scope. The accepted
extraction trust boundary and cryptographic exclusions remain unchanged.
