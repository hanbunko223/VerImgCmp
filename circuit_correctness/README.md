# Formal correctness of the `SPEG-Poseidon` application step

The full exported step has Lean proofs of **soundness, completeness, and output
determinism**, plus a theorem for 360 explicitly connected steps. The strict
acceptance command has passed with `certified: true`. See [REPORT.md](REPORT.md)
for the exact statement, validation evidence, and build measurements.

This certifies the original 16×160 RGB application step, used 360 times for HD.
It concerns the pinned, unnormalized approximate integer DCT-Q computation,
including its input hash and Horner accumulator. It does not concern
`poseidon_97_180`, conventional JPEG, Nova's augmented circuit, or Spartan.

## What Lean proves

An R1CS is a list of equations involving wires. Some wires hold the image and
incoming state; others hold intermediate results and the outgoing state.
Satisfaction means that **all** equations hold, including the constant-one
condition. Lean checks a mathematical proof about every satisfying assignment,
not merely the assignment produced by the Rust witness generator.

The specification is a separate Lean function:

```text
StepSpec(image, (h, a, r, t)) =
  (HashTransition(h, image),
   Horner(a, r, retained approximate DCT-Q coefficients of image),
   r,
   t + 1)
```

The image hash uses original bytes. The DCT branch uses centered pixels `X−128`,
the pinned 128-scale matrix, and threshold-97 reciprocal multipliers. There is
no normalization or coefficient rounding. All four state entries are field
elements; `t+1` is a field addition.

The principal declarations in
[Complete.lean](lean/CircuitCorrectness/Complete.lean) are:

- `CircuitCorrectness.Target.step_soundness`: every satisfying assignment with
  the given valid image and incoming state produces the specified output.
- `CircuitCorrectness.Target.step_completeness`: every valid image and incoming
  state has a satisfying mathematical assignment producing that output.
- `CircuitCorrectness.Target.step_determinism`: two satisfying assignments with
  the same image and incoming state have equal outgoing states.
- `CircuitCorrectness.Target.connected_steps` and `connected360`: explicitly
  connected steps implement repeated application of the specification.

Determinism here does not mean that a digest identifies a unique image, or that
a polynomial fingerprint uniquely identifies a coefficient vector. Those are
separate cryptographic questions and are outside this proof.

## A complete small example

[Example.lean](lean/CircuitCorrectness/Example.lean) uses six wires:

```text
w = (one, x, y, sum, product, output)
```

Its three rows enforce:

```text
(x + y) × one             = sum
sum × x                   = product
(product + 3 × one) × one = output
```

The satisfaction predicate also requires `one = 1`. Its matrices are:

```text
A = [0 1 1 0 0 0]    B = [1 0 0 0 0 0]    C = [0 0 0 1 0 0]
    [0 0 0 1 0 0]        [0 1 0 0 0 0]        [0 0 0 0 1 0]
    [3 0 0 0 1 0]        [1 0 0 0 0 0]        [0 0 0 0 0 1]
```

The independent specification is `spec x y = (x+y)*x+3`.

For **soundness**, assume an arbitrary assignment satisfies the rows. The first
row determines `sum`, the second determines `product`, and the third determines
`output`. Substitution gives the specification. For **completeness**, construct
`(1,x,y,x+y,(x+y)*x,(x+y)*x+3)` and prove all equations. For **determinism**, apply
soundness to two assignments sharing the same inputs.

Lean checks proof terms for these universal statements over the Pallas scalar
field. Tactics such as `simp` help construct proof terms; they do not become
correctness assumptions. The example also proves that removing the last row
admits `x=y=0, output=4`, although the specified output is `3`.

## How the production proof connects to Rust

1. The protected-source manifest identifies 106 original Rust/dependency files.
   Verification checks it without replacing it.
2. A separate Rust harness includes the production modules by source path and
   calls the actual `DctqStepCircuit::synthesize` with four incoming state wires.
3. Plain and namespace-traced `ShapeCS` exports must agree exactly. The export
   includes all 97,630 sparse rows, the constant wire, interfaces, pixel-wire
   coordinates, and Poseidon parameters.
4. Generated Lean numeral data represents those rows. The local decoder and
   `Exported.circuit.Sat` define the mathematical relation being certified.
5. Kernel-checked, chunked certificates connect the **actual rows** to byte,
   arithmetic, packing, and Poseidon operations. Poseidon round checkpoints
   prove that both actual gadget templates implement the functional rounds;
   separate certificates connect all 52 instantiated hash nodes.
6. The soundness proof composes these relations into the independently defined
   [Spec.step](lean/CircuitCorrectness/Spec.lean). Completeness constructs a byte
   seed and runs a proved straight-line program covering the remaining rows.
   The prefix and suffix account for every exported constraint.

The Python structured comparison and sample tests are additional regression
checks. They are not used as substitutes for Lean's row-to-specification proof.

## Reproduce the complete acceptance gate

Prerequisites: Rust/Cargo, Python 3, and elan. Before the first verification,
install and fetch the pinned dependencies (a network connection is needed):

```sh
elan toolchain install leanprover/lean4:v4.28.0
cd lean
lake exe cache get
cd ..
cargo fetch --locked --manifest-path exporter/Cargo.toml
```

Run from `circuit_correctness`:

```sh
python3 scripts/verify.py
```

This checks source identity, regenerates and compares exports and generated
Lean data, runs differential and mutation checks, builds the exact principal
theorem types, and audits their transitive axioms. The authoritative result is
[results/verification.json](results/verification.json). Only the full mode can
record `certified: true`; `--development` deliberately cannot.

To verify the committed mathematical artifact without a private image, Rust
synthesis, or Nova proving:

```sh
cd lean
lake build Certification
cd ..
python3 scripts/audit.py --full
```

This uses the committed Lean data and fetched pinned dependencies. Build caches
and the Lean toolchain are not included in Git. To additionally
check artifact regeneration and synthetic differential fixtures while skipping
Rust/source access, use `python3 scripts/verify.py --artifact-only`. That command
replaces `results/verification.json`; its `artifact_only` flag distinguishes it
from full source/extraction validation.

For a measured clean build with at most two local module builds at once:

```sh
python3 scripts/clean_build.py --clean --jobs 2 --sample-tree-rss
```

This removes only `lean/.lake/build`, preserves the pinned toolchain and external
mathlib caches, and checks the full `Certification` dependency closure. It
records build wall time, child high-water RSS, and sampled process-tree RSS.
If the operating system disallows process inspection, the sampled metric is
reported as unavailable. Do not run another project build simultaneously.

Individual exporter commands:

```sh
CARGO_TARGET_DIR="$PWD/target" cargo run --offline --locked --release --manifest-path exporter/Cargo.toml -- check-source
CARGO_TARGET_DIR="$PWD/target" cargo run --offline --locked --release --manifest-path exporter/Cargo.toml -- export
CARGO_TARGET_DIR="$PWD/target" cargo run --offline --locked --release --manifest-path exporter/Cargo.toml -- check-export
```

`check-export` compares regenerated artifacts without replacing them. `export`
refuses source drift. The source-manifest initialization command refuses to
replace an existing baseline.

## Toolchain and trust boundary

Lean is pinned to 4.28.0; the scripts use a project-local `.elan` when present,
otherwise `lake` on PATH (or an explicit `CIRCUIT_CORRECTNESS_LAKE` executable). mathlib and transitive revisions
are pinned in `lean/lake-manifest.json`. To install that Lean toolchain on a new
machine with elan available:

```sh
ELAN_HOME="$PWD/.elan" elan toolchain install leanprover/lean4:v4.28.0
```

Restore the pinned mathlib dependency/cache before building; do not update its
revision. Keep any new cache/build outputs inside this project. The initial
installation used mathlib's default user cache; no claim is made that this
historical dependency installation was a hermetic build.

The accepted extraction boundary includes the Rust compiler/build environment,
source-path harness, ShapeCS exporter, and conversion into Lean literals.
Fingerprints identify the extracted sources; they do not prove the exporter or
compiler correct. Within Lean, concrete row-to-specification correspondence is
proved, and the transitive audit permits only `propext`, `Classical.choice`, and
`Quot.sound`. There are no assumed Poseidon/full-step correctness lemmas or
compiler-evaluation shortcuts in the certification chain.

Excluded: collision resistance, image uniqueness, polynomial-binding error,
Fiat–Shamir, public-file parsing, Nova/Spartan/proof-verifier correctness, and
universal correctness of the Rust witness generator. The 360-step theorem
assumes connected states; it does not prove Nova establishes them. This project
also does not prove approximation quality or equivalence to conventional JPEG.
See [NOTICE.md](NOTICE.md) for mathematical-source attribution.

## Publication validation (2026-10-02)

The relocated exporter passed all 19 phases of `scripts/verify.py`, with
`certified: true`, using the unchanged 294 Lean source files and unchanged
constraint/parameter artifacts. Dependency and compiled Lean caches were reused
for this publication check; it is not a new from-scratch build measurement. The
original clean-build measurements remain in the report. No private image or Nova
proof is required for Lean verification. See `results/publication.json` for the
release validation record.
