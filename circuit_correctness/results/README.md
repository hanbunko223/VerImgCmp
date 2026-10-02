# Verification evidence

Use the status fields inside the JSON files; the presence of a file alone does
not establish that a run succeeded.

Current acceptance evidence:

- `completion_audit.json`: final requirement-by-requirement review, evidence
  fingerprints, measurement summary and source-preservation result.

- `verification.json`: complete driver result; full certification requires
  `status: pass`, `certified: true`, `mode: full`. `artifact_only: false` also
  includes protected-source checks and Rust regeneration.
- `full_axiom_audit.json`: exact five principal theorems and transitive axioms.
- `axiom_audit.json`: supporting mathematical results, including separate
  bounded-integer interpretations.
- `clean_certification.resources.json`: clean local-module build. External
  dependencies are cached; check `status`, `clean_performed`, all module exit
  codes, and the resource-semantics fields.
- `clean_build_modules/`: per-module clean-build logs.
- `rust_validation.json`, `artifact_validation.json`, `reference_validation.json`,
  `structured_check.json`, `lean_differential.log`: regression evidence.
- `final_semantic_review.json`, `final_validation_review.json`: review snapshots;
  their followups describe the state at review time, not necessarily final status.
- `environment.json`: pinned artifact identities and environment.

`standalone_lean.resources.json` and `standalone_lean.log` record a successful
fresh-process Lean check after the clean rebuild. The project README and REPORT
identify the final acceptance and completion-audit records.

Historical evidence retained for provenance:

- `progress-2026-09-28.json`: earlier incomplete milestone; it is not the current
  certification status.
- `clean_foundation.resources.json`: an earlier partial foundation build; it is
  not a full-proof clean-build measurement.
- `hash_probe.log` and `experiments/hash_monolithic_probe.lean.txt`: a stopped
  monolithic checking experiment, replaced by kernel-checked round checkpoints.
- Other construction/build logs record intermediate attempts; final acceptance
  is determined by the named current gate and audit records above.

These are Lean build and validation measurements, not cryptographic proof-system
prover/verification benchmarks. Child high-water RSS and sampled process-tree RSS
have different meanings; consult the measurement JSON descriptions.
