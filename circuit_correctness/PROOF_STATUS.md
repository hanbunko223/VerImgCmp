# Full-step certification complete

The original 97,630-row `poseidon_97` application step has completed Lean proofs:

- `CircuitCorrectness.Target.step_soundness`
- `CircuitCorrectness.Target.step_completeness`
- `CircuitCorrectness.Target.step_determinism`
- `CircuitCorrectness.Target.connected_steps`
- `CircuitCorrectness.Target.connected360`

The strict `python3 scripts/verify.py` command passed after a clean build of all
280 certification modules and a separate fresh-process standalone Lean check.
The final result is `certified: true`. All 79 supporting and five principal
axiom audits passed, allowing only `propext`, `Classical.choice`, and `Quot.sound`.
All 106 pinned production/Nova source fingerprints still match.

The statement covers the original unnormalized DCT-Q/hash/Horner computation.
Completeness covers all 69,120 byte-prefix rows and all 28,510 suffix rows.
The 360-step theorem assumes explicit adjacent-state connections and does not
verify Nova. Cryptographic security and the compiler/exporter remain outside
the formal theorem as detailed in REPORT.md.

Current evidence: `results/completion_audit.json`, `results/verification.json`,
`results/full_axiom_audit.json`, `results/clean_certification.resources.json`,
and `results/standalone_lean.resources.json`. Historical incomplete reports are
identified in `results/README.md`; they are not current certification status.
