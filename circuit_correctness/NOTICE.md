# Third-party mathematical material

`lean/CircuitCorrectness/Pratt.lean` is adapted from the first (list-certificate)
section of [CompPoly's PrattCertificate.lean](https://github.com/Verified-zkEVM/CompPoly/blob/main/CompPoly/Fields/PrattCertificate.lean).
Original copyright 2020 Bolton Bailey; Apache-2.0. The copyright/license header
is retained. The adaptation changes imports for mathlib v4.28.0 and excludes
the subsequent unused tactic implementation.

The factorization data in `scripts/generate_field.py` comes from
[CompElliptic's Pasta field certificate](https://github.com/daira/CompElliptic/blob/main/CompElliptic/Fields/Pasta.lean),
copyright 2026 the CompElliptic contributors, author Daira-Emma Hopwood,
MIT OR Apache-2.0. The downloaded reference is preserved in `references`.
Only the factorization data is used. Its other examples and native evaluation
commands are not imported into this project's proof chain.

Python's choice of factors and generators is not a primality assumption.
The generated Lean proof checks every factor product, modular exponentiation,
and recursive primality obligation using the mathematical Pratt checker.

Lean is pinned to v4.28.0; mathlib and its transitive dependencies are pinned in
`lean/lake-manifest.json`. Dependency licensing remains with the respective projects.
