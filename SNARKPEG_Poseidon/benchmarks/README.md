# Reproduce HD measurements

Build with `../build.sh`. From this directory run:

```sh
python3 run.py --threads 8
```

This script is for macOS (the preserved memory sampler uses `time -l` and
`libproc`). It enforces the historical 10 GiB memory budget, runs one warm-up
and three fresh-process HD proving runs, and verifies saved recursive and
compressed proofs in independent processes. Results are written to a new
`runs/<timestamp>/` directory; existing results are not overwritten.
Use `--threads 1` for a separate single-thread campaign. Python needs Pillow.

The generated candidate digest is suitable only for the reproducibility demo;
it is not evidence of external authorization. The full public coefficient file
is needed for verification and is not included in the proof-byte measurement.

`historical/` contains original machine-readable measurements. Recorded paths
and commands in those JSON files describe the historical machine; they are
provenance, not portable commands. `../REPORT.md` summarizes that campaign,
including its original HH control. This release's runner measures only the
published, certified `poseidon_97` implementation. Setup, recursive creation,
Spartan compression, serialization, and verification remain separate metrics;
no new benchmark performance is claimed by the publication validation.
