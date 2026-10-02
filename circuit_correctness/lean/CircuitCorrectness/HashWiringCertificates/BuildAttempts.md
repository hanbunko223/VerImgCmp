The first wiring implementation compared independently decoded complete hash
windows (388 or 238 rows). Group00 passed in 47 seconds, and the input-swap and
constant-mutation rejection checks passed in 7.9 seconds. Groups01–04 were
cancelled after over twelve minutes of concurrent work; process inspection
showed active CPU use and approximately 550–700 MiB RSS per process. This was
an optimization cancellation, not a failed mathematical check.

The replacement factors literal coefficient-ID templates, proves them equal
to the original export, and bounds each source-decoding equality check by one
128-row export chunk. Remaining list concatenations are proved structurally.
The generated modules use two dependency chains to bound concurrency.

The replacement successfully checked all52 instances. Group00 and Group01
passed in37 and38 seconds; the remaining groups completed in24–54 seconds
each. The final All module passed in2.4 seconds. These are incremental,
concurrent build timings, not a clean-project benchmark.
