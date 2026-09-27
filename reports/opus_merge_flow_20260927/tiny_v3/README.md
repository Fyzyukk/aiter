# Tiny v3: v1 flow with the grid correction

IDs21220 (64×128/S3) and21221 (64×64/S4) compiled successfully. The
version is frozen for parent-owned timing; no GPU execution ran in this task.

This version preserves the full v1 main loop, compiler unroll directive,
prefetch and wait selection, scale handling, fragment schedule and output.
Only the compact2×4 grid branch is restricted to the64-column tile, so
128-column tiles keep the original9011 general partition order. Device
entry and private candidate IDs are renamed to keep versions separate.

The source from the runtime K bound through final output is byte-identical
to v1. Traits are also byte-identical. `generation.json` records this check
and all frozen v1 input/prepared v3 output hashes. No v1/v2 or production
source was edited. ELF inspection confirmed exactly two device kernels; build and binary
hashes are recorded in `build_manifest.json` and `device_audit.json`.
