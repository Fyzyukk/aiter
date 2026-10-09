# Remaining Oct8 official integration tools

The current baseline is `reports/opus_resume_20261008/jit_formal_selected/module_deepgemm_opus.so`, SHA256
`6f6a0117b122f06b4802833c020effb4b24ff9fea56f20c8c16b9ced9787173f`.
These tools read historical records and write new files under this directory. They do not apply sources, build, or run GPU work.

`definite_selection_template.json` contains the three definite entries: existing9020 fixed384,9023 runtime and9024 fixed7168.
It references the scoped9020 snapshot plus the exact tested narrow snapshots. Root can save a final selection as a new JSON
and explicitly add any later retained entry. Rejected candidates should remain absent from that selection.

Each `source_files` item names a candidate worktree path, a reviewed snapshot and its SHA. Each `changed_entries` item names
one exact public parent/symbol and sealed baseline/candidate private libraries. The tool independently requires the private
baseline to match the current official entry. The final official entry must match the selected candidate, including all
instruction bytes, full metadata and the descriptor with only entry-offset bytes16..23 normalized.

Before the root official build, run the source-only audit with a new output name:

```bash
python3 reports/opus_remaining_20261008/formal_selected/audit_integration.py \
  --selection reports/opus_remaining_20261008/formal_selected/final_selection.json \
  --source-only \
  --output reports/opus_remaining_20261008/formal_selected/source_preflight.final.json
```

After the root build writes its existing `build_current.py` manifest to this directory:

```bash
python3 reports/opus_remaining_20261008/formal_selected/audit_integration.py \
  --selection reports/opus_remaining_20261008/formal_selected/final_selection.json \
  --build reports/opus_remaining_20261008/formal_selected/build_manifest.json \
  --output reports/opus_remaining_20261008/formal_selected/identity_audit.json
```

The complete check covers26 public parents/56 device entries, all206 build objects and every202 linked gfx950 bundle
(263 linked device entries in the current baseline). All generated implementation/instance files and normalized ninja
flags/input/link ordering must remain identical. All unselected device instructions, metadata and normalized descriptors
must match the current baseline, including the six unselected9020 entries and nonpublic kernels.

Prepare the official API queue only after that complete identity audit passes:

```bash
python3 reports/opus_remaining_20261008/formal_selected/prepare_api_gate.py \
  --identity reports/opus_remaining_20261008/formal_selected/identity_audit.json \
  --output-dir reports/opus_remaining_20261008/formal_selected/api_gate
```

Root runs `api_gate/official_api_queue.json` through its physical idle-device lock. The plan covers one representative
per26 public parents, every selected entry and each selected-entry winner K group, plus finite boundary/control cases.
Private labels are attached only when exact scalar dispatch and linked/private candidate identities agree. The retained
GPU runner imports the production Python API/reference helpers; the tool verifies those inputs equal the selected
worktree and the runner checks the actual loaded official SO SHA.

The three-entry plan includes the M17 partial helper case only for9023, whose M alignment is1.9020 and9022 require16-aligned
M. The default gate is check-only, signed seed17, eight repetitions, guarded output at256B. Split-K parents exercise their
official workspace/reducer path. Large outputs use the runner's independent FP32-bound reference in256-row chunks.

After the root queue completes:

```bash
python3 reports/opus_remaining_20261008/formal_selected/analyze_api_gate.py \
  --api-dir reports/opus_remaining_20261008/formal_selected/api_gate \
  --output reports/opus_remaining_20261008/formal_selected/api_gate/gpu_api_analysis.json
```

The result review requires every command's exact plan, unique owner marker, command fingerprint, physical GPU identity,
clean monitor epoch, actual official module SHA and signed8 reference/repeatability/guards. An unowned process causes
rejection even when a command end record says `contamination:false`. The review makes no performance claim.

Output files must be new. A failure record is preserved; use a new explicit output name for a later corrected audit.
The tools never overwrite the Oct7/early-Oct8 records or mutate production sources.
