# Independent Oct8 application review

`independent_identity_plan_review.full.json` independently rereads both official libraries and all206 build objects.
It compares actual FUNC bytes, full metadata and64-byte descriptors with only entry-offset bytes16..23 normalized.
The202 linked bundles contain263 entries: exactly the final three selected entries equal their reviewed private
candidates; all other entries equal the current Oct8 baseline. The26 public parents contain56 variants, including
the9020 fixed384 candidate and six exact-baseline9020 variants. All2691 tracked build inputs,313 generated files,
normalized ninja ordering and the sealed identity/plan/queue/runner/library hashes also passed.

The independent branch review is recorded separately in `api_gate/independent_dispatch_review.json`.
The44 API targets cover26 public parents and the exact three changed entries. The9023 fixed7168 control and9024
runtime control remain unchanged. Target32 `[9020,16,256,384]` is an unchanged runtime case because M is below1024;
its purpose string must not be interpreted as coverage of the changed fixed384 entry. The existing1472 winner
covers a legal fixed384 M tail.

The preapply snapshot `source_preapply_snapshot.json` passed: production had all2691 tracked build inputs exactly
equal to the current old-Oct8 baseline before source application. It also sealed4974 files and their file sets
under `opus_bound_analysis_20261007` and `opus_resume_20261008`, including hidden Python bytecode. The before/after
SHA expectations for all three selected sources are explicit in that snapshot. The application review requires
those historical file sets to remain unchanged. User authorization, relayed by root in this session, permits
adding this session's conclusions to the original Compute and Memory reports. For those two documents, the entire
preapply prefix remains exact; all4972 other files retain exact hashes. `authorized_documents_snapshot.json` further
freezes the bytes before section12, allowing the new section's API pending statements to be updated during closure.

After root completes the strict official API analysis and applies the three selected sources, run:

```bash
python3 reports/opus_remaining_20261008/formal_selected/independent_identity_plan_review.py \
  --application-root /root/workspace/aiter-opus-mxfp8-bpreshuffle \
  --output reports/opus_remaining_20261008/formal_selected/independent_application_review.json
```

The application review requires the passed44-target official signed8 API report and unchanged accepted evidence.
It checks every tracked `aiter/`, `csrc/` and `3rdparty/` build input against the tested selected worktree, all three
applied snapshot hashes, source HEAD, diff whitespace and the sealed official SO. It never applies sources,
builds or runs GPU work. Outputs must be new files; failure records are preserved. The accepted application report
is `independent_application_review.corrected.docs_authorized.json`. Earlier failures preserve the path-order bug
and the initially omitted explicit authorization for the two original report appends.

After the two authorized section12 texts are finalized, run the finite document/artifact review:

```bash
python3 reports/opus_remaining_20261008/formal_selected/independent_identity_plan_review.py \
  --documents-only \
  --output reports/opus_remaining_20261008/formal_selected/independent_final_documents_review.json
```

It checks each unique section12 boundary, protected old-prefix SHA, all4972 frozen artifact hashes and both complete
historical file sets, then binds final complete-document and section12 hashes. It does not rerun binary/GPU gates.

9020 retention preserves the large isolated negative Event observation. Seven shape medians are positive and
34/35 paired observations are faster, with no five-round stable loser, but `[12288,7168,384]` round2 has speedup
0.834618 and call time+19.815%. That shape's AB geometric mean is negative and the whole seven-shape round2
geometric mean is negative. The review does not discard that value or assert its cause was proved to be noise.

These CPU reviews establish source/binary identity and the finite gate protocol. Performance conclusions remain
the separate retention decisions, and correctness acceptance remains the strict root-owned GPU API analysis.
