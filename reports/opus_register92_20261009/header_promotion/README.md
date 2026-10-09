The 64 previously audited candidates now use public production declarations in
`csrc/opus_gemm/include/gfx950`. Eight new pipeline headers and four grouped
traits headers preserve their frozen device arithmetic and synchronization.
The two additional 64x64 tile-order configurations reuse the existing 64x64
pipeline, and the six fine-M configurations reuse the existing small-LDS
pipeline. The hybrid direct-B and register split-K pipelines include the
existing small-LDS reducer instead of defining another copy.

`final_receipt.json` records successful offline aggregate compilation with the
exact Clang 23 and pin-enabled Clang 24 code-generation flags from the audited
main-variants receipt. The Clang 23 aggregate explicitly instantiates 53 kernels;
the pin-enabled aggregate instantiates 11 and includes all promoted and existing
MXFP8 B-preshuffle pipeline headers to check helper and symbol collisions. Both
translation units instantiate all 64 traits and pass host syntax checks. CPU ELF
inspection verifies all 64 expected kernels have zero scratch bytes and zero
VGPR/SGPR spills. No GPU query, device execution, or HIP/HSA library load occurs.

`receipt.json` is the retained first attempt. Both compilations and host syntax
checks passed, but its initial `llvm-readelf` commands were given the Clang
offload bundle rather than the contained gfx950 ELF. `finalize.py` extracts the
existing compiled image independently and with LLVM, verifies both copies are
equal, and records the successful ELF inspection without recompilation.

`source_identity.json` verifies the nine promoted traits declarations are byte
identical after public symbol renames, pin helper definitions are byte identical
inside their scoped namespace, and the shared reducer is byte identical to the
frozen candidate copy. `final_receipt.json` also records normalized byte identity
of all eight promoted pipeline device bodies.

`template_groups92.json` filters the generated 105-entry metadata table to the
92 public tuning candidates and verifies every generated device TU hash. The 92
configurations share 18 primary GEMM function templates and one separate split-K
reducer. The retained 28 configurations use ten primary templates; the 64 added
configurations introduce eight and reuse two existing templates.

The production `opus_gemm_bpreshuffle_variants.py` module is the runtime and
code-generation parameter source. The promoted-variants JSON records frozen
candidate provenance for CPU checks. Offline compile success and static resource
metadata do not establish numerical correctness or performance; those GPU
checks remain pending while GPU testing is stopped.
