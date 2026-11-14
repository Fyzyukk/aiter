# MXFP4 × MXFP4 NT small-tile pipelines (gfx950) — dev / test directory

Small-M (M ≤ ~64) and medium-M (M ≈ 128 – 1024) tiles; the 2×2 big-tile pipeline (`../mxfp4_nt_dev`) covers M ≳ 1024.
Two pipelines, one per input layout, with the same structure (1×4 waves, LDS-DMA ring, one step ahead, counted vmcnt,
KS split; the row-major one also sends B and the scales through the LDS ring); **standard ROCm toolchain** (`/opt/rocm/llvm/bin/clang++`): no pin attributes →
built by the aiter JIT, no `.co`.  Both pipelines are written with opus (`../../include/opus`) and are independent of each
other (own traits / kargs / namespace, no shared headers; both can be included in one TU).

Files (`../include/gfx950/`):

| file | contents |
|---|---|
| `opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh` | shuffle kargs `opus_gemm_mxfp4_small_kargs_gfx950`, traits `opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950<BM, BN, D, KS>` (1×4 waves; T_M/T_N/T_K, E_M/E_N/E_K, LDS / VMEM-per-step sizes) |
| `opus_gemm_traits_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh` | row-major kargs `opus_gemm_mxfp4_small_noshuffle_kargs_gfx950`, traits `opus_gemm_a4w4_mxfp4_nt_small_noshuffle_traits_gfx950<BM, BN, D, KS>` (as the shuffle traits) |
| `opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh` | shuffle pipeline (opus: `make_layout_ga/sa/ra`, `gb`, `gsfa/gsfb`, `gc`), body `gemm_a4w4_mxfp4_nt_small_body`, kernel `gemm_a4w4_mxfp4_nt_small_kernel<Traits>` |
| `opus_gemm_pipeline_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh` | row-major pipeline, same structure / names as the shuffle file (namespace `opus_mxfp4_small_noshuffle`: `make_layout_ga/sa/gas/ra` (A), `gb/sb/gbs/rb` (B, the same 8-row LDS-DMA chunks as A), `gsfa/gsfb` + `make_offset_ssf` + `rsfa/rsfb` (scales: block-wide 4-byte LDS-DMA image, `sf_pack` v_perm), `gc`; `issue` / `read` lambdas), body `gemm_a4w4_mxfp4_nt_small_noshuffle_body`, kernel `gemm_a4w4_mxfp4_nt_small_noshuffle_kernel<Traits>` |
| `splitk_reduce_gfx950.cuh` (shared with a16w16) | `splitk_reduce_kernel`: the split-K reduce of both small-tile pipelines (as a16w16 flatmm splitk) |

Inputs:
- shuffle (`-l 0`): as aiter `gemm_a4w4(bpreshuffle=True)`: A `[M, K/2]` row-major fp4x2, B `shuffle_weight(16,16)`,
  scaleA / scaleB `e8m0_shuffle` (1x32 UE8M0), C `[M, N]` bf16.  Any M, N % 16 == 0, K % 256 == 0.
- row-major (`-l 1`): as Triton `gemm_afp4wfp4`: A `[M, K/2]`, B `[N, K/2]`, scales `[rows, K/32]`, no padding.
  Any M, N % 8 == 0, K % 256 == 0.

Tails are handled in-kernel (no padding).  Non-persistent: grid = (⌈N/BN⌉, ⌈M/BM⌉, batch × split).

All tiles use a K tile (BK) of 256.  `-d` D = ring depth (prefetch); `-s` KS = in-block K split of the 4 waves.

| pipeline | tiles (`-t`) | other knobs | meant for |
|---|---|---|---|
| shuffle (`-l 0`), 1×4 waves: the 4 waves split BN and share A; A global → LDS-DMA D-deep ring, each wave's B + scales → a D-deep register ring, asm-style (A read one step ahead, copies interleaved with MFMAs) | 16×32, 16×64, 16×128, 32×32, 32×64, 32×128, 64×128, 32×256 | `-s` KS 1/2/4 (waves split K, summed via LDS once), `-d` 3–10 ((D−1)·G ≤ 63) | M ≤ 64 (16×32 KS2/KS4, 16×64), M ≈ 128 – 512 (beat the former shuffle B_LDS by 12–30% and B_DIRECT by up to 18% at M ≤ 32, see prof/alds_vs_direct_smi3) |
| row-major (`-l 1`), the same 1×4-wave pipeline; A, B (8-row × 128 B chunks, XOR swizzle) and the scales ([row][8 B] image per K tile, block-wide 4-byte DMA) all go global → LDS-DMA D-deep ring, every wave ds_reads its operands one step ahead | 16×16 / 16×32 / 32×32 (KS 2/4), 16×64, 32×64, 64×64 (KS 1/2), 16×128, 32×128, 64×128, 16×256, 32×256, 64×256 (see `pick_rm`) | `-s` KS 1/2/4, `-d` ((D−1)·G ≤ 63, LDS ≤ 160 KB) | M ≤ 16 (16×32 KS2 D4), M ≈ 64 (32×32 KS2 D4), M ≈ 128 – 512 (32×64, 64×64, 64×128 D4) |
| both | | `-p S` cross-block split-K (≤ 8): fp32 partials → workspace, then `splitk_reduce_kernel` (as the aiter launcher) | few tiles |

```bash
make                                    # -> /tmp/opus_mxfp4_small/small.exe   (OUT=..., CLANG=...)
E=/tmp/opus_mxfp4_small/small.exe
$E -t 32x128 -d 4 -m 256 -n 4096 -k 4096 -v 1               # correctness (exact GPU reference)
$E -t 32x128 -d 4 -m 256 -n 4096 -k 4096 -v 0 -i 100        # timing: hipGraph of 100 launches, median of 5
prof/check.sh $E <HIP gpu> 2:16x32:2:4 2:32x128:1:4 1:16x16:4:6:1 ...    # full correctness matrix per MODE:TILE:KS:D[:LAYOUT]
```

Performance sweep (one command, portable; see `prof/perf_all.py` header):
```bash
AITER_USE_SYSTEM_TRITON=1 python3 prof/perf_all.py --small-exe $E \
    --big-exe <2x2 harness: ../mxfp4_nt_dev/build_var.sh cur "" -> /tmp/opus_mxfp4/cur/build/gemm_a4w4_mxfp4_4wave.exe> \
    --gpu <HIP index> --out perf_out
```
Row-major: `prof/sweep_rowmajor.py` (every `pick_rm` config × S on M × (N, K); how kids 30500-30504 were chosen,
result `prof/sweep_rowmajor_HIP6.csv`), `prof/perf_ns_vs_triton.py` (through aiter: no-shuffle 2×2 kids, row-major small kids
and Triton gemm_afp4wfp4 on identical inputs, each checked against a reference before timing).
`perf_all.py` waits for the GPU to be idle before each shape (amd-smi CU_OCCUPANCY, HIP→smi index matched by PCI bus), re-runs a shape
that got disturbed, times every small-tile config (+ split-K 2/4), every 2×2 tile (persistent / one-tile-per-block / split-K)
and aiter asm (CUDA graph), and writes `results.csv`, `table.txt` and `select.csv` ((M, N, K) → best pipeline + config).

## Interface for integration (kids / dispatch)

**Template**: shuffle `gemm_a4w4_mxfp4_nt_small_kernel<opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950<BM, BN, D, KS>>`;
row-major `gemm_a4w4_mxfp4_nt_small_noshuffle_kernel<opus_gemm_a4w4_mxfp4_nt_small_noshuffle_traits_gfx950<BM, BN, D, KS>>`.
`__launch_bounds__(256, 1)`, block = 256, standard hipcc (no pin → plain aiter JIT, no `.co`).
- `KS` 1/2/4 (waves split along K inside the block), `D` = ring depth.
- Kid names (opus_gemm_common.py): shuffle `opus_gemm_gfx950_a4w4_mxfp4_nt_d<D>ks<KS>_256x<BM>x<BN>x256_1x4_16x16x128_1x1x32`
  (kids 30410-30416); row-major `opus_gemm_gfx950_a4w4_mxfp4_nt_rm_d<D>ks<KS>_256x<BM>x<BN>x256_1x4_16x16x128_1x1x32` (kids 30500-30504,
  S = 1); split-K variants (not built) add `_sk<S>` to the tag.
- Shuffle file: gemm_a4w4(bpreshuffle=True) inputs (B shuffle_weight(16,16), scales e8m0_shuffle).  Noshuffle file: Triton
  gemm_afp4wfp4 inputs (B [N, K/2], scales [rows, K/32], no padding).  The harness `-l 0/1` picks the file.
- Constraints: K % 256 == 0; N % 16 (shuffled) / N % 8 (row-major); any M; batch ≥ 1.

**kargs** (shuffle `opus_gemm_mxfp4_small_kargs_gfx950` / row-major `opus_gemm_mxfp4_small_noshuffle_kargs_gfx950`, see the traits headers; same fields, separate structs): byte strides for A (`stride_a` = K/2), elements for C (`stride_c` = N);
`stride_sfa / stride_sfb` = bytes per 32-row block (shuffled: 32 × round_up(K/32, 8)) or bytes per row (row-major: K/32);
batch strides in bytes (C in elements).  Shuffled scale batch stride = padded rows × round_up(K/32, 8) (aiter pads rows to 256).

**grid** = (⌈N/BN⌉, ⌈M/BM⌉, batch × S), S = `split_k` (1–8, K/256 ≥ S); non-persistent.

**split-K (S > 1)** (both layouts): `ws_handle` → `opus_splitk_ws_handle` whose `ptr` is an fp32 workspace [S][batch][Mp][Np]
(Mp, Np = M, N rounded up to BM, BN; read by the kernel at entry, so captured graphs survive a workspace grow); every block
writes its partial tile, then `splitk_reduce_kernel<16, 64, __bf16, false, __bf16, HAS_OOB>` (HAS_OOB = true for row-major:
N % 8 leaves a partial 16-column tail) with grid (⌈N/1024⌉, batch × M),
64 threads, sums the S partials into C.  The aiter launcher takes the slot from `opus_splitk_ws_get` (per stream, as the
a16w16 flatmm splitk launcher).  With S = 1 `ws_handle` is not read.  No shuffle split-K kids are built: on the M ≤ 64
kid configs (16×32 KS2/KS4, 16×64; M 1–64, (N, K) ∈ {(8192, 1024) … (4096, 8192)}, HIP 6) S = 2 / 4 was 1.3–4× slower than
S = 1 (second kernel + workspace round trip + two reads of the host-coherent slot, ≈ 2 µs of it), and on 32×128 / 64×128 at
M 64–256 at most 5% faster in two shapes; the support stays (opus_gemm_common.py: `_a4w4_mxfp4_nt_small(..., s)` with s > 1
→ tag `a4w4_mxfp4_nt_d<D>ks<KS>_sk<S>`).

No row-major split-K kids either: in the config sweep (prof/sweep_rowmajor_HIP6.csv) S = 2 / 4 never beat S = 1.
