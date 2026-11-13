# MXFP4 × MXFP4 NT small-tile pipeline (gfx950) — dev / test directory

One file for the small-M (M ≤ ~64) and medium-M (M ≈ 128 – 1024) tiles; the 2×2 big-tile pipeline (`../mxfp4_nt_dev`)
covers M ≳ 1024.

- kernel: `../include/gfx950/opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh`
  (`gemm_a4w4_mxfp4_nt_small_kernel<opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950<BM, BN, D, KS, BMODE>>`)
- traits / kargs: `../include/gfx950/opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh`
- **standard ROCm toolchain** (`/opt/rocm/llvm/bin/clang++`): no pin attributes, no opus → can go through the aiter JIT, no `.co`.

Inputs as aiter `gemm_a4w4(bpreshuffle=True)`: A `[M, K/2]` row-major fp4x2, B `shuffle_weight(16,16)`, scaleA / scaleB
`e8m0_shuffle` (1x32 UE8M0), C `[M, N]` bf16.  Any M, N % 16 == 0, K % 256 == 0; tails handled in-kernel (no padding).
Non-persistent: grid = (⌈N/BN⌉, ⌈M/BM⌉, batch × split).

| B path (`-a`) | tiles (`-t`) | other knobs | meant for |
|---|---|---|---|
| 0 `B_DIRECT`: A, B, scales global → registers, D-deep register ring, no LDS / barrier in the K loop | 16×32, 16×64, 32×32, 32×64 | `-s` KS 2/4 (waves split K, summed via LDS once), `-d` 2–4 | M ≤ 64 (128 when K is not small) |
| 1 `B_LDS`: A and B global → LDS by LDS-DMA, D-deep LDS ring (one `__shared__` object per stage), one barrier per K tile; scales → registers | 32×128, 64×128 (1×4 waves), 128×128 (2×2 waves), 32×256 (D ≤ 3), 64×256 (D 2) | `-d` 2–4 | M ≈ 128 – 1024 |
| both | | `-p S` cross-block split-K (last-arriving block reduces, ≤ 8) | few tiles |

```bash
make                                    # -> /tmp/opus_mxfp4_small/small.exe   (OUT=..., CLANG=...)
E=/tmp/opus_mxfp4_small/small.exe
$E -a 1 -t 32x128 -d 3 -m 256 -n 4096 -k 4096 -v 1          # correctness (exact GPU reference)
$E -a 1 -t 32x128 -d 3 -m 256 -n 4096 -k 4096 -v 0 -i 100   # timing: hipGraph of 100 launches, median of 5
prof/check.sh $E <HIP gpu> 0:16x32:4:3 1:32x128:1:3 ...      # full correctness matrix per MODE:TILE:KS:D
```

Performance sweep (one command, portable; see `prof/perf_all.py` header):
```bash
AITER_USE_SYSTEM_TRITON=1 python3 prof/perf_all.py --small-exe $E \
    --big-exe <2x2 harness: ../mxfp4_nt_dev/build_var.sh cur "" -> /tmp/opus_mxfp4/cur/build/gemm_a4w4_mxfp4_4wave.exe> \
    --gpu <HIP index> --out perf_out
```
It waits for the GPU to be idle before each shape (amd-smi CU_OCCUPANCY, HIP→smi index matched by PCI bus), re-runs a shape
that got disturbed, times every small-tile config (+ split-K 2/4), every 2×2 tile (persistent / one-tile-per-block / split-K)
and aiter asm (CUDA graph), and writes `results.csv`, `table.txt` and `select.csv` ((M, N, K) → best pipeline + config).

## Interface for integration (kids / dispatch)

**Template**: `gemm_a4w4_mxfp4_nt_small_kernel<opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950<BM, BN, D, KS, BMODE, LAYOUT>>`,
`__launch_bounds__(256, 1)`, block = 256, standard hipcc (no pin, no opus → plain aiter JIT, no `.co`).
- `BMODE`: `OPUS_MXFP4_SMALL_B_DIRECT` (0) or `OPUS_MXFP4_SMALL_B_LDS` (1).  `KS` only for B_DIRECT (2/4), 1 for B_LDS.  `D` = ring depth.
- `LAYOUT` (default `OPUS_MXFP4_SMALL_SHUFFLED` = 0): 0 = gemm_a4w4(bpreshuffle=True) inputs (B shuffle_weight(16,16), scales e8m0_shuffle),
  1 = `OPUS_MXFP4_SMALL_ROWMAJOR` = Triton gemm_afp4wfp4 inputs (B [N, K/2], scales [rows, K/32], no padding).
  The shuffled instances are instruction-identical to the 5-parameter version (00bc51f); row-major uses separate bodies.
- Constraints: K % 256 == 0; N % 16 (shuffled) / N % 8 (row-major); any M; batch ≥ 1.  Shuffled B_LDS: BM ∈ {32, 64, 128}, BN ∈ {128, 256}
  (128 only with BM 128).  Row-major B_LDS additionally BM 16 and BN 64.

**kargs** (`opus_gemm_mxfp4_small_kargs_gfx950`, see the traits header): byte strides for A (`stride_a` = K/2), elements for C (`stride_c` = N);
`stride_sfa / stride_sfb` = bytes per 32-row block (shuffled: 32 × round_up(K/32, 8)) or bytes per row (row-major: K/32);
batch strides in bytes (C in elements).  Shuffled scale batch stride = padded rows × round_up(K/32, 8) (aiter pads rows to 256).

**grid** = (⌈N/BN⌉, ⌈M/BM⌉, batch × S), S = `splitk` (1–8, K/256 ≥ S); non-persistent.

**split-K (S > 1)** needs
- `ws`: fp32 workspace of batch × S × Mp × Np floats (Mp, Np = M, N rounded up to BM, BN);
- `cnt`: int counters, batch × ⌈M/BM⌉ × ⌈N/BN⌉, **zeroed once at allocation**; the reducing (last-arriving) block resets its counter to 0,
  so the buffer is reusable across launches on the same stream without re-zeroing (not safe for concurrent launches sharing it).
With S = 1 neither is touched (may be any valid pointer).
