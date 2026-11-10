# OPUS gfx950 GEMM 对标 DeepGEMM：交接文档

整理日期：2026-11-10
分支：`Fyzyukk/aiter` 的 `fyz/aiter-blockscale-mxfp8-nn/nt/tt`
相关文档：[opus_mxfp8_layouts_summary_20261104.md](opus_mxfp8_layouts_summary_20261104.md)（NN/TT/TN 扩展的流程、优化记录和性能）

---

## 1. 目标与范围

**目标：** 在 gfx950（MI355X）上，用 OPUS 实现对标 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 的 **dense GEMM**，覆盖它支持的数据类型和 layout。做法是写几条通用 pipeline，而不是一个组合写一个 kernel。

**范围内：** dense GEMM，包括 FP8、FP4、BF16 的各种组合，以及 NT/NN/TN/TT 四种 layout。

**暂不做：** MoE 相关的 `m_grouped_*`、`k_grouped_*`、`*_masked`、`mega_moe`；MQA logits；`mega_mhc` / `mega_gate`；`cublaslt_*` 封装（aiter 已有 hipBLASLt）。

---

## 2. 当前状态（这个分支里有什么）

### 2.1 已完成：FP8 blockscale 的 NT/NN/TT/TN

- **对外：** A 的 scale 是 1×128，B 的 scale 是 128×128，类型是 E8M0；输出 BF16。
- **对内：** 四个方向都用 MXFP8 的 scaled MFMA `v_mfma_scale_f32_16x16x128_f8f6f4`。这条指令要求每 32 个 K 一个 scale，所以每个 128 粒度的 scale 复用到 4 个 32 子块。
- **NT：** 现有的主力路径，kid 0–55，含 pin_agpr、b_vgpr、b_vgpr_overlap 等多条 pipeline，已经 tune 过。**要求 B 事先 shuffle**（`shuffle_weight(WQ, layout=(16, 16))`）。
- **NN/TT/TN：** 每个方向只有一个 256×256×128、4 wave 的 kernel，要求 M 是 256 的倍数，K 不超过 16384，B 不需要事先 shuffle。在 8192³ 上分别比 NT 慢 8.87%、7.07%、9.09%，详见第 0 节提到的 summary 文档。

### 2.2 layout 和 stride 的约定

第一个字母描述 A，第二个字母描述 B；A 是 `[M,K]`，B 是 `[N,K]`，输出 Y 是 `[M,N]`，并且必须连续存放。

| 方向 | XQ stride | WQ stride | w_scale `[N/128, K/128]` 的 stride |
|---|---|---|---|
| NT | (K, 1) | (K, 1)，事先 shuffle | 连续 |
| NN | (K, 1) | (1, N) | (1, N/128) |
| TT | (1, M) | (K, 1) | (K/128, 1) |
| TN | (1, M) | (1, N) | (1, N/128) |

x_scale 统一是 `[M, K/128]`，stride 为 (1, M)。

### 2.3 调用流程与代码位置

```text
aiter.ops.opus.opus_gemm(XQ, WQ, Y, kid=..., family="a8w8_blockscale_bpreshuffle", ...)
  -> aiter/ops/opus/dispatch.py:252
       在 kernel 名字里匹配 MXFP8_DIRECTION_PATTERN = r"_mxfp8_(nn|tt|tn)_"，匹配不到就当作 NT
  -> aiter/ops/opus/gemm_op_a8w8.py:392 起
       opus_gemm_a8w8_blockscale_bpreshuffle_{nt,nn,tt,tn}
       -> _launch_a8w8_blockscale_bpreshuffle_gemm (:329)：检查 kid 和方向是否匹配
       -> _launch_a8w8_backend -> pybind -> 生成的 C++ launcher
  -> 生成的 launcher：csrc/opus_gemm/codegen/gen_instances_gfx950.py:3202
       gen_mxscale_bpreshuffle_instance：按方向生成 stride 和 scale 检查，并填入 kargs
  -> CO 镜像：csrc/opus_gemm/gen_co/gfx950/*.co（不在 git 里，见 2.5）
```

| 作用 | 位置 |
|---|---|
| 方向匹配规则、CO 注册表加载 | `csrc/opus_gemm/opus_gemm_common.py:2896`、`:2899` |
| 按 shape 列候选 kid（会排除带方向的 kernel） | `csrc/opus_gemm/opus_gemm_common.py:3210` |
| 默认编译集合（kid 0 加上所有带方向的 CO） | `csrc/opus_gemm/opus_gemm_common.py:3402` |
| CO 注册表 | `csrc/opus_gemm/gen_co/co_kernels_gfx950.json` |
| CO 构建、AGPR pin 校验（带方向的 kernel 走单独分支） | `csrc/opus_gemm/gen_co/build_co.py:322` |
| NN/TT/TN 的 pipeline 和 traits | `csrc/opus_gemm/include/gfx950/opus_gemm_{pipeline,traits}_mxfp8_{nn,tt,tn}_4wave_gfx950.cuh` |
| NT 的主力 pipeline | `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_*_gfx950.cuh` |

**kernel 侧的关键做法：** 不是 K 方向连续的那个操作数，从全局内存按 M/N 方向每次读 16B 写进 LDS，再用转置读 `__builtin_amdgcn_ds_read_tr8_b64_v2i32`（每个 lane 读 8 字节）取出 K-major 的 MFMA 操作数。A/B 的操作数片段固定在 AGPR 里（pin_mask=3）。

### 2.4 ⚠️ 已知问题：NN/TT/TN 目前没有注册

- 在 `co_renumber56` 那一步，NN/TT/TN 的 kid 105/106/107 被移出了 `co_kernels_gfx950.json`。现在的 JSON 只有 kid 0–55（pin_agpr 0–33，prefetch_b_vgpr 34–44，prefetch_b_vgpr_overlap 45–55）。
- 原来的三条配置存在本地 `reports/opus_runtime_recovery_20261105/co_renumber56/GENERIC3.json`。`reports/` 不在 git 里，配置原文见附录 A。
- **结果：** 现在调用 NN/TT/TN 的 API 会因为 kid 没注册而报错。代码路径都还在：API、launcher 代码生成、build_co 的校验、头文件。**下一步的第一件事就是把这三条重新注册。**
- 另外，NN/TT/TN 还没有按方向自动 tune、写入 CSV 和自动查表。

### 2.5 git 里没有的内容

| 内容 | 本机位置 | 原因 |
|---|---|---|
| `reports/`（约 18 万个文件） | 工作区下的 `reports/` | 实验日志；其中有单个超过 100MB 的文件 |
| `.rocprofv3/` | 工作区下的 `.rocprofv3/` | profiler 输出 |
| 所有 `*.co` | `csrc/opus_gemm/gen_co/gfx950/` 等目录 | 二进制产物，需要用 `build_co.py` 重新生成 |
| 2026-11-04 的验证记录 | `/tmp/opus_*_20261104_*` | summary 文档里的链接指向这些位置，换机器后就没有了 |

本机工作区是 `/root/workspace/aiter-opus-upstream-compare-20261016T171653Z`。它的 git 数据在 `/root/workspace/.aiter-upstream-compare.git`，工作区里的 `.git` 文件指向这里，**不能删**。

---

## 3. DeepGEMM 支持范围调研

依据 DeepGEMM main 分支（v2.8.2）的 `deep_gemm/__init__.py`、`tests/generators.py`、`deep_gemm/utils/math.py`、`csrc/apis/layout.hpp`、`impls/sm100_fp8_fp4_gemm_1d1d.cuh`。

### 3.1 dense GEMM 接口

所有 GEMM 的计算都是 `D = C + A @ B`：C 可以不给，`accumulate=True` 时结果累加到 D 上。

| 接口 | A × B | 默认 scale 粒度 | layout | 输出 |
|---|---|---|---|---|
| `fp8_gemm_{nt,nn,tn,tt}` | FP8 × FP8 | 前向：A 1×128，B 128×128；wgrad：A、B 都是 1×128 | 4 种 | BF16；FP32（支持累加） |
| `fp8_fp4_gemm_{nt,nn,tn,tt}` | FP8 × FP4 | A 1×128，B 1×32 | 4 种 | BF16；FP32 |
| `fp4_gemm_nt` | FP4 × FP4 | A、B 都是 1×32 | 只有 NT | BF16 |
| `bf16_gemm_{nt,nn,tn,tt}` | BF16 × BF16 | 无 | 4 种 | BF16；FP32（支持累加） |
| `einsum` / `fp8_einsum` | BF16 / FP8 | 见 3.4 | `bhr,hdr->bhd`、`bhd,hdr->bhr`、`bhd,bhr->hdr`、`bmk,bnk->mn` | BF16；FP32 |
| `tf32_hc_prenorm_gemm` | TF32 | 无 | hyper-connection 专用 | 优先级低 |

### 3.2 scale 的格式和 SM100 上的处理

- **UE8M0：** 只有 8 位指数，表示 2^(e−127)。量化时先算 FP8 的 `amax/448` 或 FP4 的 `amax/6`，再用 `ceil_to_ue8m0` 向上取整到 2 的幂。
- **主机端转换（`transform_sf_into_required_layout`）：**
  - 接收 FP32 scale 时，会先转成 UE8M0，再 4 个打包成一个 int32，用 MN-major 排布（M/N 方向连续）。
  - **128×128 的 B scale 会在 M/N 方向展开成每行一个**（`index_select(-2, arange(mn) // gran_mn)`）。也就是说，DeepGEMM 的 kernel 只认识每行一个的 scale。
  - 已经打包好的 int32 scale 也可以直接传，只接受 (1,32) 和 (1,128)。
  - 4 个 UE8M0 在 int32 里沿哪个方向打包，相关函数的源码这次没有抓到，**实现 SF 转换前需要再读 C++ 源码确认**。
- **kernel 内部：**
  - FP8 相关的组合用 `SM100_MMA_MXF8F6F4`，每条 MMA 的 K=32，每 32 个 K 用一个 scale。
  - 粒度 128 时，同一个 scale 连续用 4 次（`kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4`）。**这和我们“复用到 4 个 32 子块”是同一种做法。**
  - A、B 的 K 方向粒度分别都可以是 32 或 128（`kGranKA/kGranKB ∈ {32, 128}`）。所以 **FP8 用 1×32 的真正 MXFP8 也是支持的**。
  - FP4×FP4 走专用的 MXF4 MMA（每条 K=64），**只支持 K-major**，两侧都必须是粒度 32。

### 3.3 FP4 的格式：MXFP4，不是 NVFP4

| | MXFP4（DeepGEMM 的 FP4 kernel 用的） | NVFP4（只出现在 `cublaslt_nvfp4_gemm_nt`） |
|---|---|---|
| 数据 | E2M1，值域 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 加正负号 | E2M1 |
| 打包 | 1 字节存 2 个，偶数下标的放低 4 位 | 同左 |
| 多少个元素共用一个 scale | 32 | 16 |
| scale 类型 | UE8M0 | FP8 E4M3，另有一个 FP32 的全局 scale |
| gfx950 硬件支持 | **原生支持**（MFMA scale 指令） | 不支持 |

### 3.4 einsum 的测试用例

- `einsum`（BF16）：`bmk,bnk->mn`（要在 b 维度上归约）、`bhr,hdr->bhd`、`bhd,hdr->bhr`、`bhd,bhr->hdr`（FP32 累加）。另外，输出可能是一个更大 tensor 的切片，即非连续写出。
- `fp8_einsum`：
  - `bhr,hdr->bhd`：x 按 token 量化，y 按 128 分块量化；输出 BF16，SM100 上也可以输出 FP8，此时 gran_k=32。
  - `bhd,hdr->bhr`：同上的量化方式，输出 BF16。
  - `bhd,bhr->hdr`：按 channel 量化，recipe=(1,1,128)，用 FP32 累加。

### 3.5 测试 shape（`enumerate_normal`）

- M：前向取 {1, 128, 4096}，反向取 {4096}
- (N, K)，输出 BF16：(2112,7168)、(576,7168)、(24576,1536)、(32768,512)、(7168,16384)、(4096,7168)、(7168,2048)
- (N, K)，输出 FP32（仅 BF16 GEMM）：(256,7168)、(129280,7168)
- 精度阈值：FP4×FP4 为 0.02，FP8×FP4 为 0.01，其余为 0.001（`calc_diff`）

可以看出，**小 M（1、128）、N 不是 256 的倍数（576、2112、129280）、FP32 输出**都必须支持。

---

## 4. 路线规划

| 优先级 | 路线 | A | B | layout | 内部指令 | 状态 |
|---|---|---|---|---|---|---|
| — | FP8 blockscale | FP8，1×128 | FP8，128×128 | NT/NN/TT/TN | MFMA scale 指令，一个 scale 用 4 次 | ✅ 已完成（NN/TT/TN 需重新注册） |
| **1** | MXFP4 × MXFP4 | FP4，1×32 | FP4，1×32 | NT | 同一条指令，A、B 都按 FP4 | 待做 |
| **2** | MXFP8 × MXFP8（FP8 扩展到 1×32） | FP8，1×32 | FP8，1×32 | NT/NN/TT/TN | 同一条指令，每 32 个 K 用各自的 scale | 待做 |
| **3** | MXFP8 × MXFP4 | FP8，**默认 1×128，也支持 1×32** | FP4，1×32 | NT/NN/TT/TN | 同一条指令，A 按 FP8、B 按 FP4（cbsz/blgp） | 待做 |
| **4** | BF16 | BF16 | BF16 | NT/NN/TT/TN | `v_mfma_f32_16x16x32_bf16` | 待做 |

MXFP8 × MXFP4 中 FP8 那一侧：DeepGEMM 的标准测试配置是 `(128, 32, False, True)`，即 A 为 1×128，这也是它在 MoE 推理里实际用的组合；但 kernel 也允许 A 用 1×32。我们两种都支持，默认 1×128。

---

## 5. pipeline 设计

### 5.1 P1：scaled-MFMA 通用 pipeline（覆盖路线 0–3）

A、B 各自用以下参数描述，作为 traits 传入：

| 参数 | 取值 | 决定什么 |
|---|---|---|
| `Dtype` | fp8 / fp4 | MFMA 的 cbsz/blgp、每行字节数、转置读用哪条指令 |
| `Major` | K / MN | K-major：`ds_read_b128`；MN-major：FP8 用 `ds_read_tr8_b64`，FP4 用 `ds_read_tr4_b64` |
| `SfGranK` | 32 / 128 | 128：一个 scale 用 4 次（现有逻辑）；32：用 op_sel 从打包的 int32 里选第 0–3 个字节 |
| `SfGranMN` | 1 / 128 | 128：B 的 blockscale，scale 读取量小；1：每行一个，用于 FP4、MXFP8 和 DeepGEMM 展开后的格式 |
| `BPreshuffle` | 是 / 否 | 现有 NT 的高性能路径要求 B 事先 shuffle；为了和 DeepGEMM 接口一致，需要支持不 shuffle |

公共参数：`Out ∈ {bf16, fp32}`；`Epilogue ∈ {直接写出, C+=}`。

**代码整合：** 现有 NN/TT/TN 三份 pipeline 各约 900 行，大量重复。把它们合并成一份，A/B 的 load policy 和 SF policy 抽成 traits，NT 也作为其中一种组合。

**tile 分三档：**
- 大：256×256×128，4 wave，从现有 kernel 迁过来。
- 中：128×128、128×256。
- 小 M：16/32/64×N，配合 split-K，参考 b_vgpr / m_repeat 的做法。

**对齐：** M 用 pad 或越界检查处理；N 支持对齐到 128 的情况，必要时加 N 方向的越界检查。

**FP4 路线最主要的新工作：** B 的 scale 从“每 128 行共用一个”变成“每行一个”，scale 的读取量约为原来的 128 倍。所以 SFB 在 LDS 里的缓存区和预取方式要重新设计。

### 5.2 P2：BF16 通用 pipeline（路线 4）

- 在 a16w16 现有 pipeline 的基础上，加入 MN-major 的 load policy，转置读用 `ds_read_tr16_b64`。现在 a16w16 的 launcher 要求两个输入都是 K 方向连续，需要放开。
- 加 FP32 输出和 C 累加的 epilogue。
- tile 分档同 P1，尽量复用 P1 的 load policy 和 epilogue 组件。

### 5.3 P3：batched / einsum（底层用 P1/P2）

- `bhr,hdr->bhd`、`bhd,hdr->bhr`：本质是以 h 为 batch 的 BMM。需要支持任意 batch stride、row stride，以及非连续的输出写出。
- `bhd,bhr->hdr`、`bmk,bnk->mn`：需要在 batch 维度上归约。把 K 循环改成 batch 和 K 两层（相当于 K' = b×k），配合 FP32 累加的 epilogue。

### 5.4 公共部分

- **SF 转换工具（暂不做，排在 kernel 之后）：** 类似 DeepGEMM 的 `transform_sf_into_required_layout`。支持把 FP32 scale 转成 UE8M0，把 128×128 展开成每行一个，打包，转成 MN-major。kernel 本身只认 E8M0。
- **兼容接口：** 提供和 DeepGEMM 同名的 Python 接口，比如 `fp8_gemm_nt(a=(x, sfa), b=(w, sfb), d, c=None, recipe=...)`。内部按（方向、dtype、recipe、输出类型、M/N/K）查 tune 好的 CSV 得到 kid，再走现有的 exact-kid 路径。
- **kid 分配：** 按 pipeline 和方向划分 kid 号段。现有方向匹配规则 `_mxfp8_(nn|tt|tn)_` 要扩展成能同时表示 dtype 和方向。
- **tuner：** 按方向、按 dtype 组合分开 tune，CSV 保存和自动查表复用现有流程。

---

## 6. 分阶段计划

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| 0（准备） | 把 NN/TT/TN 的 kid 重新注册（配置见附录 A），重新构建这三个 CO；按 3.5 的 shape 写测试框架，配 torch 参考结果和 benchmark | 现有四个方向都能调用；测试框架能跑通 |
| **1** | **MXFP4 × MXFP4（NT）kernel**：先把 kernel 做出来，测试里直接构造 E2M1 数据和 UE8M0 scale，不单独做量化和 SF 转换工具。用 P1 通用 pipeline 的框架先做 FP4 的 A/B load policy（两侧都是 K-major，只需普通 `ds_read`，暂不需要 `ds_read_tr4`）；SF 实现每 32 个 K 一个、每行一个（`SfGranK=32`、`SfGranMN=1`），重新设计 SFA/SFB 在 LDS 里的缓存区和预取；大 tile、中 tile、小 M tile；tune 并自动查表 | `calc_diff < 0.02`；性能目标待定 |
| **2** | **FP8 扩展到 1×32（MXFP8 × MXFP8）**：把阶段 1 的 `SfGranK=32`、`SfGranMN=1` 的 SF 逻辑用到 FP8 上；把现有 NN/TT/TN 三份 pipeline 合并进 P1 的框架，NT 也作为其中一种组合；四个方向都要支持 | `calc_diff < 1e-3`；现有 1×128 / 128×128 的结果和性能不能退化 |
| **3** | **MXFP8 × MXFP4**：A 用 FP8（默认 1×128，也支持 1×32），B 用 FP4（1×32）；四个方向；MN-major 的 FP4 要加 `ds_read_tr4_b64` 转置读 | `calc_diff < 0.01` |
| **4** | **BF16**：P2，四个方向，加 FP32 输出和 C 累加 | `calc_diff < 1e-3`，并与 hipBLASLt 对比 |
| 后续 | P3：einsum / fp8_einsum；`tf32_hc_prenorm_gemm`；MoE grouped GEMM | — |

训练场景相关的功能（不 shuffle B 的 NT、FP32 输出、C 累加、wgrad recipe）不单独列阶段，等第 7 节的问题确认后，再插入对应阶段。

**为什么 FP4×FP4 放第一位也能顺利推进：**
- DeepGEMM 的 FP4×FP4 只支持 NT，两个操作数都是 K 方向连续。所以第一阶段不需要处理转置读，可以专心做 FP4 的数据读入和每行一个的 scale，这正是后面几个阶段都要复用的核心部分。
- 阶段 1 做出来的“每 32 个 K、每行一个 scale”的 SF 逻辑，到阶段 2 直接给 FP8 用，就完成了 MXFP8 1×32 的扩展。
- 阶段 3 是把前两个阶段的成果组合起来（A 走 FP8 的读入，B 走 FP4 的读入），只需要额外补上 MN-major FP4 的转置读。

---

## 7. 待确认事项

1. **性能目标：** 各方向是按“比 NT 慢不超过 X%”来定，还是和 hipBLASLt 对比？
2. **训练场景要不要做：** 即 FP32 输出、C 累加、wgrad recipe。如果只做推理，阶段 1 的工作量能减少很多。
3. **不 shuffle B 的 NT 要不要做：** 要和 DeepGEMM 接口完全一致就必须做；只追求性能的话，可以继续要求 B 事先 shuffle。
4. **DeepGEMM 打包 UE8M0 的方向：** 需要读 C++ 源码确认（见 3.2）。

已确定：优先级为 MXFP4×MXFP4 → FP8 扩展到 1×32 → MXFP8×MXFP4 → BF16（见第 6 节）。

---

## 附录 A：NN/TT/TN 原来的注册配置

原本在 `co_kernels_gfx950.json` 的 `gfx950` 列表里，在 `co_renumber56` 时被移除：

```json
{"pipeline": "mxfp8_nn", "tile": [256, 256, 128], "wave_layout": [2, 2], "variant": "", "device_flags": [], "wide_output": false, "traits": {"name": "opus_gemm_mxfp8_nn_4wave_traits"}, "body_func": "opus_gemm_mxfp8_nn_4wave", "body_args": ["CoTraits"], "pipeline_header": "opus_gemm_pipeline_mxfp8_nn_4wave_gfx950.cuh", "pin_variant": true, "pin_mask": 3, "pad_m": false, "m_align": 256, "max_k": 16384, "kid": 105}
{"pipeline": "mxfp8_tt", "tile": [256, 256, 128], "wave_layout": [2, 2], "variant": "", "device_flags": [], "wide_output": false, "traits": {"name": "opus_gemm_mxfp8_tt_4wave_traits"}, "body_func": "opus_gemm_mxfp8_tt_4wave", "body_args": ["CoTraits"], "pipeline_header": "opus_gemm_pipeline_mxfp8_tt_4wave_gfx950.cuh", "pin_variant": true, "pin_mask": 3, "pad_m": false, "m_align": 256, "max_k": 16384, "kid": 106}
{"pipeline": "mxfp8_tn", "tile": [256, 256, 128], "wave_layout": [2, 2], "variant": "", "device_flags": [], "wide_output": false, "traits": {"name": "opus_gemm_mxfp8_tn_4wave_traits"}, "body_func": "opus_gemm_mxfp8_tn_4wave", "body_args": ["CoTraits"], "pipeline_header": "opus_gemm_pipeline_mxfp8_tn_4wave_gfx950.cuh", "pin_variant": true, "pin_mask": 3, "pad_m": false, "m_align": 256, "max_k": 16384, "kid": 107}
```

重新注册时要注意：现在的 kid 已经重新编号成连续的 0–55，加回去之前要先确定新的 kid 号段，避免和现有号段冲突。

## 附录 B：参考来源

- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM
  - `deep_gemm/__init__.py`：接口列表
  - `tests/generators.py`：量化配置、layout 组合、测试 shape
  - `deep_gemm/utils/math.py`：FP8/FP4 量化、`ceil_to_ue8m0`
  - `csrc/apis/layout.hpp`：`transform_sf_into_required_layout`
  - `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`：MMA 类型、scale 粒度的处理
