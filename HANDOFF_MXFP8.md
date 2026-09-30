# MXFP8 B-preshuffle 优化交接

## 2026-09-29：最新分支迁移与重新 tune

分支为 `Fyzyukk/aiter:aiter-opus-mxfp8-bpreshuffle`。本节是最新测试入口；
下方按日期保留的记录描述各自当时的源码、候选编号和测试状态。

当前分支包含此前的大 M pipeline 优化，以及通用小 M 候选 **9040–9046**。
每个小 M ID 固定一个 tile 和 pipeline，支持 M=1–512，使用运行时 K。
本次最后一轮改动是 9020/9022 的 M 对齐由 64 放宽到 16、9043 跨 M 片段
配对写回，以及新增 9046（64×128 small-LDS、4 wave）。
这些最后改动已通过 CPU 覆盖/索引/边界检查和离线编译，GPU 正确性及性能待新机验证。
修改前的正式通用版基线是 234/290 胜出、几何平均加速 1.159373×；
273/290 是更早的按 shape 分发版本，不能作为当前代码的成绩。

- [本轮源码与验证状态](reports/opus_mle512_general_opt_20260929/WORK_STATE.md)
- [CPU 检查及源码 SHA256](reports/opus_mle512_general_opt_20260929/cpu_ready/validation.json)
- [修改前通用版实测摘要](reports/opus_mle512_generic_tune_20260929/full290_r3/summary.md)
- [修改前逐 shape 比较](reports/opus_mle512_generic_tune_20260929/full290_r3/comparison.csv)

### 到新机后看哪些文件

| 文件 | 用途 |
|---|---|
| `csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py` | 正式 tune 和保存结果回放入口 |
| `csrc/opus_gemm/opus_gemm_common.py` | 15 个公开候选的注册及 shape 支持范围 |
| `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | 生成入口、边界检查和 kernel launch |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh` | 9040–9046 的固定 tile 和 pipeline 参数 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh` | 9040–9042 的寄存器预取 pipeline |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh` | 9043–9046 的 LDS pipeline |
| `csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh` | 小 M MFMA16 配对写回 |

### 获取分支和准备环境

在已有 ROCm/PyTorch/Triton 开发环境中执行。目标 GPU 须为 gfx950，PyTorch 须提供
原生 `torch.float8_e8m0fnu`。初始化仓库记录的 CK submodule 版本。

```bash
git clone --branch aiter-opus-mxfp8-bpreshuffle --single-branch \
  git@github.com:Fyzyukk/aiter.git aiter-opus-mxfp8-bpreshuffle
cd aiter-opus-mxfp8-bpreshuffle
git submodule update --init --recursive
git rev-parse HEAD

BUILD_TARGET=rocm AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  python -m pip install -e . --no-build-isolation

export ROCR_VISIBLE_DEVICES=0
export OPUS_HIP_CLANG_PATH=/absolute/path/to/llvm-pin-build/bin
export AITER_JIT_DIR="$(mktemp -d /tmp/aiter-opus-mxfp8-20260929.XXXXXX)"
export OPUS_RETUNE_DIR="$PWD/reports/opus_remote_retune_20260929"
mkdir -p "$OPUS_RETUNE_DIR"
set -o pipefail
```

`OPUS_HIP_CLANG_PATH` 必须替换成目标机的实际目录。9000 和共享 helper 仍要求
支持 `clang::amdgpu_pin_agpr` 的编译器；已用版本是
[`yuyzhang512/llvm-project`](https://github.com/yuyzhang512/llvm-project)
的 `49c41889681640665400cb01c9fbb4c0a024cde4`。下方历史“新服务器准备”一节
保留了该工具链的构建命令。专用 tuner 只在 OPUS 编译期间切换此编译器，
CK/CKTile/ASM 沿用环境的 ROCm 编译器。新的 `AITER_JIT_DIR` 用于在目标机重新构建，
首次运行包含 JIT 构建时间。

### 全量重新 tune：745 个 gfx950 shape

在仓库根目录执行正式模块入口：

```bash
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  -i aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
  -o "$OPUS_RETUNE_DIR/full745_tuned.csv" \
  -o2 "$OPUS_RETUNE_DIR/full745_profile.csv" \
  --libtype all --splitK --shape_grouped --mp 1 \
  --warmup 5 --iters 51 --all \
  2>&1 | tee "$OPUS_RETUNE_DIR/full745.log"
```

原始 CSV 有 1042 行，tuner 按当前 `gfx/cu_num` 筛出 gfx950/256 CU 的 745 个 shape，
忽略输入中的旧 `libtype/kernelId/us`。这里省略 `--opus-kids`，自动枚举当前全部
26 个默认候选：9000、9010、9020–9024、9030、9040–9047、9049、9051–9055、9060–9063，
并按各自支持范围筛选。
`--libtype all` 比较 OPUS、CK、CKTile、ASM；`--all` 强制重新测量已有 shape。
每个候选都执行数值比较，`-o` 保存最快有效选择，`-o2` 保存候选 profile，日志保存失败详情。
OPUS 使用原生随机 E8M0 scale，其他后端使用原 tuner 的随机 FP32 scale，分别对照各自 reference。

### 使用最新完整调优表

最新的 [745 行调优表](aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv)
采用基线的 13 列格式，每个 shape 保留最快有效候选。也可以将此文件传给上方命令的 `-i`，
tuner 会读取 shape 并重新枚举候选。

只看 OPUS 时可将 `--libtype all` 改为 `--libtype opus`。全量 745 项包含 290 个 M≤512 shape。
上述 CSV 的 `cu_num` 为 256；其他 CU 规格的 gfx950
可提供仅含 `M,N,K` 的 CSV，tuner 会填入当前 GPU 的 `gfx/cu_num`。

四卡时将 `ROCR_VISIBLE_DEVICES` 改成实际空闲的四张卡，例如 `0,1,2,3`，并用 `--mp 4`；
八卡对应 `--mp 8`。保留 `--shape_grouped`，让同一 shape 的候选在同一张卡上比较。

### 回放保存的选择

```bash
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  --run_config "$OPUS_RETUNE_DIR/full745_tuned.csv" \
  --mp 1 --warmup 5 --iters 51 \
  2>&1 | tee "$OPUS_RETUNE_DIR/full745_replay.log"
```

回放重新做数值检查并输出耗时，不搜索新候选。
需要与历史三轮中位数比较时，分别用不同的输出文件名重新 tune 三轮，再按候选汇总中位数。
保留每轮 `tuned.csv`、`profile.csv`、日志，以及当前 commit、GPU 和编译器版本。

## 2026-09-28：9021/9022/9023/9024/9030 全面整理与复用

五个候选的 pipeline/traits 已按用户要求整理：能直接复用的 helper 直接复用，
不同的布局采用 9000 的 shape/dim/unfold 结构。9021/9022 的 RA 复用 9000；
9023/9024 的 GA 在原 lane XOR 后复用 9000；9030 的 RA 复用 9020。
9021/9022/9030 的原始字节 Scale A reader 统一复用 9020 的参数化 helper。
所有候选保留各自独立的 kernel body 和 traits，未修改本轮冻结的 9000/9020。

五个 gmem 基址均保留 batch stride 和 tile row/col；matrix/scale layout 前置，
四个 scale offset 与矩阵 offset 集中声明，SFA/SFB 分为两个 panel helper。
traits 只放几何、存储、派生常量和断言；寄存器、C layout/offset、K tile/M-N repeat
及下半段 helper 命名统一，说明注释用单行。9021/9022 的 A/C pin、9023/9024 的
XOR A 与预打包 scale、9030 的 byte SFB/64-bit C/tile-local bounds 均保持。

9023/9024 的 scale layout 表达组内位置，K group/panel 推进放到动态 offset；
避免将完整的 group 字节地址跨主循环缓存。初版 VGPR 曾增至 233/151，未采用；
最终 VGPR 为 138/112（整理前 144/114）。9021/9022/9030 最终 VGPR 为
200/244/205（整理前 200/240/207）。五个最终 device TU 均通过 CPU 编译，
ABI/LDS 保持，private segment 与 VGPR/SGPR spill 均为 0。
9021/9022 部分 paired LDS writes 被拆成 single writes，不能称为机器码完全相同。
本轮没有 GPU 数值或性能验证，也没有重新 tune；此前 305-shape 实测不代表本次整理后的实测结果。

本次提交准备同时保留当前 8 个候选的注册、独立头文件命名、旧文件删除及必要报告。
相对与上游的共同基线 `b3d0cf4e`，最终非 reports 净变更 78 个文件，其中 50 个属于
保留实验包；共享 generic tuner 没有净改动。当前上游已前进，本轮没有合并上游。

- [五个候选的整理、源码和 CPU 证据](reports/opus_9021_9030_style_20260928/RESULTS.md)
- [分支整体 OPUS tune 集成文件清单](reports/opus_9021_9030_style_20260928/INTEGRATION_FILES.md)
- [最终 host 地址/guard 检查](reports/opus_9021_9030_style_20260928/host_layout_final/validation.json)

## 当前代码约定（2026-09-28，用户确认）

后续 layout helper 按 9000 的风格显式接收 `lane_id` 和所需的
`wave_id_m` / `wave_id_n`，通过 shape/dim/unfold 表达线程分工，
不要为了简化参数合成 `thread_id` 再拆解。配置常量、派生常量及对应
`static_assert` 集中在 traits；pipeline 直接引用 `T::VEC_SCALE_A` 等常量。
tile 的 `row` / `col` 起点放在 `g_*` 基址，layout 只表达 tile 内偏移。
全局视图按 9000 保留 `batch_id = block_id_z()` 及每个指针对应的
`batch_id * stride_*_batch`，即使现有入口只允许 batch=1，也不省略这一项。
9020 的 A/B/C 寄存器用默认 `typename decltype(mma)::vtype_*` 声明，
累加器显式 `clear(v_c)`；按实际流水线需要决定份数，不为外观照搬多份 tile。

9020 的 `gsfa` / `ssfa` 已采用上述接口，shape 为 `(2,4,16,3,4,16)`，
自由维仍是三次 M pass 与 16 字节向量。`rsfa` 保留原有 lane/wave_m 接口。
随后已将 SFA 的 row、SFB 的 col 移入各自 gmem 基址，SFB 三个 layout
统一为 shape/dim/unfold 和 lane/wave 接口；RA、swizzle、输出及预取固定
常量迁入 traits，C layout 前置到 MMA 初始化后。M 尾判断、Scale A 后打包、
Scale B 两字节打包及流水线顺序保持。
CPU 地址核对覆盖全部 512 个线程的 14848 项有效地址，数量和顺序一致。
完整 kernel 编译通过：VGPR204、LDS143360、private/spill0；SGPR53→52，
指令字节7812→7808，内存指令计数与 ABI 一致。本次没有运行 GPU 或重新 tune。
详见[风格整理及验证](reports/opus_9020_style_consistency_20260928/RESULTS.md)。

9020 的 B gmem 构造随后统一为 9000 的 typed-pointer 加 tile 偏移写法，
删除 `b_address/b_lo/b_hi` 及源码中的两次指针 `readfirstlane`。
当前编译器仍生成标量描述符，没有 descriptor waterfall；内存指令计数一致，
VGPR204、LDS143360、private/spill0，SGPR52→51，指令字节7808→7812。
入口限制 `N*K <= INT32_MAX` 且 `stride_b=K`，支持范围内的乘法保持等价。
A/C 的 M 尾资源长度保留。仅做 CPU 编译及 ISA 核对，未运行 GPU 或重新 tune；
详见[B gmem 整理记录](reports/opus_9020_gmem_style_20260928/RESULTS.md)。

随后按用户要求，9020 的 A/B/C/SFA/SFB 五个 gmem 基址全部补齐 batch 偏移，
保留原有 A/C 资源长度。layout、traits、入口和其他候选未改；现有 grid.z=1，
五个新增项实际均为零。一次 CPU 编译通过，内存指令计数与 ABI 一致，
没有 descriptor waterfall；VGPR204、LDS143360、private/spill0，SGPR51→54，
指令字节7812→7896。未运行 GPU 或 tune，不以地址等价宣称性能相同。
详见[五个 gmem 的 batch 偏移整理记录](reports/opus_9020_batch_gmem_style_20260928/RESULTS.md)。

9020 随后将手工 fragment 别名、array 容器与 a_chunks/b_chunks 指针改为默认
MMA 类型的一份 v_a/v_b/v_c。逐片加载、MFMA 和最终 BF16 转换通过静态
slice/set_slice 访问相同元素，保留 swap_ab、scale selector 和原调度。
9020 本来没有 pin AGPR；一次 CPU 编译后仍为 AGPR0、private/spill0，
VGPR204→203、SGPR54、LDS143360、指令字节7896。内存/MFMA/wait/barrier
计数与 ABI 一致，但机器码不同；未运行 GPU，不据此宣称性能提升。
详见[默认 MMA 寄存器类型整理记录](reports/opus_9020_mma_vtypes_20260928/RESULTS.md)。

9020 下半段随后按 9000 风格整理：C 命名为 p_coord_c/u_gc/gc_offsets，
c_offset 集中表达输出合作读取时的 LDS 行列地址；四个 gsfa/gsfb/ssfa/ssfb
offset 与 A/B offset 一起声明，并用于实际读写。SFA/SFB 分成两个 panel
helper，仍在原位置按序仅调用一次(0)，没有新增 refill。寄存器读片、K tile、
M/N repeat 和输出 helper 命名统一，cached offsets 用 layout_to_offsets 提取。
原地址、scale 打包与 MFMA/LDS 交错保持；显式 wait/barrier 顺序和参数一致。
一次 CPU 编译后完整 metadata 一致：AGPR0、VGPR203、SGPR54、LDS143360、
private/spill0、指令字节7896；内存/MFMA/wait/barrier 数量相同但机器码不同。
未运行 GPU 或 tune，详见[下半段 offset 和 helper 整理记录](reports/opus_9020_pipeline_offsets_style_20260928/RESULTS.md)。

此前已完成当前 9020 的 Scale A 前/后打包六轮对照，生产仍保留 LDS
原始字节、读后打包方式，详见
[对照结果](reports/opus_9020_sfa_pack_ab_20260928/RESULTS.md)。

## 2026-09-28 最新：9020 RA按9000的shape/dim形式展开

仅将9020的`make_layout_ra_scale`改为显式`ra_block_shape`、`ra_block_dim`，
通过`unfold_x_stride`和`unfold_p_coord`构造布局。八wave坐标映射保留，
源码展开核对仍为shape(3,8,8,2,4,16)、stride(8448,1056,128,64,16,1)，
自由维仍为repeat/chunk/byte；函数之外的pipeline正文未改。
本轮为等价表达整理，只做源码核对，没有重复编译或运行GPU。

- [源码核对](reports/opus_9020_ra_shape_dim_20260928/source_check.json)
- [本轮差异](reports/opus_9020_ra_shape_dim_20260928/change.diff)

## 2026-09-28：9020恢复非XOR A加载，Scale A改为常规layout

按用户要求，仅修改9020 pipeline。A的u_ga/u_sa直接复用9000已有helper，
恢复`async_load<16>(g_a, ..., u_ga, u_sa, ...)`；八wave reader用纯
shape/stride/coord表达原来的协作LDS映射。32字节LDS段间padding恢复使用，
A_STAGE仍为25344，无需改变traits或扩大LDS分配。

Scale A的global/LDS复制共用`make_layout_sfa_panel`，仅group stride不同。
每个K128 group由4线程在3次64行pass中搬运192行，覆盖原[group][192行]字节panel；
读端仍按三个M repeat读取并打包。全部layout均不再手工改写cached offsets。
这次SFA的线程搬运分工实际改变，保留原存储格式、边界判断及0x7f尾行填充。
5个g_*和12个u_*仍前置；B/SFB、MFMA、主循环和输出代码保持。

9020确实支持M尾块，例如M1472最后一个192行tile只有128行有效；
已有g_a有效范围以及SFA/C尾行判断保留。LDS的32字节padding不等同于M尾行处理，
没有新增PAD_M分支、输入补齐或额外候选。

**一次实际device TU CPU编译通过，VGPR208→204，SGPR52、LDS143360、private0、spill0，
名称、registry、生成TU/impl和96字节ABI保持；机器码7880→7656 bytes。**
另一次host-only检查编译及一次CPU运行通过：A覆盖18个M尾/K tile/stage场景，
SFA覆盖9个M尾/有效K group场景，搬运唯一覆盖、LDS padding、尾行填充及读端标签均正确。
host检查使用实际Opus布局和当前traits，不调用HIP runtime API。
无GPU数值/性能测试或重新tune；不能由寄存器减少推断实测更快。其余36个受保护生产文件未改。

- [实际device TU编译与资源](reports/opus_9020_restore_a_layout_20260928/validation.json)
- [CPU布局覆盖检查](reports/opus_9020_restore_a_layout_20260928/host_layout/validation.json)
- [源码范围核对](reports/opus_9020_restore_a_layout_20260928/source_check.json)
- [本轮差异](reports/opus_9020_restore_a_layout_20260928/change.diff)

## 2026-09-28：9020恢复完整的前置global view和layout声明

按用户要求，只整理9020 pipeline：在文件前部定义9个候选专用layout factory，
kernel内先集中声明g_a/g_b/g_c/g_sfa/g_sfb，再列出A/B/SFA/SFB共12个u_*布局，
随后才定义LDS、寄存器和加载helper。所有新layout均为实际Opus cached layout并被使用。
B的原64-bit地址及readfirstlane前置后构造g_b；A XOR映射和scale映射移入各自factory。
A三次预取、六个fragment地址、SFA边界/填充、SFB两个byte打包及MFMA selector保留。
SFB LDS读写继续使用scalar-offset overload。Prologue至输出段源码逐字节不变。

两次9020实际device TU CPU编译均成功，第二次恢复了上述scalar-offset调用形式。
最终完整metadata与基线一致：VGPR208、SGPR52、LDS143360、private0、spill0；
公开名称、registry、生成TU/impl及ABI不变。最终机器码7816→7880 bytes，
因此**并非ISA逐字节一致，也没有新的GPU数值/性能验证**。
其余36个受保护生产文件未改，保留当前用户对traits的编辑；没有重新tune。

- [最终CPU编译对照](reports/opus_9020_frontmatter_20260928/attempt2/validation.json)
- [源码范围核对](reports/opus_9020_frontmatter_20260928/source_check.json)
- [本轮差异](reports/opus_9020_frontmatter_20260928/change.diff)

## 2026-09-28：将A的LDS地址计算放回pipeline

按用户纠正，9020/9030的traits只保留常量和静态断言。
9020当前pipeline在`load_a`中直接计算XOR A布局地址，traits内旧`a_lds_offset`
没有调用，已删除；9020 pipeline未改。9030仍使用该映射，已将完整公式移到
pipeline的地址helper区、`sb_offset`之后，作为无捕获局部lambda，唯一调用同步改名。
公式仅为常量引用加`T::`，stage偏移仍保留在调用点。

**仅对9030实际device TU进行一次CPU编译，7884字节机器指令和完整metadata
（含全部名称及ABI字段）完全一致；registry、公开名称、生成TU/impl及host/device符号不变。**
两份traits各43个常量及全部断言保留，其余34个受保护源码未改。
无GPU执行或重新tune；下方保留历史位置说明。

- [CPU编译与机器码对照](reports/opus_9030_lds_helper_20260928/validation.json)
- [源码范围核对](reports/opus_9030_lds_helper_20260928/source_check.json)
- [本轮差异](reports/opus_9030_lds_helper_20260928/change.diff)

## 2026-09-28：9020/9030 traits按9000的声明结构整理

两个traits均按9000的顺序分组：线程/wave → B/T/W → HALF及几何断言 →
E/VEC/GROUP → runtime-K扩展参数 → LDS → scale存储及容量断言。
每个常量独立一行，八wave专用`a_lds_offset`完整放在末尾。
逐项源码核对确认两份各43个常量的名称和初始化表达式、全部断言均保留，
地址映射函数正文逐字节不变；其它35个受保护源码未改。
本轮仅整理声明与空行，采用源码对照，没有重复编译或运行GPU。

- [源码核对](reports/opus_traits_9000_style_20260928/source_check.json)
- [本轮差异](reports/opus_traits_9000_style_20260928/change.diff)

## 2026-09-28：9020/9030 traits展开为各自完整定义

按用户纠正，9020的traits不能只列一个继承通用模板的入口。
已将9020及同样结构的9030展开为非模板、无继承的完整struct，保留原类型名。
各自文件包含全部几何参数、LDS/scale大小、向量配置、`a_lds_offset`正文及静态断言。
9020仍为packed B-scale：SFB512、B_SCALE_PACKS=1、LDS143360；
9030仍为byte B-scale：SFB256、B_SCALE_PACKS=2、LDS143104。

旧`opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<Waves, ScalePanel>`
来自此前实验族，保留供retained源码使用；当前9020/9030不再包含或继承它。
基础traits头仅继续提供公共宏和kargs等定义。

**两个实际device TU各进行一次CPU编译，机器指令7816/7884 bytes逐字节一致，
完整metadata（包括名称）、registry、公开名称、生成TU/impl及host实现均一致。**
只有这两个生产traits文件改变，其余35个受保护源码哈希不变，包含全部pipeline及9000/9010。
无GPU、重新tune、提交或推送。

- [CPU编译与机器码对照](reports/opus_9020_9030_traits_20260928/validation.json)
- [实际源码差异](reports/opus_9020_9030_traits_20260928/change.diff)
- [源码范围核对](reports/opus_9020_9030_traits_20260928/source_check.json)

## 2026-09-28：消除9010 spill，保留两个独立候选

只改9010 padded-M pipeline的`load_sfa_panel`：将已有的
`sfa_panel_raw[pass] = {};`移到`first_k_group < scale_k_groups`判断之前。
无效K pass不会被publish，提前定义其临时值可以切断旧scale数据跨主循环的活跃区间；
有效K/M数据的加载和发布不变。M边界、输出row guard、MFMA次序、wait/barrier均保留。

**生产实际device TU编译后：VGPR spill 14→0，private segment 44→0 bytes，
scratch load/store静态指令17→0；SGPR spill仍为0，LDS仍为152064 bytes。**
机器指令24312→23872 bytes；SGPR 76→80，metadata VGPR总数512→497、AGPR256→241。
旧scratch指令位于主循环入口/出口及尾声衔接处，热主循环内本来就没有scratch读写。

共5次CPU device TU编译（基线、三个局部实验、最终生产）；最终产物与成功实验逐字节一致。
另外两个实验未整合：改scale受检offset仍有spill；删除输出row guard仍有spill，且大N下
无效行的32-bit字节offset可能回绕，不能只依赖buffer bound，故完整保留原guard。
36个其它受保护源码哈希不变，9010生成的impl/device TU、公开名称、shape契约及96-byte ABI不变。
无GPU执行或重新tune，尚无新的GPU数值/性能结果；未提交或推送。

- [编译资源、源码语义核对与保护哈希](reports/opus_9010_spill_20260928/summary.json)
- [实际代码差异](reports/opus_9010_spill_20260928/change.diff)
- [原spill位置诊断](reports/opus_9010_spill_20260928/diagnosis/summary.md)

## 2026-09-28：9000无padding，9010独立承担padded-M流程

按用户纠正，将此前混在9000中的五处`T::PAD_M`分支完全分离。
9000 pipeline只保留无padding流程，traits删除`PAD_M=false`。
9010 pipeline由wrapper改为完整独立主体，函数为
`gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel`，codegen使用该函数。
A/C有效范围、受检矩阵预取、A-scale尾行零填充、Prologue边界加载、输出行检查均在9010内。
9010仍继承同一几何traits，并从9000头文件复用layout/AGPR helpers；不调用9000 kernel。
当前共**8个pipeline入口、8个独立计算模板、8个候选**。

保留本轮开始前用户对9000的注释、格式及traits末尾静态断言的删除；
9000共享helper区逐字节不变，其余六个候选和registry未改。
基线复用核对确认：除用户已删除的`sizeof(kargs)==96`静态断言外，
本轮开始前9000 pipeline/traits的C++ token与已验证版本一致。

**只对9000和9010的实际device TU各进行一次CPU编译，两项均通过；
机器指令逐字节一致，完整资源metadata除符号名称外一致，96-byte ABI不变。
公开名称、registry、host launcher检查一致。**
无GPU执行或重新tune，无提交或推送；下方历史记录保留当时的共享结构说明。

- [9000/9010编译与机器码对照](reports/opus_9000_9010_separate_20260928/validation.json)
- [本轮源码差异](reports/opus_9000_9010_separate_20260928/refactor.diff)
- [当前pipeline和traits文件表](csrc/opus_gemm/README.md#mxfp8-b-preshuffle-pipeline-and-traits-headers)

## 2026-09-28：所有候选pipeline内部按9000组织，并整理逐候选优化步骤

9020、9021、9022、9023、9024、9030六个独立pipeline已统一内部组织：
types/coordinates → global views → layouts → LDS → MMA/registers → helpers →
Prologue → Main loop → Epilogue → Output writeback。
实际移动了声明/布局/LDS视图，补齐并使用统一类型和坐标名，抽取地址/operand/output helpers；
9021/9022的advance正文内联进原runtime for，9020/9030的advance_tile定义移到Prologue之前。
9010直接继承9000的完整执行流程，wrapper补充说明；9000原计算主体和共享helper未改。

原stage数、wait/barrier、AGPR pin、scale布局、尾部处理和输出策略保留。
首轮6项均编译成功，但声明求值位置变化影响了readfirstlane与K/128的指令生成；
定位后恢复wave坐标和runtime-K的原求值位置，保留新的组织结构。
**第二轮6/6机器指令逐字节一致，完整资源metadata及公开名称/registry/host launcher全部一致。**
两轮共12次CPU device TU编译；无GPU执行、重新tune、提交或推送。
所有traits、注册和codegen保持不变，之前305项调优仍是最近一次性能测量。

- [最终六项机器码与资源对照](reports/opus_pipeline_structure_20260928/attempt2/validation.json)
- [逐候选优化步骤总结](reports/opus_pipeline_structure_20260928/OPTIMIZATION_SUMMARY.md)
- [当前源码架构和候选文件](csrc/opus_gemm/README.md#mxfp8-b-preshuffle-pipeline-and-traits-headers)

## 2026-09-28：9021–9024拆成四套独立pipeline和traits

按用户要求将两组共享入口拆开，每个候选拥有自己的pipeline计算主体和固定geometry traits，
文件/type/device kernel继续使用9000风格的前缀与以下独立后缀：

| ID | 后缀 | NUM_STAGES | LDS bytes |
|---|---|---:|---:|
| 9021 | `4wave_128x128` | 3 | 105504 |
| 9022 | `4wave_160x128` | 2 | 81184 |
| 9023 | `4wave_64x128` | 3 | 80384 |
| 9024 | `4wave_64x64` | 4 | 71936 |

四个旧共享头文件已移除，替换为八个独立头文件；traits不再使用TileM/TileN模板参数。
codegen同步生成四套独立类型与函数。现在共8个pipeline入口、7个计算模板、8个候选，
其中9010继续复用9000的计算模板。其余候选、公共helper、registry和shape支持不变。

**本轮只编译9021–9024的4个实际device TU，各一次，全部通过；
四个候选的机器指令逐字节一致，完整资源metadata除名称字段外一致，
公开kernelName、registry和host launcher检查全部一致。**
计算与调度保留原样；无GPU执行、重新tune、提交或推送。前轮报告保留历史文件名。

- [当前独立pipeline和traits文件表](csrc/opus_gemm/README.md#mxfp8-b-preshuffle-pipeline-and-traits-headers)
- [本轮四项CPU编译和对照](reports/opus_split_9021_9024_20260928/validation.json)

## 2026-09-28：当前8个候选按9000的pipeline/traits结构统一组织

所有生产入口保留在 `csrc/opus_gemm/include/gfx950/`，按wave数、tile几何和必要变体命名。
9000原文件和共享helper不动；其余五组pipeline/traits使用同一后缀：

| 当前ID | 文件和内部类型的后缀 | 实例化方式 |
|---|---|---|
| 9010 | `4wave_256x256_padded_m` | PAD_M traits，复用9000 kernel |
| 9020 | `8wave_192x256` | 固定192×256、8-wave |
| 9021/9022 | `4wave_m128_160_n128` | 同一模板，TileM=128/160 |
| 9023/9024 | `4wave_m64_n64_128` | 同一模板，TileN=128/64 |
| 9030 | `8wave_192x256_large_output` | 独立大输出地址pipeline |

内部device函数统一为 `gemm_a8w8_mxfp8_scale_<suffix>_kernel`；
9000/9010继续使用 `gemm_a8w8_mxfp8_scale_kernel`。现在共6个pipeline入口、5个计算模板、8个实例。
9020的packed B-scale storage wrapper移入自身traits，pipeline直接使用传入Traits；
9030改为直接继承原192×256几何base，保留原scale布局和143104-byte LDS，
9020仍为143360-byte LDS。两者互不继承。generic192×256 base仍保留供retained源码使用。

codegen同步映射新文件/type/device符号；候选ID、公开kernelName、shape支持、96-byte ABI、
所有计算和流水线调度保持不变。旧五组头文件已由新路径替代。
本轮验证采用CPU生成与编译8个实际device TU，并对照上一轮完整305项测量保存的JIT；
**8/8编译通过，机器指令逐字节一致，除名称字段外的完整资源metadata一致；
8/8公开名称、registry和host launcher检查一致。**
没有新增GPU调优或新的性能结论，没有提交或推送。历史报告和冻结二进制保留原名称。

- [当前8个候选的完整源码链接与命名规则](csrc/opus_gemm/README.md#mxfp8-b-preshuffle-pipeline-and-traits-headers)
- [本轮编译、机器码与资源对照](reports/opus_layout_9000_20260928/validation.json)
- [重组前源码快照](reports/opus_layout_9000_20260928/before_sources)

## 2026-09-28：优化后的9020及9030已完成完整305项全后端重调优

用户要求完整重tune、包含新增10项，并列每个OPUS候选的文件和优化目标。
本轮 **305/305得到有效最优，退出码0**，profile29,180条，tuned305条；
原tuner成功扫描计时653.8101秒，GPU0–7、warmup5、iters51、all backends及合法splitK。
所有1,955条OPUS候选/shape记录数值通过，支持枚举完全匹配当前8个ID。
整体中选：**OPUS299、CK2、CKTile4、ASM0**。
OPUS各ID中选：9000×130、9010×29、9020×44、9021×12、9022×51、9023×16、9024×7、9030×10。

新增的是10个shape，不是10个kernel；**10/10全部中选9030**。
其中5项有有效外部候选，OPUS耗时降低18.250%–19.660%；另5项没有有效外部可比较。
所有300个可比shape中，OPUS快294、慢6，等权几何平均耗时降低12.398%。

**本轮旧9020慢16项为11快/5慢，不能沿用先前局部三轮的16/16结论。**
本轮仍慢的6项：`1536,7168,768`（9020，+1.460%）、`10240,768,7168`（9020，+0.721%）、
`1536,16384,1536`（9020，+0.589%）、`1536,768,7168`（9024，+0.390%）、
`1536,7168,3072`（9020，+0.241%）、`1664,7168,768`（9020，+0.066%）。
以上均比较本轮有效最优，不混用历史微秒数；本轮只有一次全量扫描，没有继续优化或复测。

首次启动在测量前发现完整JIT的device TU缺少uintptr_t；已将9020中4处类型替换为
OPUS已有u64_t，地址算法不变。首次失败日志已归档，随后完整重启并完成。
最终JIT main机器指令与前次局部实测完全相同（7816bytes、VGPR208/SGPR52/LDS143360、无spill）；
原输入/基线、其它kernel、注册/codegen和共享tuner保持不变。未提交或推送。

- [完整结果和6项慢shape明细](reports/opus_full305_after9020_20260928/RESULTS.md)
- [8个候选文件/traits/优化目标](reports/opus_full305_after9020_20260928/CANDIDATES.md)
- [305项最优CSV](reports/opus_full305_after9020_20260928/tuned.csv)
- [全候选profile](reports/opus_full305_after9020_20260928/profile.csv)
- [校验与统计](reports/opus_full305_after9020_20260928/summary.json)

## 2026-09-28：继续优化 9020，16 项三轮中位数全部胜出（已整合）

已将本轮 `mid_packb_seed0_bptr` 整合到生产 9020 的 main pipeline。
最终同场三轮：**16/16 中位数快于实测外部候选，14/16 每轮都快于当轮所有外部候选**；
`(1536,7168,3072)` 和 `(1344,16384,1536)` 各赢 2/3 轮，不能声称每轮稳定 16/16。
相对外部中位数耗时降低 0.808%–4.057%；相对同场原 9020，15/16 更快，
几何平均耗时降低 2.745%，最佳降低 5.749%，最差回退仅 0.119%。
原 9020 同场本身已赢外部 6/16，以上优化收益均来自本轮新旧同场比较。

改动为 A XOR LDS 映射、B-scale 双字节打包、中点发布流水线、
K0 与 scale panel 重叠，并在预取处重建 B 指针描述符以消除编译器 waterfall。
仍为同一个 9020、192×256×128、8-wave、runtime K128–16384；没有新增候选。
只改一个生产 pipeline 文件。packed storage 类型限定在该文件，原 main traits 不动，
避免影响继承 main traits 的 9030。9000、9023、9030、共享 helpers、注册/codegen、
mp_tuner、原输入/baseline CSV 的保护哈希均未变。

最终 155 个启动检查和 465 次计时后检查全部 error=0，保留原数值判据与 profiler。
整合后重新编译的生产机器指令与实测版本完全相同：VGPR208、SGPR52、LDS143360、无 spill。
6 项边界/原有赢家回归的28条记录全部通过；四个旧赢家耗时变化为
−5.113%、+0.021%、−1.766%、−1.898%，没有追加全量 295/305 项调优。
忽略项 `(1536,768,7168)` 的 9023 未改。GPU 工作已结束，未提交或推送。

- [本轮完整结果与16项表](reports/opus_9020_resume_20260928/RESULTS.md)
- [最终三轮比较](reports/opus_9020_resume_20260928/bptr16_r3_comparison.csv)
- [汇总/保护哈希](reports/opus_9020_resume_20260928/summary.json)
- [生产机器码核对](reports/opus_9020_resume_20260928/final/device_audit.json)
- [边界与已有赢家回归](reports/opus_9020_resume_20260928/final_regression_run.json)

## 2026-09-28：按用户要求直接尝试优化 9020 的 16 个慢项

用户将范围改为优化现有 9020，忽略 `(1536,768,7168)` 的 9023。
已完成隔离实验及 16 项同 GPU 三轮比较，**尚未达到 16/16，未替换生产 9020**。
最好的三轮统一改法 `prefetch_first` 胜外部 7/16、比原 9020 快 9/16，
相对原版几何平均耗时仅降低 0.125%，最差回退 2.438%。
原版本次同场也已经赢 6/16，不能把旧记录与新计时之差算作优化收益。
155 个候选/shape 的启动检查和 465 次计时后的数值检查全部通过。
输出重排、scale 打包、循环调度、最终转换错开和新的 M 分组均已留存探索结果；
没有得到足以覆盖生产 9020 的稳定收益。所有 GPU 工作已结束。
9000、9023、9030、共享 helpers、注册/codegen、mp_tuner 和原始 CSV 的保护哈希未变。
没有新增注册候选、全量调优、提交或推送。

- [本次结果及 16 项明细](reports/opus_9020_opt_20260928/RESULTS.md)
- [三轮逐项比较](reports/opus_9020_opt_20260928/target16_r3_comparison.csv)
- [汇总与保护哈希结论](reports/opus_9020_opt_20260928/summary.json)

## 2026-09-28 最新：新增 9030，额外 10 项全部由 OPUS 覆盖

新增独立 `large_output` 候选 **9030**，192×256×128、8 wave、runtime K。
以 9020 的计算流程为基础，仅在独立 pipeline 中将 C 基址改为 64 位计算，
buffer 范围限制为当前 tile 的有效行跨度，batch=1 的未使用输出 batch stride 设为 0。
A/B 和局部 tile 字节跨度仍受 signed 32-bit 限制；原七个候选的大小限制不变。
新候选在本次 305 项输入中恰好支持此前额外的 10 项。

**10/10 项 GPU 数值检查通过，全部 errRatio=0，退出码 0。**
使用原专用 tuner，仅扫描 9030 和这 10 个目标，物理 GPU 0–7，warmup 5、iters 51。
本批包含 2/2.5/3/3.5/4/5/6/7/8 GiB 输出，最大 `(65536,65536,1536)` 也通过。
全部目标 K=1536；没有追加其它边界形状或 295 项全量复测。
本批调优器统计耗时 20.6128 秒（含 OPUS JIT），测量耗时范围 1828.6272–7750.0115 us。
外部后端未重测，不能用此前 CK 微秒数作同场性能结论。

当前注册为 **9000、9010、9020–9024、9030，共 8 项**。
结合原 295 项的已通过结果和本批 10 项，现在 **305 项全部有数值通过的 OPUS 候选**。
这不代表 305 项全部性能胜出，也不改写先前 300 成功/5 失败的全后端扫描记录。

一次 CPU 核对确认原七个生成 launcher 完全相同、1945 个候选/shape 支持关系未变、
59 个已有头文件哈希未变。9000、共享 helpers、现有七个 kernel 的计算代码和
`mp_tuner.py` 未改；本批已结束，没有后台 GPU 工作，未提交或推送。

- [9030 结果报告](reports/opus_large_output_20260928/RESULTS.md)
- [10 项实测 CSV](reports/opus_large_output_20260928/tuned.csv)
- [GPU 汇总](reports/opus_large_output_20260928/summary.json)、[CPU 接线核对](reports/opus_large_output_20260928/preflight.json)
- [运行命令](reports/opus_large_output_20260928/run.sh)、[完整日志](reports/opus_large_output_20260928/tune.log)

17 个慢 shape 的当前最佳 OPUS 仍为 9020×16、9023×1。
优化建议是另建独立候选，以 9020 为起点先处理 9 项 N7168/K384或768 的短 K 组；
这组占差距最大的前 8 项。窄 N 的 `(1536,768,7168)` 单独看 9023，
0.064%/0.012% 两项尚无跨轮稳定性证据，不为这些微小差距新增专用实现。
本次落实的是额外 10 项覆盖，尚未开展 17 项的新性能优化实验。

## 2026-09-28 最新：删除旧三项并重编号，列出 17 个慢 shape

当前注册仅七项：**9000、9010、9020、9021、9022、9023、9024**。
9000 不变，旧 9020（padded-M）改为 9010，旧 9060–9064 依次改为 9020–9024。
旧 9010/9011/9012 的实现、六个专用 pipeline/traits 头及生成分支已删除。
以下较早调优记录中的 ID 仍表示实测时的旧编号。

一次 CPU 核对通过：七个保留项的元数据、生成 launcher 和实例化代码逐项完全相同，
59 个保留头文件哈希未变，包含 9000 及共享依赖。未执行 HIP 编译或 GPU 复测。
默认目录的旧 OPUS JIT 二进制及 receipt 已移入本轮归档，避免新编号误用旧映射；
后续执行需要重建与当前注册一致的 OPUS JIT。历史实测 JIT、CSV、计时均保留原样。

已提供[新编号的 300 项 tuned CSV](reports/opus_mxfp8_renumber_20260928/tuned.csv)，
只改 OPUS 的 kernelId，原 kernelName、shape、耗时与其它后端记录均保持。
编号映射及 CPU 验证见
[id_mapping.json](reports/opus_mxfp8_renumber_20260928/id_mapping.json)、
[validation.json](reports/opus_mxfp8_renumber_20260928/validation.json)。

集合核对：旧 295 项是当前 305 项的真子集，**恰好多 10 项，没有多 19 项**。
当前 305 项与原始 baseline 的 gfx950/256CU、M>=1024 唯一 shape 集合完全一致。
新增项为 `(M,65536,1536)`，M=16384/20480/24576/28672/32768/40960/49152/57344/65536，
以及 `(65536,16384,1536)`；均超出 OPUS 现有输出字节范围。
精确差集：[added_shapes.csv](reports/opus_mxfp8_renumber_20260928/added_shapes.csv)。

现有实测中 OPUS 慢于有效外部候选的 **17 项**完整列表见
[slow_shapes.csv](reports/opus_mxfp8_renumber_20260928/slow_shapes.csv)，
同时保留 `opus_measured_id` 旧编号和 `opus_kernelId` 当前编号。
它们全在原 295 项中，不含那 10 项无 OPUS 候选的大 shape；
16 项的最快 OPUS 为当前 9020，另 `(1536,768,7168)` 为当前 9023。
未解决的 5 项不算作这 17 个计时慢项。

## 2026-09-28：原专用 tuner 已重扫全部 305 项，300 项有有效选择

按 strict-task-scope 继续昨日记录，使用下方原专用 tuner 命令，物理 GPU 0–7、
`--mp 8 --shape_grouped --warmup 5 --iters 51 --all`，OPUS 限定
9000/9020/9060–9064，CK/CKTile/ASM 全候选及合法 ASM splitK。
本轮从头测量，独立 JIT 和结果目录为
[`reports/opus_m_ge1024_retune_20260928/`](reports/opus_m_ge1024_retune_20260928/)。
调优器统计耗时 **1076.54 秒（约 17.94 分钟，含扫描期间的 JIT 构建）**。
四批全部执行完，29,170 条候选记录覆盖全部 305 项；
**300 项获得有效最优，5 项无有效候选，进程退出码为 1，不能称为 305 项全部通过。**

整体中选：**OPUS 278、CK 7、CKTile 15、ASM 0**。
300 条已保存选择均为本批最小有效耗时，且 `errRatio=0`。
OPUS 的 1,945 条合法候选记录全部通过；全部七个 ID 均有中选：
9000×132、9020×30、9060×31、9061×14、9062×50、9063×11、9064×10。
本轮只完成调优及记录，没有追加 kernel 优化、删除或再次测量。

在 OPUS 与外部均有有效候选的 **295 项**上，OPUS **278 快 / 17 慢**，
相对最快有效外部的等权几何平均耗时降低 **11.867965%**。
相对同批仅 9000/9020，七项候选池有 **133 项更快、162 项相同**，
几何平均耗时降低 **15.876562%**。这些是本轮结果；
51 次计时不是 51 个独立调优轮次，也不与历史微秒数混合选型。
最差外部差距为 `(1472,7168,768)`：OPUS 17.2066 us，外部 16.5248 us，
OPUS 慢 4.125920%。

五个未解决 shape 均为 **N=65536、K=1536**，
**M=32768/40960/49152/57344/65536**（BF16 输出分别为 4/5/6/7/8 GiB）。
每项均为：OPUS 被现有支持范围过滤，CKTile 的 33 项明确不支持，
CK 的 18 项有正耗时但 `errRatio=1`，ASM 的 30 项也被数值检查拒绝。
其余五个无 OPUS 候选的大 shape 均得到有效 CK 选择。
保留原始 305 项输入与所有失败记录，不把输入缩回 295 项。

ASM 本轮 **11,670 条记录全部有正耗时且被严格数值检查拒绝**。
只读核对未发现专用适配层明显的 ASM 数据布局或参考复用错误；
底层数值差异原因仍未确定。CK/CKTile/ASM 保持原 generic 的独立随机 FP32 scale，
OPUS 保持原生随机 E8M0 scale，各自计算参考；跨后端计时不是同一组数学输入。
没有放宽数值阈值或用旧计时替代被拒绝候选。

交付：

- [本轮结果与限制](reports/opus_m_ge1024_retune_20260928/RESULTS.md)
- [300 项有效最优](reports/opus_m_ge1024_retune_20260928/tuned.csv)
- [全部 29,170 条候选记录](reports/opus_m_ge1024_retune_20260928/profile.csv)
- [305 项逐 shape 比较](reports/opus_m_ge1024_retune_20260928/comparison.csv)
- [候选用量](reports/opus_m_ge1024_retune_20260928/candidate_usage.csv)
- [完整性与来源汇总](reports/opus_m_ge1024_retune_20260928/summary.json)
- [本轮命令](reports/opus_m_ge1024_retune_20260928/run.sh)、[运行记录](reports/opus_m_ge1024_retune_20260928/run.json)、[日志](reports/opus_m_ge1024_retune_20260928/tune.log)

71 个记录文件的 SHA256 均保持一致，包含原始 CSV、305 项输入、专用 tuner、
`mp_tuner.py`、9000/9020 及相关头文件。未改 kernel、共享框架或生产 dispatch CSV；
本轮无后台 GPU 工作，未提交或推送。

## M >= 1024 的 305 项输入

运行文件恢复为
[`csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py`](csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py)。
原始输入和历史基线是
[`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`](aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv)，
本次未修改其内容。原表共 1042 条数据，其中 gfx950/256 CU 共 745 个唯一 shape。
其中 `M >= 1024` 的全部 **305 个唯一 shape** 已包含在
[2026-09-28 的逐 shape 对比记录](reports/opus_m_ge1024_retune_20260928/comparison.csv) 中，
可直接用作该范围的 shape 输入；tuner 忽略其中的旧候选和耗时。
它包含旧 295 项子集之外的全部 10 项。该轮 GPU 调优结果见上节。

输入按后端生成：CK/CKTile/ASM 直接复用原 blockscale tuner 的
`generate_data`，A/B 为 `rand(FP16) / 10` 后转 FP8，scale 独立随机生成 FP32；
OPUS 使用相同的 A/B 生成方式，scale 独立生成原生 E8M0 指数字节。
外部后端不使用 E8M0 解码的 scale。调优和回放均使用各自数据计算参考结果，
保留现有误差判定；专用 tuner 使用调优器已有的参考重算路径处理两类输入，
不修改 `aiter/utility/mp_tuner.py`。

在仓库根目录执行，将编译器路径换成目标机器的实际路径：

```bash
ROCR_VISIBLE_DEVICES=0 \
OPUS_HIP_CLANG_PATH=/absolute/path/to/llvm-pin-build/bin \
python -u -m csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune \
  -i reports/opus_m_ge1024_retune_20260928/comparison.csv \
  -o /tmp/dsv4_m_ge1024_tuned.csv \
  -o2 /tmp/dsv4_m_ge1024_profile.csv \
  --opus-kids 9000,9010,9020,9021,9022,9023,9024,9030 \
  --libtype all --splitK --shape_grouped --mp 1 \
  --warmup 5 --iters 51 --all
```

这是原 tuner 的单命令入口，OPUS JIT 在正式扫描前自动补编候选，CK/CKTile/ASM
沿用原后端的 JIT 路径。八卡时使用物理 0–7 卡，将环境变量改为 `ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，
并将参数改为 `--mp 8`。
`--shape_grouped` 将每个 shape 的候选放在同一张卡比较。

新五项使用正式 ID，避免旧独立库 ID 与其它架构的全局 ID 区间冲突：

| Family | 正式 ID | 历史独立库 ID | Tile M×N×K |
|---|---:|---:|---|
| main | 9020 | 21000 | 192×256×128 |
| small | 9021 | 21310 | 128×128×128 |
| small | 9022 | 21311 | 160×128×128 |
| narrow | 9023 | 21220 | 64×128×128 |
| narrow | 9024 | 21221 | 64×64×128 |

上面的命令在 OPUS 侧选择 9000/9010、五个合并候选及大输出 9030，并与 CK/CKTile/ASM 比较。
旧 9010/9011/9012 实现已删除；省略 `--opus-kids` 时原 tuner 枚举当前 26 个默认候选。
不支持某个 shape 的 OPUS 候选按原规则跳过；原始 shape 仍由有效的外部候选参与比较。

`-i` 读取新 untuned CSV 的 shape；`-o` 保存新的逐 shape 最优选择，`-o2` 保存
各候选记录。原始基线的历史微秒数不会参与本轮选型。若需要扫描完整 745 项，
将 `-i` 换回原始基线 CSV；tuner 会忽略其中已有的 `libtype/kernelId/us`。

环境需要 gfx950 ROCm/PyTorch（原生 `torch.float8_e8m0fnu`）、已初始化的 CK submodule，
以及支持 `clang::amdgpu_pin_agpr` 的定制 clang；已验证工具链源为
`yuyzhang512/llvm-project` 提交 `49c41889681640665400cb01c9fbb4c0a024cde4`。

以下为历史阶段记录。此前新增的 `reports/opus_remote_tune_20260927/` 仅保留为
独立实验工具；它的 prepare/build/launch 和 295 项子集不是当前推荐入口。

## 2026-09-27 最新：合并 kernel 统一为 9000/9020 的源码样式

按用户最新要求，三个新 family 已整理成 `include/gfx950/` 下的正式
`opus_gemm_{pipeline,traits}_a8w8_mxscale_bpreshuffle_{main,small,narrow}_gfx950.cuh`。
入口统一为 `template<class Traits>`；small 用 M128/160 两个 Traits 实例，
narrow 用 N128/64 两个实例，main 也由固定入口改为 Traits 模板。
计算流程、runtime K、私有 ID 和原启动约束保持。来源为 main_v1/21000、
small_v4/21310+21311、tiny_v3/21220+21221；加原9000/9020，最终目标仍最多7候选。

三库五个kernel已CPU编译通过，与来源版本逐ID比较，指令字节和归一化资源描述
全部一致；94个既有文件（9000/9020、helpers、注册、codegen及来源文件）哈希未变。
新薄测量adapter和验证证据在
[opus_native_style_20260927](reports/opus_native_style_20260927/README.md)。
未运行GPU，也未把私有ID加入全局注册。

**合并调优尚未完成。** 当前可用完整基线仍是下方16候选、OPUS291/295。
第三批 `pilot62_v34_r3` 于08:32 UTC启动后中断：恢复检查时相关进程已不存在，
八份summary均0数据行，JSON的running/launching为残留状态。须以新batch重启测量，
完成合并版295项比较，再最终选型、替换和删除。原自动续跑/安装工具仍绑定旧版本
路径和符号，使用新正式头前需更新其来源映射。

## 2026-09-27 最新完成：8 卡全量重新 tune，仅保留实际中选 kernel

按最新要求，从头完成完整 295 项、8 卡、三轮的全后端扫描（CK/CKTile/ASM 全枚举），耗时22.20分钟；没有复用旧计时或1%子集阈值。8个worker均passed。**OPUS291胜、CKTile4胜；实际中选16个OPUS（原5+11通用）**。相对有效外部几何平均耗时降低12.9453%，相对同批原5降低4.2688%；4575条合法OPUS候选均通过目标数值检查。

保留原5：9000、9010、9011、9012、9020。保留通用：13163、20000、20010、20011、20020、20100、20124、20125、20126、20128、20131。移除未选通用9661、9663、20120、20121、20122、20123、20127、20129、20130，并删除旧9030–9051的9个注册和对应生成分支/5个无依赖头。9000及共享helpers未改，原5的生成代码逐字节一致。

中选通用已迁入 `csrc/opus_gemm/mxfp8_bpreshuffle_retained/`，独立C ABI构建，9库仅11个device kernel，机器码与实测版本完全相同。共清理50个纯MXFP8实验目录的380个实现/生成器/二进制文件，并替换旧测量JIT为原5子集、删除788个旧生成缓存。历史CSV/JSON/日志/文档和更早混合快照保留；下方旧实验源码链接可能因本次按要求删除而失效。

入口：[本次完成报告](reports/opus_retune_prune_20260927/README.md)、[16候选模板清单](reports/opus_retune_prune_20260927/SELECTED_CANDIDATES.md)、[现存候选配置](reports/opus_retune_prune_20260927/selected_experiments.json)、[295整体选择](reports/opus_retune_prune_20260927/full295_r3/results/overall_selection.csv)、[删除清单](reports/opus_retune_prune_20260927/deleted_experiment_files.json)。

4个外部负项仍是N7168/K384、M1536/1600/1664/1728，慢1.569%/3.237%/1.442%/0.376%，均0/3轮胜。不额外追加测试，不修改默认dispatch CSV；本次无后台GPU工作，未提交/推送。下文原8候选结论属于前一轮历史。

## 2026-09-27 本轮完成：原5保留，其余实验收敛为8个通用候选

用户要求保留 **9000、9010、9011、9012、9020**，将其余实验合并成真正的runtime-K候选，不要求恰好4个。已恢复昨天记录、完成295项重新tune，并继续做了短K循环、融合输出和cache策略优化。9000及影响它的共享helpers、生产注册和默认dispatch均未改；本轮只做kernel、目标shape数值检查及性能比较。

**最终结果：8个通用候选+原5，共291/295项快于同卡重测的有效外部finalist。** 原5选中200项，8个通用选中95项。相同shape等权几何平均耗时：相对原5降低4.147%，相对外部降低12.973%；相对旧25实验+原5增加0.114637%，最坏增加3.951%。固定K对照没有进入精简集合。20个通用候选的精确子集搜索证明：逐shape相对完整通用池最多1%损失、且保留其全部外部胜项时，最少需要8个；实际几何损失0.008106%、最坏0.956643%。

最终数据来自完整v6的178行与后续v7的117行，均为三轮；每个shape整行选择其最新完整worker，包含新旧候选、外部参考、来源和轮次。所有shape保持v6的物理GPU4–7分配，**两批全部passed，数值检查0拒绝**。两个队列均complete、benchmark/analysis退出码均0，本轮没有后台GPU工作。较早v5的119项五轮也全部通过，但不混入最终计时。

| 通用ID | 实现 | runtime K | 最终选择次数 |
|---|---|---|---:|
| 20000 | 192×256、8 wave、N-first | 128–16384，步长128 | 12 |
| 20100 | 同主体，窄N/大M使用group-M4 | 128–16384，步长128 | 31 |
| 20128 | 20000融合最终MFMA/BF16输出，vec8/cache2 | 128–16384，步长128 | 21 |
| 20010 | 128×128、4 wave | 128–1536，步长128 | 7 |
| 20011 | 160×128、4 wave | 128–1536，步长128 | 14 |
| 20020 | 192×224、8 wave、融合输出 | 128–1536，步长128 | 2 |
| 20125 | 192×256统一U2/drain、非融合vec8输出 | 128–1536，步长128 | 6 |
| 20131 | 192×256统一U2/drain、fused vec4、group-M4/cache2 | 128–1536，步长128 | 2 |

原5的使用次数：9000×129、9010×10、9011×23、9012×7、9020×31。各通用ID对应单个runtime-K入口，不按K实例化。

**有效与无效改动：** 20100的通用网格策略和20010/20011小tile保留主要贡献；20125在部分大M短K有用。20128相对20000在同源295项中200项更快、几何平均快0.165%，属于小幅互补收益。20131相对20127在117合法项上几何平均快2.305%，但只58项严格更快，不能全局替换；它在(1472,7168,384)为12.280889µs，快于同场外部12.379533µs，3/3轮胜，解决一个原通用池外部负项。同shape去掉20131的最快通用为20128，12.691872µs。

20129/20130统一小tile循环后，实际VGPR分配升至264/320；虽无spill，117项全部比20010/20011慢，几何平均慢55.76%/49.47%，已剔除。20120–20123无最终贡献；20124/20126等局部小收益在精简约束下可移除。所有失败实验、源代码和中间结果保留。

**剩余4个外部负项**都为N7168/K384，且本批各0/3轮胜：M1536慢2.888%、M1600慢2.882%、M1664慢2.110%、M1728慢0.905%。另有8项相对旧25+原5慢超过3%，最坏为(1152,7168,384)的3.951%；完整列表在最终报告。未把这些差距当作已解决，也未反复测到获胜。

最终交付入口：

- [完整中文结果与剩余慢项](reports/opus_generalize_20260926/refine117_20260927_v7_r3/review_results/RESULTS.md)
- [8候选配置及原5声明](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/selected_experiments.json)
- [295项选择、GPU与轮次来源](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/choices295.csv)
- [候选用量及移除影响](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/candidate_usage.csv)
- [同场优化配对结果](reports/opus_generalize_20260926/refine117_20260927_v7_r3/review_results/candidate_effects.csv)
- [精确子集与0%/1%/2%取舍](reports/opus_generalize_20260926/refine117_20260927_v7_r3/exact_generic_selection/report.md)
- [测量与来源审计](reports/opus_generalize_20260926/refine117_20260927_v7_r3/runtime_analysis/summary.json)
- [最终独立选型复核](reports/opus_generalize_20260926/v7_final_selection_review.json)、[原始数据与结果复核](reports/opus_generalize_20260926/v7_results_review.json)：最小集合、同卡来源、固定K隔离及原5保护均通过。

配置是实测候选池，库目录相对配置文件解析；registered_opus_ids只声明原5成员，旧loader不消费该字段，不是已计时验证的生产dispatch。外部对照只从历史全后端扫描取有效finalist身份，全部耗时在当前worker重新测量；本轮没有重扫全部CK/CKTile/ASM。

阶段记录均保留：v3完整295三轮；v4短117三轮；v5短117+长2共119五轮；v6再做完整295三轮；v7最后117三轮。v6/v7使用同一物理GPU分配，v7以v6为base，禁止跨批次为各候选取最小值。源码、生成器、metadata、构建二进制和保护文件在各次计时期间冻结；无额外边界、接口或单元测试。本轮未提交或推送。

## 2026-09-26 最新恢复：减少实际候选，runtime-K 通用化

用户在完整295项调优后明确要求：把大量实验候选做成少量通用 kernel，**不是合并源文件**。最新工作入口为 [opus_generalize_20260926](reports/opus_generalize_20260926/README.md)。已编译4个实际runtime-K kernel：20000（192×256长短K）、20010（128×128短K）、20011（160×128短K）、20020（192×224融合输出）。仍冻结9000，只做kernel及目标shape数值/性能比较。

最新 `generalize99_v1_r3` 在99项目标测试中断：0–3因外部占用失败，4–7各仅完成1项；恢复时旧进程均已不存在。旧记录原样保留，不把其残留 `running` 当作当前任务。已准备 [resume_launch.py](reports/opus_generalize_20260926/resume_launch.py)，把99项重新分片到物理GPU4–7；复用历史有效外部候选身份，所有对照耗时在实际新卡重测，源卡与实际卡分别记录。新的完整通用化测量尚未完成，不能引用下文293/295作为通用候选池结果。

恢复时GPU0–3满载、4–7原本空闲；启动前复查发现后四卡也新增外部任务。08:38 UTC已启动独立等待队列PID11966，自动等待GPU4–7连续空闲后运行99项并汇总；当时状态为waiting，尚无新GPU结果。实时状态见 [queue_state.json](reports/opus_generalize_20260926/queue_resume99_v1_r3/queue_state.json)。不使用前四卡，不终止他人任务，不删除占用检测。

## 2026-09-26 最新完成：完整295项重新调优

**任务已完成：295/295全量三轮调优，加58项同GPU五轮确认；最终OPUS293胜、CKTile2胜、CK/ASM均0胜。** `full295_r3`与`close295_r5`两批各8个worker全部passed，已无后台GPU工作。最终237项采用三轮、58项采用五轮，每个shape取较新完整批次整行，不跨批次选最小耗时。

**9000、共享helpers、注册和默认配置完全未改。** 本次最终127项仍选中9000。`experiment`表示独立编译的OPUS实验实现，算法仍为OPUS。最终选型共30个ID：原注册9000×127、9020×29、9011×23、9010×10、9012×7，共196项；25个实验ID覆盖99项（其中97项整体胜出）。相对最快有效外部几何平均耗时下降13.01294%，相对同批旧OPUS下降4.34455%。

入口：

- [最终报告](reports/opus_cover87_20260926/display295_v1/README.md)
- [295项逐shape最佳OPUS选择](reports/opus_cover87_20260926/display295_v1/usage/opus_choices.csv)
- [30个候选的使用次数](reports/opus_cover87_20260926/display295_v1/usage/opus_candidate_usage.csv)
- [295项整体最快实现](reports/opus_cover87_20260926/display295_v1/overall_winners.csv)
- [汇总与来源](reports/opus_cover87_20260926/coverage295_v1/summary.json)
- [精选25实验ID配置](reports/opus_cover87_20260926/selected_experiments295.json)；全扫描实际输入为[25库109实验ID](reports/opus_cover87_20260926/experiments295.json)，另行保留全部合法注册OPUS。

剩余两项（均五轮）：`(1600,7168,384)` OPUS14540 12.784568us vs CKTile11 12.760023us，慢0.192362%、0/5轮胜；`(1536,7168,768)` OPUS12641 17.179160us vs CKTile30 17.168143us，慢0.064172%、2/5轮胜。本轮原87项为85胜/2负，其余208项全胜；历史v11的87/87只代表之前目标批次，不能替代本次全量结果。没有把噪声反复复测到获胜，也没有额外边界/接口/单元测试。

最后一次kernel改进为OPUS15940（192×224、cache0、融合最终MFMA/BF16 LDS输出）。完整295调优后，15940仍在`(1536,7168,384)`选中；相关源码在`reports/opus_resume_20260926/shortk_n224_exp/`。新候选全部独立实验实现；生产注册未改。

全部OPUS全扫描候选记录（注册2626、实验5650）及五轮确认候选均通过原目标数值检查。58项确认中56项中位数领先，37项每轮都领先。精确295成员复用旧`shapes.csv`，另10项原寻址排除未纳入。旧阶段记录、失败实验和所有候选源/二进制哈希继续保留。

后续若继续优化，优先上述两项；沿用已编译JIT及目标shape检查，不改9000。不得使用旧`selected_experiments.json`替代新全量候选配置，也不得用旧87快照替代295结果。

## 2026-09-26 v10历史阶段：原87项优化

**9000保持冻结。** 用户再次明确不得修改9000；本轮没有修改9000本体、共享helpers、生产注册或默认调度。新实现均位于 `reports/opus_resume_20260926/` 的独立实验目录。14640只是隔离的9000派生副本，无收益，已从后续候选配置移除。

按每个shape最新完整五轮批次，原87项 **86胜 / 1未胜**。相对最快有效CK/CKTile/ASM候选，几何平均耗时下降 **4.63%**；相对同批旧正式OPUS下降 **13.00%**。后续54项全部中位数领先。56项五轮均领先，另30项中位数领先但未每轮胜出，不能把全部中位数胜项称为稳定大幅领先。

唯一未胜项 `(1536,7168,384)`：本批最佳OPUS15040 **12.665778us**，cktile_11_split0 **12.660383us**，差 **0.0426%**，接近持平但保留为未胜。所有测量已经结束，最后批次为 `targeted1_v10_r5`，status=passed；没有后台GPU工作待收尾。

入口：[完整结果](reports/opus_cover87_20260926/RESULTS.md)、[87项选型](reports/opus_cover87_20260926/coverage87/best_opus_selection.csv)、[逐shape比较](reports/opus_cover87_20260926/coverage87/comparison.csv)、[汇总与来源](reports/opus_cover87_20260926/coverage87/summary.json)、[精选候选配置](reports/opus_cover87_20260926/selected_experiments.json)。每个shape整行使用最新完整批次；不跨批选最小耗时。

已完成顺序：`finalists87_r5` → `targeted35_v3_r5` → `targeted27_v4_r5` → `targeted3_v5_r5` → `targeted2_v6_r5` → `targeted2_v7_r5` → `targeted1_v8_r5` → `targeted1_v9_r5` → `targeted1_v10_r5`。全扫描来源为 `full87_r3`：完整枚举CK/CKTile/ASM，数值不合格候选剔除；后续只同卡重测有效外部最快者5%内对手。原87成员和295项历史基线保留。

本轮只进行kernel优化、编译、目标shape随测数值检查和性能比较，没有新增边界、接口、单元测试或295项全量回归。下文33项结果和额外验证流程为历史记录。

## 2026-09-26 前一阶段：33 项 kernel 实验

本轮按用户最新要求，只优化 kernel、测目标 shape、比较效果；停止扩展边界/接口测试和全后端扫描。完成四轮 kernel 实验，均在原 **25 个短 K + 8 个 K1536** 目标上同卡同批测五轮，随目标计时检查原 FP32 误差界。

**有效突破是 8-wave BF16 LDS 输出重排**：先按 pitch=264 将累加结果转为 BF16 放入复用的 LDS，再连续 `load/store<8>` 写回。最终候选 **9640/9641/9642/9651** 对应 K384/768/1024/1536；本轮 **33/33 超过同批旧最快 OPUS，18/33 超过同批 CKTile 对照**。相对原固定 K kernel，各组几何平均耗时下降 **31.29% / 24.10% / 23.42% / 23.83%**；相对旧最快 OPUS 下降 **10.98% / 13.07% / 15.32% / 10.75%**。CKTile 的多数胜负差距较小，18/33 是本批五轮中位数结果。

最新结果和复跑入口见 **[RESULTS.md](reports/opus_resume_20260926/RESULTS.md)**；最终逐 shape 比较在 [epilogue_r5](reports/opus_resume_run_20260926/epilogue_r5/variant_comparison/variants_by_shape.csv)。新 kernel 为独立实验实现，位于 [shortk_epilogue_exp](reports/opus_resume_20260926/shortk_epilogue_exp/) 和 [k1536_epilogue_exp](reports/opus_resume_20260926/k1536_epilogue_exp/)，保留同库原版控制组；生产注册、9000 和默认选型均未修改。

本轮 CKTile 对照是同批重测的 27/28/29 和原 shape 的最快 CKTile，不是重新穷举所有后端。未重测完整 87/295 项，不能将 18 项直接相加得到新的全量胜负。下文的 **209/86 完整基线**及原 87 项成员继续保留；旧“33 项未完成”指当时中断的全后端 sweep，新完成的是上述目标 kernel 比较。下一步应沿用已经证明有效的输出重排方向，勿把无收益的 scale/XOR 微调当作主线。

## 2026-09-25 迁移交接记录

本分支：`Fyzyukk/aiter:aiter-opus-mxfp8-bpreshuffle`。实验源码提交为 `b7df6147`。本次交接把当前实验源码、最新完整基线、未完成实验记录及复跑工具提交到同一分支，供换服务器继续优化。本次没有启动 GPU 测试，也没有等待原机空闲。

## 先看结论和未完成事项

- **最新完整基线为 209 胜 / 86 负**：295 个支持的 M≥1024 模型 shape，另 10 项超出既有单张量有符号 32 位字节寻址范围。基线代码是 `10ab50645d1f25e11844b814b66002b27181dfbf`，已同步 upstream `b3d0cf4e`。此前 208/87、206/89 属于旧批次。
- **优化范围仍保留原 87 项**：上述 86 个负项，以及 `(4096,2048,7168)` 这个仅领先约 0.033% 的近平局项。不能因为最新基线少了一个负项就删除它。
- **9030–9033** 已接入注册和代码生成，GPU 边界检查通过；87 项五轮全后端扫描未完成，不能据此给出完整胜负统计。
- **9040–9042、9050–9051** 已实现，离线编译、CPU 检查和首次 GPU 正确性验证均通过；33 项五轮性能扫描未完成，尚未关闭任何优化目标。
- 9000/9010/9011/9012/9020 保持原实现；新 ID 均为可显式调用的实验候选，未加入默认编译集合，未更改生产调度 CSV。

最先补测 **25 个短 K + 8 个 K1536**；再按原 87 项分组继续，最终完整复核 295 项及原有胜项。要分别记录“超过旧 OPUS”和“超过最快有效外部后端”。

## 已恢复的证据

| 记录 | 当前可用结论 | 入口 |
|---|---|---|
| 当前分支全量重建、扫描、确认、回放 | 295 项；OPUS 209 胜、CKTile 86 胜；16 项追加五轮确认；295 项选择回放通过 | [完整结果](reports/opus_local_gap_current_20260925/RESULTS.md)、[最终比较](reports/opus_local_gap_current_20260925/final_comparison.csv)、[落后清单](reports/opus_local_gap_current_20260925/final_remaining_shapes.csv) |
| 当前全量候选数 | 27,690；OPUS 1,275 全有效；CK 5,160/5,310 有效；CKTile 9,535/9,735 有效；ASM 0/11,370 有效 | [全部候选记录](reports/opus_local_gap_current_20260925/profile.csv)、[拒绝记录](reports/opus_local_gap_current_20260925/rejected_candidates.csv) |
| 9030–9033 GPU 验证 | 156 项数值/保护区检查通过，12 项非法对齐被拒绝 | [validation.json](reports/opus_9030_targets87_20260925/validation.json) |
| fixed-K GPU 验证 | 96 项数值检查通过：41 项目标调用、50 项边界、5 项 raw E8M0；40 项非法 shape 被拒绝 | [validation.json](reports/opus_fixedk_gpu_20260925/validation.json) |
| fixed-K CPU/编译 | 短 K 集成检查 103/103；K1536 集成检查 127/127；布局/调度检查及 gfx950 编译通过 | [短 K](reports/opus_shortk_20260925/README.md)、[K1536 集成](reports/opus_k1536_20260925/integration/REVIEW.md)、[K1536 编译资源](reports/opus_k1536_20260925/offline/README.md) |

短 K / K1536 目录内较早文档的“GPU 待验证”描述对应当时的 CPU 阶段。以较新的 `opus_fixedk_gpu_20260925/validation.json` 和本交接页为准；性能验收仍未完成。本次交接重新执行 K1536 标准库集成检查，127/127 通过。

**中断记录的解释：** `opus_9030_targets87_20260925` 的四个 `gpu*_r5_run.json`，以及 `opus_fixedk_gpu_20260925` 的三个同名文件都残留 `status=running`。本次恢复时当前进程空间没有对应测试进程；这些是未正常收尾的记录，不能当作仍在后台执行或已完成。

- 903x 原始计时各卡分别为 103/100/99/100 行，都只触及分片的第一个 shape，未完成五轮。
- fixed-K 原始计时各卡分别为 327/315/361 行，每卡触及两个 shape；每卡仅首个 shape 有五轮汇总。不要把三个局部结果外推成 33 项结论。
- 原机器持续有外部占用，保留部分记录仅用于追溯。新服务器从全新 prefix 开始，同一个 shape 的所有候选在同卡同批重新测量。

## 源码入口和优化方向

注册：[opus_gemm_common.py](csrc/opus_gemm/opus_gemm_common.py)；生成：[gen_instances_gfx950.py](csrc/opus_gemm/codegen/gen_instances_gfx950.py)；调用：`aiter.ops.opus.opus_gemm(..., kid=..., layout="bpreshuffle", x_scale=..., w_scale=...)`。

pipeline/traits 位于 `csrc/opus_gemm/include/gfx950/`，公共文件名前缀为 `opus_gemm_{pipeline,traits}_a8w8_mxscale_bpreshuffle_`。

| ID | 几何/调度 | 文件后缀和说明 |
|---|---|---|
| 9000 | 256×256×128，4 wave | `4wave_gfx950.cuh`；冻结基准，保留 pipeline、traits 和影响它的共享 helper |
| 9010 / 9011 / 9012 | 128×128 / 64×128 / 64×64；K128 | 对应尺寸后缀；正式配置分别为 S3/K+2、S3/K+2、S4/K+3，直接 BF16 |
| 9020 | padded M | `padded_m_gfx950.cuh`，复用 9000 主体 |
| 9030 / 9031 | 192×256×128；4 / 8 wave；S64 | `192x256_gfx950.cuh` |
| 9032 / 9033 | 同几何；4 / 8 wave；S128 | 同上，scale panel 更大 |
| 9040 / 9041 / 9042 | 192×256×128，8 wave；固定 K384 / K768 / K1024 | `shortk_gfx950.cuh`；完整展开 3/6/8 步，只准备实际 scale |
| 9050 / 9051 | 同几何；固定 K1536 | `k1536_gfx950.cuh`；完整展开12步 / 两步循环 |

固定 K 入口要求精确 K，正 M/N、M%64=0、N%256=0，保留 FP8/BF16、原生 E8M0 布局、对齐和字节范围检查。固定 K 不能用于其他 K；过滤逻辑为 `a8w8_mxscale_bpreshuffle_supports_shape`，generated launcher 也有精确 K guard。

优先关注寄存器压力：9041 为 256 VGPR、无 spill；9050 为 256 VGPR、22 VGPR spills、76 private bytes；9051 为 206 VGPR、无 spill。编译资源只提供优化线索，不能替代 GPU 性能比较。不要为了减少等待而删除必要同步，也不要从短 K 结果外推长 K。

原 87 项分组为 **25 + 8 + 12 + 26 + 10 + 3 + 3**：短 K、K1536、N7168/K3072、N6144或7168/K7168、N7168/K16384、窄 N768、大 M/N2048。精确成员与旧基线冻结在 [分组台账](reports/opus_shortk_20260925/plan/targets87_ledger.csv) 和 [cohorts.csv](reports/opus_shortk_20260925/plan/cohorts.csv)。

## 新服务器准备

```bash
git clone --branch aiter-opus-mxfp8-bpreshuffle --single-branch \
  git@github.com:Fyzyukk/aiter.git
cd aiter
git submodule update --init --recursive
git rev-parse HEAD
git -C 3rdparty/composable_kernel rev-parse HEAD
```

CK submodule 应为 `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`。本交接已把当前修改提交进分支，不需要再应用旧 overlay 或旧 patch。

历史环境是 MI355X / gfx950 / 256 CU，Python 3.12、PyTorch `2.11.0+rocm7.14.0`、HIP `7.14.60850`。需要原生 `torch.float8_e8m0fnu`、可用的 ROCm/PyTorch 开发环境，以及 `rocm-smi`。新机环境不同，应保存版本并重建该环境的基线。已有匹配的 ROCm/PyTorch/Triton 环境中，可按仓库安装方式执行：

```bash
BUILD_TARGET=rocm AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  python -m pip install -e . --no-build-isolation
```

**OPUS 需要定制 clang。** 9000 使用 `clang::amdgpu_pin_agpr`；新候选引用相关布局 helper，也会遇到该编译器要求。普通 ROCm clang 不能直接代替。历史 OPUS 编译器为 `https://github.com/yuyzhang512/llvm-project.git` 的 `49c41889681640665400cb01c9fbb4c0a024cde4`（clang 24）；CK/CKTile/ASM 使用 ROCm clang 23。可迁移已构建工具链，或在新机按旧配置构建：

```bash
git clone https://github.com/yuyzhang512/llvm-project.git llvm-pin-src
git -C llvm-pin-src checkout 49c41889681640665400cb01c9fbb4c0a024cde4
cmake -G Ninja -S llvm-pin-src/llvm -B llvm-pin-build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_PROJECTS='clang;lld' -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build llvm-pin-build --target clang lld --parallel 12
```

保留 `aiter/jit/optCompilerConfig.json` 的编译语义。完整编译器版本记录见 [build.json](reports/opus_fixedk_gpu_20260925/build.json)。工具链、ROCm/PyTorch 和 `.so` 不随 Git 推送，必须在目标机器准备。

## 生成适配新机的入口

使用 [prepare_remote.py](reports/opus_remote_handoff_20260925/prepare_remote.py)，只读归档模板，在 `reports/` 下生成新目录。它不导入 torch/aiter、不查询 GPU、不编译、不测试，也不会覆盖已有目录。

先在新机核对同一张空闲卡的 **rocm-smi 物理编号、ROCr UUID 和 PCI bus**。ROCr 数字编号可能与物理编号不同。下面三个设备值必须替换；`85` 是 PCI `0000:85:00.0` 的十六进制 bus 字节。当前 harness 限定 PCI domain/device 为 0、gfx950/256CU。

```bash
python reports/opus_remote_handoff_20260925/prepare_remote.py \
  --output-dir reports/opus_remote_run \
  --gpu-index 0 --gpu-uuid GPU-0123456789abcdef --pci-bus 85 \
  --opus-clang-path /absolute/path/to/llvm-pin-build/bin \
  --stock-clang-path /opt/rocm/llvm/bin

# 仅列清单，不访问 GPU。
python reports/opus_remote_run/benchmark.py \
  --shapes reports/opus_remote_run/shapes33.csv --list-shapes
```

生成目录包含全新 `build.py`、`validate.py`、`benchmark.py`、最新 tuner adapter，以及 25/8/33/87/295 项 shape CSV。构建脚本从源码构建五个后端模块，OPUS 显式包含全部 14 个候选；不会复制旧机的 `.so`。计时脚本保留空闲检测、实际加载路径和源码/二进制哈希检查。

这些迁移入口已做 CPU 生成、语法、shape 清单检查；新机编译和 GPU 运行仍需执行。不要直接启动归档目录里的 `launch.py`，也不要直接跑归档 `build.py`：旧脚本有固定设备/路径，部分还需要旧 JIT 缓存。

## 新机重建、正确性和计时

在仓库根目录依次执行；每个命令成功后再继续。构建较重，编译时不要同时计时。

```bash
python -u reports/opus_remote_run/build.py
python -u reports/opus_remote_run/validate.py

# 先补完整 25+8 项，每个候选先检查数值，合格才计时。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 5 \
  --shapes reports/opus_remote_run/shapes33.csv \
  --jit-dir reports/opus_remote_run/jit --prefix fixedk_r5

# 后续覆盖原87项，包括全部合法旧/新OPUS和参考后端。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 5 \
  --shapes reports/opus_remote_run/shapes87.csv \
  --jit-dir reports/opus_remote_run/jit --prefix targets87_r5

# 候选优化完成后，完整回归，避免丢掉已有胜项。
python -u reports/opus_remote_run/benchmark.py --sweep --rounds 3 \
  --shapes reports/opus_remote_run/shapes295.csv \
  --jit-dir reports/opus_remote_run/jit --prefix all295_r3
```

`validate.py` 覆盖 fixed-K 的目标、边界、raw E8M0 和非法 K/对齐拒绝。benchmark 对每个合法候选独立检查数值并复查输出；修改 903x 或其他旧 kernel 时，还应适配其专项边界回归，不能只看性能目标。旧 903x 验证脚本在 [validate.py](reports/opus_9030_targets87_20260925/validate.py)，使用前需另行调整其固定 GPU 和输出目录。

若 GPU 空闲检查失败，保留此次输出，以新 prefix 重跑该 shape 的完整批次；不要删除检测或混合两次计时。prefix 相对路径落在新生成目录下，输出文件若已存在会拒绝覆盖。`validate.py` 和 `build.py` 的结果也应使用新目录保存，避免覆盖上一版控制组。

接近 ±3% 或轮次胜负不一致的 shape，从 `*_choices.csv` / `*_raw.csv` 另建包含 M/N/K 的 CSV，然后用 **同一次完整 sweep、同卡、同一套源码/二进制** 追加五轮：

```bash
python -u reports/opus_remote_run/benchmark.py \
  --final reports/opus_remote_run/all295_r3_run.json --rounds 5 \
  --shapes reports/opus_remote_run/close_shapes.csv \
  --jit-dir reports/opus_remote_run/jit --prefix close_r5
```

`--final` 保留全部合法 OPUS，以及有效外部对手中距最快者≤5%的候选。确认批次整体替换该 shape 的旧批次，不能挑两批最小值。完成性能选型后，还需把最终选择通过原生 E8M0 接口回放；旧 [replay.py](reports/opus_local_gap_current_20260925/replay.py) 可作实现参考，其历史路径和 GPU 映射需要适配。

## 必须保留的比较口径

- 原 FP32 逐元素误差界：`1e-4 + 5e-5 * sum(abs(A_i*B_i))`，区间端点舍入 BF16，`error==0`，保留 NaN 预填和输出保护区。数值失败不参与性能选优。
- 使用原 `run_perftest` profiler，warmup=5、iters=51、自动输入轮换，轮换候选顺序。公共 B shuffle、scale 解码和参考计算不计时；后端内部转换计时；CPU launch 开销不计时。
- 每个 shape 的全部对比在同一张实际空闲 GPU 上完成。新机器的耗时不与旧机器逐项拼接；旧值只作参考。
- 差距为 `(OPUS_us / reference_us - 1) * 100%`，正值表示 OPUS 更慢。上游 CSV 的历史 us 不直接替代同批重测值。计时条件差异分析见 [调查记录](reports/cktile_timing_diagnosis_20260925/README.md)。
- 9000 保持冻结；新方案用独立候选和独立 JIT 目录。新候选通过前不改默认选型；最终保留原 87 项成员、10 项寻址排除和全 295 项覆盖。

## 交回时保留什么

源码 commit、环境/编译命令、新控制组与候选的源码/二进制 SHA256、设备 UUID/PCI、完整 raw/correctness/summary/run 文件、最终 295 项比较与选择、87 项进度和仍未解决项。报告区分数值通过、超过旧 OPUS、超过最快外部对手三种结论。

本分支保存 2026-09-25 的文本报告、原始 CSV/JSON、验证源码及此前 9030 实验记录；更早的完整本地实验目录不在本次交接范围。文件清单和 SHA256 见 [manifest.json](reports/opus_remote_handoff_20260925/manifest.json)。`reports/.gitattributes` 保留证据原始字节，避免 CSV 换行归一化破坏历史哈希。

旧 JSON 内的绝对路径与二进制哈希描述原机，不要求新编译产物匹配旧哈希。未上传 JIT 缓存、编译二进制及完整工具链；新生成的运行入口以新哈希记录本次执行。


## Historical snapshot: 2026-09-27 single-flow queue before interruption

Latest scope: leave 9000/9020 and shared helpers unchanged; consolidate every other
retained candidate, including 9010/9011/9012, into one main loop and output flow per
geometry. Work and live status: `reports/opus_merge_flow_20260927/RUNNING.md`.
Two 62-shape/3-round pilots passed numerical checks. Five new geometries in three
families are implemented; small v3/v4 and tiny v3 await numerical/performance data.
At that earlier handoff, eight cards had outside activity and a queue was waiting.
Recovery subsequently confirmed that those processes no longer exist and all
eight pilot v34 summary files contain zero data rows. The JSON queue states are
historical remnants. Use the new-machine entry at the top of this document;
full295 selection and final cleanup remain pending.
