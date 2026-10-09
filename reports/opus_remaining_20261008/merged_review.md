# 9020 / 9022 / 9030：Oct8 selected 之后的诊断准备

基线为刚应用的 Oct8 正式模块 `6f6a0117b122f06b4802833c020effb4b24ff9fea56f20c8c16b9ced9787173f`，9021 issue/publish 和 Oct7 scoped traits 保留。当前用户明确纳入 9020，覆盖旧冻结约束；9000、共享 helpers 和已应用 9021 保持冻结。本记录只读源码、历史资料和 ELF，不构建、不跑 GPU、不修改 kernel。

[完整机器记录](merged_review.json)包含 245 个实际赢家、九个 device variant 的完整 metadata、指令 SHA、代表 shape 和历史 PC。九个历史 `.co` FUNC 指令、完整 metadata 和仅 entry-offset 归一化 descriptor 都匹配 Oct8 selected；PC 属于历史代码对象，采集新 ATT 后须重定位，不能直接套历史地址。745 条历史赢家表中的时间不当作新 Oct8 计时。

CPU 核验通过：28 个引用文件 SHA、245 个赢家条目、15 个代表 shape、九个 Oct8 variant 身份、两个 9030 支持域对照和两个本地链接均一致。

| Parent | 当前实际赢家 / variant | 首选 short | 首选 long / 实际函数 | 关键区别 |
|---|---:|---|---|---|
| 9020 | 76 / 7 | `1536×7168×384`，224 WG，`8wave_traits<192,256,8,384>` | `1536×7168×16384`，224 WG，`8wave_traits<192,256,128,0>`；均为 `gemm_a8w8_mxfp8_scale_8wave_192x256_kernel` | 同 grid，但 fixed/runtime body 与 scale panel 不同；已采用 K0→scale→K1 partial wait 与 midpoint |
| 9022 | 159 / 1 | `800×7168×384`，280 WG | `800×7168×16384`，280 WG；`gemm_a8w8_mxfp8_scale_4wave_160x128_kernel<...160x128_traits...>` | 同 body，同 grid；scale-first，32-tile refill，两槽 consumer barrier |
| 9030 | 10 / 1 | 实际赢家只有 K1536，先 `65536×16384×1536`，21888 WG | 可选同 M/N K384、K16384 支持域机制对照；`gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel<...large_output_traits...>` | 全 K scale upfront、旧 end-wait；C64位/2GiB 输出域，必须先量 C 物理写流量 |

9020 的 76 个赢家分成七个真实分支：runtime192×256/scale128 为31个，fixed384为7个，fixed768为5个，fixed1536为9个，fixed3072为5个，fixed7168为16个，高M窄N128×128/scale64为3个。除首对外，`8192×768×7168` 是窄 N 分支代表；`192×65536×1536` 与 `1344×16384×1536` 分别代表 runtime128 与 fixed32；`6144×2048×7168` 代表 fixed64。逐项 branch 条件和实际 symbol 在 JSON 中，不能只按 parent 默认 tile 解释。

三者均存在真实静态 scale producer 串行，但与 9021 的起步和分工不同。9020 fixed384 在 K0 首 load `0x70f0` 后，SFA load/wait/store 为 `0x72e0/0x72e8/0x7390`，然后两条 SFB byte load `0x73e8/0x73f0`，wait/store `0x7400/0x7414`，K1 后 partial wait/barrier `0x7598/0x759c`。SFB 两半本来就一起请求；候选应针对 SFA→SFB 依赖发布，不能重复“提前 K0”这个已有优化。runtime128 还有多个 SFA pass/panel 条件块，必须由动态 ATT 确认执行路径。

9022 的两次 SFA pass 各自 load/wait/store：`0x1c48/0x1c50/0x1cf8`、`0x1df0/0x1df8/0x1ea0`，之后 SFB `0x1f08/0x1f18/0x1f1c`，K0 到 `0x2038` 才发出。可提出保持原 scale-first matrix prologue 的 raw issue/publish 单机制；必须保存两次 SFA guard、尾部 `0x7f`、panel32 和全部同步。Oct7 全局 K0→scale→K1 曾覆盖159赢家，但20个 K16384 geomean −0.468%，16个 negative median、9个5/5慢，所以已拒绝；这不是新 issue/publish 已失败或已获准的证据。`800×7168×16384` 本身是旧5/5慢信号，优先纳入定位。

9022 有第五 M-repeat scale。直接每坐标两 u32 会使 SFA5120→8192B，LDS81184→84256B，越过80KiB预算。Oct8 9021 packed SFA十项全慢，同样不能自动扩展。除非新 consumer ATT/counter 显示明确损失，先不做 packed 布局；两槽 consumer barrier 涉及 slot 退休，不能因局部间隙删除。

9030 的三次 SFA pass 分别为 `0x1e10/0x1e18/0x1ec0`、`0x1f84/0x1f8c/0x2034`、`0x20f4/0x20fc/0x21a4`，SFB 两 byte `0x21fc/0x2204` 后等待并 packed store，K0 到 `0x2340` 才发出。其整128 tile scale24576B+512B一次预载，没有稳态 refill；旧 `advance_tile` 在开头发 K+2，末尾 full wait/barrier。Oct7提出但未试的独立方向是仅迁移9020式 midpoint publish/retire，保留C64位基址、tile resource、bounds、所有地址与输出，起步顺序另行评估。先看 interior ATT 是否真受 next-tile LDS/end-wait限制，以及2–8GiB C物理写服务是否占主要时间，不能把一个 WG startup 优势推成整核收益。

9030 接受 M%64、N%256、K%128、K≤16384，A/B 与 tile-local bytes≤INT32_MAX，且2MN>INT32_MAX。`65536×16384` 输出恰为2147483648B，因此同 M/N 的 K384/K16384 可作为合法对照，但不是历史赢家。十个真实赢家均K1536、输出2–8GiB；51地址Event池对8GiB输出会超过显存，root应先预算 buffers，不能机械展开全部计时。

按[最新合并诊断说明](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)的路径采集：先无profiler时间/工作量/物理流量，再GL2/SoC、TCP/TA/UTCL1、wave wait/issue，最后SPI与资源。JSON列出具体gfx950候选counter pass。优先核对 SQ_WAVES/F8工作量和TCC32B读写请求；同pass LEVEL/REQ与TCP latency/REQ说明接口累计等待，不能当纯HBM返回延迟。wave quad-cycle wait/issue比例与墙钟损失分开；LDS issue、bank count和返回等待分开。

ATT 分开 kernargs、K0/scale/K1/publication、首MFMA前LDS准备、普通tile边界、仅9022的32tile refill、末轮MFMA/BF16/输出/endpgm。争取9020/9030八波、9022四波完整拼接时间线，并记录CU/SIMD和采集范围；waves及重复tile相关。gfx9成功issue=time+stall，duration已含stall；wait依赖只建立队列成员关系。EXEC0下指令事件不代表有效lane访问。

沿用Oct8已确认的时钟限制：rawCSV GRBM是八值求和，不是max；JSON max仍与kernel timestamp不一致。保留raw instances/维度，不猜XCC序号，不套固定倍率，不给绝对MFMA利用率或动态occupancy。历史MI300模型常量不得直接代入MI355X。静态串行是候选机会，关键路径和完整调用收益留给root统一GPU诊断及受控Event决定。
