# 9040 之后的小矩阵与 fine 系列有限优化审查

本轮已收口：fixed9062 initial handoff、9046/9055 ring issue次序和runtime9051 scale-first候选均拒绝，small原selected保留。33个实际config各自数值门、7个自身ATT和4config/21个actualwinner candidate Event已审查；18个public parent可追溯，未测config的keep不声明性能增益。没有新增small source/deviceentry采用，也没有待执行候选。最终分项结果及证据范围见文末。

基线为 Oct8 已应用 selected。9021本轮已完成；9042/9053/9054三个runtime B-scale复用alias保持已采用状态。本文件由CPU后处理生成；所有GPU实验由root执行。以下原始诊断建议和控制点保留为准备时点记录，当前执行和收口状态见JSON及文末。

278 个历史实际赢家覆盖 18 个公开 parent、30 个 actual ID，但对应 **33 个实际 producer symbol/config**，另有 5 个 reducer。9060 的 runtime/K3072/K7168，以及 9063 的 M80 runtime split4/fixed K16384，不能仅按 ID 合并。历史耗时属于 9 月 30 日选型上下文；全部 symbol 已经经过 Oct7 selected 映射并匹配 Oct8 正式模块，不能把旧耗时当作 Oct8 测量。

先建议 6 个 ATT 代表、4 个有针对性的 counter pass；其余只在首轮证据提出明确问题时追加。

| 顺序 | parent → actual | shape M,N,K | 实际配置 | 首轮 counter | 需要定位的 PC/阶段 |
| --- | --- | --- | --- | --- | --- |
| 1 | 9046 → 9046 | 256,7168,16384 | 64×128, 8 wave, S6/C2/split1; 224 WG, LDS 160384 B | sq_ea | 起步 scale/matrix wait；普通两 tile group 的 VMEM/LDS 返回与 barrier 到达差 |
| 2 | 9055 → 9055 | 64,7168,7168 | 32×64, 4 wave, S12/C4/split1; 224 WG, LDS 153912 B | lds_issue | 早期 scale prefix 与 12-tile 起步突发；四 tile 同步、group 内 MFMA 间隙和未来请求 |
| 3 | 9042 → 9071 | 192,768,7168 | 16×48, 4 wave, Q4/wave-K4; 192 WG, LDS 12288 B | ta_lds | 每 wave 14 tile / Q4 / tail2；矩阵与 byte scale issue/wait，以及 N128 scale 地址 |
| 4 | 9062 → 9062 | 144,7168,16384 | 80×128, 4 wave, S4/C1/split2; 224 WG, LDS 115008 B | sq_ea | M80 每 wave 6/7 matrix 请求差；每 split 64 tile 的 barrier 与操作数等待 |
| 5 | 9040 → 9040 | 8,7168,384 | 16×32, 1 wave, Q6/wave-K1; 224 WG, LDS 0 B | 先 ATT | 首 MFMA 前的参数、矩阵与 scale 突发；3 tile / Q6 的 VMEM tail 与直接输出 |
| 6 | 9047 → 9047 | 64,7168,384 | 32×64, 4 wave, S4/C1/split1; 224 WG, LDS 38016 B | 先 ATT | 初始 matrix/scale VMEM 与 barrier；register scale 消费和 matrix LDS 返回 |

9046 的长 K 代表覆盖 8-wave/S6/C2、224 WG、160384 B 动态 LDS，优先区分 matrix group 返回、消费者 LDS 依赖与 barrier 到达差。9055 的 S12/C4 是另一种深 ring：早期 scale prefix、12-tile 起步突发、四 tile 同步和 drain 必须分别核算。此时 LDS 预算和 WG 数只是软件供给事实，不能据此宣布 occupancy 或外部带宽已限制。

9071 使用 parent9042 的 `192×768×7168`，保留 N48/4-wave/OUTPUT4。每个 wave 有 14 个 K128 tile、prefetch4、tail2；需要分别查看矩阵与 byte scale issue/wait、FP32 partial/barrier/分布式归约，以及最后 BF16 输出。9062 的 `144×7168×16384` 是 M80/fixed K/split2，6/7 条矩阵请求的 wave 差异需和 matching reducer 分开看；完整时间必须包含 workspace 写读、reducer 和设备调度空隙。

9040 的 K384 只有 3 tile，小于 queue6，选 `8×7168×384` 是当前实际赢家，M padding 比例 1/2；旧全局 reuse 拒绝 case `16×7168×768` 只是限制 blanket opt-in 的证据，不能混称当前赢家。9047 的 `64×7168×384` 是 full-M、register-scales、短 K 3-tile 路径，适合区分 matrix LDS 返回和寄存器 scale 消费。两者先看 ATT，省去相同四 counter 套餐。

旧 9051 `1×65536×1536` 的四组 clean counter 与当前 entry 指令身份保持相同，可复用 GL2 miss、LFIFO、translation in-flight 的线索。若首轮后决定优先此路径，追加 exact-PC ATT；只有出现相应返回/issue 等待才补 latency/UTCL1，不重复完整四组。

已验证方向的限制：

| 方向 | 已有结果 | 本轮限制 |
| --- | --- | --- |
| 全局 register ReuseBScale 默认 | 9040 5/5 轮慢 27.36%；9051/9052 无稳定收益 | 默认 false 保留；不因少 load/VGPR 扩大开启范围 |
| 三个 runtime reuse alias | 6/6 实际赢家，均 5/5 更快；非赢家 M8/N16384/K1536 两项慢 0.81%/0.64% | 9042/9053/9054 true 保留，fixed9070/9072 true、9071/9073 false 保留 |
| fine_wait 全局 helper+3 wait site | 九例中五例每轮慢，约 1.38–1.94%；其余接近零/混合 | 不复制该修改；当前 conservative wait 没有被证明错误 |
| fine_n64 | M144/M160 fixedK16384 split2 均 0/5 更快，慢 14.04%/13.49% | BN128 保留；污染记录仍排除 |
| fixed_n32 | 9071/9073 N48→N32 慢 45.20%/46.54% | N48 保留；不能从退步认定某个硬件接口根因 |

分层测量先核对 useful/executed MFMA 工作量、WG/wave/grid 和 producer/reducer 真实入口，再沿 SQ 等待 → TA/TCP/UTCL1 请求回压 → GL2/EA 服务定位。`sq_ea` 给 9046 与 9062；`lds_issue` 给 9055；`ta_lds` 给 9071。下一层 L2/translation/请求驻留只由首轮证据触发。不同 pass 不能合成同一 dispatch 的墙钟分解；非 windowed gate/busy/latency 保留原始范围。raw GRBM CSV 求和错误与窗口不一致仍不支持绝对 MFMA 利用率/动态 occupancy/cycle→ns。

条件追加最多六个已列出的代表：9051 external ATT、9046 同 M/N 的 K1024、9055 同 M/N 的 K3072、fine9060 fixed S1、内部9056 S8/C2、balanced M96 实际9066 split4。每项的触发条件、精确 symbol 和 reducer 均见 JSON；未触发不执行。

所有 33 producer 配置的覆盖表如下。精确 mangled symbol、full metadata、指令/描述符 SHA、全部 278 shape 与条件保存在 [small_family_review.json](small_family_review.json)。

| actual ID | family | 赢家数 | 公开 parent | 选中 K | tile / wave / split / queue | Oct8 VGPR / 指令 B |
| --- | --- | ---: | --- | --- | --- | --- |
| 9040 | register_queue | 11 | 9040 | 384,768,1536 | 16×32/1w/split1/Q6 | 190 / 4944 |
| 9041 | register_queue | 1 | 9041 | 7168 | 16×16/8w/split1/Q2 | 49 / 1724 |
| 9042 | register_queue | 1 | 9042 | 3072 | 32×32/4w/split1/Q3 | 142 / 3892 |
| 9043 | small_lds_ring | 2 | 9043 | 1536,7168 | 32×64/4w/split1/S8/C2 | 48 / 7844 |
| 9044 | small_lds_ring | 9 | 9044 | 3072,7168 | 64×64/4w/split1/S4/C1 | 58 / 9840 |
| 9045 | small_lds_ring | 17 | 9045 | 384,3072,7168 | 96×64/4w/split1/S4/C1 | 82 / 12796 |
| 9046 | small_lds_ring | 32 | 9046 | 1024,1536,3072,7168,16384 | 64×128/8w/split1/S6/C2 | 80 / 8044 |
| 9047 | small_lds_ring | 27 | 9047 | 384,768,1024 | 32×64/4w/split1/S4/C1 | 62 / 3588 |
| 9049 | small_lds_ring | 6 | 9049 | 768,1536 | 32×128/4w/split1/S4/C2 | 114 / 6328 |
| 9050 | register_queue | 16 | 9041 | 7168,16384 | 16×16/8w/split1/Q3 | 66 / 2396 |
| 9051 | register_queue | 17 | 9051 | 768,1024,1536,3072,16384 | 16×32/4w/split1/Q3 | 104 / 3012 |
| 9052 | register_queue | 7 | 9052 | 1536,3072,16384 | 16×32/8w/split1/Q2 | 75 / 2508 |
| 9053 | register_queue | 2 | 9053 | 1024,16384 | 32×32/8w/split1/Q2 | 105 / 3372 |
| 9054 | register_queue | 3 | 9054 | 768,1024 | 32×64/4w/split1/Q2 | 157 / 4292 |
| 9055 | small_lds_ring | 17 | 9055 | 3072,7168 | 32×64/4w/split1/S12/C4 | 56 / 10356 |
| 9056 | small_lds_ring | 17 | 9044 | 1536,3072,7168 | 64×64/4w/split1/S8/C2 | 66 / 12700 |
| 9060 | fine_lds_split | 2 | 9060 | 1536 | 80×128/4w/split1/S4/C1 | 88 / 15888 |
| 9060 | fine_lds_split | 2 | 9060 | 3072 | 80×128/4w/split1/S4/C1 | 112 / 8660 |
| 9060 | fine_lds_split | 5 | 9060 | 7168 | 80×128/4w/split1/S4/C1 | 112 / 10776 |
| 9061 | fine_lds_split | 19 | 9061 | 1536,3072,7168,16384 | 96×128/8w/split1/S4/C1 | 70 / 11664 |
| 9062 | fine_lds_split | 2 | 9062 | 16384 | 80×128/4w/split2/S4/C1 | 86 / 11052 |
| 9063 | fine_lds_split | 7 | 9063 | 7168 | 80×128/4w/split4/S4/C1 | 88 / 11440 |
| 9063 | fine_lds_split | 2 | 9063 | 16384 | 80×128/4w/split4/S4/C1 | 86 / 8988 |
| 9064 | fine_lds_split | 4 | 9062 | 7168,16384 | 96×128/8w/split2/S4/C1 | 66 / 9380 |
| 9065 | fine_lds_split | 8 | 9063 | 7168 | 96×128/8w/split4/S4/C1 | 66 / 9380 |
| 9066 | fine_lds_split | 1 | 9063 | 16384 | 96×128/4w/split4/S4/C1 | 110 / 12488 |
| 9067 | fine_lds_split | 1 | 9063 | 16384 | 128×128/4w/split4/S4/C1 | 130 / 14824 |
| 9068 | fine_lds_split | 1 | 9063 | 16384 | 112×128/4w/split4/S4/C1 | 106 / 13992 |
| 9069 | fine_lds_split | 1 | 9063 | 16384 | 48×128/4w/split4/S4/C1 | 68 / 8884 |
| 9070 | register_queue | 17 | 9052 | 7168 | 16×32/8w/split1/Q3 | 96 / 2172 |
| 9071 | register_queue | 7 | 9042 | 7168 | 16×48/4w/split1/Q4 | 168 / 3956 |
| 9072 | register_queue | 7 | 9053 | 7168 | 32×32/4w/split1/Q4 | 166 / 4392 |
| 9073 | register_queue | 7 | 9042 | 7168 | 32×48/4w/split1/Q3 | 185 / 4200 |

首轮代表的 exact symbols：

- `lds9046_long`：`_Z38gemm_a8w8_mxfp8_scale_small_lds_kernelI33opus_gemm_small_lds_traits_gfx950ILi64ELi128ELi4ELi2ELi6ELi2ELi2ELb0ELb0ELb0ELb0ELb0ELi1ELi4ELi128ELi0ELi0ELb0EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950`；FUNC SHA `0921cff0c791a2824087e9bb7e5b4206fcc27f7b2f53a3f50599139282e22d9c`。
- `lds9055_deep_cluster`：`_Z38gemm_a8w8_mxfp8_scale_small_lds_kernelI33opus_gemm_small_lds_traits_gfx950ILi32ELi64ELi1ELi4ELi12ELi4ELi2ELb0ELb0ELb1ELb0ELb0ELi1ELi4ELi128ELi0ELi0ELb0EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950`；FUNC SHA `1b535a3aea56a57412eeb81acd67101d357e2b1b23c8893ee946e4759ca1d3f3`。
- `register9071_fixed_n48`：`_Z43gemm_a8w8_mxfp8_scale_small_register_kernelI38opus_gemm_small_register_traits_gfx950ILi16ELi48ELi1ELi1ELi4ELi4ELi4ELi3ELi7168ELb1ELb0EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950`；FUNC SHA `d056f09b8256bdbe2f093a9467ee6908a5c0e25d1f2fa2489f101e95c95abb82`。
- `fine9062_fixed_split2`：`_Z38gemm_a8w8_mxfp8_scale_small_lds_kernelI48opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950ILi80ELi1ELi4ELi4ELi1ELi2ELi4ELi128ELi0ELi16384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950`；FUNC SHA `f5de3a4b595c29d1bf8da013ddeb30376ae9964ca2892c20139f05f28972954b`。
  匹配 reducer：`_Z43opus_gemm_mxscale_bpreshuffle_reduce_kernelILi2ELi4ELi128EEvPKfPDF16bi`；FUNC SHA `1b6aac04ed18da418f95eeffb7b14c2c53406404dd5c9bd85391e310a3960c4a`。
- `register9040_short`：`_Z43gemm_a8w8_mxfp8_scale_small_register_kernelI38opus_gemm_small_register_traits_gfx950ILi16ELi32ELi1ELi1ELi6ELi1ELi3ELi3ELi0ELb0ELb0EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950`；FUNC SHA `bf2be9d964afcf07730626f2c3b4df68ba34ca92da83829856098f214154cb28`。
- `lds9047_short_register_scales`：`_Z38gemm_a8w8_mxfp8_scale_small_lds_kernelI33opus_gemm_small_lds_traits_gfx950ILi32ELi64ELi2ELi2ELi4ELi1ELi2ELb1ELb0ELb0ELb0ELb0ELi1ELi4ELi128ELi0ELi0ELb0EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950`；FUNC SHA `95b67c88caa02e3dcd46c223b6a79aed88142b6423d7393a2a61d2ad6d83d72e`。

正式 9040+ parent 共 40 个 device entry，其中 35 producer、5 reducer；两个 producer 没有历史 winner symbol 交集，分别保留为兼容/runtime 路径，不为凑齐 ID 盲跑。完整吞吐改动以 Oct8 selected 为 baseline，clean 物理卡、共享地址 AB/BA 完整 GPU Event 与 signed 数值/重复/guard 决定是否采用；ATT 仅用于局部机制定位。

依据：[Compute 文档](../opus_bound_analysis_20261007/COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md)、[Memory 文档](../opus_bound_analysis_20261007/MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md)、[最新带宽总结](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md)、[Oct8 正式身份](../opus_resume_20261008/formal_selected/identity_audit.json)。各文件 SHA 记录在 JSON，旧实验与 manifest 不作改写。

主任务随后选择了七个实际 ATT 代表（diagnostics/plan.json 的 target14–20）：上表六项，加 9051 external。原推荐保留作为准备时点；执行与 CPU 后处理另存 small_att_analysis.json。33 个 config 现各有 processing 状态、同指令历史证据复用范围和后续触发条件；18 个 public parent 可追溯到全部实际 config。采用三 alias 复用 Oct7 exact candidate Event，9051 复用原四组 counter，所有配置现均有明确keep或候选reject收口决定，不把共享family代表视作其性能实测。

<!-- OCT8_VALID_ATT_UPDATE -->

本轮有限small系列已收口：fine_startup、ring_overlap两个exacttype和runtime9051 scale-first均拒绝，原selected保留。没有新增small source或deviceentry采用，没有待跑small候选。已准备但未build的9071草稿停止；N48/Q4/OUTPUT4/falseBscaleReuse保持原策略，自己的旧N32 Event、局部ATT和数值门分别列为证据。

七个有限代表的精确ATT后处理见 [small_att_analysis.json](small_att_analysis.json)。9055原CU0 `code=null`/零wave排除，独立CU1补采的四个完整wave与正式symbol/FUNC/fullmetadata/descriptor、owner/物理卡、stats/ISA吻合。9051的32个完整wave跨多个WG，同PC/occurrence聚合barrier跨度不能推断WG内到达差。所有shader clocks只用于该CU局部机制，不组成全GPU墙钟百分比。

| 本轮候选 | 自身完整Event范围 | 结果 | 决定 |
| --- | --- | --- | --- |
| fixed9062 initial matrix-before-scale | M144/M160两个actualwinner，含split2 matching reducer | speedup0.996596/0.998854，0/5和1/5更快 | 拒绝，保留原initial handoff |
| 9046 steady ring future issue-before-read | [256,7168,16384]一个actualwinner | speedup0.980340，0/5更快 | 拒绝该exacttype候选，保留原ring |
| 9055 steady ring future issue-before-read | [64,7168,7168]一个actualwinner | speedup0.991871，0/5更快 | 拒绝该exacttype候选，保留原ring |
| runtime9051 same-tile scale-first issue | 全部17个冻结actualwinner | 8项median正、8项退、1项平；仅3/17全部5轮更快、7/17两个order组正 | 全exacttype统一拒绝，保留原issue顺序 |

依据分别为 [fine_startup/results_analysis.json](fine_startup/results_analysis.json)、[ring_overlap/results_analysis.json](ring_overlap/results_analysis.json) 和 [register_issue_order/coverage_analysis.json](register_issue_order/coverage_analysis.json)。各候选40entry基线精确匹配Oct8；fine/register候选39未选entry不变，ring候选38未选entry不变。全部自身signed8repeat/reference/guard及每shape51共享池5ABBA完整Event后的重复/reference/guard通过。9051早期两个正代表保留为screen证据，但全17覆盖的mixed结果决定统一拒绝；没有按shape新增阈值或重复coverage寻找正结果。

[small_config_gate_audit.json](small_config_gate_audit.json) 验证33/33 small实际config各自一个actualwinner的signed2repeat/reference/output/workspaceguard，正式symbol和指令SHA均匹配；all44rootgate ownclaim干净。sealedplan中9042/9053/9054的descriptive trait label旧false已在audit校正，实际mangledsymbol和hash都指向当前true alias，执行身份正确。已采用三runtimealias各自六个历史actualwinner的正Event和Oct8selected身份复用仍保留；fixed9070/9072原true策略属于自身source/identity事实，没有冒称本轮新采用性能。

33config/18parent/278历史winner完整映射保留在 [small_family_review.json](small_family_review.json)。每config现在分开记录自身Oct8 candidate Event形状、自身局部ATT、自身历史exactconfig Event、共享机制解释和自身数值门；每parent映射相同字段。当前candidate性能实测为4config/21个actualwinner形状，局部ATT为7config，其余keep依靠各自selectedtype/source/identity/numerical与存在时的自身历史实验。共享ring/register/fine机制解释以及别的config的reject都明确不算该config性能；没有33config全性能实测声明。

| public parent | actual ID / selected fixedK | 自身本轮candidate Event形状数 | 自身ATT配置数 | 自身历史exactconfig Event行数 | 决定 |
| --- | --- | ---: | ---: | ---: | --- |
| 9040 | 9040 | 0 | 1 | 1 | keep_current |
| 9041 | 9041, 9050 | 0 | 0 | 0 | keep_current |
| 9042 | 9042, 9071, 9073 | 0 | 1 | 3 | keep_current, retain_prior_accepted_runtime_reuse_alias |
| 9043 | 9043 | 0 | 0 | 0 | keep_current |
| 9044 | 9044, 9056 | 0 | 0 | 0 | keep_current |
| 9045 | 9045 | 0 | 0 | 0 | keep_current |
| 9046 | 9046 | 1 | 1 | 0 | reject_finite_candidate_keep_current |
| 9047 | 9047 | 0 | 1 | 0 | keep_current |
| 9049 | 9049 | 0 | 0 | 0 | keep_current |
| 9051 | 9051 | 17 | 1 | 1 | reject_finite_candidate_keep_current |
| 9052 | 9052, 9070 | 0 | 0 | 1 | keep_current |
| 9053 | 9053, 9072 | 0 | 0 | 2 | keep_current, retain_prior_accepted_runtime_reuse_alias |
| 9054 | 9054 | 0 | 0 | 3 | retain_prior_accepted_runtime_reuse_alias |
| 9055 | 9055 | 1 | 1 | 0 | reject_finite_candidate_keep_current |
| 9060 | 9060/K0, 9060/K3072, 9060/K7168 | 0 | 0 | 2 | keep_current |
| 9061 | 9061/K0 | 0 | 0 | 1 | keep_current |
| 9062 | 9062/K16384, 9064/K0 | 2 | 1 | 4 | keep_current, reject_finite_candidate_keep_current |
| 9063 | 9063/K0, 9063/K16384, 9065/K0, 9066/K0, 9067/K0, 9068/K0, 9069/K0 | 0 | 0 | 3 | keep_current |
