# Clang 23 + pin-AGPR LLVM 全后端重测

完整 **745/745** 个 gfx950/256-CU shape 在 8 张物理 GPU 上测量完成。
全部 28 个 OPUS 候选保留，**13,027/13,027 个合法配置均有效，errRatio=0**。
CK/CKTile/ASM 候选重新完整 tune，总计 79,504 条原始记录。
同一 shape 的所有后端候选在同一 GPU 上依次比较；8 个进程都核对实际加载库，
KFD/AMD-SMI监控未发现非所属进程干扰。

| 最优后端 | Shape 数 |
| --- | ---: |
| OPUS | 693 |
| CK | 8 |
| CKTILE | 10 |
| ASM | 34 |

OPUS 相对本轮 CK/CKTile/ASM 最快有效者，几何平均加速 **1.139670×**，
693 项更快、52 项更慢。
这是本轮同 shape/同 GPU 数据的比较，公共shuffle和reference不计时；
OPUS使用FP8与native E8M0 scale，外部后端保留原tuner的FP32 scale和输入分布。
小差距受单轮筛选噪声影响；不把历史时间混入新表选型。

## 编译器记录

CK、CKTile、ASM主机封装及24个不使用pin AGPR的OPUS候选使用最小
Clang23 `46fcb339fb61119b337f973c7ca9e710a319fdd0`，
路径 `/opt/rocm-llvm23-46fcb339/bin`。
只有9000/9001/9010/9011使用pin LLVM Clang24
`49c41889681640665400cb01c9fbb4c0a024cde4`，
路径 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin`。
ASM设备kernel为既有预编译`.co`。同一`/opt/rocm` SDK；
pin24仅使用ROCm7.0的clang20 resource headers，Clang23使用自己的resource headers。
原上游CSV的tune编译器仍未找到可靠收据，本轮创建新的可复查基线。

28个OPUS对象逐个核对编译器；4个pin候选的设备机器码与上轮完全一致。
88个外部/core对象为Clang23；6个ASM设备二进制身份已保存。
全28候选signed/cancellation校验加5个尾部样例通过，共66次实际输出校验；
60个CPU/JIT检查及10个tuner编译环境检查通过。

## 与上轮重测的描述性比较

此前CK/CKTile/ASM封装用Clang20，OPUS全部用pin24。
下列“上轮时间/新时间”的几何平均同时受GPU分片、地址池、时间漂移及编译器影响，
不能将所有差异只归因于编译器。

| 比较 | 对应Shape | 旧/新GM加速 | 新延迟GM变化 |
| --- | ---: | ---: | ---: |
| CK最优 | 740 | 1.000958× | -0.096% |
| CKTILE最优 | 735 | 1.508080× | -33.691% |
| ASM最优 | 745 | 0.996151× | +0.386% |

外部后端最优整体旧/新GM 1.084254×；
OPUS最优旧/新GM 0.984529×；
全后端最优旧/新GM 0.990033×。
各shape明细、原候选对照及无效记录均保留。
OPUS最优的延迟GM相对上轮为 **+1.571%**。
同候选对照中，四个机器码不变的pin24候选共1312项，延迟GM-0.227%；
其余改用Clang23的候选共11715项，延迟GM+0.870%。
该跨轮数据不能替代同卡同池编译器A/B，详见`opus_pin_vs_unpin_historical_change.json`。
CKTile已摆脱此前Clang20的严重spill退化；样例M4096 N2048 K7168 ID27
初次profiler数值检查80.0586µs，本轮全量profile为78.2885µs，旧Clang20约3.2ms。
该样例不同Event/profiler时间不混用；全量选型以本轮profile为准。

## 与上游保存候选对应

| 上游后端 | 保存Shape | 同执行已测 | 有效 | 新延迟相对上游GM变化 |
| --- | ---: | ---: | ---: | ---: |
| ASM | 133 | 133 | 133 | -0.189% |
| CK | 409 | 409 | 409 | -0.057% |
| CKTILE | 178 | 178 | 178 | -5.680% |
| TRITON | 25 | 0 | 0 | 未测 |

ASM按kernelName/splitK匹配，数字ID为枚举序号。CKTile同时匹配数字ID，
因为部分候选名字相同。CK B-preshuffle封装接收splitK并计算KBatch，
实际调用未使用它；因此原4行splitK标签不一致不是kernel执行缺失，
全部409个上游CK执行都有对应记录。严格CSV参数匹配仍单独保存。
上游25个Triton shape未在本次候选集合中测量。
这些上游历史时间的对照不证明已复现原工具链或原测试环境。

## 输出与核验

- `tuned_all.csv`：本轮全部后端最优745行；`tuned_all_config.csv`将bw从GB/s转为TB/s，已保存到仓库模型配置CSV。
- `tuned_opus.csv`：OPUS28内部最优745行。
- `tuned_external.csv`：CK/CKTile/ASM最快有效者745行；另有各后端最优CSV。
- `profile.csv`、`batches/*`：完整raw、逐卡进度、合法候选清单。
- `shape_comparison.csv`、`old_new_same_call_comparison.csv`：同轮和历史对照。
- `upstream_same_execution_comparison.csv`、`upstream_same_call_comparison.csv`：上游实际执行/CSV参数对照。
- `compiler_manifest.json`、`build_manifest.json`、`*_commands.txt`、`*_per_object_compiler_audit.json`：构建记录。
- `pin_machine_identity.json`、`asm_device_binary_hashes.json`：设备指令和二进制身份。
- `plan.json`、`queue_eight.json`、`claim_eight.jsonl`、`runtime_mapped_libraries_audit.json`：执行和实际库加载。
- `summary.json`：完整统计；`previous_published_all_backend.csv`保留此前已发布745行。

测量warmup5/iters51、原tuner的torch profiler；大输出地址池上限8并按空闲显存限额，
reference和误差检查按256行分块，表达式不变。OPUS完整FP32/BF16累加范围零outlier；
外部后端保留原误差门槛。无效项按后端统计为 `{'cktile': 874, 'ck': 498}`，全部留在raw中。
本次未枚举Triton。tune时间不能直接作为compute/memory bandwidth/memory latency/dispatch分类证据。
旧 `reports/opus_retune28_20261008/` 原始记录保持不变。
