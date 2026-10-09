本轮将9个MXFP8 B-preshuffle配置的global split-K改为启动参数。默认调优集合从92个
静态配置缩为 **89个**；完整注册仍为 **105个ID**，其中 **16个历史兼容/内部ID**
保持可调用。查看[89个调优配置](registry89.csv)、[105个注册ID](registry105.csv)及
[目录导出状态](catalog_summary.json)。清单是文档导出，运行时仍以
[`opus_gemm_common.py`](../../csrc/opus_gemm/opus_gemm_common.py)和
[`opus_gemm_bpreshuffle_variants.py`](../../csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py)为准。

9个runtime配置为register 92310/92311/92320/92321/92330/92340，以及fine
92410/92420/92430。fine 92411/92421/92431分别保留相同geometry的历史固定split2
路径，移出默认调优。register六个ID的原global split都是4；它们的tile/local WaveK不同，
因此仍是六个静态配置。

| 配置 | tile M×N | local WaveK | `split_k=0`历史默认 |
|---|---|---:|---:|
| 92310 | 16×16 | 1 | 4 |
| 92311 | 16×16 | 2 | 4 |
| 92320 | 16×32 | 1 | 4 |
| 92321 | 16×32 | 2 | 4 |
| 92330 | 32×32 | 1 | 4 |
| 92340 | 32×64 | 1 | 4 |
| 92410 | 48×64 | 1 | 1 |
| 92420 | 64×128 | 1 | 1 |
| 92430 | 96×128 | 1 | 1 |

框架根据M/N/K和请求值生成一份启动计划，统一决定producer grid、dynamic LDS、
FP32 workspace容量与reducer的split参数。tile、MFMA布局、寄存器队列、local
WaveK与reducer vector/block geometry仍在编译期确定。global K128 tiles均衡分配；
register每个global partition再在静态local WaveK之间均衡分配。

| runtime请求值 | 启动语义 |
|---|---|
| `1..16` | literal global partition数；必须不超过`K/128`，并通过shape/byte/workspace检查 |
| `0` | 保留历史默认fine1/register4；短K register可包含空global/local partition |
| `-1` | 可选M/N/K与调用方CU数的grid heuristic；这是未实测的策略，不代表最快参数 |

split1直接写BF16；split>1每个partition完成FP32 local-wave reduction后写一份FP32
workspace，再由shared runtime reducer完成一次BF16转换。fine只编译direct/partial
两种输出模式；register每个静态geometry一个producer；split数不进入producer traits或
模板实例数量。旧96-byte kargs保持原样，runtime入口使用派生类型追加`int split_k`；
包含对齐后的runtime kargs为104 bytes。

tuner保留`(kid, splitK)`搜索，枚举runtime候选的全部合法positive split并记录实际正数，
供精确回放。清理split-only ID不减少合法启动参数的搜索空间。其余固定split候选继续用
CSV `splitK=0`；兼容fine奇数ID继续调用历史固定split2路径。

runtime注册、调用、调优和离线集成验证已完成；GPU数值与性能仍未运行：

| 验证项 | 状态与证据 |
|---|---|
| 5个新C++ headers；旧42个bpreshuffle headers字节保持 | 通过，[冻结收据](headers/receipt.json) |
| 14个runtime入口离线device编译及host syntax | 通过；scratch/VGPRspill/SGPRspill均0，[ELF审计](headers/elf_audit.json) |
| actual C++ partition/layout/LDS | 通过；2064 global cases、17544 nested WaveK2 cases、6192 fine LDS cases，[CPU输出](headers/isolated_cpu_run.log) |
| Python注册、计划、调优/回放、dispatch/manifest | 31项通过；另6项mixed compiler检查通过，[CPU检查](cpu_python_audit.json) |
| 完整105个generated HIP TU | 全部fresh编译通过，90 baseline23、15 pin24，[编译收据](full_build/build_receipt.json) |
| fused host、实际router、完整pybind TU | 编译通过；新绑定符号与router定义匹配，[构建范围](full_build/verification_scope.json) |
| router/fused host/105对象 `--no-undefined`链接 | 通过；127个host launch引用闭合，12个runtime producer+1个runtime reducer均在fatbin，[链接与ELF检查](full_build/final_receipt.json) |
| runtime scratch/spill | 9个配置所有producer/reducer均0；旧9023一个既有specialization有SGPR spill30，详见完整资源记录 |
| 96个非runtime配置保持 | 生成源码、kernel指令和metadata逐项一致；整份ELF含路径等元数据有字节差，[对照](full_build/binding_and_nonruntime_comparison.json) |
| GPU数值与性能 | 未运行；用户停止GPU测试的要求继续有效 |

一次初始CPU检查使用HIP driver直接链接并执行，自动加载了libamdhip64；该事件已在
[headers说明](headers/README.md)和冻结收据披露。最终CPU检查改为compile-only HIP
parser加plain C++ link，执行前通过readelf确认无HIP/HSA/CUDA/Torch依赖。本轮没有
新GPU数值或性能结论，旧745选型与性能数据不能作为runtime split实现的验证结果。

[上一阶段92候选报告](../opus_register92_20261009/README.md)及其manifest保持冻结。
它的生成代码、字段和105对象链接描述的是当时固定split路径；本次新目录分别记录runtime
改动。目录中的CSV把旧generated SHA放入`historical_generated_*`字段，当前generated
SHA已按本次实际生成结果填写，可由codegen/all105_metadata.json逐项复核。

标量启动示例见[launch_examples.json](launch_examples.json)。例如`M=16,N=128,K=384`、
`kid=92310`，同一个注册ID可调优`splitK=1/2/3`；split3的grid为`(8,1,3)`，
workspace为`3*16*128`个FP32元素。计算均衡分区和改变启动参数不增加split专属traits。

最终汇总见[verification.json](verification.json)与[artifact_manifest.json](artifact_manifest.json)。
新报告记录本次源码与产物哈希；旧92报告501个文件、此前全系列优化报告838个文件及正式CSV/库均核验未变。
