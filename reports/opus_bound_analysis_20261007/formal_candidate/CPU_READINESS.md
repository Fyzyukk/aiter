# 隔离正式 Opus candidate 的 CPU 构建与身份核对

状态：CPU正式身份核对及15项正式GPU API smoke均通过，仍为`pending_no_adoption`。原分支生产源码未修改，未提交或推送；正式candidate已在空闲GPU窗口实际加载并执行，性能采用仍待当前赢家Event结论与最终保留范围核对。

## 1. 源码范围

Detached worktree：`/root/workspace/opus-bound-candidate-20261007`，基于 `b152ab834e68f8fde18adbd7b200c587770b9fb5`。

只修改三个文件，共21行新增、9行删除：

- 9021的128×128 prologue，与私有 `compute_prologue/candidate` 完全一致。
- 9022的160×128 prologue，与该私有candidate完全一致。
- small register traits中9042、9053、9054对应三个alias显式 `ReuseBScale=true`；模板默认值仍为false，其他alias及生成器条件保持原值。

源码哈希和精确差异见 [source_manifest.json](source_manifest.json) 与 [source_changes.diff](source_changes.diff)。没有把未采用的 narrow unroll、fixed_n32或fine_wait带入此候选。

## 2. 构建隔离

正式candidate JIT目录：`reports/opus_bound_analysis_20261007/jit_formal_candidate`。新模块：

```text
module_deepgemm_opus.so
SHA256 51d4b3d7dd033cc99d391d87e89c25e8afc501749bc919a786af492eb717d84a
```

原 `jit_baseline/module_deepgemm_opus.so` 仍为SHA `093bc9cdde5c96931007f51167880e5a6e1e8b3be2ffd9971957fa1b1bd4e9cb`；原 `current_build.json` 未覆盖。HIP/ROCR/CUDA visibility均为空。实际导入的JIT core来自隔离worktree。

初次直接导入tuner时，它通过runtime gfx选择Python dtype，在隐藏GPU时因rocminfo无GPU失败；没有执行候选kernel。最终预构建直接调用正式 `core.get_args_of_build` / `core.build_module`，使用与tuner同样的 `--extra_kids` 和编译参数，避免tuner导入时的GPU枚举及CUDA分配尝试。保留已有LLVM backend flag过滤与ROCm resource目录适配，未改kernel或生产JIT源码。

构建记录见 [build_manifest.json](build_manifest.json) 和 [build.log](build.log)。从原checkout复现：

```bash
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
/opt/venv/bin/python3 reports/opus_bound_analysis_20261007/build_current.py \
  --source-root /root/workspace/opus-bound-candidate-20261007 \
  --output-dir /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_bound_analysis_20261007/formal_candidate \
  --jit-dir /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_bound_analysis_20261007/jit_formal_candidate \
  --manifest /root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_bound_analysis_20261007/formal_candidate/build_manifest.json \
  --pending-candidate
```

## 3. 正式机器码核对

CPU审查见 [identity_audit.json](identity_audit.json)，脚本 [audit_formal_candidate.py](../audit_formal_candidate.py)。

| 核对范围 | 结果 |
|---|---|
| 26个当前9000系列parent | 全部正式编译完成 |
| 实际device entry | 共56个，包含5个split-K reducer entry |
| 改变的5个entry | 指令字节SHA、完整metadata、归一化descriptor匹配对应已测private candidate |
| 其他51个entry | 指令字节SHA、完整metadata、归一化descriptor匹配正式baseline |
| 26个generated device TU及host launcher impl | 与baseline字节相同，选择条件、grid和参数传递未变 |
| kargs | producer仍96B，shared reducer仍20B |
| 全构建对象 | 检查206个CUDA对象，其中202个有device bundle；未改变设备kernel身份全部匹配，无非预期增删symbol |

Descriptor只归一化 `kernel_code_entry_byte_offset` 的16..23字节。Register三个改变entry的C++symbol按最后的bool参数从false变true，其他metadata包含参数/资源完整相等。

隔离worktree编译使**206个完整对象文件的SHA均不同**，涉及源码路径、host对象和布局信息。因此完整对象SHA列明确保留差异；未改变设备身份由指令、metadata及归一化descriptor逐项证明，不能写成“全部对象字节相同”。最终linked.so的SHA也是新的，后续正式API必须验证实际加载路径与此SHA。

## 4. 正式GPU API smoke已完成

15项正式API smoke已全部完成；先前 [CPU支持域/分支/私有身份核验](smoke_plan_cpu_audit.json) 与实际GPU结果由 [严格结果审查](gpu_smoke_analysis.json)、[CPU汇总脚本](../analyze_formal_candidate_smoke.py)连接。它覆盖五个改变entry、短K/长K/M-tail及四个不变entry对照，每个命令为check-only8rep。Official共120次、相同device身份的private共96次，合计216次独立FP32误差区间、repeatability、output/workspace guard全部通过；private的指令字节、完整metadata及归一化descriptor重新逐项匹配。

UTC `2026-10-07 20:06:01.681–20:07:42.529`，设备MI355X/gfx950、PCI`0000:05:00.0`、HIP visible1。每项使用自身最近的command start及其后的clean end，returncode0且`contamination=false`，GPU ordinal/PCI与物理claim相同。全部15项实际Python API加载的module路径与上述新正式SHA`51d4b3d7dd033cc99d391d87e89c25e8afc501749bc919a786af492eb717d84a`一致。此批没有Event或profiler计时，只证明数值、入口和实际模块身份。

当前公开9000系列family是GEMM-only/rank2，不能从底层generated launcher含rank3检查推定公开支持rank3；本次采用不增加rank3正向GPU条件。显式16B-offset仍使用同一对齐与相对地址表达式，此次没有改指针/stride/bounds，亦无需强制追加。

## 5. 仍待采用决策

机制正确性、当前赢家共享池Event及现有异常筛选复核由主线程在空闲GPU窗口执行。额外支持域计划保留partial范围，完成全部1426/646组合不是采用硬条件；充分验证后停止可选穷举。这里的CPU身份通过只证明已经测试的私有device body正确落入隔离正式构建，最小有限条件见 [MINIMUM_REMAINING_GATES.md](MINIMUM_REMAINING_GATES.md)。

正式rank2/native E8M0 API smoke与实际加载module身份已完成。若最终保留全部三个文件差异，可引用现成new.so身份及15项结果；若拒绝其中某些parent/alias，须移除对应差异，CPU重建并核对最终改变/不变身份与必要正式入口。当前仍为pending，不能视为已经采用。
