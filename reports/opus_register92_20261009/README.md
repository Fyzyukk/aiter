# 92 个 MXFP8 B-preshuffle 候选注册与模板整理

92 个保留配置已全部加入 OPUS 注册、exact-kid 代码生成和默认调优候选集合：原 28 个公开配置、上一轮全系列优化 62 个、较早的 shortK 原型 2 个。另保留 13 个历史兼容/内部 ID，因此完整注册表为 105 个 ID。

查看 [92 个公开候选完整清单](registry92.csv)、[105 个完整注册清单](registry105.csv) 和 [机器可读汇总](summary.json)。清单包含 ID、系列、模板、traits 参数、tile、shape/指针对齐限制、workspace 和编译器。它们是文档导出，不是第二套运行时配置。

## 模板重复与参数配置

**92 个公开配置共用 18 个 GEMM 主模板，加 1 个 split-K 归约模板。** 原 28 个配置已经共用 10 个 GEMM 主模板；本次 64 个新配置只新增 8 个主模板。编译时，同一个模板仍会按不同参数生成多个实例。

大量重复的执行主体已经合并为共享 pipeline，差异通过 traits 参数表达：tile 大小、A/B 队列深度、scale panel、固定 K、tile 调度顺序、split-K、寄存器预取、输出策略。新 64 个配置集中在 [Python 参数表](../../csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py)，原 41 个入口及合并注册表仍由 [common.py](../../csrc/opus_gemm/opus_gemm_common.py) 定义。新增 4 个分组 traits 头；[provenance JSON](../../csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_promoted_variants.json) 仅供来源追溯和 CPU 检查，不参与运行时或代码生成选型。

下面按主模板统计，而非按用户筛选系列统计。small LDS 的 17 个配置跨 `small_lds`、`fine_lds` 两个系列；large-output 系列使用三个不同主模板。

| 共享 GEMM 主模板 | 配置数 | 公开 ID |
|---|---:|---|
| 原 pin | 2 | 9000、9001 |
| 原 padded pin | 2 | 9010、9011 |
| 8-wave main | 1 | 9020 |
| 128×128 main | 1 | 9021 |
| 160×128 main | 1 | 9022 |
| 64×128 narrow | 1 | 9023 |
| 64×64 narrow / tile order | 3 | 9024、92011、92012 |
| 原 large output | 1 | 9030 |
| small register | 7 | 9040–9042、9051–9054 |
| small LDS / fine | 17 | 9043–9047、9049、9055、9060–9063、92410/11、92420/21、92430/31 |
| geometry | 11 | 92000–92010 |
| shortK160 | 2 | 92020、92021 |
| pin fixed-K | 10 | 92100–92104、92110–92114 |
| padded pin fixed-K | 1 | 92120 |
| small direct-B | 24 | 92200–92223 |
| register split-K | 6 | 92310/11、92320/21、92330、92340 |
| large output panel16 | 1 | 92500 |
| large output direct-B | 1 | 92501 |

模板计数由 92 个实际生成的 device TU 核验，详见 [template_groups92.json](header_promotion/template_groups92.json)。独立复核展开 traits 的默认值、别名、fine/hybrid 参数后，92 个公开配置签名全部不同；去除生成名称、注释和空白后，完整 launcher + device 源码也没有完全重复，见 [read_only_template_review.json](read_only_template_review.json)。仅比较 `.device.cu` 薄壳会遗漏被 include 的参数。不同配置是否在某些 shape 上生成等效机器码、是否值得性能剪枝，仍需进一步证据。

## 候选系列与筛选

默认调优保留全部 92 个合法配置，并先按 shape、K、字节范围和输出 dtype 过滤。候选池没有因为共享模板而删掉参数组合。可以通过新增 `--opus-families` 限定优化方向，并与 `--opus-kids` 取交集：

```text
--opus-families small_direct_b,register_split,fine_lds
--opus-families geometry,shortk,pin_fixed,pad_pin_fixed
--opus-families large_output
```

| 可选系列 | 配置数 | 主要参数或用途 |
|---|---:|---|
| `pin` / `padded_pin` | 2 / 2 | 原 pin、scale reset、padded M、展开参数 |
| `main` / `small_main` / `narrow` | 1 / 2 / 2 | 原 main 和较小 tile |
| `register` / `small_lds` | 7 / 7 | 原小 M 寄存器或 LDS 路径 |
| `fine_lds` | 10 | 原 4 个加新 6 个，tile 与 split-K 参数 |
| `geometry` | 11 | BM64/96/128/160、panel8/16/32、runtime/fixed K |
| `tile_order` | 2 | K7168，M 方向 tile 分组 8/16 |
| `shortk` | 2 | BM160、panel8，精确 K384/K768 |
| `pin_fixed` | 10 | 5 种精确 K × scale reset 开/关 |
| `pad_pin_fixed` | 1 | padded M、精确 K7168 |
| `small_direct_b` | 24 | 8 种 A 路径 × B ahead1/2/3 |
| `register_split` | 6 | register tile、wave-K、split-K4 |
| `large_output` | 3 | 原 9030 加 K1536 panel16/direct-B 两个配置 |

只读标量筛选历史 745 个 shape，每个 shape 有 3–73 个合法公开配置，中位数 48，总计 35,666 个 shape/config 组合。这是待测规模，不是已完成计时数量。13 个历史兼容 ID 不进入默认集合；显式 `--opus-kids` 仍可以选取。

## 注册实现与验证

统一 descriptor emitter 在 [gen_instances_gfx950.py](../../csrc/opus_gemm/codegen/gen_instances_gfx950.py) 中生成新配置，并复用原 tensor / raw E8M0 scale ABI。固定 K、M 上限、tile/指针对齐、输入输出重叠检查均保留。split-K 配置生成 FP32 partial 与共享 BF16 reducer 的完整调用；large-output 配置保留 64 位 C base 和 tile 局部 signed-32 位限制。workspace 查询按注册 split-K 元数据判断。

混合编译器路由覆盖全部 15 个 pin 配置（原 4 + 新 11）。公开候选其余 77 个使用 baseline Clang 23；完整 105 个 ID 中为 90 个 baseline + 15 个 pin Clang 24。

| 检查 | 结果 | 记录 |
|---|---|---|
| CPU 注册/契约与混合编译器测试 | 18 项通过；其中注册测试 12 项包含 36,864 次边界契约比较 | [cpu_audit.json](cpu_audit.json)、[verification.json](verification.json) |
| 原 41 个 ID | 注册字段、名称、生成 impl/device/host 元数据保持原样 | [codegen receipt](codegen/codegen_receipt.json) |
| 新共享 header 主体 | 来源实验设备主体在符号/辅助函数作用域规范化后保持一致 | [header receipt](header_promotion/final_receipt.json) |
| 全 105 个实际 generated device TU | 完整 HIP 对象编译通过，fused host + 全对象离线链接通过 | [完整链接 receipt](linked_codegen_final/final_receipt.json) |
| HIP launch stub 闭合 | host 所需 123 个引用全部由对象及链接库提供，未解析 kernel stub 为 0 | 同上 |
| 新 64 个实际逐 ID 设备实例 | scratch、VGPR spill、SGPR spill 全部为 0 | 同上 |

逐 ID 编译与 aggregate header 编译的 TU 不同；部分 pin 配置的寄存器分配和指令发生变化，最终 receipt 记录实际逐 ID 结果及差异，不宣称两种编译的机器码一致。首次过严的 inspection assertion 已记录并被最终检查替代；编译和链接本身没有失败。

本次仅创建、离线链接和读取 ELF，没有加载生成的 `.so`。**GPU 查询、数值检查、benchmark 和自动等待均保持停止。** 新 64 个配置尚无 GPU 正确性或性能验收结果，不能据此声称已经快于 FlyDSL。

## 冻结结果与复现

正式 745 项 tuning CSV、正式 OPUS 库和冻结 FlyDSL CSV 保持 SHA256；本次没有更新性能选型。上一轮 all/gap 实验报告的文件产物也保持不变。它们记录的 external-source hashes 对应注册前源码快照，本次授权修改 common/codegen 后不再要求这些历史 hashes 与当前源码相同。

只读清单可用 `python reports/opus_register92_20261009/build_registry_report.py` 导出；它禁止导入 torch/aiter/triton/flydsl，不编译或运行 GPU。最终验证与文件哈希见 [verification.json](verification.json) 和 [artifact_manifest.json](artifact_manifest.json)。离线编译脚本及完整命令保留在新报告子目录；旧报告不重写。
