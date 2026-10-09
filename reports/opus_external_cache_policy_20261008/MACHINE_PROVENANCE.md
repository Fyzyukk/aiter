# 上游745行测量设备出处

本机PyTorch报告AMD Instinct MI355X / gfx950 / 256CU。
`gfx950`是ISA架构，不能仅凭它区分MI350X与MI355X。

对745行原始us来源的10个提交逐个复查公开GitHub PR：
704行的来源PR明确写MI355X，33行来源PR#3339写MI355（未展开X后缀），
8行来源PR#4202未明确设备型号。没有检出MI350X来源说明。
这些是PR作者描述，不是上游实机硬件/构建/时钟收据；不能据此保证所有条件相同。
原始us分组保存在`../opus_clang23_mixed_retune_20261008/original_timing_row_commits.json`。
本次重新请求的完整PR描述在`upstream_gpu_model_receipts.json`，提取在
`upstream_gpu_model_summary.json`。

| PR | 当前保留us行数 | PR设备表述 |
| --- | ---: | --- |
| [#3383](https://github.com/ROCm/aiter/pull/3383) | 217 | 8x MI355X |
| [#4326](https://github.com/ROCm/aiter/pull/4326) | 160 | MI355X / gfx950 / 256CU |
| [#3108](https://github.com/ROCm/aiter/pull/3108) | 150 | 8x MI355X |
| [#3238](https://github.com/ROCm/aiter/pull/3238) | 74 | 8x MI355X |
| [#5283](https://github.com/ROCm/aiter/pull/5283) | 60 | MI355X gfx950 |
| [#3339](https://github.com/ROCm/aiter/pull/3339) | 33 | MI355 / 256CU |
| [#4264](https://github.com/ROCm/aiter/pull/4264) | 17 | MI355X |
| [#5279](https://github.com/ROCm/aiter/pull/5279) | 16 | MI355X |
| [#5440](https://github.com/ROCm/aiter/pull/5440) | 10 | MI355X / gfx950 / 256CU |
| [#4202](https://github.com/ROCm/aiter/pull/4202) | 8 | 未确认 |

CKTile4096x2048x7168 / ID27 /62us由#5283写入，明确是MI355X；
同组8192/14336也属于该PR，不能用“上游MI350、本机MI355”解释已知几项慢例。
当前本机固定Clang23 CKTile二进制，改变地址轮换/复用和warmup/iters，
79.23us下降到58.71us（上游62）；这是同机器的直接协议证据。
CK/ASM同二进制地址测试也显示既有可解释的差异，也有无法仅由缓存解释的残差。
同型号不同服务器仍可能有功耗上限、时钟、温度、驱动、LLVM/SDK与运行协议区别，
这些上游原始收据仍未知，没有为剩余差异选定单一原因。
