# CK编译器影响与上游基线来源（2026-10-08）

CK和CKTile都经LLVM生成GPU机器码，Clang20可能改变CK指令调度、寄存器分配及spill。目前没有做CK同源码Clang20/23 A/B，因此不能排除编译器影响，也不能将本轮与上游的全部差异归因于编译器。

对保存profile和上游CSV的405个相同CK调用：时间几何平均变化+0.02920098%，中位数+2.28087377%；184个在±5%内、141个慢超过5%、80个快超过5%。这是跨轮观测，整体平均接近不能说明逐项不受影响。

本轮CK/CKTile的SO .comment均显示AMD Clang20，ROCm7.0 f4087f6b428f0e6f575ebac8a8a724dab123d06e；冻结build_external.py也显式选择/opt/rocm/llvm/bin。本地9月25日保存构建收据的外部后端是Clang23（46fcb339，SDK产物另有补丁），OPUS是pin Clang24（49c41889）。这些本地信息不能作为上游原基线工具链的证明。

上游745行现有内容的git blame分布在6个提交，其中包含仅修改bw单位的提交；不能把它解读为6次计时或已知统一工具链。已读取相关PR的GPU/计时记录，未找到明确Clang版本收据，上游原始二进制也未取得。目前上游基线Clang版本应标为未知，不能断言是20或23。

[结构化核对](CK_compiler_baseline_provenance_20261008.json)、[原逐shape调用对照](corrected_call_comparison_745.csv)、[CKTile已确认的Clang20/23诊断](cktile_diagnosis_20261008/README.md)。本次没有新的GPU计时，原CSV/冻结报告保持。
