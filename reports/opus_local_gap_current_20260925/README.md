# 当前 upstream 分支的本机性能验收

状态：测量、确认和回放已完成。OPUS 209 胜 / 86 负，详见 [RESULTS.md](RESULTS.md)。

当前分支 HEAD 为 `10ab50645d1f25e11844b814b66002b27181dfbf`。
所有测量模块在本目录的独立 `jit/` 目录从当前源码构建。
CK/CKTile/ASM host wrapper 使用当前 ROCm 工具链；OPUS 使用其所需的
`amdgpu-pin-op-dst` 工具链，版本记录在 `build.json`。

基线 CSV 仅提供 gfx950/256CU/M≥1024 的 305 个 shape；295 项受支持，10 项超出
现有字节寻址范围，见 `deferred_shapes.csv`。旧的 208/87 仅用作状态变化对照。

最终有效计时来自物理 GPU 4–6 的空闲时段，按 UUID 绑定。GPU 7 初始批次和 GPU 4 后续中断的补测批次整批排除；迁移的 shape 在 GPU 5/6 完整重测。每个 shape 的全部候选和最终确认都在同一张卡上比较。最终数值回放使用空闲 GPU 5/6，回放时间不替代计时。
扫描全部 27,690 个 CK/CKTile/ASM/OPUS 候选，数值合格后每项测三轮；每轮采用
`run_perftest` GPU profiler 时间、warmup=5、iters=51、自动参数轮换。
共同的数据准备、B shuffle、scale 解码和参考计算排除在计时外，后端内部转换计入。
各 shape/轮次检查 GPU 空闲状态；测量期间禁止构建并核对源码与二进制哈希。

差距≤3%或轮次胜负不一致的 shape 在原 GPU 上追加五轮确认，复测所有合法 OPUS
和全量扫描中距最快有效对手≤5%的参考候选。确认批次整体替换原批次，不跨批次取最小值。
最终导出的全部选择经原生 E8M0 接口回放验证。最终结果见 `RESULTS.md`。

差距统一定义为 `(OPUS_us / 对手_us - 1) × 100%`，正数表示 OPUS 更慢。
这些是本机所述输入和缓存模式下的 GEMM 接口 GPU 时间，不等同于模型端到端延迟。

本目录的 `tune_adapter.py` 是归档 MXFP8 tuner 的外置适配，不重新加入生产源码。
