# 迁移包验证记录（2026-09-27）

验证使用从 Git 暂存内容建立的隔离 checkout，不依赖原工作区未跟踪的实验文件。

- 离线 `prepare.py` 成功生成新的运行目录；全部生成 Python 文件通过语法解析。
- `benchmark.py --list-shapes` 返回 295 个唯一 shape。
- 两卡 `launch.py --plan-only` 分为 148 / 147 项，完整覆盖且无重复。
- `build.py --plan-only` 包含五个 JIT 模块和十二个独立库；编译命令中的源码路径
  均指向隔离 checkout。
- 实际执行 `build.py --private-only`，十二个独立库均编译成功；设备代码审计确认
  共十六个入口，即旧十一项控制和新五项候选。最终脚本修订后，已核对这些编译
  输入和 flags 未变。该步骤只编译，没有运行 GPU kernel。
- 94 项既有文件与样式整理时的保护哈希一致。隔离 checkout 包含其中 88 项；
  另外六项是三份历史 `.so` 和三份历史构建日志，没有打包。
- Shape 和历史基线文件与 `baseline/provenance.json` 的 SHA256 一致；保留 CSV
  原始 CRLF，以便按字节核对。Git 空白检查使用 `cr-at-eol`。
- 独立审阅确认构建和测量均固定当前 checkout 的源码目录，并在导入后检查实际
  AITER / META / CSRC / ASM / CK 路径；README 命令与最终 CLI 一致。

本次迁移验证没有执行完整后端 JIT 构建，也没有进行 GPU 数值或性能测量。
新七候选池的完整 295 项结论需在目标机器运行 README 中的完整流程后获得。
历史 OPUS 291 胜 / CKTile 4 胜属于旧十六候选池。
