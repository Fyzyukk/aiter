本次提交包含截至2026-10-09的MXFP8 B-preshuffle实现、mixed compiler支持、调优结果、89个活动配置/105个注册ID的runtime split-K集成，以及报告源码、CSV、汇总和验证收据。

CPU回归：91项通过，命令为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q op_tests/test_jit_cache_transaction.py op_tests/test_opus_mixed_compiler.py op_tests/test_opus_bpreshuffle_registry_cpu.py op_tests/test_opus_bpreshuffle_runtime_splitk_cpu.py`。修正了独立参数表引入后，旧runpy注册读取测试的同目录导入路径；实现源码与已验证HIP代码保持原样。

[最新runtime报告](../opus_runtime_splitk_20261009/README.md)记录105个实际HIP对象fresh离线编译、host/router/pybind编译、127个launch引用闭合与链接检查。GPU测试仍由用户暂停，runtime实现尚未完成GPU数值及性能验证。

Git中保存可审查的源码、测试依赖、表格、文字结论和收据。编译对象/动态库、无后缀ELF可执行文件、崩溃转储、压缩档案、CK与Python安装副本、profiler缓存和原始ATT/device metadata留在原本本地目录。此前报告manifest仍描述完整本地实验快照，不应按Git中的精简文件集合重新生成旧manifest；Git检出不包含这些完整二进制实验包。`published_files.json`列出本次新增报告文件及哈希。

远程目标为 `https://github.com/Fyzyukk/aiter.git` 的 `aiter-opus-mxfp8-bpreshuffle` 分支。使用普通push，包含该分支已有的upstream/main合并。
