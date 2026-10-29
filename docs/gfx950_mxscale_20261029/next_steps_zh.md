# 新机器使用与后续优化

## 拉取新分支

在已有的 aiter 仓库内执行：

```bash
git fetch origin fyz/gfx950-co-spill-restore-20261029
git switch --track origin/fyz/gfx950-co-spill-restore-20261029
```

没有仓库时：

```bash
git clone --branch fyz/gfx950-co-spill-restore-20261029 https://github.com/Fyzyukk/aiter.git
cd aiter
```

使用已有兼容 gfx950 的 ROCm、PyTorch、aiter/FlyDSL 环境。分支已经带有 262 个 gfx950 CO，直接使用这些镜像不需要重新构建 pin 编译器。源码侧 JIT host launcher 和非 pin kernel 仍由新机器的正常 aiter 构建流程处理。

若环境里曾设置 `OPUS_GEN_CO_DIR` 指向其他 checkout，先清除；它必须指向包含 `gfx950/` 的父目录，不能指向 `gfx950/` 本身。`OPUS_CO_LLVM_BIN`、`HIP_CLANG_PATH` 等旧机器编译器路径也不要直接沿用。

```bash
unset OPUS_GEN_CO_DIR
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AITER_JIT_DIR="$PWD/.jit_gfx950_co_restore"
```

该分支的默认模型表仍选择 FlyDSL。拉取分支后不会自动将 192/207 个 OPUS 胜出项部署到默认 GEMM；需要后续生成并使用新的 tuned CSV。

## 可选：重建 CO

仅当修改 pin kernel 或需要换编译器时重建。需要支持当前显式 AGPR pin 的 LLVM；普通 ROCm clang 不能自动视为具备该扩展。本次使用 `llvm-amdgpu-pin-op-dst-49c41889-build/bin` 与 ROCm 7.2.4，以下路径应换成新机器实际路径。

```bash
python3 csrc/opus_gemm/gen_co/build_co.py \
  --arch gfx950 \
  --llvm-bin /path/to/pin-llvm/bin \
  --device-flag=--target=x86_64-unknown-linux-gnu \
  --device-flag=-B/path/to/rocm/lib/llvm/bin \
  --out-dir csrc/opus_gemm/gen_co/gfx950 \
  --jobs 16
```

默认保留通过 pin 校验的 spill/scratch 配置。仅需重新验收已有镜像时可加 `--skip-existing`；进行源码重编译时不要用它。候选筛选实验使用独立输出目录，避免严格开关删除当前可用镜像。

## 后续工作顺序

1. **先补候选覆盖。** 已实现 `--opus_top_geometries`：默认 Top8，`0` 表示全部保留，`4` 可使用原几何截断范围。另有 `--opus_pipelines`、`--opus_routes`，可集中搜索 m_repeat JIT。CPU 检查中，22 个漏测旧几何有 4 个进入 Top8，全部 22 个进入全量模式。
2. **先覆盖 22 个遗漏旧几何的落后 shape。** 输入在 [priority_missing_old_geometry_22.csv](priority_missing_old_geometry_22.csv)。先确认旧几何候选进入测量列表，再评价是否超过基线。旧数字 kid 已变化，应以 kernel 参数映射当前候选，不能直接搬旧 CSV 中的 ID。
3. **验证恢复 CO。** 优先 kids 8/10，再 kids 9/11；它们没有 VGPR spill/scratch。检查包含 M 尾块、K 尾轮和长 K 的正确性与输出边界，然后测性能。kids 1/7 有大量 VGPR spill，kids 3/5 只有少量 VGPR spill，kids 78–80 有 private segment；分别以实测结果决定取舍。
4. **再优化仍然慢的 pipeline。** 窄 N、K=7168 优先 BM/BN/wave 覆盖，再比较 stage、B 预取距离。长 K 的 4wave 可有限枚举 GroupM=0/4/8/16 和输出方式；用 profiling 判断 MFMA、LDS/memory wait、驻留 wave 和寄存器压力，避免仅凭 shape 推断瓶颈。
5. **保留比较口径。** 保存的确切 FlyDSL kernel、同轮 FlyDSL 最佳和旧 OPUS 等价配置分别比较。前几名交替重复测量，记录中位数与离散度；最终结合调用频率和绝对微秒差判断收益。

## 先在 CPU 导出候选

候选导出与 GPU tuner 共用筛选函数，不导入 PyTorch，不枚举 GPU，不编译或测量 kernel。下面集中导出 22 个优先 shape 的 m_repeat JIT 候选：

```bash
python3 -m csrc.opus_gemm.opus_gemm_search \
  -i docs/gfx950_mxscale_20261029/priority_missing_old_geometry_22.csv \
  -o /tmp/gfx950_m_repeat_candidates.csv \
  --opus_top_geometries 0 \
  --opus_pipelines pingpong_lds --opus_routes jit
```

候选 CSV 是待测清单，包含 kernel ID、名称、几何和路由，没有耗时。已保存的 CPU 验证结果见 [cpu_search_optimization_zh.md](cpu_search_optimization_zh.md)。历史 seed 现在按当前 kernel 名称匹配，避免旧数字 kid 指向其他配置；旧名称没有等价匹配时仍需核对参数。

## 在 GPU 上测量

以下为后续 GPU 命令模板，本次未执行。第一条补 22 个 shape 的全 pipeline 几何，第二条集中搜索 m_repeat JIT，包括新增的分组和大 panel 配置。

```bash
HIP_VISIBLE_DEVICES=0 AITER_REBUILD=1 \
python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype both \
  -i docs/gfx950_mxscale_20261029/priority_missing_old_geometry_22.csv \
  -o /tmp/gfx950_missing_geometry_retune.csv \
  --opus_top_geometries 0 --mp 1
```

```bash
HIP_VISIBLE_DEVICES=0 AITER_REBUILD=1 \
python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \
  --mxscale_table --libtype opus \
  -i docs/gfx950_mxscale_20261029/shapes_421.csv \
  -o /tmp/gfx950_m_repeat_retune.csv \
  --opus_top_geometries 0 \
  --opus_pipelines pingpong_lds --opus_routes jit --mp 1
```

[shapes_421.csv](shapes_421.csv) 可替换为完整分析集合。输出应使用新文件，保留当前 FlyDSL 基线。当前输入固定 `gfx950 / 256 CU / 128x128`，其他 CU 数量应生成对应机器的输入，不能把该机器的性能结果直接当作新机器的排序。

候选恢复和几何补齐是两项工作：恢复 CO 只让候选可用，Top4 是否将其加入某个 shape 的测量仍由 tuner 决定。目前没有给恢复项赋予新的胜出数量，也没有改变默认后端选择。
