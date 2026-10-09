# OPUS MXFP8 B-preshuffle 参数配置流程

用户确认“少数 pipeline，配置作为参数调优”后，本阶段补齐参数生成、独立按需编译和按 shape 选型。
当前为 **5 个计算 pipeline、1 个 traits 头、89 组默认配置、105 个历史兼容 ID**。
合法新参数组合直接实例化这五个模板，不需要注册新 ID 或增加 kernel 文件。

| Pipeline | 默认配置 | 兼容配置 | 已离线验证的新组合 |
|---|---:|---:|---|
| `pin` | 15 | 15 | 256×256，scale panel 32，runtime K |
| `tiled` | 20 | 20 | 96×128，stage 3，panel 32 |
| `register` | 13 | 18 | 16×32，prefetch 5，runtime global split-K |
| `lds` | 38 | 49 | 64×128，stage 5，runtime global split-K |
| `large_output` | 3 | 3 | 192×256，direct B，C chunk 64，fixed K1536 |

89 是默认调优集合，不是可编译配置总数。新的合法 tuple 数量可以继续增加；shape、参数之间的
约束和模板能力仍然限制合法范围。离线编译成功不能证明新的 tuple 数值正确或性能更快。

## 代码位置

| 文件 | 职责 |
|---|---|
| [canonical catalog](../../csrc/opus_gemm/opus_gemm_bpreshuffle_catalog.json) | 默认完整参数、active 状态、旧名称与 ABI metadata |
| [configuration model](../../csrc/opus_gemm/opus_gemm_bpreshuffle_config.py) | 构造/验证合法 tuple、traits 表达式、shape 过滤、launch contract |
| [compatibility registry](../../csrc/opus_gemm/opus_gemm_common.py) | 从 catalog 派生旧注册表；split/grid/LDS/workspace 标量计划 |
| [shape policy](../../csrc/opus_gemm/opus_gemm_bpreshuffle_policy.py) | tuned CSV lookup 和合法默认配置 |
| [public API](../../aiter/ops/opus/__init__.py) | `opus_gemm_bpreshuffle` 参数入口 |
| [tensor adapter](../../aiter/ops/opus/gemm_op_a8w8.py) | 张量/scale/shape/workspace 检查及计划交给 runtime |
| [module runtime](../../aiter/ops/opus/bpreshuffle_runtime.py) | 独立配置编译、缓存、三种 raw ABI、stream/tensor conversion |
| [single-config generation](../../csrc/opus_gemm/gen_instances.py) | `--bpreshuffle_config` 专用入口，生成配置及其 dispatch 依赖 |
| [gfx950 emitter](../../csrc/opus_gemm/codegen/gen_instances_gfx950.py) | 从 canonical tuple 生成 traits 与 host/device 实例 |
| [one traits header](../../csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh) | 几何和派生布局/LDS/scale/output 常量 |
| [tuner](../../csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py) | 参数任务、prepare、测量、完整 CSV 和 replay |

五个计算头继续位于 `csrc/opus_gemm/include/gfx950/`，名称为
`opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{pin,tiled,register,lds,large_output}_gfx950.cuh`。
旧 source consolidation 记录见 [五 pipeline 报告](../opus_pipeline5_20261009/README.md)，该报告保持冻结。

## 调用与配置流程

```mermaid
flowchart LR
    A[shape 或显式参数] --> B[tuned CSV / 合法默认 / 显式 config]
    B --> C[canonical config]
    C --> D[traits 和 launch plan]
    D --> E[独立 JIT module]
    E --> F[producer 和可选 reducer]
    C --> G[tuner 保存完整 pipeline/config 与 splitK]
```

自动调用：

```python
opus_gemm_bpreshuffle(XQ, WQ_shuffled, Y, x_scale, w_scale)
opus_gemm_bpreshuffle(XQ, WQ_shuffled, Y, x_scale, w_scale, pipeline="register")
```

显式新 tuple：

```python
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config

config = construct_config(
    "register", tile_m=16, tile_n=32, prefetch=5, runtime_split_k=True,
)
opus_gemm_bpreshuffle(XQ, WQ_shuffled, Y, x_scale, w_scale, config=config, split_k=3)
```

该 literal split 要求 K≥384；此配置保留 M≤512、K≤16384 等 launch 限制。
FP8 E4M3FN XQ/WQ 和 BF16 Y 必须 contiguous、同 GPU；N/K 是128的倍数。
A scale 为 E8M0 `[M,K/128]`、stride `(1,M)`；B scale 为 contiguous E8M0 `[N/128,K/128]`。
`WQ_shuffled` 的内容应由 `shuffle_weight(WQ, layout=(16,16))` 产生。

`split_k=None` 优先使用保存的 tuned split，否则使用 grid/CU heuristic；`0` 明确保留历史默认
sentinel；`1..min(16,K/128)` 是 literal count；`-1` 请求 heuristic。固定 split 的配置启动参数
为0，固定值保存在 compile config 内。local WaveK、tile、队列深度仍为编译参数。
runtime global split count 不进入 compile tuple，不按每个 count 增加 ID 或 producer。

自动选型 key 是 `(gfx,cu_num,M,N,K)`。默认读取正式 OPUS tuned CSV；非 OPUS backend 的行
不选择 OPUS 配置。缺行或默认表过期时，用合法 catalog 默认配置。`tuned_file` 或
`OPUS_BPRESHUFFLE_TUNED_CONFIG` 指定的文件若无效，报路径/行号。显式 config 绕过 CSV。
fallback 按 pipeline、grid coverage、padding 等标量规则排序，没有经过新性能测量。

独立模块名为 `module_opus_bpreshuffle_<sha256>`，完整参数、源码内容、工具链、resource/HIP
headers 和 build settings 共同决定 key。每个模块只生成所选 tuple、必要 dispatch 目标和最小
host/router/pybind，不依赖 aggregate subset 的已加载模块。首次使用自动 prepare，热调用复用；
显式 `ensure_config(config)` 重新检查源文件和编译器身份。请求 JSON 位于
`<jit build root>/opus_bpreshuffle_requests/`，独立于可清理的模块构建目录。
pin 配置保留定制 clang 要求；其余配置可只设置 baseline compiler，不因旧 aggregate 的 pin
配置而要求加载整组。现有 exact-kid API 继续走原兼容路径。

## 参数调优和回放

`--opus_configs extra.json` 增加合法配置，`--opus_pipelines register,lds` 限制 pipeline。
示例输入见 [extra_configs.json](extra_configs.json)。计时前准备各配置，runtime split 独立枚举。
输出保存完整 canonical `pipeline/config` 和 `splitK`；旧配置保留 compatibility ID，新 tuple
为 `kernelId=-1`。参数 CSV 可省略 ID 列；正 ID 必须精确匹配完整参数。
任务中的配置对象沿用到结果 CSV 和 replay，避免仅凭 ID 恢复新配置。

支持的轴以 `BPRESHUFFLE_PIPELINES[name].tunable_axes` 及 `validate_config` 的联合约束为准。
例如 register 支持 queue prefetch1–8 与受限 MFMA wave geometry；tiled 支持主/窄 tile 对应
schedule 的 stage/panel 组合；LDS 受 ring、scale capacity、XOR、direct B 和 output arena 限制。
inactive 非中性字段、被模板忽略的组合和冲突的显式值提前拒绝。保存完整 JSON 可避免 partial
参数的 seed 选择在未来改变。traits 和代码生成消费该完整参数，旧 ID 只用于兼容 metadata。

## 验证与实际边界

- CPU 集成覆盖 configuration construction、旧 registry/ABI、shape/split/LDS、公开 adapter、
  同进程两配置缓存隔离、原 `compile_ops` conversion/stream wrapper、JSON→tasks→CSV→replay、
  profile 排序与错误候选排除、JIT cache transactions、mixed compiler。
- 正式745行表：693个OPUS行保持原配置与split选择；其余52行选择合法OPUS fallback。
- 179,685个标量 shape 边界投影与旧 shape contract 一致。
- [105 canonical device 编译收据](canonical105_build/build_receipt.json)：全部成功，含源码、
  compiler、object SHA256；采用最终 LICM policy。
- [五个独立模块收据](runtime_build/receipt.json)、[device argv](runtime_build/device_runtime_policy_result.json)、
  [host 结果](runtime_build/host_result.json)、[link 结果](runtime_build/link_result.json)：每类一个
  无注册 ID 的新 tuple，device/fused host/router/pybind 编译通过，plain C++ `--no-undefined`
  link通过，无未解析OPUS符号。模块未加载。文本重建脚本及 argv 一并保留，二进制不提交。

GPU 查询、kernel launch、`.so` load 都没有执行。用户停止测试的要求持续有效；真实首次 JIT
调用、数值/边界验证、性能调优和同卡745回归尚未完成，不能据本次工程验证宣布超过FlyDSL。
正式 measured CSV、历史报告和历史性能结论保持原样。

最终 CPU 命令、结果和源码哈希见 [validation.json](validation.json)。
