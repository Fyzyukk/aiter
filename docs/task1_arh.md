# 任务一：OPUS Split-K Workspace Torch 化修改结构

## 修改目标

```text
C++内部管理workspace
          │
          ▼
Python/Torch按实际kid分配workspace，C++只接收指针并launch
```

## 文件变更总览

```text
新增文件：10个
整文件删除：0个
修改现有文件：按Python、entry、codegen、kernel四层进行
```

### 新增文件

```text
aiter/ops/opus/
├─ _selector_a16w16.py
├─ _workspace.py
├─ _workspace_a16w16.py
└─ heuristics/
   ├─ __init__.py
   ├─ a16w16_gfx942.py
   ├─ a16w16_gfx950.py
   └─ a16w16_gfx1250.py

op_tests/
├─ test_opus_dispatch.py
├─ test_opus_workspace.py
└─ test_opus_graph.py
```

### 整文件删除

```text
无。

任务一不删除：
├─ opus_gemm_lookup.h                    # 任务二才删除
├─ opus_gemm_a16w16_tune_lookup.h        # 任务一只在其中拆两类kid表
├─ 三个opus_gemm_heuristic_dispatch头    # 任务一保留作parity golden
└─ opus_gemm.cu                          # 只删除其中旧allocator/registry代码
```

### 从现有文件中删除的内容

```text
csrc/opus_gemm/opus_gemm.cu
├─ SplitkWsRegistry / Owner / mutex map
├─ opus_splitk_ws_get()
├─ opus_splitk_ws_device_handle()
├─ opus_splitk_ws_sync_to_device()
├─ opus_gemm_workspace_init() C++实现
├─ <mutex> / <unordered_map>
└─ generic opus_gemm()的bf16生产路径

csrc/opus_gemm/include/opus_gemm.h
└─ opus_gemm_workspace_init()声明

csrc/include/rocm_ops.hpp
└─ OPUS_GEMM_WORKSPACE_INIT_PYBIND

csrc/pybind/opus_gemm_pybind.cu
└─ OPUS_GEMM_WORKSPACE_INIT_PYBIND调用

aiter/tuned_gemm.py
├─ _opus_needs_ws_prewarm
├─ capture stream猜测
├─ warmed workspace集合
└─ prewarm调用点

三个arch traits
├─ opus_splitk_ws_handle
├─ handle guard
└─ gfx942旧opus_splitk_ws_ptr(handle)

三个arch codegen launcher
├─ capture检测
├─ registry get/grow
├─ hipMalloc/hipFree workspace
├─ 4 MiB grow逻辑
└─ handle mirror/sync
```

### 必须修改的现有文件

```text
Python metadata / selector接入
├─ csrc/opus_gemm/opus_gemm_common.py
├─ csrc/opus_gemm/opus_gemm_tune.py
├─ aiter/ops/opus/gemm_op_a16w16.py
├─ aiter/ops/opus/__init__.py
└─ aiter/tuned_gemm.py

C++ entry / binding
├─ csrc/opus_gemm/include/opus_gemm.h
├─ csrc/opus_gemm/opus_gemm.cu
├─ csrc/include/rocm_ops.hpp
└─ csrc/pybind/opus_gemm_pybind.cu

Codegen / dispatch
├─ csrc/opus_gemm/gen_instances.py
├─ csrc/opus_gemm/codegen/common.py
├─ csrc/opus_gemm/codegen/gen_instances_gfx942.py
├─ csrc/opus_gemm/codegen/gen_instances_gfx950.py
├─ csrc/opus_gemm/codegen/gen_instances_gfx1250.py
├─ csrc/opus_gemm/include/opus_gemm_common.cuh
├─ csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh
├─ csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh
└─ csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh

Traits
├─ csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh
├─ csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh
└─ csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh

Main pipeline
├─ csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh
├─ csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf1.cuh
├─ csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v.cuh
├─ csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh
├─ csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh
├─ csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh
├─ csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh
└─ csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh

Reduce
├─ csrc/opus_gemm/include/gfx942/a16w16/splitk_reduce_gfx942.cuh
├─ csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh
└─ csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh

现有测试 / 文档
├─ op_tests/test_opus_a16w16_gemm.py
├─ csrc/opus_gemm/README.md
└─ aiter/ops/opus/README.md
```

### 只做回归检查，原则上不新增workspace逻辑

```text
csrc/gemm_a16w16/gemm_a16w16_tune.py
aiter/ops/deepgemm.py
```

这两个调用方继续调用公共Python wrapper，由wrapper统一分配workspace；不要各自复制planner。

## 修改前

```text
Python a16w16 API
        │
        ▼
C++ lookup / heuristic得到kid
        │
        ▼
Generated split-K launcher
        │
        ├─ SplitkWsRegistry(stream)
        ├─ hipHostMalloc(handle)
        ├─ hipMalloc(device mirror)
        ├─ hipMalloc/hipFree(workspace)
        ├─ sync_to_device
        └─ graph prewarm
                │
                ▼
       ws_handle -> workspace ptr
                │
        ┌───────┴────────┐
        ▼                ▼
   main kernel      reduce kernel
```

## 修改后

```text
Python a16w16 API
        │
        ▼
Python selector
  explicit -> tuned CSV -> heuristic
        │
        ├─ requested kid
        ├─ actual kid
        └─ allocation split-K
        │
        ▼
a16w16 WorkspacePlan(actual kid)
  tile + dtype + padded shape + batch rule
        │
        ▼
torch.empty(shape, dtype=D_WS, device)
        │
        ▼
raw a16w16 launch(tensors, workspace, kid, split-K)
        │
        ▼
C++ generated kid dispatch
        │
        ▼
workspace.data_ptr() -> ptr_ws
        │
   ┌────┴─────────┐
   ▼              ▼
main kernel   reduce kernel
写partials     读partials并写Y
```

## Python侧修改结构

```text
csrc/opus_gemm/opus_gemm_common.py
└─ 现有kernel instance唯一事实源
   ├─ tile: B_M/B_N/B_K
   ├─ arch/family/kid
   └─ splitk_workspace_dtype

aiter/ops/opus/
├─ _selector_a16w16.py
│  ├─ 解析最终kid
│  ├─ 解析gfx942 auto split-K
│  └─ 解析gfx942 requested kid -> actual kid
│
├─ heuristics/
│  ├─ a16w16_gfx942.py
│  ├─ a16w16_gfx950.py
│  └─ a16w16_gfx1250.py
│
├─ _workspace.py
│  ├─ WorkspacePlan
│  ├─ allocate_workspace
│  └─ validate_workspace
│
├─ _workspace_a16w16.py
│  └─ actual kid -> a16w16 workspace shape/dtype
│
└─ gemm_op_a16w16.py
   └─ resolve -> plan -> torch.empty -> raw launch
```

## C++侧修改结构

```text
opus_gemm.cu / opus_gemm.h / pybind
└─ a16w16 raw entry增加optional workspace
        │
        ▼
generated dispatch
├─ non-workspace kid -> 旧5参数launcher
└─ workspace kid     -> 新6参数launcher
        │
        ▼
generated split-K launcher
├─ 校验workspace device/dtype/contiguous/alignment/capacity
├─ workspace.data_ptr()
├─ 构造kargs.ptr_ws
└─ launch main + reduce
```

## Kernel ABI修改结构

```text
旧ABI
  opus_splitk_ws_handle*
      └─ device端二次解引用 -> ptr

新ABI
  main kargs: void* ptr_ws
  reduce arg: const void* ws_ptr
      └─ direct pointer
```

```text
traits
├─ D_ACC   累加类型
├─ D_WS    workspace读写类型
└─ D_OUT   最终输出类型

pipeline
└─ ptr_ws -> D_WS*

reduce
└─ ws_ptr -> const D_WS*
```

gfx942保留direct-pointer uniform helper：

```text
ptr_ws
  -> opus_gfx942_uniform_ws_ptr()
  -> 64-bit readfirstlane uniform化
  -> D_WS*
```

## 三架构adapter

```text
                        a16w16 WorkspacePlan
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
          gfx950                gfx942               gfx1250
  [S,B,padM,padN]       [S,B,padM,padN]        [S,padM,padN]
       fp32             actual kid: bf16/fp32       fp32
     multi-batch             multi-batch          batch == 1
```

gfx942 actual-kid解析：

```text
N不在 {64,128,256,384,512,1024,2048}

requested 10210 -> actual 10200
requested 10213 -> actual 10203
requested 10216 -> reject

WorkspacePlan只读取actual kid
```

## 删除结构

```text
删除
├─ SplitkWsRegistry
├─ opus_splitk_ws_handle
├─ opus_splitk_ws_get/device_handle/sync
├─ workspace hipMalloc/hipFree grow
├─ host/device handle mirror
├─ opus_gemm_workspace_init C++/pybind
└─ tuned_gemm graph prewarm

保留
├─ 三架构kernel数值算法
├─ split-K clamp
├─ reduce算法
├─ non-workspace launcher ABI
└─ C++ kernel物理安全检查
```

## 最终结构结论

```text
Python：确定actual kid + 分配typed workspace
C++：strict kid dispatch + direct pointer + kernel launch
Kernel：只读写调用方传入的workspace
```
