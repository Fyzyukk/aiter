# OPUS GEMM 两项任务架构汇报版

> 核心原则：**Python负责选核、校验和内存；C++负责按kid dispatch并launch kernel。**

## 总体目标架构

```text
┌──────────────────────── Python / Torch ────────────────────────┐
│ Public family API                                               │
│   a16w16 / a8w8 / a8w8_scale / a8w8_blockscale                │
│                            │                                    │
│                            ▼                                    │
│ Family adapter                                                   │
│   normalize layout/dtype                                        │
│   explicit kid -> tuned CSV -> heuristic                        │
│   resolve split-K / actual kid                                  │
│   allocate output / typed workspace                             │
└────────────────────────────┬────────────────────────────────────┘
                             │ tensors + kid + split-K + workspace
                             ▼
┌──────────────────────────── C++ ────────────────────────────────┐
│ Family-specific raw launch                                      │
│   arch check + tensor safety check                              │
│                            │                                    │
│                            ▼                                    │
│ Generated kid dispatch                                          │
│   (arch, family, kid) -> launcher                               │
│                            │                                    │
│                            ▼                                    │
│ kargs -> main kernel [-> reduce kernel]                         │
└─────────────────────────────────────────────────────────────────┘
```

## 任务一：split-K workspace Torch 化

### 改造前

```text
Python调用
   │
   ▼
C++ launcher
   │
   ├─ per-stream SplitkWsRegistry
   ├─ hipMalloc / hipFree
   ├─ host pinned handle
   ├─ device mirror + sync
   └─ graph prewarm
          │
          ▼
   ws_handle -> ptr -> main/reduce
```

### 改造后

```text
Python selector
   │
   ├─ requested kid -> actual kid
   ├─ resolve split-K
   └─ WorkspacePlan(actual kid)
          │
          ▼
torch.empty(shape, dtype=D_WS, device)
          │
          ▼
C++ workspace kid dispatch
          │
          ▼
direct ptr_ws
   ├─ main kernel写partials
   └─ reduce kernel读partials并写Y
```

```text
共享层                         a16w16 adapter
┌─────────────────────┐       ┌────────────────────────────┐
│ WorkspacePlan       │       │ kid -> workspace dtype    │
│ allocate_workspace  │◄──────│ kid -> padded shape       │
│ validate_workspace  │       │ arch-specific batch rule  │
└─────────────────────┘       └────────────────────────────┘
```

任务一完成后删除：

```text
SplitkWsRegistry
opus_splitk_ws_handle
hipMalloc workspace grow
host/device mirror
workspace_init / prewarm
```

## 任务二：OPUS GEMM 接口重构

### 改造前

```text
                       ┌─ Python tuned CSV
Python API ────────────┤
                       └─ C++ opus_gemm mega entry
                              ├─ dtype/family路由
                              ├─ C++ (M,N,K) lookup
                              ├─ C++ heuristic
                              ├─ kid区间能力判断
                              └─ kernel launch
```

### 改造后

```text
Python唯一策略层
   │
   ├─ a16w16 selector: explicit -> CSV -> heuristic
   ├─ a16w16 layout/dtype validation
   ├─ workspace/output allocation
   └─ family-specific API
          │
          ▼
C++ family-specific raw entry
   │
   ├─ opus_gemm_a16w16_launch
   ├─ opus_gemm_a8w8_launch
   ├─ opus_gemm_a8w8_scale_launch
   └─ opus_gemm_a8w8_blockscale_bpreshuffle_launch
          │
          ▼
generated kid dispatch -> launcher -> kernel
```

```text
删除                              保留
┌────────────────────────┐       ┌────────────────────────────┐
│ generic opus_gemm()    │       │ Python tuned CSV lookup    │
│ C++ shape lookup       │       │ Python heuristics          │
│ C++ heuristics         │       │ generated kid dispatch     │
│ C++ framework fallback │       │ C++ kernel safety checks   │
└────────────────────────┘       └────────────────────────────┘
```

## 最终模块结构

```text
csrc/opus_gemm/opus_gemm_common.py
└─ kernel instance唯一事实源

aiter/ops/opus/
├─ gemm_op_a16w16.py              高层a16w16 API
├─ gemm_op_a8w8.py                a8w8 family API
├─ _selector_a16w16.py            explicit/CSV/heuristic
├─ heuristics/
│  ├─ a16w16_gfx942.py
│  ├─ a16w16_gfx950.py
│  └─ a16w16_gfx1250.py
├─ _layout_a16w16.py              dtype/layout规范化
├─ _workspace.py                  通用workspace生命周期
└─ _workspace_a16w16.py           a16w16 workspace计划

csrc/opus_gemm/
├─ opus_gemm.cu                   family raw launch + safety check
├─ include/gfx*/opus_gemm_arch_*  strict kid dispatch
├─ generated *_kid_dispatch.h     kid -> launcher
└─ traits/pipeline/reduce          kargs与kernel实现
```

## 汇报结论

```text
任务一：workspace所有权从C++ raw HIP allocator迁到Torch。
任务二：dispatch policy从C++迁到Python，C++收敛为kid dispatch与kernel launch。
```
