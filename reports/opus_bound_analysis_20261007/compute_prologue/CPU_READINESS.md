# 9021 / 9022 Prologue partial wait 实验

2026-10-07。私有基线与候选的 CPU 编译和 ISA 审计完成；没有执行 GPU。生产文件未修改。

## 私有改动

仅修改两个 pipeline 的 prologue：K0 矩阵请求 → 编译调度屏障 → scale producer → 编译调度屏障 → K1 请求 → 编译调度屏障 → `loops > 1 ? vmcnt(tile_requests) : vmcnt(0)`。全部原 `lgkmcnt(0)`、硬件 barrier、main loop、consumer retirement、scale refill 和输出保留。

9021 使用 `vmcnt(8)`，9022 使用 `vmcnt(9)`。实际 ISA 顺序与源码一致。Scale load 的依赖等待包含 `vmcnt(0)`，会先等待已经发出的 K0。因此预期作用是 K0 与 scale 部分加载重叠、让 K1 保持在途，实际能否改善耗时须 GPU 验证。

## 编译与资源

指定工具链 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin`，ROCm `/opt/rocm`，gfx950。完整命令在 `build_manifest.json`。参数取 retained manifest 冻结配置，未加入 `-amdgpu-coerce-illegal-types=1`。

首次编译 host pass 因旧 `hip_fp8/hip_bf16` header 与 clang 24 OCML 声明不兼容失败，日志保存为 `build_attempt1.log`。修复仅在相同的两个私有 launcher 里、完整 HIP runtime include 后定义 `__HIPCC_RTC__`，让 OPUS traits 的 host pass 使用 minimal declarations。设备 header 与生产基线相同。最终两个 .so 均编译成功，开启 `-verify-machineinstrs`。

| ID | VGPR baseline→candidate | AGPR | SGPR | LDS bytes | private / VGPR spill / SGPR spill | 指令 bytes |
|---|---:|---:|---:|---:|---|---:|
| 9021 | 164→164 | 0→0 | 96→98 | 105504→105504 | 全部0 | 8316→8588 |
| 9022 | 194→194 | 0→0 | 66→63 | 81184→81184 | 全部0 | 6888→6944 |

两者 MFMA、矩阵 load、LDS operand read、硬件 barrier 静态数量相同。等待静态数量和其他 SALU/VALU 会随新控制流重排变化，不能由静态字节量认定性能改善。

## 调用接口

`baseline/experiments.so` 与 `candidate/experiments.so` 提供相同 C ABI：

```cpp
extern "C" int launch(int variant, const void* a, const void* b,
                      const void* sfa, const void* sfb, void* c,
                      int m, int n, int k, void* stream);
```

`variant` 为 9021 或 9022。返回 HIP error code，不同步；调用者负责同步和完整测量。

- 9021：任意正 M；9022：正 M 且 M%16=0。
- N 为正的128倍数；K 为128..16384内128倍数。
- A/B/C/SFA 基址16字节对齐；A/B元素字节和C字节分别不超过 INT32_MAX。
- A 原 FP8连续 `[M,K]`，B标准 `(16,16)` 预排 FP8，SFA 原生 E8M0 `[K/128,M]`，SFB `[N/128,K/128]`，C BF16连续 `[M,N]`。
- batch=1；所有 pointer batch strides 在 kargs 中完整设置并由 kernel 保留。

## 等 GPU 空闲后的最小验证范围

同样输入/地址比较 baseline、candidate 与独立数学参考。覆盖 K128/K256、奇偶 loops、K4096/K4224、K8192/K8320、K16384，M尾块和SFA16字节对齐；9021补充 M1/M15/M17。优先计时 `(512,6144,7168)`、`(608,7168,7168)` 及短K384/768，两版本AB/BA多轮交替。

证据：`source_manifest.json`、`changes.diff`、`build_manifest.json`、`isa_metadata_comparison.json`、每variant的 `metadata.txt/device.s/kid*_prologue.txt`。两个 launcher完全相同；两个目标header是唯一修改差异；审计时生产源码哈希保持。
