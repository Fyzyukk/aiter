# Fine-M 每 wave VMEM 等待阈值实验（2026-10-07）

CPU 编译和机器码检查已通过；GPU 正确性与性能由主代理串行执行。本页不报告未经实测的加速。

## 假设与唯一改动

Fine-M A copy每组覆盖8行。`B_M/8`不能被wave数整除时，前`(B_M/8)%NUM_WAVES`个wave每个K128 tile多发一条direct-LDS load。原`VMEM_TILE`为整除floor，统一等待阈值会使这些wave多等未来tile的一条load。候选新增`wait_for_future_tiles`lambda，严格限定`FINE_M_LOADS && !REGISTER_SCALES && EARLY_SCALE_LOADS && partial_waves != 0`，选择每wave实际load数。

替换三个原steady-state wait调用点，所有`vmcnt(0)`、LGKM wait、barrier、ring覆盖保护、drain和scale顺序保留。

## 编译证据

- 两侧均有20个gfx950 device入口：15个producer、5个reducer。
- M80: vmcnt12变为wave0/1用14、wave2/3用12。
- M96/8wave: vmcnt6变为wave0..3用8、wave4..7用6。
- M112: 14/16；M48: 10/12。
- VGPR、AGPR、direct-LDS static指令数、barrier static指令数各配置均一致。
- 受影响producer SGPR增加2；所有producer/reducer scratch、VGPR spill、SGPR spill均为0。
- M96/4wave、M128/4wave及五个reducer的指令字节哈希保持一致。

见`static_validation.json`、`static_summary.json`、各侧`resource_isa.json`、`device_audit.json`。分支代价可能抵消减少等待的收益，须实测决定保留。

## 私有 ABI

`launch_workspace(int variant, const void* a, const void* b, const void* sfa, const void* sfb, void* c, void* workspace, int m, int n, int k, void* stream)`。

调用者预先分配`split_count(variant)*M*N`个FP32元素。split2/4一次调用包括producer及生产同款reducer；split1可以workspace传NULL。`launch`省略workspace，只支持split1。

Public 9060..9069严格复刻当前traits、fixed-K及父候选dispatch。强制runtime ID仅用于诊断：29060=M80split1、29061=M96/8wavesplit1、29062=M80split2、29063=M80split4、29069=M48split4。这些私有ID不能写入生产tuned CSV。

## 重建

```bash
python3 reports/opus_bound_analysis_20261007/fine_wait/build.py
```

编译器固定`/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin`，build禁用GPU可见性，无GPU调用。baseline和candidate使用相同build flags、同一当前source快照。
