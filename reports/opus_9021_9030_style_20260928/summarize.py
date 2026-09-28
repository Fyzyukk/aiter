#!/usr/bin/env python3
"""Summarize frozen CPU compilation and address checks without running kernels."""
import collections
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "csrc/opus_gemm/include/gfx950"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


snapshot = json.loads((HERE / "snapshot.json").read_text())
tags = snapshot["candidate_tags"]
spec = importlib.util.spec_from_file_location(
    "device_audit", HERE.parent / "opus_9020_restore_a_layout_20260928/validate.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.HERE = HERE
elf = audit.audit_helpers()
allowed = set()
sources = {}
comparisons = {}
patch = []

for kid, tag in tags.items():
    directory = HERE / kid / "final"
    p_name = f"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{tag}_gfx950.cuh"
    t_name = f"opus_gemm_traits_a8w8_mxscale_bpreshuffle_{tag}_gfx950.cuh"
    assert (SRC / p_name).read_text().replace(f'"{t_name}"', '"traits_runtime.cuh"') == (directory / "pipeline_runtime.cuh").read_text()
    assert (SRC / t_name).read_bytes() == (directory / "traits_runtime.cuh").read_bytes()
    assert json.loads((directory / "build_manifest.json").read_text())["returncode"] == 0
    final = audit.inspect_one(directory / "experiments.so", elf)
    final.pop("body")
    _, code = next(audit.objects(directory / "experiments.so", elf))
    (directory / "device.co").write_bytes(code)
    isa = subprocess.check_output([str(audit.LLVM / "llvm-objdump"), "-d", str(directory / "device.co")], text=True)
    (directory / "device_isa.txt").write_text(isa)
    final["instruction_counts"] = dict(collections.Counter(re.findall(r'^\s+([a-z][a-z0-9_]+)\s+.*//', isa, re.M)))
    save(f"{kid}/final/device_audit.json", final)
    before = json.loads((HERE / kid / "device_comparison.json").read_text())["before"]
    abi_keys = (".name", ".args", ".kernarg_segment_align", ".kernarg_segment_size", ".max_flat_workgroup_size", ".wavefront_size")
    assert all(before["resources"][key] == final["resources"][key] for key in abi_keys)
    assert all(final["resources"][key] == 0 for key in (".private_segment_fixed_size", ".vgpr_spill_count", ".sgpr_spill_count"))
    assert final["resources"][".group_segment_fixed_size"] == before["resources"][".group_segment_fixed_size"]
    keys = set(before["instruction_counts"]) | set(final["instruction_counts"])
    differences = {key: [before["instruction_counts"].get(key, 0), final["instruction_counts"].get(key, 0)]
                   for key in sorted(keys) if key.startswith(("buffer_", "ds_", "v_mfma", "s_waitcnt", "s_barrier", "scratch_"))
                   and before["instruction_counts"].get(key, 0) != final["instruction_counts"].get(key, 0)}
    comparisons[kid] = dict(before=before, final=final, abi_unchanged=True,
                            memory_mfma_sync_count_differences=differences)
    for name in (p_name, t_name):
        path = SRC / name
        allowed.add(str(path))
        text = path.read_text()
        assert not any(line.rstrip() != line for line in text.splitlines())
        sources[str(path)] = sha(path)
        old = HERE / kid / "before" / name
        patch.extend(difflib.unified_diff(old.read_text().splitlines(True), text.splitlines(True),
                     fromfile=f"before/{name}", tofile=str(path.relative_to(ROOT))))

unexpected = [path for path, digest in snapshot["source_sha256"].items()
              if path not in allowed and sha(Path(path)) != digest]
assert not unexpected, unexpected
host = json.loads((HERE / "host_layout_final/validation.json").read_text())
assert host["compile_returncode"] == 0 and host["run_returncode"] == 0
save("final_device_comparison.json", comparisons)
save("source_check.json", dict(cpu_only=True, gpu_executed=False,
     final_production_sha256=sources, compiled_snapshots_match_production=True,
     protected_source_count=len(snapshot["source_sha256"]) - len(allowed),
     protected_sources_unchanged=True, unexpected_changes=unexpected,
     baseline="snapshot.json, captured before this five-candidate cleanup"))
(HERE / "change.diff").write_text("".join(patch))

lines = ["# 9021 / 9022 / 9023 / 9024 / 9030 整理记录", "",
         "五个候选的 pipeline 与 traits 已整理。直接复用已有且适用的 helper；不同的布局保留各自实现，并使用 shape/dim/unfold、前置 gmem/layout、集中 offset 和单行注释。", "",
         "| 候选 | 复用与保留的区别 |", "|---|---|",
         "| 9021 / 9022 | GA/SA/RA/GB/SB/RB 复用 9000；原始字节 SFA reader 复用 9020。保留 A/C pin、原 scale producer 分工、M 尾行处理及 3/2-stage ring。 |",
         "| 9023 / 9024 | GA 经原 lane XOR 后复用 9000，SA/GB/SB/RB 同样复用；XOR RA 用本地 shape/dim。保留 SFA u16 打包、SFB u32 复制、直接 C 输出及 3/4-stage ring。 |",
         "| 9030 | GA/SA/GB/SB/RB 复用 9000，RA/raw SFA reader 复用 9020。保留两个 128-byte SFB 面板、64-bit C 基址、tile-local C bounds 和原 U2 调度。 |", "",
         "五个候选的 traits 独立列出几何、存储、派生常量和断言；地址计算均在 pipeline。A/B/C/SFA/SFB 基址包含 batch stride，tile row/col 放入对应基址。9021/9022 保留原 pin 容器；9023/9024/9030 使用默认 MMA vtype 和 clear(v_c)。", "",
         "9023/9024 的 scale layout 表达组内位置，四个 scale offset 表达 K group/panel 推进。SFA 动态地址保留 (panel_begin + group) * stride_sfa 的计算顺序；不在主循环前缓存完整 group 字节地址。最终 source 与被检查的编译快照逐字一致。", "",
         "## CPU 验证", "",
         "| ID | AGPR 前→后 | VGPR 前→后 | SGPR 前→后 | LDS bytes | 指令 bytes 前→后 |", "|---|---:|---:|---:|---:|---:|"]
for kid, item in comparisons.items():
    b, f = item["before"], item["final"]
    br, fr = b["resources"], f["resources"]
    lines.append(f"| {kid} | {br['.agpr_count']}→{fr['.agpr_count']} | {br['.vgpr_count']}→{fr['.vgpr_count']} | {br['.sgpr_count']}→{fr['.sgpr_count']} | {fr['.group_segment_fixed_size']} | {b['instruction_bytes']}→{f['instruction_bytes']} |")
lines += ["", "五个最终 device TU 编译通过，ABI 与 LDS 大小保持，private segment 和 VGPR/SGPR spill 全为 0。9021/9022 的部分 paired LDS writes 被拆为 single writes；机器码不是逐字相同，不能声称所有指令计数完全一致。具体差异保存在 final_device_comparison.json。", "",
          "最终 host-only 检查使用实际 helper，对照整理前的地址公式，覆盖各线程、RA repeat/chunk/stage、scale pass/pack、K panel 边界与 guard、tile 基址和 C 布局。结果见 [host validation](host_layout_final/validation.json)。此前第一版地址检查也通过，但 9023/9024 的编译 VGPR 增至 233/151；这些版本未作为最终源码交付。独立诊断产物保留，最终使用组内 layout 与动态 K offset 分工。", "",
          "本轮未运行 GPU 数值测试、benchmark 或重新 tune。寄存器计数只说明编译资源，不代表实测性能。之前完整 305-shape sweep 发生在这些源码整理之前。", "",
          "## 文件与证据", "",
          "[分支整体 OPUS tune 集成清单](INTEGRATION_FILES.md)；[最终编译对照](final_device_comparison.json)；[源码范围与快照核对](source_check.json)；[8 个候选的 CPU codegen 依赖核对](integration_codegen.json)。", ""]
for kid, tag in tags.items():
    lines.append(f"**{kid}**")
    lines.append("")
    for typ in ("pipeline", "traits"):
        p = SRC / f"opus_gemm_{typ}_a8w8_mxscale_bpreshuffle_{tag}_gfx950.cuh"
        lines.extend([f"[{p}]({p})", ""])
(HERE / "RESULTS.md").write_text("\n".join(lines))
print("PASS: final sources, ABI, no spills, protected files and host validation recorded")
