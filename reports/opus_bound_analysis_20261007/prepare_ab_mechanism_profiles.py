#!/usr/bin/env python3
"""Prepare independent private A/B PMC passes; never import or execute GPU code."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
GROUPS = ("sq_ea", "ta_lds")
SELECTIONS = (
    ("prologue", "compute_prologue_confirmation_priority.json", 0, 9021, [480, 7168, 384]),
    ("register_scoped", "register_scoped_positive_winners.json", 1, 9042, [32, 7168, 3072]),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = json.loads((OUT / "counter_evidence.json").read_text())
    inventory_path = OUT / "shape_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    commands, targets, source_hashes = [], [], {}
    runner_path = OUT / "experiment_runner.py"
    parser_path = OUT / "parse_counters.py"
    aliases_path = OUT / "gfx950_extra_counters.yaml"
    for path in (runner_path, parser_path, aliases_path, inventory_path, OUT / "counter_evidence.json"):
        source_hashes[str(path)] = sha(path)
    for family, plan_name, index, kid, shape in SELECTIONS:
        plan_path = OUT / "plans" / plan_name
        plan = json.loads(plan_path.read_text())
        assert not plan["workspace"]
        assert set(plan["libraries"]) == {"baseline", "candidate"}
        target = plan["targets"][index]
        assert target["kid"] == kid and target["shape"] == shape and target["signed"]
        rows = [row for row in inventory["rows"]
                if [row["M"], row["N"], row["K"]] == shape]
        assert len(rows) == 1
        row = rows[0]
        assert row["parent_id"] == row["actual_configuration_id"] == kid
        expected = {
            "family": family, "plan": str(plan_path), "plan_sha256": sha(plan_path),
            "target_index": index, "kid": kid, "shape": shape, "signed": True,
            "BM": row["BM"], "BN": row["BN"], "waves_per_workgroup": row["waves"],
            "wave_k": row["wave_k"], "split_k": row["split_k"],
            "producer_grid": row["producer_grid"], "workgroups": row["producer_workgroups"],
            "waves": row["producer_waves_launched"],
            "dynamic_scaled_mfma": row["padded_mfma_instruction_count_estimate"],
            "useful_f8_operations": row["useful_flops"],
            "padded_f8_operations": row["padded_tile_flops_estimate"],
            "official_baseline_instruction_sha256": row["official_instruction_sha256"],
            "libraries": {
                label: {"path": str(Path(path).resolve()), "sha256": sha(Path(path))}
                for label, path in plan["libraries"].items()
            },
        }
        assert expected["dynamic_scaled_mfma"] * 65536 == expected["padded_f8_operations"]
        targets.append(expected)
        source_hashes[str(plan_path)] = sha(plan_path)
        for group in GROUPS:
            group_path = OUT / "pmc" / f"{group}.txt"
            counter_names = group_path.read_text().strip().removeprefix("pmc:").split()
            assert counter_names == evidence["groups"][group]["aggregate_counters"]
            assert evidence["groups"][group]["static_budget_passed"]
            source_hashes[str(group_path)] = sha(group_path)
            # A/B for the same group remain adjacent in the serialized queue.
            for label in ("baseline", "candidate"):
                name = f"ab_{family}_target{index}_kid{kid}_{label}_{group}"
                folder = OUT / "ab_profiles" / name
                commands.append({
                    "name": name, "family": family, "group": group, "label": label,
                    "expected_target": expected,
                    "expected_library": expected["libraries"][label],
                    "argv": [
                        "/opt/rocm/bin/rocprofv3", "-E", str(aliases_path), "-i", str(group_path),
                        "--kernel-trace", "--kernel-include-regex",
                        "gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel",
                        "-d", str(folder), "-o", name, "--output-format", "csv", "--",
                        "/opt/venv/bin/python3", str(runner_path), "--plan", str(plan_path),
                        "--output", str(folder / "application.json"),
                        "--profile", "--label", label, "--target-index", str(index),
                        "--iters", "51", "--profile-rotation", "0",
                    ],
                    "log": str(folder / "rocprof.log"),
                    "parse_after_success_argv": [
                        "/opt/venv/bin/python3", str(parser_path), str(folder),
                        "--output", str(folder / "parsed.json"), "--last", "51",
                    ],
                })
    assert len(commands) == 8 and len({command["name"] for command in commands}) == 8
    queue = {
        "status": "prepared_cpu_only_no_gpu_execution",
        "gpu_execution_owner": "root serialized physical-device idle lock",
        "source_head": inventory["source_head"],
        "env": {
            "AITER_AOT_IMPORT": "1", "AITER_JIT_DIR": str(OUT / "jit_baseline"),
            "GPU_ARCHS": "gfx950", "CU_NUM": "256",
            "OPUS_HIP_CLANG_PATH": "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin",
            "HIP_CLANG_PATH": "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin",
        },
        "precondition": "Root has clean same-source correctness evidence; verify frozen plan/library/runner hashes before execution. Profiling mode itself performs no reference/guard check.",
        "source_file_sha256": source_hashes,
        "targets": targets,
        "comparison_scope": {
            "sq_ea": [
                "matched-grid SQ_WAVES and padded F8 operations", "mean_wave_life_cycles_estimate",
                "SQ wait/issue/active matched wave-cycle ratios", "DRAM read/write bytes",
                "raw TCP LFIFO, TA-facing and TCR stall counts plus same-scope gate ratios",
            ],
            "ta_lds": [
                "raw TA buffer/direct-LDS read events", "raw SQ LDS wait/bank-conflict counts",
                "raw SQ VMEM address FIFO counts", "raw TCP RFIFO/tag-conflict counts",
            ],
        },
        "execution_notes": [
            "Eight commands use -i with one PMC file; A/B labels and both groups are separate subprocesses.",
            "Root selects an idle physical GPU, sets HIP_VISIBLE_DEVICES and OPUS_EXPECTED_GPU_BDF, and excludes any contaminated file in full.",
            "Keep source/process/agent/queue/dispatch identity and last51; never join PMC passes into synthetic dispatches.",
            "The same seed gives equal input values across labels; independent processes do not imply equal physical pointers/cache state.",
            "Profile durations and bytes/time describe instrumented observations, not adoption speedup. Final speedup uses clean shared-pool Event results.",
            "Do not pass CU/SIMD normalization to parser; short-kernel GRBM window remains unresolved.",
            "Unwindowed FIFO/TCC/TCP probes have scope limits; raw stall differences are clues, not exact elapsed-time savings or confirmed root causes.",
            "Core groups have runtime acceptance on official representatives; these private target/label combinations remain pending until collected.",
            "Prologue M480/BM128 executes M512 padding; validate issued F8 against padded2818572288, not useful2642411520.",
            "The register-scoped baseline/candidate are private isolated libraries; production formal integration requires a separate identity and regression audit.",
        ],
        "commands": commands,
    }
    output = OUT / "ab_mechanism_profile_queue.json"
    output.write_text(json.dumps(queue, indent=2) + "\n")
    print(json.dumps({"status": queue["status"], "commands": len(commands),
                      "output": str(output), "groups": list(GROUPS),
                      "expected_waves": [target["waves"] for target in targets]}))


if __name__ == "__main__":
    main()
