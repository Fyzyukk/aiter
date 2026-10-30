#!/usr/bin/env python3
"""Check published identities, artifact membership and new CO correctness."""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    import torch
    from aiter.test_common import checkAllclose
    from aiter.jit import core
    from aiter.jit.utils.jit_cache import compiled_kids_are_current
    from csrc.opus_gemm.opus_gemm_common import KernelKey, default_compiled_keys, _opus_sidecar_path
    from prepare_opus_repeat import registry_metadata, pure_common, key

    plan = json.loads((campaign / "result/public421_plan.json").read_text())
    md, _ = registry_metadata()
    registry = pure_common().get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    assert len(plan["shapes"]) == 421
    assert len({key(i["shape"]) for i in plan["shapes"]}) == 421
    for item in plan["shapes"]:
        candidate = item["candidates"][0]
        assert candidate["kernelName"] == md[candidate["kid"]]["kernelName"]
        assert pure_common().a8w8_mxscale_bpreshuffle_supports_shape(registry[candidate["kid"]], *key(item["shape"])[2:5])
    kids = {c["kid"] for item in plan["shapes"] for c in item["candidates"]}
    required = {k.encode() for k in {KernelKey("a8w8_blockscale_bpreshuffle", "gfx950", 256, kid) for kid in kids}
                | default_compiled_keys("gfx950", 256)}
    artifact = Path(core.get_user_jit_dir()) / "module_deepgemm_opus.so"
    assert compiled_kids_are_current(_opus_sidecar_path(), str(artifact), required)
    path = ROOT / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
    spec = importlib.util.spec_from_file_location("_public_kernel_campaign_smoke", path)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    torch.cuda.set_device(0)
    selected = json.loads((campaign / "integration/receipt.json").read_text())["selected_ids"]
    checks = []
    with torch.inference_mode():
        for kid in selected:
            items = [i for i in plan["shapes"] if i["candidates"][0]["kid"] == kid]
            item = min(items, key=lambda i: i["shape"]["M"] * i["shape"]["N"] * i["shape"]["K"])
            shape = item["shape"]
            data = tuner.generate_data(shape["M"], shape["N"], shape["K"], 0, "128x128", device="cuda:0")
            reference = tuner.run_torch(data["x_deq"], data["w_deq"])
            data["out"].fill_(float("nan"))
            tuner.run_gemm_opus(*(data[k] for k in ("x", "weight_shuffle", "x_scale", "w_scale", "out")), kid)
            torch.cuda.synchronize()
            assert bool(torch.isfinite(data["out"]).all())
            error = float(checkAllclose(reference, data["out"], rtol=1e-2, atol=.01,
                tol_err_ratio=.05, printLog=False, catastrophic_check=True))
            assert error <= .05
            checks.append({"kid": kid, "shape": shape, "errRatio": error, "correct": True})
            del data, reference
            torch.cuda.empty_cache()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    validation = {"public_artifact_membership_verified": True, "required_routes": len(required),
        "public_module_sha256": sha(artifact), "correctness_only_new_co_checks": checks,
        "performance_statistics_unchanged": True, "source_sha256": {
            str(ROOT / rel): sha(ROOT / rel) for rel in ("csrc/opus_gemm/opus_gemm_common.py",
                "csrc/opus_gemm/gen_co/co_kernels_gfx950.json", "csrc/opus_gemm/gen_co/gen_co_kernels_gfx950.py")}}
    (campaign / "result/public_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    metrics = json.loads((campaign / "result/public_metrics.json").read_text())
    actual, incorrect = 0, 0
    for part in ("screen", "confirm"):
        for file in (campaign / "final" / part / "repeats").glob("gpu*.jsonl"):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                actual += 1
                incorrect += int(not row["correct"] or bool(row["failure"]))
    completion = {**metrics, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "public_validation": validation, "actual_measurements": actual,
        "incorrect_experimental_measurements": incorrect, "pending": 0}
    (campaign / "completion_receipt.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n")
    state = json.loads((LOCAL / "workflow_state.json").read_text())
    for name, record in state.items():
        if name.startswith("active_") and isinstance(record, dict) and record.get("directory") == str(campaign):
            record["phase"] = "complete"
            record["result"] = str(campaign / "result/public_result_zh.md")
    state["latest_completed_kernel_campaign"] = {"directory": str(campaign),
        **{k: metrics[k] for k in ("combined421_wins", "combined421_losses", "stable_candidate_improved_shapes",
            "new_co_integrated_ids", "new_co_best_shapes")}, "public_jit_verified": True}
    (LOCAL / "workflow_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"combined421_wins": metrics["combined421_wins"], "combined421_losses": metrics["combined421_losses"],
        "correctness_only_new_co_checks": len(checks), "public_jit_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
