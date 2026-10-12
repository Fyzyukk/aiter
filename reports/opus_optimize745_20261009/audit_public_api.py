#!/usr/bin/env python3
"""CPU contract checks for the actual benchmark's public/prepared adapters."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent


def main():
    source = HERE / "benchmark.py"
    tree = ast.parse(source.read_text())
    root = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    helpers = [n for n in ast.walk(root) if isinstance(n, ast.FunctionDef)
               and n.name in ("variant", "check_public")]
    assert {n.name for n in helpers} == {"variant", "check_public"}
    calls = []
    m, n, k = 3, 128, 384
    config = SimpleNamespace(pipeline="register")
    output = torch.full((m, n), float("nan"))
    workspace = torch.full((16,), float("nan"))
    pool = [([torch.ones(m, k), torch.ones(n, k), torch.ones(m, k//128),
              torch.ones(n//128, k//128), output, workspace],)]
    task = {"config":"candidate", "split_k":3}
    entry = {"public_api_checks":{}}
    guard = {"output":True, "workspace":True}

    def public(a, b, out, sa, sb, *, config, split_k, workspace):
        calls.append(("public", (a, b, out, sa, sb), config, split_k, workspace))
        out.fill_(7)

    def prepared(a, b, sa, sb, out, ws, split):
        calls.append(("prepared", (a, b, sa, sb, out), config, split, ws))
        out.fill_(7)
        return out

    def compare(ref, actual, strict):
        assert strict
        if not torch.equal(ref, actual):
            raise ValueError("numerical mismatch")
        return 0.0

    env = {"fused_configs":{}, "configs":{"candidate":config},
           "prepared":{"candidate":SimpleNamespace(launch=prepared)},
           "kernel_instance_from_config":lambda c:c,
           "bpreshuffle_launch_plan":lambda *args:SimpleNamespace(workspace_elements=3),
           "opus_gemm_bpreshuffle":public, "m":m, "n":n, "k":k,
           "torch":SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)),
           "compare":compare, "pool_guard_ok":lambda p:guard["output"],
           "workspace_tail_ok":lambda tag, values:guard["workspace"], "entry":entry}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"), env)
    tag, public_run = env["variant"](task, pool, public_api=True)
    assert tag == "candidate:3"
    assert public_run(0) is output
    _, tensors, passed_config, split, ws = calls[-1]
    assert all(a is b for a,b in zip(tensors,
        (pool[0][0][0],pool[0][0][1],output,pool[0][0][2],pool[0][0][3])))
    assert passed_config is config and split == 3 and ws.numel() == 3
    assert ws.data_ptr() == workspace.data_ptr()
    _, prepared_run = env["variant"](task, pool)
    assert prepared_run(0).data_ptr() == output.data_ptr()
    assert calls[-1][0] == "prepared" and calls[-1][1][0].shape == (1,m,k)
    assert calls[-1][1][1].shape == (1,n,k) and calls[-1][1][-1].shape == (1,m,n)
    ref = torch.full_like(output, 7)
    for seed in (29,101):
        env["check_public"](task,pool,ref,seed)
    assert [c["seed"] for c in entry["public_api_checks"][tag]] == [29,101]
    rejected = []
    for failure in ("numerical", "output_guard", "workspace_extent"):
        guard["output"] = failure != "output_guard"
        guard["workspace"] = failure != "workspace_extent"
        previous = len(entry["public_api_checks"][tag])
        try:
            env["check_public"](task,pool,ref+1 if failure == "numerical" else ref,29)
        except (ValueError,AssertionError):
            rejected.append(failure)
        else:
            raise AssertionError(f"Accepted {failure}")
        assert len(entry["public_api_checks"][tag]) == previous
    env["bpreshuffle_launch_plan"] = lambda *args:SimpleNamespace(workspace_elements=0)
    _, direct_run = env["variant"]({"config":"candidate","split_k":1},pool,public_api=True)
    direct_run(0)
    assert calls[-1][3] == 1 and calls[-1][4] is None
    env["fused_configs"]["private"] = {}
    try:
        env["variant"]({"config":"private","split_k":3},pool,public_api=True)
    except AssertionError:
        rejected.append("private_as_public")
    else:
        raise AssertionError("Private experiment accepted as production API")
    report = {"status":"passed", "benchmark_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "public_argument_order":"checked", "prepared_3d_argument_order":"checked",
              "exact_workspace_slice":"checked", "direct_workspace_none":"checked",
              "seed_receipts":[29,101], "rejected":rejected, "kernel_launches":0,
              "scope":"CPU adapter contract and error handling; GPU correctness/performance pending"}
    (HERE / "public_api_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
