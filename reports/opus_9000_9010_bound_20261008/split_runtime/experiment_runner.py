#!/usr/bin/env python3
"""Validate and time isolated kernel libraries with the current accumulation contract.

Uses the repository run_perftest timing and automatic argument rotation.
Profiling is separate, with kernel filtering in rocprofv3. GPU selection and
the physical-device lock belong to the caller. Saves every round, never the
minimum of independent runs. Libraries do no allocation inside timed launch.
"""
import argparse
import ctypes
import ctypes.util
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/workspace/opus-9000-9010-split-20261008')
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AITER_AOT_IMPORT", "1")

import torch
from aiter.test_common import run_perftest, device_memory_profiling
from aiter.ops.shuffle import shuffle_weight
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

def normalize_bdf(value):
    match=re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",value.strip())
    if not match: raise ValueError(f"Invalid PCI BDF {value!r}")
    domain,bus,device,function=(int(part,16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"

def visible_gpu_identity():
    if torch.cuda.device_count()!=1:
        raise RuntimeError("Caller must bind exactly one GPU using the HIP index from idle picker")
    props=torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx950"):
        raise RuntimeError(f"Expected gfx950, got {props.gcnArchName}")
    hip=ctypes.CDLL(ctypes.util.find_library("amdhip64") or "/opt/rocm/lib/libamdhip64.so")
    query=hip.hipDeviceGetPCIBusId
    query.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    query.restype=ctypes.c_int
    buffer=ctypes.create_string_buffer(64)
    error=query(buffer,len(buffer),0)
    if error: raise RuntimeError(f"hipDeviceGetPCIBusId returned {error}")
    bdf=normalize_bdf(buffer.value.decode())
    expected=os.environ.get("OPUS_EXPECTED_GPU_BDF")
    if expected and normalize_bdf(expected)!=bdf:
        raise RuntimeError(f"Visible HIP device0 PCI mismatch: expected {expected}, got {bdf}")
    return {"properties":str(props),"pci_bdf":bdf,"expected_pci_bdf":expected,
            "HIP_VISIBLE_DEVICES":os.environ.get("HIP_VISIBLE_DEVICES")}

class Runner:
    def __init__(self, path, workspace=False):
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        self.workspace = workspace
        self.path = Path(path).resolve()
        self.split_count = getattr(self.library, "split_count", None)
        if self.split_count is not None:
            self.split_count.restype = ctypes.c_int
            self.split_count.argtypes = [ctypes.c_int]
        self.launch = getattr(self.library, "launch_workspace" if workspace else "launch")
        self.launch.restype = ctypes.c_int
        self.launch.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * (6 if workspace else 5)
                               + [ctypes.c_int] * 3 + [ctypes.c_void_p])

    def __call__(self, x, w, out, x_scale, w_scale, workspace, kid):
        m, k = x.shape
        n = out.shape[1]
        pointers = [x.data_ptr(), w.data_ptr(), x_scale.data_ptr(), w_scale.data_ptr(), out.data_ptr()]
        if self.workspace:
            pointers.append(workspace.data_ptr())
        rc = self.launch(kid, *pointers, m, n, k, torch.cuda.current_stream().cuda_stream)
        self.last_workspace = workspace
        if rc:
            raise RuntimeError(f"HIP launch returned {rc}, kid={kid}, shape={(m,n,k)}")
        return out

def make_data(m,n,k,seed,signed):
    data = tune.generate_data(m,n,k,seed,device="cuda")
    if signed:
        g = torch.Generator(device="cuda").manual_seed(seed+31)
        for name in ("x", "w_reference"):
            value = data[name]
            sign = torch.randint(0,2,value.shape,device="cuda",generator=g)
            data[name] = (value.float()*(sign*2-1)).to(value.dtype)
        data["w"] = shuffle_weight(data["w_reference"],layout=(16,16))
    guard = torch.full((m*n+256,),42,device="cuda",dtype=torch.bfloat16)
    data["out"] = guard[128:-128].view(m,n)
    return data,guard

def guarded_workspace(split, m, n):
    storage = torch.full((split * m * n + 256,), 73, device="cuda", dtype=torch.float32)
    return storage[128:-128].view(split, m, n), storage


def storage_guards(tensor, value):
    # Check the actual backing storage of a returned rotated tensor.
    offset=tensor.storage_offset()
    capacity=tensor.untyped_storage().nbytes()//tensor.element_size()
    if offset < 128 or capacity < offset + tensor.numel() + 128: return False
    backing=torch.empty((0,),device=tensor.device,dtype=tensor.dtype).set_(
        tensor.untyped_storage(),0,(capacity,),(1,))
    return bool(torch.all(backing[offset-128:offset]==value)) and bool(
        torch.all(backing[offset+tensor.numel():offset+tensor.numel()+128]==value))

def check_result(reference, out, guard, workspace, workspace_guard, *, label, kid, shape):
    error = tune.compare_outputs(reference, out, printLog=False)
    output_guard_ok = storage_guards(out,42)
    workspace_guard_ok = (workspace_guard is None or
        (storage_guards(workspace,73) and bool(torch.isfinite(workspace).all())))
    if error or not output_guard_ok or not workspace_guard_ok:
        raise RuntimeError(f"Numerical/guard failure {label} {kid} {shape}: "
                           f"errRatio={error}, output_guard={output_guard_ok}, workspace_guard={workspace_guard_ok}")
    return {"errRatio": 0, "output_guards": True, "workspace_guards": workspace_guard_ok}


def profile_arguments(runner, call_args, iters, requested_rotation, *, with_metadata=False):
    # Match aiter.run_perftest's automatic physical input/output storage rotation,
    # without nesting a torch profiler inside rocprofv3 counter collection.
    if requested_rotation:
        count = requested_rotation
        metadata={"mode":"explicit","requested_count":requested_rotation}
    else:
        device = torch.cuda.current_device()
        used, input_size, _, _ = device_memory_profiling(runner, *call_args)
        prop = torch.cuda.get_device_properties(device)
        free = torch.cuda.mem_get_info(device)[0]
        cache_size = max(0, min(getattr(prop, "L2_cache_size", 4096 * 1024) * 64 * 128,
                                (free - used + input_size) * 0.9))
        count = int((cache_size + input_size - 1) // input_size)
        metadata={"mode":"aiter_automatic","input_logical_bytes":input_size,
                  "iteration_used_bytes":used,"free_bytes_at_sizing":free,
                  "rotation_budget_bytes":cache_size}
    count = max(1, min(count, iters))
    pool=[copy.deepcopy(call_args) for _ in range(count - 1)] + [call_args]
    metadata["count"]=count
    return (pool,metadata) if with_metadata else pool


def event_confirmation(runners, call_args, reference, guard, workspace_guard, *,
                       kid, shape, iters, result, save):
    """Compare complete captured launch batches with one shared physical pool.

    GPU calls happen only when the caller runs --event-confirm on an idle,
    locked GPU. All allocation, initialization and checks are outside events
    and capture. The same input values and pointers feed every label.
    """
    labels=list(runners)
    result.update(status="preparing",method="CUDA Graph batch replay timed by HIP-backed torch events",
                  rounds=5,iters_per_graph=iters,direct_warmup_calls=5,graph_warmup_replays=1,
                  includes="complete private launch; split-K producer and reducer",
                  numerical_contract="unchanged independent FP32 accumulation interval",
                  measurements=[])
    save()
    expected={}
    # Warm current stream and prepare exact same-label expected outputs before
    # pool sizing, so their allocation is accounted in available memory.
    for label,runner in runners.items():
        for _ in range(5): runner(*call_args)
        torch.cuda.synchronize()
        check_result(reference,call_args[2],guard,call_args[5],workspace_guard,
                     label=label,kid=kid,shape=shape)
        expected[label]={"out":call_args[2].clone(),
                         "workspace":call_args[5].clone() if workspace_guard is not None else None}
    torch.cuda.synchronize()
    pool,rotation=profile_arguments(runners[labels[0]],call_args,iters,0,with_metadata=True)
    result["rotation"]=rotation
    result["shared_pool"]=True
    result["pool_pointers"]=[{name:args[index].data_ptr() for index,name in
        enumerate(["a","b","c","sfa","sfb","workspace"])} for args in pool]
    result["pool_storage_bytes"]=sum(args[index].untyped_storage().nbytes()
        for args in pool for index in range(6))
    save()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs={}
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    # Instantiate both lazy event handles before any measured replay.
    start.record(stream)
    end.record(stream)
    end.synchronize()

    def initialize():
        # Every output and physical partial buffer must be overwritten by the
        # captured kernels. Guards are preserved rather than restored.
        for args in pool:
            args[2].fill_(float("nan"))
            if workspace_guard is not None: args[5].fill_(float("nan"))

    def checks(label):
        rows=[]
        for index,args in enumerate(pool):
            checked=check_result(reference,args[2],guard,args[5],workspace_guard,
                                 label=label,kid=kid,shape=shape)
            if not torch.equal(expected[label]["out"],args[2]):
                raise RuntimeError(f"Graph event nonrepeatable output {label} {kid} {shape} pool{index}")
            if workspace_guard is not None and not torch.equal(expected[label]["workspace"],args[5]):
                raise RuntimeError(f"Graph event nonrepeatable workspace {label} {kid} {shape} pool{index}")
            rows.append({"pool_index":index,"output_repeatable":True,
                         "workspace_repeatable":workspace_guard is not None,**checked})
        torch.cuda.synchronize()
        return rows

    try:
        for label,runner in runners.items():
            with torch.cuda.stream(stream):
                # Warm this exact stream and every pool address before capture.
                for args in pool: runner(*args)
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for iteration in range(iters): runner(*pool[iteration%len(pool)])
            graphs[label]=graph
            with torch.cuda.stream(stream):
                initialize()
                graph.replay()
            stream.synchronize()
            checks(label)
        result["status"]="running"
        save()
        for round_index in range(5):
            order=labels if round_index%2==0 else list(reversed(labels))
            for label in order:
                with torch.cuda.stream(stream):
                    initialize()
                    start.record(stream)
                    graphs[label].replay()
                    end.record(stream)
                end.synchronize()
                total_ms=start.elapsed_time(end)
                if not 0<total_ms<float("inf"):
                    raise RuntimeError(f"Invalid event time {total_ms} ms {label} {kid} {shape}")
                row={"round":round_index,"label":label,"order":order,
                     "event_total_ms":total_ms,"iters":iters,"us_per_call":total_ms*1000/iters,
                     "rotation_count":len(pool),"all_pool_checks":checks(label)}
                result["measurements"].append(row)
                print(json.dumps({"method":"graph_event_confirmation","kid":kid,"shape":shape,
                                  **{name:row[name] for name in ["round","label","event_total_ms","iters","us_per_call","rotation_count"]}}),flush=True)
                save()
        result["median_us"]={label:statistics.median(row["us_per_call"] for row in result["measurements"]
                                                     if row["label"]==label) for label in labels}
        if "baseline" in labels:
            result["median_speedup"]={label:result["median_us"]["baseline"]/result["median_us"][label]
                                      for label in labels if label!="baseline"}
        result["status"]="passed"
        save()
    except BaseException as error:
        result["status"]="interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
        result["error"]={"type":type(error).__name__,"message":str(error)}
        save()
        raise
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values(): graph.reset()
        del graphs,pool,expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--iters",type=int,default=51)
    parser.add_argument("--repetitions",type=int,default=8)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--event-confirm",action="store_true",
                        help="after trace screening, confirm with 5 AB/BA rounds of a shared-rotation CUDA Graph batch and events")
    parser.add_argument("--label",help="run one library label, required for counter profiling")
    parser.add_argument("--target-index",type=int,help="run one target by zero-based plan index")
    parser.add_argument("--profile-rotation",type=int,default=0,
                        help="0 matches run_perftest automatic storage rotation; 1 explicitly profiles warm repeated buffers")

    args=parser.parse_args()
    if args.rounds < 1 or args.iters < 2 or args.repetitions < 1 or args.profile_rotation < 0:
        parser.error("rounds/repetitions must be positive, iters >= 2, profile-rotation >= 0")
    if args.profile and (not args.label or args.target_index is None):
        parser.error("counter profiling requires --label and --target-index to identify one library and shape")
    if args.profile and args.event_confirm:
        parser.error("--event-confirm and --profile are separate measurement methods")
    plan=json.loads(args.plan.read_text())
    if args.label and args.label not in plan["libraries"]:
        parser.error(f"Unknown --label {args.label}")
    if args.target_index is not None and not 0 <= args.target_index < len(plan["targets"]):
        parser.error("target-index is outside the plan")
    workspace_mode=plan.get("workspace",False)
    libraries={label:path for label,path in plan["libraries"].items() if not args.label or label==args.label}
    if args.event_confirm and len(libraries)<2:
        parser.error("--event-confirm requires at least two library labels sharing a pool")
    identity={label:{"path":str(Path(path).resolve()),"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}
              for label,path in libraries.items()}
    result={"status":"running","started":time.time(),"source_head":subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"python":sys.executable,
        "torch":torch.__version__,"hip":torch.version.hip,"libraries":identity,"plan":plan,
        "plan_sha256":hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rounds":args.rounds,"warmup":5,"iters":args.iters,
        "profiling_only":args.profile,"label_filter":args.label,"target_index":args.target_index,
        "event_confirmation_requested":args.event_confirm,
        "timing":"aiter.run_perftest with automatic input/output/workspace rotation","rows":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=args.output.with_suffix(args.output.suffix+".tmp")
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        temporary.replace(args.output)
    save()
    try:
        result["gpu"]=visible_gpu_identity()
        runners={label:Runner(path,workspace_mode) for label,path in libraries.items()}
        for index,target in enumerate(plan["targets"]):
            if args.target_index is not None and index!=args.target_index: continue
            kid=target["kid"]; m,n,k=target["shape"]
            if workspace_mode:
                reported={label:int(runner.split_count(kid)) for label,runner in runners.items()
                          if runner.split_count is not None}
                if any(value <= 0 for value in reported.values()) or len(set(reported.values()))>1:
                    raise RuntimeError(f"Inconsistent library split_count for {kid}: {reported}")
                split=next(iter(reported.values())) if reported else target.get("split")
                if split is None or ("split" in target and split!=target["split"]):
                    raise RuntimeError(f"Plan split does not match library split_count: {target}, {reported}")
            else: split=1
            data,guard=make_data(m,n,k,target.get("seed",17),target.get("signed",True))
            uses_workspace=workspace_mode and split>1
            if uses_workspace:
                workspace,workspace_guard=guarded_workspace(split,m,n)
            else:
                workspace=torch.empty((0,),device="cuda",dtype=torch.float32)
                workspace_guard=None
            call_args=(*(data[name] for name in tune._BENCH_KEYS),workspace,kid)
            row={"target_index":index,"kid":kid,"shape":[m,n,k],"split":split,"correctness":{},"timings":[]}
            result["rows"].append(row)
            if args.profile:
                label,runner=next(iter(runners.items()))
                rotated=profile_arguments(runner,call_args,args.iters,args.profile_rotation)
                for _ in range(5): runner(*call_args)
                torch.cuda.synchronize()
                for iteration in range(args.iters): runner(*rotated[iteration%len(rotated)])
                torch.cuda.synchronize()
                row.update(profiling_only=True,profile_label=label,profile_rotation=len(rotated),
                           profile_iterations=args.iters,profile_warmup=5)
                del rotated
            else:
                reference=tune.run_torch(*(data[name] for name in tune._REF_KEYS),with_bounds=True)
                for label,runner in runners.items():
                    first=None
                    for rep in range(args.repetitions):
                        data["out"].fill_(float("nan"))
                        if uses_workspace: workspace.fill_(float("nan"))
                        out=runner(*call_args)
                        torch.cuda.synchronize()
                        checks=check_result(reference,out,guard,workspace,workspace_guard,
                                            label=label,kid=kid,shape=(m,n,k))
                        if first is None: first=out.clone()
                        elif not torch.equal(first,out):
                            raise RuntimeError(f"Nonrepeatable result {label} {kid} {(m,n,k)} rep{rep}")
                    row["correctness"][label]={"repetitions":args.repetitions,**checks,"repeatable":True}
                    del first
                labels=list(runners)
                for rnd in range(args.rounds):
                    order=labels if rnd%2==0 else list(reversed(labels))
                    for label in order:
                        data["out"].fill_(float("nan"))
                        if uses_workspace: workspace.fill_(float("nan"))
                        out,us=run_perftest(runners[label],*call_args,num_warmup=5,num_iters=args.iters)
                        if not 0<us<float("inf"): raise RuntimeError(f"Invalid timing {us}")
                        checks=check_result(reference,out,guard,runners[label].last_workspace,workspace_guard,
                                            label=label,kid=kid,shape=(m,n,k))
                        row["timings"].append({"round":rnd,"label":label,"us":us,**checks})
                        print(json.dumps({"kid":kid,"shape":[m,n,k],"round":rnd,"label":label,"us":us}),flush=True)
                    save()
                row["median_us"]={label:statistics.median(v["us"] for v in row["timings"] if v["label"]==label) for label in labels}
                if "baseline" in labels:
                    row["median_speedup"]={label:row["median_us"]["baseline"]/row["median_us"][label] for label in labels if label!="baseline"}
                if args.event_confirm:
                    row["event_confirmation"]={}
                    event_confirmation(runners,call_args,reference,guard,workspace_guard,
                        kid=kid,shape=(m,n,k),iters=args.iters,result=row["event_confirmation"],save=save)
                del reference
            save()
            del data,guard,workspace,workspace_guard,call_args
            if "out" in locals(): del out
            torch.cuda.empty_cache()
        for label,item in identity.items():
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()!=item["sha256"]:
                raise RuntimeError(f"Library changed during measurement: {label}")
        result["status"]="passed"
    except BaseException as exc:
        result["status"]="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        result["finished"]=time.time()
        save()

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        if "--output" in sys.argv:
            try:
                output=Path(sys.argv[sys.argv.index("--output")+1])
                if not output.exists():
                    output.parent.mkdir(parents=True,exist_ok=True)
                    output.write_text(json.dumps({"status":"interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                        "error":{"type":type(exc).__name__,"message":str(exc)},"finished":time.time()},indent=2)+"\n")
            except Exception: pass
        raise
