#!/usr/bin/env python3
"""Inspect the already compiled gfx950 ELF inside a Clang offload bundle on CPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
PARSER = ROOT / "reports/opus_flydsl_all_20261009/hybrid_small/cpu_audit_v4/elf_parser.py"
spec = importlib.util.spec_from_file_location("cpu_elf_parser", PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    final = HERE / "final_receipt.json"
    if final.exists():
        raise SystemExit("Refusing to overwrite final receipt")
    previous = json.loads((HERE/"receipt.json").read_text())
    mapping = json.loads((ROOT/"csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_promoted_variants.json").read_text())
    variants = {row["id"]:row for row in mapping["variants"]}
    small_traits = (ROOT/"csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh").read_text()
    promoted_small = (ROOT/"csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_variants_gfx950.cuh").read_text()
    baselines = dict(re.findall(r"small_baseline_traits<(\d+)>\s*\{\s*using type = (\w+);",promoted_small))
    aliases = dict(re.findall(r"using (opus_gemm_mxscale_bpreshuffle_small_\w+_traits_gfx950)\s*=\s*([^;]+);",small_traits))
    def canonical(v):
        trait=v["traits"].replace(" ","")
        if v["family"]=="geometry" and trait.count(",")==2:
            trait=trait[:-1]+",0>"
        if v["family"]=="small_direct_b":
            match=re.fullmatch(r"opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<(\d+),(\d+)>",trait)
            base=aliases[baselines[match[1]]].replace(" ","").replace("\n","")
            args=base[base.index("<")+1:-1].split(",")
            defaults=[None]*5+["1","2","false","false","false","false","false","1","4","128","0","0","false"]
            args+=defaults[len(args):]
            base="opus_gemm_small_lds_traits_gfx950<"+",".join(args)+">"
            trait="opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits<"+base+","+match[2]+">"
        return v["kernel"]+"<"+trait+">"
    audit = []
    for build in previous["builds"]:
        assert build["compile"]["returncode"] == build["host_syntax"]["returncode"] == 0
        folder = HERE/build["side"]
        obj = ROOT/build["object"]
        assert sha(obj) == build["object_sha256"]
        data = obj.read_bytes()
        magic = b"__CLANG_OFFLOAD_BUNDLE__"
        assert data.startswith(magic)
        count, = struct.unpack_from("<Q",data,len(magic))
        offset = len(magic)+8
        records = []
        for _ in range(count):
            begin,size,length = struct.unpack_from("<QQQ",data,offset)
            offset += 24
            target = data[offset:offset+length].decode()
            offset += length
            records.append(dict(target=target,offset=begin,size=size))
        device_record = next(r for r in records if r["target"] in ("hip-amdgcn-amd-amdhsa--gfx950", "hipv4-amdgcn-amd-amdhsa--gfx950"))
        device_bytes = data[device_record["offset"]:device_record["offset"]+device_record["size"]]
        device = folder/"device.co"
        device.write_bytes(device_bytes)
        compiler = Path(build["compile"]["argv"][0])
        command = [str(compiler.parent/"clang-offload-bundler"),"-type=o","-unbundle",
                   "-targets="+device_record["target"],"-input="+str(obj),"-output="+str(folder/"llvm_device.co")]
        proc = subprocess.run(command,cwd=ROOT,env=ENV,capture_output=True,text=True)
        (folder/"unbundle.log").write_text(proc.stdout+proc.stderr)
        assert proc.returncode==0, proc.stderr
        assert (folder/"llvm_device.co").read_bytes()==device_bytes
        tools = []
        for label,args in (("device_headers",["-h"]),("device_notes",["--notes"]),("device_symbols",["--symbols"])):
            cmd = [str(compiler.parent/"llvm-readelf"),*args,str(device)]
            proc = subprocess.run(cmd,cwd=ROOT,env=ENV,capture_output=True,text=True)
            path = folder/(label+".txt")
            path.write_text(proc.stdout+proc.stderr)
            assert proc.returncode==0,proc.stderr
            tools.append(dict(argv=cmd,returncode=proc.returncode,log=str(path.relative_to(ROOT)),log_sha256=sha(path)))
        elf = parser.Elf(device_bytes)
        assert elf.machine == 224
        metadata,symbols = elf.metadata(),elf.symbols()
        kernels = metadata["amdhsa.kernels"]
        names = subprocess.check_output(["c++filt",*[k[".name"] for k in kernels]],cwd=ROOT,env=ENV,text=True).splitlines()
        assert len(names)==len(kernels)==len(build["instantiated_ids"]),(build["side"],len(kernels))
        expected = {canonical(variants[kid]):kid for kid in build["instantiated_ids"]}
        rows=[]
        for kernel,name in zip(kernels,names):
            compact=name.replace(" ","")
            matches = [kid for key,kid in expected.items() if key in compact]
            assert len(matches)==1,(name,matches)
            resources = {key:kernel.get("."+key,0) for key in ("agpr_count","vgpr_count","sgpr_count",
                        "group_segment_fixed_size","private_segment_fixed_size","vgpr_spill_count","sgpr_spill_count")}
            assert all(resources[key]==0 for key in ("private_segment_fixed_size","vgpr_spill_count","sgpr_spill_count")),name
            code=symbols[kernel[".name"]]
            assert code["type"]==2 and code["size"]>0
            rows.append(dict(id=matches[0],name=name,resources=resources,instruction_bytes=code["size"],
                             instruction_sha256=hashlib.sha256(code["bytes"]).hexdigest(),metadata=kernel))
        assert sorted(row["id"] for row in rows)==build["instantiated_ids"]
        (folder/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
        audit.append(dict(side=build["side"],bundle_records=records,llvm_independent_unbundle_equal=True,
                          unbundle_argv=command,device_elf=str(device.relative_to(ROOT)),device_sha256=sha(device),
                          kernel_count=len(kernels),kernels=rows,inspections=tools,
                          all_scratch_and_spills_zero=True))
    assert len({row["id"] for side in audit for row in side["kernels"]})==64
    assert previous["inputs"] == {path:sha(ROOT/path) for path in previous["inputs"]}
    result=dict(status="offline_aggregate_headers_passed",gpu_operations=0,hip_or_hsa_library_loads=0,device_execution=0,
                numerical_validation="not_run_gpu_stopped",performance_validation="not_run_gpu_stopped",
                promoted_variant_count=64,new_pipeline_header_count=8,grouped_traits_header_count=4,
                source_identity=previous["source_identity"],inputs_unchanged=True,
                successful_compile_and_host_syntax_receipt="reports/opus_register92_20261009/header_promotion/receipt.json",
                prior_receipt_sha256=sha(HERE/"receipt.json"),
                prior_inspection_error="llvm-readelf was first given a Clang offload bundle; retained first receipt unchanged, extracted contained ELF without recompilation",
                elf_parser_sha256=sha(PARSER),finalize_script_sha256=sha(__file__),builds=audit)
    final.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"kernel_counts":{b["side"]:b["kernel_count"] for b in audit},
                      "all_scratch_and_spills_zero":True,"inputs_unchanged":True},indent=2))


if __name__ == "__main__":
    main()
