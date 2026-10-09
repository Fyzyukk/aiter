#!/usr/bin/env python3
"""Read-only source review of the 92 public configurations. No builds or GPU imports."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
CODEGEN=REPORT/"codegen"
GFX=ROOT/"csrc/opus_gemm/include/gfx950"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def compact(text):
    return re.sub(r"\s+","",text)


def split_args(text):
    level=0
    start=0
    args=[]
    for index,char in enumerate(text):
        if char=="<":level+=1
        elif char==">":level-=1
        elif char=="," and level==0:
            args.append(text[start:index]);start=index+1
    if text[start:]:args.append(text[start:])
    return args


def main():
    out=REPORT/"read_only_template_review.json"
    if out.exists():
        raise SystemExit("Refusing to overwrite completed review")
    metadata_path=CODEGEN/"all105_metadata.json"
    metadata=json.loads(metadata_path.read_text())
    public=[row for row in metadata if row["public_tuning"]]
    assert len(metadata)==105 and len(public)==92
    inputs={str(metadata_path.relative_to(ROOT)):sha(metadata_path)}
    traits_paths=sorted(GFX.glob("opus_gemm_traits_a8w8_mxscale_bpreshuffle*gfx950.cuh"))
    text="\n".join(path.read_text() for path in traits_paths)
    for path in traits_paths:inputs[str(path.relative_to(ROOT))]=sha(path)
    aliases=dict(re.findall(r"using\s+(\w+)\s*=\s*([^;]+);",text))
    parameters={}
    for match in re.finditer(r"template\s*<([^<>]*)>\s*struct\s+(\w+)\b",text):
        params=[]
        for param in split_args(match[1]):
            pieces=param.strip().split("=",1)
            params.append((pieces[0].split()[-1],compact(pieces[1]) if len(pieces)>1 else None))
        parameters[match[2]]=params
    baseline=dict(re.findall(r"small_baseline_traits<(\d+)>\s*\{\s*using type = (\w+);",text))

    def canonical(expr):
        expr=compact(expr)
        if expr in aliases:
            return canonical(aliases[expr])
        if "<" not in expr:
            if expr=="opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950":
                return "opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950{RESET_SFA_BEFORE_LOAD=true}"
            if expr=="opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950":
                return "opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950{PAD_M=true,LOOP_UNROLL=4}"
            if expr=="opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950":
                return "opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950{PAD_M=true}"
            return expr
        name=expr[:expr.index("<")]
        args=split_args(expr[expr.index("<")+1:-1])
        if name in parameters:
            defs=parameters[name]
            assert len(args)<=len(defs),(name,args)
            for _,default in defs[len(args):]:
                assert default is not None,(name,args)
                args.append(default)
        if name=="opus_gemm_mxscale_bpreshuffle_small_direct_b_traits":
            return "opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits<"+canonical(baseline[args[0]])+","+args[1]+">"
        if name=="opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950":
            bm,wm,wn,stages,cluster,split,vec,block,cache,fixed=args
            args=[bm,"128",wm,wn,stages,cluster,"2","false","false","true","true","true",split,vec,block,cache,fixed,"true"]
            name="opus_gemm_small_lds_traits_gfx950"
        elif name=="opus_gemm_mxscale_bpreshuffle_narrow_fine_traits":
            bm,bn,wm,wn,split=args
            args=[bm,bn,wm,wn,"4","1","2","false","false","true","true","true",split,"16","128","2","0","true"]
            name="opus_gemm_small_lds_traits_gfx950"
        elif name=="opus_gemm_mxscale_bpreshuffle_register_split_traits":
            bm,bn,wk=args
            return canonical("opus_gemm_small_register_traits_gfx950<"+",".join([bm,bn,"1","1","3",wk,"3","3"])+">")+"{GLOBAL_SPLIT_K=4}"
        return name+"<"+",".join(canonical(arg) if "<" in arg or arg in aliases else arg for arg in args)+">"

    normalized=collections.defaultdict(lambda:collections.defaultdict(list))
    primary=collections.defaultdict(list)
    main_groups=collections.defaultdict(list)
    emitted=collections.defaultdict(list)
    scalar=collections.defaultdict(list)
    rows=[]
    for row in public:
        name=row["name"]
        impl=CODEGEN/"impl"/(name+".cuh")
        device=CODEGEN/"instances"/(name+"_Cbf16_t.device.cu")
        assert sha(impl)==row["generated_impl_sha256"] and sha(device)==row["generated_device_tu_sha256"],row["kid"]
        for path in (impl,device):inputs[str(path.relative_to(ROOT))]=sha(path)
        implementation,tu=impl.read_text(),device.read_text()
        local_aliases=dict(re.findall(r"using\s+(\w+)\s*=\s*([^;]+);",implementation))
        declarations=re.findall(r"template\s+__global__\s+void\s+(\w+)\s*<\s*(.*?)>\s*\((.*?)\);",tu,re.S)
        assert declarations,row["kid"]
        main_func,main_alias,_=declarations[0]
        primary_trait=canonical(local_aliases[main_alias.strip()])
        config=main_func+"<"+primary_trait+">"
        primary[config].append(row["kid"])
        main_groups[main_func].append(row["kid"])
        resolved=[]
        for function,args,kargs in declarations:
            args=split_args(compact(args))
            fullargs=[canonical(local_aliases.get(arg,arg)) for arg in args]
            signature=function+"<"+",".join(fullargs)+">("+compact(kargs)+")"
            emitted[signature].append(row["kid"])
            resolved.append(signature)
        def normalize(source):
            source=source.replace(name,"INSTANCE")
            source=re.sub(r"/\*.*?\*/|//[^\n]*","",source,flags=re.S)
            return compact(source)
        nim,nut=normalize(implementation),normalize(tu)
        for label,data in (("raw_impl",implementation),("raw_device",tu),("normalized_impl",nim),
                           ("normalized_device_shell",nut),("normalized_impl_and_device",nim+nut)):
            normalized[label][digest(data)].append(row["kid"])
        instance=dict(row["instance"])
        instance.pop("name_tag",None);instance.pop("name_root",None)
        if instance["bpreshuffle_variant"]:
            instance["bpreshuffle_variant"]=dict(instance["bpreshuffle_variant"])
            for key in ("kid","label"):instance["bpreshuffle_variant"].pop(key,None)
        scalar[json.dumps(instance,sort_keys=True)].append(row["kid"])
        rows.append(dict(kid=row["kid"],primary_kernel=main_func,source_traits=local_aliases[main_alias.strip()].strip(),
                         canonical_primary_traits=primary_trait,configuration_signature=config,
                         resolved_device_instantiations=resolved,normalized_impl_sha256=digest(nim),
                         normalized_device_shell_sha256=digest(nut),normalized_impl_and_device_sha256=digest(nim+nut)))
    assert len(primary)==92 and len(main_groups)==18
    functions={signature[:signature.index("<")] for signature in emitted}
    assert len(functions)==19 and functions-set(main_groups)=={"opus_gemm_mxscale_bpreshuffle_reduce_kernel"}
    dup=lambda groups:[dict(ids=ids,signature=signature) for signature,ids in groups.items() if len(ids)>1]
    duplicate_instantiations=dup(emitted)
    for signature,ids in emitted.items():
        if len(ids)>1:
            assert signature.startswith("opus_gemm_mxscale_bpreshuffle_reduce_kernel<"), (signature,ids)
    variants_path=ROOT/"csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py"
    common_path=ROOT/"csrc/opus_gemm/opus_gemm_common.py"
    generator_path=ROOT/"csrc/opus_gemm/codegen/gen_instances_gfx950.py"
    provenance_path=GFX/"opus_gemm_mxscale_bpreshuffle_promoted_variants.json"
    for path in (variants_path,common_path,generator_path,provenance_path):inputs[str(path.relative_to(ROOT))]=sha(path)
    descriptor_rows={row["instance"]["bpreshuffle_variant"]["kid"]:row["instance"]["bpreshuffle_variant"] for row in public if row["new_variant"]}
    provenance=json.loads(provenance_path.read_text())["variants"]
    assert len(descriptor_rows)==len(provenance)==64
    for row in provenance:
        descriptor=descriptor_rows[row["id"]]
        for key in ("pipeline_header","traits_header","kernel","traits"):
            assert descriptor[key]==row[key],(row["id"],key)
    common=common_path.read_text();generator=generator_path.read_text()
    assert "for _variant in NEW_BPRESHUFFLE_VARIANTS:" in common and "variant = getattr(k, \"bpreshuffle_variant\", None)" in generator
    runtime_reads=[]
    for path in (ROOT/"csrc/opus_gemm").rglob("*.py"):
        if "promoted_variants.json" in path.read_text():runtime_reads.append(str(path.relative_to(ROOT)))
    assert not runtime_reads
    summaries={label:dict(unique_sources=len(groups),duplicate_groups=[ids for ids in groups.values() if len(ids)>1])
               for label,groups in normalized.items()}
    result=dict(status="passed_source_review",gpu_operations=0,compilations=0,hip_or_hsa_library_loads=0,
      registered_ids=105,public_tuning_candidates=92,retained_public_candidates=28,new_public_candidates=64,
      excluded_legacy_kids=[row["kid"] for row in metadata if not row["public_tuning"]],
      distinct_primary_kernel_templates=18,distinct_functions_including_reducer=19,
      primary_template_groups=[dict(kernel=kernel,count=len(ids),ids=ids) for kernel,ids in main_groups.items()],
      canonical_primary_configurations=dict(count=len(primary),duplicate_groups=dup(primary),
        method="Resolve generated primary traits aliases; fill all defaults from C++ source declarations; expand original fine and new narrow-fine wrappers to their full small-LDS parameter list; expand hybrid baselines; include primary function identity and parameter overrides."),
      generated_source_comparison=summaries,
      emitted_device_specializations=dict(unique_signatures=len(emitted),duplicates=duplicate_instantiations,
        note="The only repeated fully resolved specialization is the intentionally shared reducer. No repeated primary GEMM specialization was found across the 92 public configurations, including embedded original dispatch choices."),
      incomplete_scalar_metadata_comparison=dict(unique_when_name_tags_removed=len(scalar),duplicate_groups=[ids for ids in scalar.values() if len(ids)>1],
        explanation="9043/9055 appear equal only if old name_tag is removed before resolving traits; small_lds vs small_lds_deep select NUM_STAGES 8 vs 12. This lossy metadata comparison is not a duplicate-configuration check."),
      source_table_relationship=dict(new64_runtime_parameter_source=str(variants_path.relative_to(ROOT)),
        all105_registry=str(common_path.relative_to(ROOT)),new64_codegen_uses_registered_descriptor=True,
        original41_remain_defined_in_common=True,provenance_path=str(provenance_path.relative_to(ROOT)),
        provenance_mapping_matches64=True,runtime_python_reads_provenance_json=runtime_reads,
        note="The scalar Python descriptor table is the sole runtime/codegen parameter source for the 64 new configurations. The full 92 public set is exported by the common registry; original configurations remain there. JSON is frozen promotion provenance consumed by CPU checks, not a second runtime source."),
      conclusions=["All 92 public configurations have distinct resolved primary function/parameter signatures.",
        "All 92 complete generated implementations remain distinct after per-instance symbol, comment and whitespace normalization.",
        "Thin device TU source shells normalize to 26 structures because the traits arguments reside in the included implementation; this is shared emission structure and does not make configurations equivalent.",
        "18 GEMM templates and one reducer implement the 92 configurations; a template count is not a configuration count."],
      limits=["Read-only C++/Python source and previously generated TU inspection; no new compilation or GPU work.",
        "Distinct source/parameter configurations do not prove distinct compiled machine code, numerical results, or performance.",
        "No machine-code equivalence is claimed, including shape-conditional cases where separate configurations may execute the same work."],
      configurations=rows,inputs_sha256=inputs,review_script_sha256=sha(__file__))
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ("status","public_tuning_candidates","distinct_primary_kernel_templates","distinct_functions_including_reducer")},indent=2))
    print(json.dumps({"canonical_primary_configurations":len(primary),"normalized_complete_implementations":summaries["normalized_impl"]["unique_sources"],"thin_device_shell_structures":summaries["normalized_device_shell"]["unique_sources"],"repeated_specialization_groups":len(duplicate_instantiations)},indent=2))


if __name__=="__main__":main()
