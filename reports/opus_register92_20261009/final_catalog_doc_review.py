#!/usr/bin/env python3
"""Independent read-only review of final public/all catalogs and current documentation."""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(value):return re.sub(r"\s+","",value)


def main():
    output=REPORT/"final_catalog_doc_review.json"
    if output.exists():raise SystemExit("Refusing to overwrite final review")
    paths=[REPORT/name for name in ("registry92.csv","registry105.csv","summary.json","read_only_template_review.json")]
    paths += [ROOT/"csrc/opus_gemm/README.md",ROOT/"HANDOFF_MXFP8.md",REPORT/"README.md",
              ROOT/"csrc/opus_gemm/opus_gemm_common.py",REPORT/"codegen/all105_metadata.json",
              REPORT/"codegen/codegen_receipt.json",REPORT/"cpu_audit.json",REPORT/"linked_codegen_final/final_receipt.json"]
    inputs={str(path.relative_to(ROOT)):sha(path) for path in paths}
    metadata={row["kid"]:row for row in json.loads((REPORT/"codegen/all105_metadata.json").read_text())}
    public=list(csv.DictReader((REPORT/"registry92.csv").open()))
    full=list(csv.DictReader((REPORT/"registry105.csv").open()))
    summary=json.loads((REPORT/"summary.json").read_text())
    independent=json.loads((REPORT/"read_only_template_review.json").read_text())
    common=ast.parse((ROOT/"csrc/opus_gemm/opus_gemm_common.py").read_text())
    legacy=next(ast.literal_eval(node.value) for node in common.body if isinstance(node,ast.Assign) and
                any(isinstance(target,ast.Name) and target.id=="A8W8_BPRESHUFFLE_LEGACY_KIDS" for target in node.targets))
    assert len(legacy)==13
    assert len(public)==len({row["kid"] for row in public})==92
    assert len(full)==len({row["kid"] for row in full})==105
    assert {int(row["kid"]) for row in public}==set(metadata)-set(legacy)
    assert {int(row["kid"]) for row in full}==set(metadata)
    assert [row for row in full if row["role"]!="legacy_internal"]==public
    original={9000:"pin",9001:"pin",9010:"padded_pin",9011:"padded_pin",9020:"main",9021:"small_main",9022:"small_main",9023:"narrow",9024:"narrow",9030:"large_output"}
    original.update({kid:"register" for kid in (9040,9041,9042,9051,9052,9053,9054)})
    original.update({kid:"small_lds" for kid in (9043,9044,9045,9046,9047,9049,9055)})
    original.update({kid:"fine_lds" for kid in (9060,9061,9062,9063)})
    pin={9000,9001,9010,9011}|{kid for kid,row in metadata.items() if row["new_variant"] and row["instance"]["bpreshuffle_variant"]["pin_agpr"]}
    checked=0
    for row in full:
        kid=int(row["kid"]);saved=metadata[kid];instance=saved["instance"];variant=instance["bpreshuffle_variant"]
        name=saved["name"];impl=REPORT/"codegen/impl"/(name+".cuh");device=REPORT/"codegen/instances"/(name+"_Cbf16_t.device.cu")
        assert sha(impl)==saved["generated_impl_sha256"] and sha(device)==saved["generated_device_tu_sha256"]
        text,tu=impl.read_text(),device.read_text()
        aliases=re.findall(r"using\s+(\w+)\s*=\s*([^;]+);",text)
        primary=dict(aliases)[name+"_Traits"].strip()
        family=variant["family"] if variant else original[legacy.get(kid,kid)]
        pipeline=variant["pipeline_header"] if variant else re.search(r'#include "([^"]*opus_gemm_pipeline_[^"]+)"',text)[1]
        expected=dict(kid=str(kid),role="new_tuning" if saved["new_variant"] else "retained_tuning" if saved["public_tuning"] else "legacy_internal",
            family=family,label=variant["label"] if variant else instance["name_tag"] or "default",
            compiler="pin_clang24" if kid in pin else "baseline_clang23",pipeline_header=pipeline,
            traits_header=re.search(r'#ifdef OPUS_FUSED_HOST_TU\s*#include "([^"]+)"',text)[1],
            primary_kernel=re.findall(r"template\s+__global__\s+void\s+(\w+)",tu)[0],primary_traits=primary,
            additional_traits=json.dumps([value.strip() for alias,value in aliases if alias!=name+"_Traits"]),
            tile_m=str(instance["B_M"]),tile_n=str(instance["B_N"]),tile_k=str(instance["B_K"]),block_size=str(instance["BLOCK_SIZE"]),
            m_alignment=next(iter(re.findall(r"m % (\d+) == 0",text)),"1"),
            n_alignment=next(iter(re.findall(r"n % (\d+) == 0",text))),
            max_m="" if instance["max_m"] is None else str(instance["max_m"]),
            max_k="" if instance["max_k"] is None else str(instance["max_k"]),
            fixed_k=str(instance["bpreshuffle_fixed_k"]),split_k=str(instance["bpreshuffle_split_k"]),
            workspace_dtype=instance["splitk_workspace_dtype"] or "none",
            sfa_pointer_alignment=next(iter(re.findall(r"reinterpret_cast<uintptr_t>\(x_scale.data_ptr\(\)\) % (\d+)",text)),"1"),
            c_pointer_alignment=re.findall(r"output % (\d+)",text)[0],max_tensor_bytes=str(instance["max_tensor_bytes"]),
            large_output_only=str(family=="large_output"),output_dtype="bf16",internal_dispatch=json.dumps(instance["bpreshuffle_dispatch"]),
            legacy_parent=str(legacy[kid]) if kid in legacy else "",name=name,
            generated_impl_sha256=saved["generated_impl_sha256"],generated_device_tu_sha256=saved["generated_device_tu_sha256"])
        assert set(expected)==set(row),(kid,set(expected)^set(row))
        for key,wanted in expected.items():
            assert canon(row[key])==canon(wanted),(kid,key,row[key],wanted)
            checked+=1
        assert (ROOT/"csrc/opus_gemm/include"/row["pipeline_header"]).is_file()
        assert (ROOT/"csrc/opus_gemm/include"/row["traits_header"]).is_file()
    counts=collections.Counter(row["family"] for row in public)
    assert sum(counts.values())==92
    assert {name:data["count"] for name,data in summary["families"].items()}==dict(counts)
    for family,data in summary["families"].items():
        selected=[row for row in public if row["family"]==family]
        assert data["kids"]==[int(row["kid"]) for row in selected]
        assert set(data["primary_kernel_templates"])=={row["primary_kernel"] for row in selected}
    assert summary["compiler_counts_public"]==dict(collections.Counter(row["compiler"] for row in public))
    assert summary["compiler_counts_all"]==dict(collections.Counter(row["compiler"] for row in full))
    assert summary["primary_gemm_templates"]==len({row["primary_kernel"] for row in public})==18
    assert summary["shared_reducer_templates"]==1 and independent["distinct_functions_including_reducer"]==19
    assert summary["tuning_candidates"]==92 and summary["total_registered_ids"]==105
    assert summary["retained_tuning_candidates"]==28 and summary["new_tuning_candidates"]==64
    assert summary["earlier_all_series_variants"]==62 and summary["earlier_shortk_variants"]==2
    assert summary["legacy_internal_ids"]==13
    for filename,wanted in summary["catalog_sha256"].items():assert sha(REPORT/filename)==wanted
    for filename,wanted in summary["source_sha256"].items():assert sha(ROOT/filename)==wanted
    formal=ROOT/"aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv"
    assert len(list(csv.DictReader(formal.open())))==summary["selection_745_scalar_only"]["shapes"]==745
    assert sha(formal)==summary["selection_745_scalar_only"]["source_sha256"]
    assert not summary["selection_745_scalar_only"]["measured"]
    assert summary["runtime_sources"]==dict(new64="csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py",original41_and_combined_registry="csrc/opus_gemm/opus_gemm_common.py")
    assert summary["provenance_json_runtime_use"] is False
    readme=(REPORT/"README.md").read_text();handoff=(ROOT/"HANDOFF_MXFP8.md").read_text().split("## 2026-10-09：全部落后系列")[0]
    production=(ROOT/"csrc/opus_gemm/README.md").read_text()
    assert "18 个 GEMM 主模板" in readme and "1 个 split-K 归约模板" in readme
    assert "原 41 个入口及合并注册表" in readme and "不参与运行时或代码生成选型" in readme
    assert "64新模板" not in handoff and "18个GEMM主模板加1个shared reducer" in handoff
    assert "92 tuning configurations" in production and "105 registered IDs" in production
    assert "GPU validation of those 64 variants remains pending" in production
    cpu=json.loads((REPORT/"cpu_audit.json").read_text());linked=json.loads((REPORT/"linked_codegen_final/final_receipt.json").read_text())
    assert cpu["tests"]==12 and cpu["boundary_contract_comparisons"]==36864
    assert linked["registered_ids"]==linked["compile_count"]==105 and linked["public_tuning_candidates"]==92
    assert linked["new_variants"]==64 and linked["host_kernel_stub_reference_count"]==123
    assert linked["linked_kernel_stub_undefined_count"]==0 and linked["all64_new_scratch_and_spills_zero"]
    for count in ("15","77","90","123","36,864","105","64"):
        assert count in readme,count
    docs_numeric=dict(public_candidates=92,total_ids=105,new_candidates=64,retained_candidates=28,
      main_templates=18,reducer_templates=1,new_main_templates=8,retained_main_templates=10,
      compiler_public=dict(pin_clang24=15,baseline_clang23=77),compiler_all=dict(pin_clang24=15,baseline_clang23=90),
      cpu_registration_tests=12,boundary_contract_comparisons=36864,host_launch_stub_references=123,
      linked_stub_undefined=0,source_scalar_745_shapes=745)
    after={filename:sha(ROOT/filename) for filename in inputs}
    assert inputs==after,"Inputs changed during read-only review"
    result=dict(status="passed_final_catalog_and_document_review",gpu_operations=0,compilations=0,
      production_source_edits=0,old_report_edits=0,hip_or_hsa_library_loads=0,
      public_csv_rows=92,all_csv_rows=105,fields_per_row=len(public[0]),catalog_field_comparisons=checked,
      public_is_exact_all_registry_minus_legacy=True,all_header_paths_exist=True,
      all_generated_source_hashes_match=True,new64_pipeline_paths_match_descriptors=True,
      all_catalog_fields_match_metadata_and_generated_launch_contracts=True,
      family_and_compiler_counts_match_summary=True,numeric_documentation_checks=docs_numeric,
      primary_configuration_duplicates=independent["canonical_primary_configurations"]["duplicate_groups"],
      normalized_complete_impl_duplicates=independent["generated_source_comparison"]["normalized_impl"]["duplicate_groups"],
      new64_descriptor_runtime_source_only=True,provenance_json_is_cpu_check_and_documentation_only=True,
      read_only_template_review_sha256=sha(REPORT/"read_only_template_review.json"),
      conclusions=["No duplicate resolved primary configuration among the 92 public candidates; 92 complete generated implementations remain distinct after normalization.",
        "Template sharing is 18 primary GEMM functions plus one shared reducer, not 92 copied primary function bodies.",
        "Both final catalogs accurately identify the underlying pipeline and traits headers, including all 64 corrected descriptor pipeline paths.",
        "Current documentation distinguishes the sole new64 Python descriptor source from the original41/common registry and provenance JSON."],
      limits=["Read-only metadata/source/documentation comparison; no new tests, builds, GPU queries or device execution.",
        "No claim of machine-code equivalence or performance equivalence.",
        "The summary's 35,666 candidate/shape count is a scalar selection export and is not a measured benchmark count."],
      inputs_unchanged=True,inputs_sha256=inputs,review_script_sha256=sha(__file__))
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ("status","public_csv_rows","all_csv_rows","fields_per_row","catalog_field_comparisons")},indent=2))


if __name__=="__main__":main()
