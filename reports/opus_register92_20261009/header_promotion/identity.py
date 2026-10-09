#!/usr/bin/env python3
"""Record promoted trait declaration and shared helper identity against frozen sources."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GFX=ROOT/"csrc/opus_gemm/include/gfx950"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def declaration(text,name):
    begin=text.index("struct "+name)
    if text[:begin].rstrip().endswith(">"):
        begin=text.rfind("template",0,begin)
    brace=text.index("{",begin)
    depth,end=1,brace+1
    while depth:
        depth+=(text[end]=="{")-(text[end]=="}")
        end+=1
    assert text[end]==";"
    return text[begin:end+1]


def main():
    output=HERE/"source_identity.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite identity receipt")
    pairs=[
      ("reports/opus_flydsl_all_20261009/main_variants/candidate/traits.cuh","opus_gemm_traits_a8w8_mxscale_bpreshuffle_main_variants_gfx950.cuh",{
       "opus_private_geometry_traits":"opus_gemm_mxscale_bpreshuffle_geometry_traits",
       "opus_private_pin_fixed_traits":"opus_gemm_mxscale_bpreshuffle_pin_fixed_traits",
       "opus_private_pad_pin_fixed_traits":"opus_gemm_mxscale_bpreshuffle_pad_pin_fixed_traits"}),
      ("reports/opus_flydsl_gap_20261009/shortk9022/candidate/traits.cuh","opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh",{
       "opus_private_shortk9022_panel8_traits":"opus_gemm_mxscale_bpreshuffle_shortk_traits"}),
      ("reports/opus_flydsl_all_20261009/hybrid_small/candidate/traits.cuh","opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_variants_gfx950.cuh",{
       "opus_private_small_direct_b_traits":"opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits"}),
      ("reports/opus_flydsl_all_20261009/small_split/candidate/traits.cuh","opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_variants_gfx950.cuh",{
       "opus_private_register_split4_traits":"opus_gemm_mxscale_bpreshuffle_register_split_traits",
       "opus_private_narrow_fine_traits":"opus_gemm_mxscale_bpreshuffle_narrow_fine_traits"}),
      ("reports/opus_flydsl_all_20261009/large9030/traits.cuh","opus_gemm_traits_a8w8_mxscale_bpreshuffle_large_variants_gfx950.cuh",{
       "opus_private_9030_fixed1536_panel16_traits":"opus_gemm_mxscale_bpreshuffle_large_output_panel16_traits",
       "opus_private_9030_directb_a3_chunk96_traits":"opus_gemm_mxscale_bpreshuffle_large_output_direct_b_traits"}),
    ]
    rows=[]
    for oldpath,newname,replacements in pairs:
        original=(ROOT/oldpath).read_text()
        promoted=(GFX/newname).read_text()
        normalized=original
        for old,new in replacements.items():
            normalized=normalized.replace(old,new)
        normalized=normalized.replace("opus_private_small_gcd","opus_gemm_mxscale_bpreshuffle_small_gcd")
        for old,new in replacements.items():
            olddecl=declaration(normalized,new)
            newdecl=declaration(promoted,new)
            assert olddecl==newdecl,(old,new)
            rows.append(dict(original=oldpath,promoted=str((GFX/newname).relative_to(ROOT)),
                             old_symbol=old,new_symbol=new,declaration_sha256=digest(olddecl.encode()),
                             status="byte_identical_after_public_symbol_rename"))
    private=(ROOT/"reports/opus_flydsl_all_20261009/main_variants/candidate/pin.cuh").read_text()
    public=(GFX/"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_pin_fixed_gfx950.cuh").read_text()
    begin='using opus::operator""_I;'
    helpers_old=private[private.index(begin):private.index("template<class Traits>",private.index(begin))].rstrip()
    helpers_new=public[public.index(begin):public.index("} // namespace opus_gemm_mxscale_bpreshuffle_pin_detail")].rstrip()
    assert helpers_old==helpers_new
    hybrid=(ROOT/"reports/opus_flydsl_all_20261009/hybrid_small/candidate/pipeline.cuh").read_text()
    smalllds=(GFX/"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh").read_text()
    marker="#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)"
    reducer_old=hybrid[hybrid.index(marker):]
    reducer_shared=smalllds[smalllds.index(marker):]
    assert reducer_old==reducer_shared
    newhybrid=(GFX/"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_direct_b_gfx950.cuh").read_text()
    newreg=(GFX/"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_split_gfx950.cuh").read_text()
    include='#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"'
    assert include in newhybrid and include in newreg
    assert "void opus_gemm_mxscale_bpreshuffle_reduce_kernel(" not in newhybrid+newreg
    for file in GFX.glob("*variants*gfx950.cuh"):
        assert "opus_private" not in file.read_text(),file
    mapping=json.loads((GFX/"opus_gemm_mxscale_bpreshuffle_promoted_variants.json").read_text())
    newheaders=sorted({GFX/row[key].split("/")[-1] for row in mapping["variants"] for key in ("pipeline_header","traits_header")
                      if "variants" in row[key] or any(word in row[key] for word in ("_geometry_","_shortk_","_pin_fixed_","_small_direct_b_","_register_split_","_large_output_panel16_","_large_output_direct_b_"))})
    assert len(newheaders)==12,len(newheaders)
    result=dict(status="passed",gpu_operations=0,trait_declarations=rows,
                pin_helpers_byte_identical=True,pin_helper_sha256=digest(helpers_old.encode()),
                shared_reducer_byte_identical=True,reducer_sha256=digest(reducer_old.encode()),
                shared_reducer_header=str((GFX/"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh").relative_to(ROOT)),
                reducer_duplicate_definitions_removed=True,new_production_header_count=12,
                production_header_sha256={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in newheaders},
                mapping_sha256=digest((GFX/"opus_gemm_mxscale_bpreshuffle_promoted_variants.json").read_bytes()),
                identity_script_sha256=digest(Path(__file__).read_bytes()))
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"passed","trait_declarations":len(rows),"shared_reducer_byte_identical":True,
                      "pin_helpers_byte_identical":True,"new_production_headers":len(newheaders)},indent=2))


if __name__=="__main__":
    main()
