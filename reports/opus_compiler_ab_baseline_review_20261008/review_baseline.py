#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];RUN=ROOT/'reports/opus_clang23_mixed_retune_20261008'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    upstream=ROOT/'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv'
    published=ROOT/'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv'
    assert sha(published)==sha(RUN/'tuned_all_config.csv')
    old=pd.read_csv(upstream);old=old[old.gfx.eq('gfx950')&old.cu_num.eq(256)]
    latest=pd.read_csv(published)
    assert len(old)==len(latest)==745
    compare=pd.read_csv(RUN/'upstream_same_execution_comparison.csv')
    compare['new_latency_change_pct']=(compare.us_new/compare.us_upstream-1)*100
    compare.to_csv(OUT/'upstream_same_execution_review.csv',index=False)
    stats={}
    for name,rows in compare.groupby('libtype'):
        valid=rows[rows.same_execution_valid.eq(True)];ratios=valid.us_new/valid.us_upstream;pct=(ratios-1)*100
        stats[name]={'upstream_rows':len(rows),'same_execution_valid':len(valid)}
        if len(valid):
            stats[name].update(geomean_new_latency_change_pct=float((np.exp(np.log(ratios).mean())-1)*100),median_new_latency_change_pct=float(pct.median()),within_5pct=int(pct.between(-5,5).sum()),slower_over5pct=int((pct>5).sum()),faster_over5pct=int((pct< -5).sum()),within_10pct=int(pct.between(-10,10).sum()),slower_over10pct=int((pct>10).sum()),faster_over10pct=int((pct< -10).sum()))
    outliers=compare[compare.same_execution_valid.eq(True)&compare.new_latency_change_pct.abs().gt(5)].sort_values(['libtype','new_latency_change_pct'],ascending=[True,False])
    outliers.to_csv(OUT/'upstream_outside_5pct.csv',index=False)
    selection=old.merge(latest,on=['gfx','cu_num','M','N','K'],suffixes=('_upstream','_latest'),validate='one_to_one')
    selection['latest_latency_change_pct']=(selection.us_latest/selection.us_upstream-1)*100
    selection.to_csv(OUT/'upstream_vs_latest_selection.csv',index=False)
    transitions=pd.crosstab(selection.libtype_upstream,selection.libtype_latest)
    transitions.to_csv(OUT/'backend_selection_transitions.csv')
    result={'status':'completed','latest_745_file':str(published),'latest_file_sha256':sha(published),'upstream_file':str(upstream),'upstream_file_sha256':sha(upstream),'latest_backend_counts':latest.libtype.value_counts().to_dict(),'upstream_backend_counts':old.libtype.value_counts().to_dict(),'same_execution_comparison':stats,'identity_rules':{'asm':'same kernelName and splitK; numeric ID is enumeration ordinal, 42/133 IDs remapped','cktile':'same numeric ID, kernelName and splitK; IDs8/9 have same name but distinct instantiations','ck':'same numeric ID/name; splitK parameter ignored by unchanged C++ bpreshuffle wrapper, so 4 label differences execute same candidate','triton':'25 upstream rows not part of requested CK/CKTile/ASM/OPUS tune'},'upstream_original_toolchain':'unconfirmed; Clang23 fallback is freshly measured build, not proven reproduction of upstream binaries','bandwidth_units':{'config':'TB/s','raw_profile_and_tuned':'GB/s'},'selection_transition_counts':transitions.to_dict()}
    (OUT/'baseline_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(stats,indent=2));print(transitions.to_string())
if __name__=='__main__':main()
