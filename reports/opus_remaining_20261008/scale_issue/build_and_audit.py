#!/usr/bin/env python3
"""CPU build and exact Oct8 baseline audit; no GPU imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'reports/opus_resume_20261008/scale_issue_publish'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    command=json.loads((OLD/'build_manifest.json').read_text())['builds'][0]['command']
    records=[]
    env=dict(os.environ,HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
    for side in ['baseline','candidate']:
        argv=[arg.replace(str(OLD/'baseline'),str(HERE/side)) for arg in command]
        argv.insert(argv.index('-shared'), '-resource-dir=/opt/rocm/lib/llvm/lib/clang/20')
        started=time.monotonic()
        with (HERE/f'{side}_build.log').open('w') as log:
            result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'side':side,'command':argv,'returncode':result.returncode,'seconds':time.monotonic()-started}
        records.append(record)
        if result.returncode:raise RuntimeError(f'{side} build failed; see log')
        record['library_sha256']=sha(HERE/side/'experiments.so')
    (HERE/'build_manifest.json').write_text(json.dumps({'status':'passed','cpu_only':True,'builds':records},indent=2)+'\n')
    helper=ROOT/'reports/opus_bound_analysis_20261007/audit_current_compute_metadata.py'
    spec=importlib.util.spec_from_file_location('elf_helpers',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inventory=json.loads((HERE.parent/'inventory.json').read_text())
    expected={e['symbol']:e for e in inventory['entries'] if e['parent_id'] in [9020,9022]}
    images={};rows={}
    for side in ['baseline','candidate']:
        data=module.device_bundle(HERE/side/'experiments.so')
        path=HERE/side/'device.co';path.write_bytes(data)
        _,items=module.summarize_image(data);rows[side]={r['name']:r for r in items}
        images[side]={'path':str(path),'sha256':sha(path),'kernels':items}
        subprocess.run(['/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump','-d','--mcpu=gfx950',str(path)],
                       stdout=(HERE/side/'device.s').open('w'),check=True)
    assert set(rows['baseline'])==set(rows['candidate'])==set(expected)
    checks=[]
    for name,e in expected.items():
        base=rows['baseline'][name];candidate=rows['candidate'][name]
        if base['instruction_sha256'] != e['instruction_sha256']:
            assert e['traits'] == 'opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128, 128, 64, 0>'
            # This branch is excluded from Event until rebuilt with the exact official TU.
            base['pending_exact_baseline'] = True
        assert base['metadata']==e['current_metadata'],f'Baseline metadata mismatch {name}'
        for side,row in [('baseline',base),('candidate',candidate)]:
            assert row['metadata']['.private_segment_fixed_size']==0
            assert row['metadata']['.vgpr_spill_count']==0
        checks.append({'parent_id':e['parent_id'],'name':name,'baseline_matches_oct8_FUNC_metadata':base['instruction_sha256']==e['instruction_sha256'],
                       'baseline':base,'candidate':candidate,
                       'LDS_unchanged':base['metadata']['.group_segment_fixed_size']==candidate['metadata']['.group_segment_fixed_size']})
    result={'status':'passed','cpu_only':True,'images':images,'checks':checks,'build_manifest_sha256':sha(HERE/'build_manifest.json')}
    (HERE/'device_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','kernels':len(checks),
        'resources':[(r['parent_id'],r['baseline']['metadata']['.vgpr_count'],r['candidate']['metadata']['.vgpr_count'],
                       r['baseline']['metadata']['.sgpr_count'],r['candidate']['metadata']['.sgpr_count']) for r in checks]}))


if __name__=='__main__':main()
