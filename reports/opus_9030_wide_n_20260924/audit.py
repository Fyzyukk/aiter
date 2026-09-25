"""Save device ISA and allocation metadata for the exact pilot library."""
import json
from pathlib import Path
import re
import subprocess
import yaml

HERE=Path(__file__).resolve().parent
LLVM=Path('/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin')
build=json.loads((HERE/'build_manifest.json').read_text())
assert build['status']=='passed'
out=subprocess.check_output([str(LLVM/'llvm-objdump'),'--offloading',str(HERE/'experiments.so')],text=True)
(HERE/'offloading.txt').write_text(out)
objects=list(HERE.glob('experiments.so.*.hip-amdgcn-amd-amdhsa--gfx950'))
assert len(objects)==1,objects
metadata=subprocess.check_output([str(LLVM/'llvm-readobj'),'--notes',str(objects[0])],text=True)
assembly=subprocess.check_output([str(LLVM/'llvm-objdump'),'-d','--mcpu=gfx950',str(objects[0])],text=True)
(HERE/'metadata.txt').write_text(metadata)
(HERE/'experiments.disasm').write_text(assembly)
start=metadata.index('---\n');end=metadata.index('...',start)+3
kernels=yaml.safe_load(metadata[start:end])['amdhsa.kernels']
rows=[]
for kernel in kernels:
    match=re.search(r'ILi([48])ELi(64|128)EEE',kernel['.name'])
    assert match,kernel['.name']
    waves,panel=map(int,match.groups())
    row=dict(waves=waves,scale_panel=panel)
    for key in ['.agpr_count','.vgpr_count','.sgpr_count','.group_segment_fixed_size',
                '.max_flat_workgroup_size','.private_segment_fixed_size','.vgpr_spill_count','.sgpr_spill_count']:
        row[key[1:]]=kernel.get(key,0)
    row['symbol']=kernel['.name']
    assert row['group_segment_fixed_size']==2*(192+256)*128*33//32+192*panel+2*panel
    assert row['max_flat_workgroup_size']==waves*64
    assert row['private_segment_fixed_size']==0
    assert row['vgpr_spill_count']==row['sgpr_spill_count']==0
    rows.append(row)
assert len(rows)==4
result=dict(status='passed',binary_sha256=build['binary_sha256'],variants=rows,
            note='vgpr_count includes AGPR reservation; 8-wave CTAs share each SIMD register file between two waves')
(HERE/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
