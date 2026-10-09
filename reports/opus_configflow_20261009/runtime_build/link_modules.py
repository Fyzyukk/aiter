"""Link scalar module validation artifacts, without loading them."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
results=[]
for pipeline in ('pin','tiled','register','lds','large_output'):
    directory=OUT/pipeline
    device=directory/'device.o'
    artifact=directory/('module_offline_'+pipeline+'.so')
    argv=['c++','-shared','-Wl,--no-undefined','-Wl,--allow-shlib-undefined',
          str(directory/'all_instances_host_gfx950.o'),str(directory/'bpreshuffle_config_dispatch.o'),
          str(directory/'bpreshuffle_config_pybind.o'),str(device),
          '-L/opt/rocm/lib','-lamdhip64','-lpython3.10','-o',str(artifact)]
    result=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (directory/'link_argv.json').write_text(json.dumps(argv,indent=2)+'\n')
    (directory/'link.log').write_text(result.stdout)
    receipt={'pipeline':pipeline,'returncode':result.returncode,'artifact':str(artifact),'loaded':False}
    if result.returncode==0:
        receipt['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
        symbols=subprocess.check_output(['nm','-D','--undefined-only',str(artifact)],text=True)
        (directory/'dynamic_undefined.txt').write_text(symbols)
        unresolved=[line for line in symbols.splitlines() if 'opus_gemm' in line]
        receipt['unresolved_opus_symbols']=unresolved
    results.append(receipt)
(OUT/'link_result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
sys.exit(any(result['returncode'] or result.get('unresolved_opus_symbols') for result in results))
