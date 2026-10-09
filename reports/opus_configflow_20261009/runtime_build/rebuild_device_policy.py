"""Recompile device validation TUs with the runtime's RTC/LICM policy."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ORIGINAL=json.loads((OUT.parent/'device_build/build_receipt.json').read_text())
def build(row):
    directory=OUT/row['pipeline']
    argv=list(row['argv'])
    source=next((directory/'instances').glob('*.device.cu'))
    object_path=directory/'device.o'
    argv[argv.index('-c')+1]=str(source)
    argv[argv.index('-o')+1]=str(object_path)
    argv=[arg for arg in argv if not arg.startswith('-I'+str(OUT.parent/'device_build'))]
    argv+=['-I'+str(directory),'-D__HIPCC_RTC__','-mllvm','--disable-machine-licm']
    result=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (directory/'device_runtime_policy.log').write_text(result.stdout)
    return {'pipeline':row['pipeline'],'argv':argv,'object':str(object_path),'returncode':result.returncode,
            'gpu_queries':0,'kernel_launches':0,'library_loads':0}
with ThreadPoolExecutor(max_workers=3) as executor:results=list(executor.map(build,ORIGINAL['builds']))
(OUT/'device_runtime_policy_result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps([{key:value for key,value in row.items() if key!='argv'} for row in results],indent=2))
sys.exit(any(row['returncode'] for row in results))
