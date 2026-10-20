import json,os,sys,inspect,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
from report_environment import configure
configure();sys.path.insert(0,str(root))
from aiter.jit import core
args=core.get_args_of_build('module_deepgemm_opus');args['blob_gen_cmd']+=' --extra_kids '+' '.join(map(str,json.loads((p/'compile_ids.json').read_text())))
args={k:v for k,v in args.items()if k in inspect.signature(core.build_module).parameters};args['md_name']='module_deepgemm_opus';core.build_module(**args)
paths=[root/'csrc/opus_gemm/opus_gemm_common.py',*sorted((root/'csrc/opus_gemm/include/gfx950').glob('*.cuh')),root/'aiter/ops/opus/gemm_op_a8w8.py']
(p/'build_fingerprint.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in paths},indent=2)+'\n')
print('Publish build complete',flush=True)
