"""CPU integration checks for per-TU compiler selection and scoped resources."""
import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
UTIL=ROOT/'aiter/jit/utils'

def load(name):
 spec=importlib.util.spec_from_file_location(name,UTIL/(name+'.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module


def functions(path,names,namespace):
 nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace


def test_ninja_executes_override_argv_and_keeps_global_flags(tmp_path):
 compiler=tmp_path/"compiler $ dollar.py"
 receipt=tmp_path/'receipt.jsonl'
 compiler.write_text("import json,sys\nfrom pathlib import Path\nwith Path(sys.argv[1]).open('a') as f:f.write(json.dumps(sys.argv[2:])+'\\n')\nPath(sys.argv[sys.argv.index('-o')+1]).touch()\n")
 argv=[sys.executable,str(compiler),str(receipt)]
 namespace={'os':os,'shlex':shlex,'get_cxx_compiler':lambda:sys.executable,'_posix_path':lambda x:x,'executable_path':lambda _:shlex.join(argv+['baseline']).replace('$','$$'),'IS_HIP_EXTENSION':False,'_is_cuda_file':lambda _:True,'_ninja_path':lambda x:x,'_maybe_write':lambda p,c:Path(p).write_text(c)}
 write=functions(UTIL/'cpp_extension.py',{'_write_ninja_file'},namespace)['_write_ninja_file']
 sources=[tmp_path/'pin.cu',tmp_path/'ordinary.cu']
 for source in sources:source.touch()
 ninja=tmp_path/'build.ninja'
 write(path=str(ninja),cflags=[],post_cflags=[],cuda_cflags=['--global-before'],cuda_post_cflags=['--global-after'],cuda_dlink_post_cflags=[],sources=[str(p) for p in sources],objects=[str(p.with_suffix('.o')) for p in sources],ldflags=[],library_target=None,with_cuda=True,extra_cuda_cflags_per_source={'pin.cu':['--pin-extra']},hip_compiler_commands_per_source={'pin.cu':argv+['pin $ literal']})
 subprocess.run([shutil.which('ninja'),'-f',str(ninja)],cwd=tmp_path,check=True,capture_output=True,text=True)
 rows=[json.loads(line) for line in receipt.read_text().splitlines()]
 assert len(rows)==2
 rows={r[0]:r for r in rows}
 assert set(rows)=={'baseline','pin $ literal'}
 for r in rows.values():assert '--global-before' in r and '--global-after' in r
 assert '--pin-extra' in rows['pin $ literal'] and '--pin-extra' not in rows['baseline']


def test_only_four_public_kernels_override_compiler(monkeypatch,tmp_path):
 module=load('opus_compiler')
 directory=tmp_path/'pin';directory.mkdir();(directory/'clang++').touch()
 monkeypatch.setenv('OPUS_BASELINE_HIP_CLANG_PATH','baseline')
 monkeypatch.setenv('OPUS_HIP_CLANG_PATH',str(directory))
 monkeypatch.setenv('OPUS_HIP_RESOURCE_DIR','pin-resources')
 commands=module.opus_compiler_commands_per_source()
 from csrc.opus_gemm.opus_gemm_common import A8W8_BPRESHUFFLE_TUNING_KIDS,a8w8_mxscale_gemm_bpreshuffle_kernels_list
 import fnmatch
 matched={kid for kid in A8W8_BPRESHUFFLE_TUNING_KIDS if any(fnmatch.fnmatch(a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].name+'_C0.device.cu',pattern) for pattern in commands)}
 assert matched=={9000,9001,9010,9011}
 assert len(commands)==4
 for command in commands.values():assert command[command.index('--compiler')+1]==str(directory/'clang++')
 monkeypatch.delenv('OPUS_BASELINE_HIP_CLANG_PATH')
 assert module.opus_compiler_commands_per_source()=={}


def test_pin_launcher_removes_baseline_resources_and_preserves_arguments(monkeypatch,tmp_path):
 module=load('opus_compiler');calls=[]
 monkeypatch.setattr(sys,'argv',['launcher','--compiler','pin clang++','--resource-dir','pin resources','--','-resource-dir=baseline','-c','source $x.cu','-o','output.o'])
 monkeypatch.setattr(module.subprocess,'call',lambda args:calls.append(args) or 0)
 assert module.main()==0
 assert '-resource-dir=baseline' not in calls[0]
 assert '-resource-dir=pin resources' in calls[0]
 assert calls[0][-4:]==['-c','source $x.cu','-o','output.o']


@pytest.mark.parametrize('fail',[False,True])
def test_mixed_environment_restores_all_resources_on_failure(monkeypatch,tmp_path,fail):
 pin=tmp_path/'pin';base=tmp_path/'base'
 for directory in [pin,base]:directory.mkdir();(directory/'clang++').touch()
 resources=tmp_path/'resources';(resources/'include').mkdir(parents=True)
 cleared=[]
 core=SimpleNamespace(hip_flag_checker=SimpleNamespace(cache_clear=lambda:cleared.append('flags')),check_LLVM_MAIN_REVISION=SimpleNamespace(cache_clear=lambda:cleared.append('revision')),get_hip_version=lambda:'7.0.0')
 # Extract the real context manager, injecting imports without importing HIP.
 import builtins
 original_import=builtins.__import__
 def importer(name,*args,**kwargs):
  if name=='aiter.jit':return SimpleNamespace(core=core)
  if name=='cpp_extension':return SimpleNamespace(ROCM_HOME='/opt/rocm')
  return original_import(name,*args,**kwargs)
 namespace={'__builtins__':{**vars(builtins),'__import__':importer},'contextmanager':contextmanager,'os':os,'Path':Path,'subprocess':SimpleNamespace(check_output=lambda *a,**kw:'clang version 24.0.0'),'re':__import__('re'),'logger':SimpleNamespace(info=lambda _:None),'_DEFAULT_HIP_CLANG_PATH':'missing'}
 context=functions(ROOT/'csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py',{'_opus_compiler_environment'},namespace)['_opus_compiler_environment']
 monkeypatch.setenv('OPUS_HIP_CLANG_PATH',str(pin));monkeypatch.setenv('OPUS_BASELINE_HIP_CLANG_PATH',str(base));monkeypatch.setenv('OPUS_HIP_RESOURCE_DIR',str(resources));monkeypatch.setenv('AITER_HIP_RESOURCE_DIR','external');monkeypatch.setenv('HIP_CLANG_PATH','original');monkeypatch.delenv('OPUS_BASELINE_HIP_RESOURCE_DIR',raising=False)
 previous=dict(os.environ)
 try:
  with context():
   assert os.environ['HIP_CLANG_PATH']==str(base)
   assert 'AITER_HIP_RESOURCE_DIR' not in os.environ
   assert os.environ['OPUS_HIP_RESOURCE_DIR']==str(resources)
   if fail:raise RuntimeError('build failed')
 except RuntimeError:
  assert fail
 assert dict(os.environ)==previous
 assert cleared==['flags','revision']*2


def test_compiler_change_invalidates_versioner(tmp_path):
 module=load('_cpp_extension_versioner');versioner=module.ExtensionVersioner()
 source=tmp_path/'source.cu';source.touch()
 args=dict(name='test',source_files=[str(source)],build_directory=str(tmp_path),with_cuda=True,is_python_module=False,is_standalone=False)
 for expected,compiler in [(0,'compiler23'),(0,'compiler23'),(1,'pin24')]:
  assert versioner.bump_version_if_changed(**args,build_arguments=[[json.dumps({'pin.cu':[compiler]},sort_keys=True)]])==expected
