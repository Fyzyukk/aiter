#!/usr/bin/env python3
"""CPU fake-runtime checks for shared-pointer event/graph orchestration.

Loads only function AST; never imports torch, aiter, HIP or experiment_runner.
This verifies ordering and records, not real GPU graph support or performance.
"""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent

class Tensor:
    next_pointer=100
    def __init__(self,value=0,size=8):
        self.value=value
        self.size=size
        self.pointer=Tensor.next_pointer; Tensor.next_pointer+=1
    def fill_(self,value): self.value=value
    def clone(self): return Tensor(self.value,self.size)
    def data_ptr(self): return self.pointer
    def untyped_storage(self): return SimpleNamespace(nbytes=lambda:self.size)

def exercise(workspace):
    state={'capture':None,'runner_order':[],'graph_replays':[],'fills':0,'resets':0,
           'pool_checked':[],'streams':[]}
    class Stream:
        def wait_stream(self,stream): pass
        def synchronize(self): pass
    class Graph:
        def __init__(self): self.calls=[]
        def replay(self):
            state['graph_replays'].append(self.calls[0][0].label)
            for runner,args in self.calls: runner(*args)
        def reset(self): state['resets']+=1
    class Capture:
        def __init__(self,graph,stream): self.graph=graph
        def __enter__(self): state['capture']=self.graph
        def __exit__(self,*args): state['capture']=None
    class Event:
        def __init__(self,**kwargs): pass
        def record(self,stream): pass
        def synchronize(self): pass
        def elapsed_time(self,end): return 0.05
    cuda=SimpleNamespace(synchronize=lambda:None,Stream=Stream,current_stream=lambda:'main',
                         stream=lambda s:nullcontext(),CUDAGraph=Graph,graph=Capture,Event=Event)
    fake_torch=SimpleNamespace(cuda=cuda,equal=lambda a,b:a.value==b.value)
    class Runner:
        def __init__(self,label,output): self.label=label; self.output=output; self.capture_pointers=[]
        def __call__(self,*args):
            if state['capture'] is not None:
                state['capture'].calls.append((self,args))
                self.capture_pointers.append(tuple(a.data_ptr() for a in args[:6]))
            else:
                args[2].value=self.output
                if workspace: args[5].value=self.output+1
                state['runner_order'].append(self.label)
            return args[2]
    call_args=tuple(Tensor() for _ in range(6))+(9062,)
    pool=[tuple(Tensor() for _ in range(6))+(9062,) for _ in range(2)]+[call_args]
    def profile_arguments(runner,args,iters,rotation,**kwargs):
        assert runner.label=='baseline' and rotation==0 and kwargs=={'with_metadata':True}
        return pool,{'mode':'aiter_automatic','count':len(pool)}
    def check_result(reference,out,guard,partial,partial_guard,**kwargs):
        assert state['capture'] is None
        assert out.value in [1,2]
        if workspace: assert partial.value==out.value+1
        state['pool_checked'].append((kwargs['label'],out.data_ptr()))
        return {'errRatio':0,'output_guards':True,'workspace_guards':True}
    tree=ast.parse((ROOT/'experiment_runner.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='event_confirmation')
    captures=[n for n in ast.walk(function) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
        and i.context_expr.func.attr=='graph' for i in n.items)]
    assert len(captures)==1
    assert set(ast.unparse(n.func) for n in ast.walk(captures[0]) if isinstance(n,ast.Call))=={
        'torch.cuda.graph','range','runner','len'}
    namespace={'torch':fake_torch,'profile_arguments':profile_arguments,'check_result':check_result,
               'statistics':statistics,'json':json,'print':lambda *args,**kwargs:None}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'event_confirmation','exec'),namespace)
    runners={'baseline':Runner('baseline',1),'candidate':Runner('candidate',2)}
    result={}
    namespace['event_confirmation'](runners,call_args,None,None,object() if workspace else None,
        kid=9062,shape=(1,128,128),iters=5,result=result,save=lambda:None)
    assert result['status']=='passed' and len(result['measurements'])==10
    assert result['shared_pool'] and result['rotation']['count']==3
    assert runners['baseline'].capture_pointers==runners['candidate'].capture_pointers
    assert len(runners['baseline'].capture_pointers)==5
    expected=['baseline','candidate','candidate','baseline','baseline','candidate',
              'candidate','baseline','baseline','candidate']
    assert [row['label'] for row in result['measurements']]==expected
    assert state['graph_replays']==['baseline','candidate']+expected
    assert all(len(row['all_pool_checks'])==3 for row in result['measurements'])
    assert all(row['event_total_ms']==0.05 and row['iters']==5 and row['us_per_call']==10
               for row in result['measurements'])
    assert state['resets']==2
    return {'workspace':workspace,'status':'passed','round_measurements':10,'rotation_count':3,
            'checks':['shared pointers','AB/BA order','5-call full replay','per-buffer checks',
                      'same-label output and workspace repeatability','event units','capture has no allocator','graph cleanup']}

def main():
    for name in ['experiment_runner.py','official_smoke.py']:
        path=ROOT/name; compile(path.read_text(),str(path),'exec')
    result={'status':'cpu_fake_runtime_passed_no_gpu_execution','cases':[exercise(False),exercise(True)],
            'limit':'Real GPU graph capture, event durations, numerical correctness and speedups are not verified by this CPU fixture.'}
    (ROOT/'event_orchestration_cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(result['cases'])}))

if __name__=='__main__': main()
