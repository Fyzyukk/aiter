"""Run one frozen, finite pilot after GPU 7 becomes idle. No GPU import while waiting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--max-wait-seconds',type=int,default=43200)
    args=parser.parse_args()
    assert Path(args.run_id).name==args.run_id
    state_path=HERE/f'{args.run_id}_queue.json'
    if state_path.exists():raise FileExistsError(state_path)
    build=json.loads((HERE/'build_manifest.json').read_text())
    layout=json.loads((HERE/'layout_check.json').read_text())
    audit=json.loads((HERE/'resources.json').read_text())
    assert build['status']==layout['status']==audit['status']=='passed'
    assert audit['binary_sha256']==digest(HERE/'experiments.so')==build['binary_sha256']
    frozen=dict(build['source_sha256'])
    frozen.update(layout['source_sha256'])
    frozen.update({str(HERE/f):digest(HERE/f) for f in
                   ['bench.py','experiments.so','variants.json','targets.csv','build_manifest.json',
                    'layout_check.json','resources.json','run_when_idle.py']})
    state=dict(status='waiting_for_gpu',pid=os.getpid(),start_time=time.time(),gpu_work_started=False,
               protected_sha256=frozen,run_id=args.run_id)
    def save():
        temporary=state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    def verify():
        for p,h in frozen.items():assert digest(p)==h,p
    verify();save()
    try:
        window=[]
        while time.time()-state['start_time']<args.max_wait_seconds:
            card=json.loads(subprocess.check_output(['rocm-smi','-d','7','--showuse','--showmemuse','--json'],text=True))['card7']
            state['last_sample']=dict(time=time.time(),card=card);save()
            clear=int(card['GPU use (%)'])==0 and int(card['GPU Memory Allocated (VRAM%)'])<=1
            window=(window+[card])[-3:] if clear else []
            if len(window)==3 and len({s['GFX Activity'] for s in window})==1:break
            time.sleep(5 if clear else 30)
        else:
            state['status']='expired_waiting_for_gpu';return
        verify()
        command=[sys.executable,str(HERE/'bench.py'),'--prefix',args.run_id,'--rounds','5']
        state.update(status='running_pilot',gpu_work_started=True,command=command);save()
        with (HERE/f'{args.run_id}.log').open('x') as f:
            result=subprocess.run(command,cwd=HERE.parents[1],stdout=f,stderr=subprocess.STDOUT)
        verify()
        state.update(returncode=result.returncode,
                     status='pilot_complete_review_pending' if result.returncode==0 else 'pilot_failed')
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        raise
    finally:
        state['end_time']=time.time();save()

if __name__=='__main__':main()
