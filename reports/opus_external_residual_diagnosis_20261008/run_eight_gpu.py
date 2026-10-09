#!/usr/bin/env python3
"""Eight physical GPU locks, eight namespace-safe KFD owners, strict monitoring."""
import argparse,fcntl,importlib.util,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owner_queue',OUT/'run_when_idle_strict.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
spec=importlib.util.spec_from_file_location('picker',ROOT/'.claude/skills/validate-kernel-pr/pick-idle-gpu.py');picker=importlib.util.module_from_spec(spec);spec.loader.exec_module(picker)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);ap.add_argument('--log',type=Path,required=True);args=ap.parse_args();queue=json.loads(args.queue.read_text())
 smi=picker.import_amdsmi();smi.amdsmi_init();locks=[];children={};streams={};fail=False
 def log(record):
  record['time']=time.time()
  with args.log.open('a') as f:f.write(json.dumps(record)+'\n')
  if record['event']!='monitor':print(json.dumps(record),flush=True)
 try:
  while True:
   devices,_=picker.sample(smi,3,1)
   eligible=[g for g in devices if g['hip_index'] is not None and g['peak_gfx'] is not None and g['peak_gfx']<=2 and g['used_gib']<=8 and g['free_gib']>=16 and not smi.amdsmi_get_gpu_process_list(smi.amdsmi_get_processor_handles()[g['smi_index']])]
   eligible.sort(key=lambda g:g['hip_index'])
   if len(eligible)>=len(queue['commands']):break
   log({'event':'waiting','eligible':len(eligible),'devices':devices});time.sleep(15)
  devices=eligible[:len(queue['commands'])]
  for g in devices:
   fd=open(f"/tmp/gpu-{g['hip_index']}.lock",'w');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
   handle=smi.amdsmi_get_processor_handles()[g['smi_index']]
   assert not smi.amdsmi_get_gpu_process_list(handle)
  log({'event':'claimed_eight','devices':devices})
  env=dict(os.environ)
  for k in ['ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:env.pop(k,None)
  env.update(queue['env'])
  # KFD registrations must be serialized so each handshake has one unique PID.
  for item,g in zip(queue['commands'],devices):
   run_env={**env,'HIP_VISIBLE_DEVICES':str(g['hip_index']),'OPUS_EXPECTED_GPU_BDF':g['bdf']}
   log({'event':'start','command':item,'gpu':g,'fingerprint':owner.command_fingerprint(item,queue['env'])})
   stream=Path(item['log']).open('w');streams[item['name']]=stream
   child,identity=owner.launch_owned_python(item,run_env,stream)
   children[item['name']]={'child':child,'owner':identity,'gpu':g}
   log({'event':'owner_identity','name':item['name'],**identity,'gpu':g})
  while children:
   for name,item in list(children.items()):
    child=item['child'];g=item['gpu'];handle=smi.amdsmi_get_processor_handles()[g['smi_index']]
    processes=smi.amdsmi_get_gpu_process_list(handle)
    foreign=[p for p in processes if p['pid']!=item['owner']['host_pid']]
    if foreign:
     log({'event':'external_work_started','name':name,'external':foreign});os.killpg(child.pid,signal.SIGTERM);child.wait();fail=True
    log({'event':'monitor','name':name,'owner_host_pid':item['owner']['host_pid'],'gpu_bdf':g['bdf'],'processes':[{'pid':p['pid'],'vram':p.get('memory_usage',{}).get('vram_mem')} for p in processes]})
    code=child.poll()
    if code is not None:
     log({'event':'end','name':name,'returncode':code,'contamination':bool(foreign),'gpu':g});streams[name].close();del children[name]
     if code:fail=True
   time.sleep(2)
  log({'event':'completed','success':not fail});return int(fail)
 finally:
  for item in children.values():
   child=item['child']
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except Exception:os.killpg(child.pid,signal.SIGKILL);child.wait()
  for s in streams.values():s.close()
  for fd in locks:fd.close()
  smi.amdsmi_shut_down()
if __name__=='__main__':raise SystemExit(main())
