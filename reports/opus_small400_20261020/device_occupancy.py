"""Read-only availability gate; user reassigned this task to GPUs 4-7."""
import datetime
import json
import re
import subprocess

# Verified with hipDeviceGetPCIBusId under ROCR_VISIBLE_DEVICES=4,5,6,7
# and rocm-smi --showbus. Workers use ROCR ordinals; rocm-smi prints DRM IDs.
ROCR_TO_DRM = {4: 7, 5: 4, 6: 6, 7: 5}
DRM_BUS = {4: '0000:85:00.0', 5: '0000:95:00.0', 6: '0000:e5:00.0', 7: '0000:f5:00.0'}

def inspect_devices(gpus=(4,5,6,7)):
    assert gpus and set(gpus) <= {4,5,6,7} and len(set(gpus))==len(gpus)
    physical = [ROCR_TO_DRM[gpu] for gpu in gpus]
    cmd=['/opt/rocm/bin/rocm-smi','-d',*map(str,physical),'--showpidgpus','--showbus']
    result=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=20)
    output=result.stdout+result.stderr
    assert 'GPUs Indexed by PID' in output, 'Cannot establish GPU process occupancy'
    assert 'Unable to get list' not in output, output
    buses = {int(i): b.lower() for i,b in re.findall(r'GPU\[(\d+)\].*PCI Bus:\s*(\S+)', output)}
    assert all(buses.get(i)==DRM_BUS[i] for i in physical), 'PCI mapping changed: '+output
    matches=re.findall(r'PID (\d+) is using (\d+) DRM device\(s\):\s*\n([\d ]+)',output)
    expected=re.findall(r'PID (\d+) is using ([1-9]\d*) DRM device\(s\)',output)
    assert len(matches)==len(expected), 'Cannot parse every active KFD process mapping: '+output
    occupied=[]
    for pid,count,indices in matches:
        devices=[int(v) for v in indices.split()]
        assert len(devices)==int(count),(pid,count,indices)
        assigned=sorted(set(devices)&set(physical))
        if assigned:occupied.append(dict(pid=int(pid),gpus=assigned))
    return dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                gpus=list(gpus),drm_gpus=physical,rocr_to_drm=ROCR_TO_DRM,occupied=occupied,command=cmd,raw=output)

def require_idle_devices(gpus=(4,5,6,7)):
    state=inspect_devices(gpus)
    # The user reassigned the back four on this turn. PID 2158722 holds
    # initialized contexts on all eight, while its workers use the front four.
    # Accept that known context holder only while these assigned devices are
    # inactive and retain their observed initialization-sized allocation.
    known = [r for r in state['occupied'] if r['pid']==2158722]
    if known:
        usage=json.loads(subprocess.check_output(['/opt/rocm/bin/rocm-smi','-d',*map(str,state['drm_gpus']),
            '--showuse','--showmeminfo','vram','--json'],text=True))
        assert all(int(v['GPU use (%)'])<=3 and int(v['VRAM Total Used Memory (B)'])<2*1024**3 for v in usage.values()), usage
        state['authorized_inactive_contexts']=known
        state['usage_before']=usage
        state['occupied']=[r for r in state['occupied'] if r['pid']!=2158722]
    assert not state['occupied'], 'Assigned GPUs already have KFD processes: '+json.dumps(state['occupied'])
    return state

if __name__=='__main__':
    state=inspect_devices();print(json.dumps(state,indent=2))
    raise SystemExit(bool(state['occupied']))
