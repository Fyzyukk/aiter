#!/usr/bin/env python3
"""Exercise per-device owner detection, missing data and watcher eligibility."""
import importlib.util
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    kfd=module('kfd745_audit',HERE/'kfd_owners.py')
    watch=module('watch745_audit',HERE/'watch.py')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        nodes=root/'topology/nodes';nodes.mkdir(parents=True)
        for name,gid,location in ((0,0,0),(1,23,0x500),(2,24,0x1500)):
            node=nodes/str(name);node.mkdir();(node/'gpu_id').write_text(str(gid))
            (node/'properties').write_text(f'location_id {location}\ndomain 0\n')
        ids=kfd.gpu_ids_by_bdf(root)
        assert ids=={'0000:05:00.0':23,'0000:15:00.0':24}
        proc=root/'proc';proc.mkdir()
        for pid,size0,size1 in ((100,2*1024*1024,0),(101,38*1024**3,0),(102,0,0)):
            p=proc/str(pid);p.mkdir();(p/'queues').mkdir()
            (p/'vram_23').write_text(str(size0));(p/'vram_24').write_text(str(size1))
        q=proc/'102/queues/1';q.mkdir();(q/'gpuid').write_text('24');(q/'type').write_text('0')
        owners=kfd.snapshot(ids.values(),root)
        assert [o['pid'] for o in owners[23]]==[101]
        assert [o['pid'] for o in owners[24]]==[102]
        assert not kfd.external(owners[23],101) and kfd.external(owners[23],102)==owners[23]
        idle={'smi_index':0,'hip_index':1,'peak_gfx':0,'used_gib':.3,'free_gib':287,
              'processes':[],'kfd_owners':[]}
        assert watch.eligible(idle)
        assert not watch.eligible({**idle,'kfd_owners':owners[23]})
        assert not watch.eligible({**idle,'kfd_owners':None})
        assert not watch.eligible({**idle,'processes':[{'pid':1}]})
        assert not watch.eligible({**idle,'smi_index':4})
        # A queue that still exists but lacks metadata is unknown, not removed.
        (q/'type').unlink()
        try:kfd.snapshot(ids.values(),root)
        except FileNotFoundError:pass
        else:raise AssertionError('Incomplete live queue treated as removed')
        (q/'type').write_text('0')
        # Missing per-device accounting for a still-live process is unknown.
        (proc/'101/vram_23').unlink()
        try:kfd.snapshot(ids.values(),root)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing owner data treated as idle')
    real_ids=kfd.gpu_ids_by_bdf();real=kfd.snapshot(real_ids.values())
    report={'status':'passed','kernel_launches':0,'gpu_contexts':0,
            'cases':['registration_only','vram_without_smi','queue_without_vram','own_pid',
                     'unknown_read_rejected','incomplete_live_queue_rejected','physical0_3_restriction'],
            'observed_gpu_ids':real_ids,'observed_owner_pids':{bdf:[o['pid'] for o in real[gid]] for bdf,gid in real_ids.items()}}
    (HERE/'kfd_owners_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
