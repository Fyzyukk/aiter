from pathlib import Path
import json,os,sys,ctypes,ctypes.util
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
import torch
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from aiter.jit import core
from aiter.utility import mp_tuner
from aiter.test_common import run_perftest
from op_tests import test_opus_mxscale_bpreshuffle as tests
from run_serial_tune import opus_ref,compare
tune.run_torch=opus_ref;tune.compare_outputs=compare
records=[]
assert tune._ensure_kids_compiled(tune.A8W8_BPRESHUFFLE_TUNING_KIDS) is False
# Signed/cancellation checks for all public families plus important row/K tails.
for kid in sorted(tune.A8W8_BPRESHUFFLE_TUNING_KIDS):
 shape={9000: (256, 7168, 384), 9001: (256, 7168, 384), 9010: (64, 7168, 384), 9011: (64, 7168, 384), 9020: (16, 7168, 384), 9021: (1, 7168, 384), 9022: (16, 7168, 384), 9023: (1, 7168, 384), 9024: (1, 7168, 384), 9030: (16384, 65536, 1536), 9040: (1, 7168, 384), 9041: (1, 7168, 384), 9042: (1, 7168, 384), 9043: (1, 7168, 384), 9044: (1, 7168, 384), 9045: (1, 7168, 384), 9046: (1, 7168, 384), 9047: (1, 7168, 384), 9049: (1, 7168, 384), 9051: (1, 7168, 384), 9052: (1, 7168, 384), 9053: (1, 7168, 384), 9054: (1, 7168, 384), 9055: (1, 7168, 384), 9060: (1, 7168, 384), 9061: (1, 7168, 384), 9062: (1, 7168, 384), 9063: (1, 7168, 384)}[kid]
 shape=(shape[0],shape[1],512 if shape[2]==384 else shape[2])
 tests._check_signed_repeated(kid,shape,2,cancellation=True)
 records.append({'kid':kid,'shape':shape,'signed_cancellation':True,'repetitions':2,'passed':True})
for kid,shape in [(9020,(208,65792,384)),(9021,(16,256,8320)),(9040,(1,256,128)),(9060,(1,512,384)),(9011,(1856,7168,768))]:
 tests._check_signed_repeated(kid,shape,2)
 records.append({'kid':kid,'shape':shape,'signed':True,'repetitions':2,'passed':True})
tuner=tune.OpusMxscaleBpreshuffleTuner();info=('gfx950',256,4096,2048,7168)
tasks=tuner.get_gemm_a8w8_blockscale_cktile_tune_task(info,True,17,True,[1,2],{'num_warmup':5,'num_iters':51})
tasks=[task for task in tasks if task[0][1]==27]
assert len(tasks)==1,len(tasks)
result=mp_tuner.work_group({os.getpid():0},True,.05,(1,()),tasks,verbose=False)
assert len(result)==1 and result[0][1]>0 and result[0][2]<=.05,result
assert result[0][1]<150,('CKTile still anomalous',result)
records.append({'cktile_known_same_candidate':'M4096/N2048/K7168/ID27/splitK0','us':result[0][1],'errRatio':result[0][2],'threshold_us':150})
(OUT/'smoke.json').write_text(json.dumps({'status':'passed','records':records,'gpu':str(torch.cuda.get_device_properties(0)),'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID')},indent=2)+'\n')
print(json.dumps(records[-1]),flush=True)
print('SMOKE_PASSED',flush=True)
