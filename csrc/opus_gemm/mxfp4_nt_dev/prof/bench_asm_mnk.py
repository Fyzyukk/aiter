# 8192^3 MXFP4 GEMM: aiter asm (bpreshuffle) vs Triton gemm_afp4wfp4, same random data as our host (random fp4 bytes, e8m0 124..130)
import torch, aiter, sys
from aiter.ops.gemm_op_a4w4 import gemm_a4w4, gemm_a4w4_asm
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
import sys; M,N,K=(int(v) for v in sys.argv[1:4]) if len(sys.argv)>3 else (8192,8192,8192); torch.manual_seed(0); dev='cuda'
x=torch.randint(0,256,(M,K//2),dtype=torch.uint8,device=dev); w=torch.randint(0,256,(N,K//2),dtype=torch.uint8,device=dev)
xs=torch.randint(124,131,(M,K//32),dtype=torch.uint8,device=dev); ws=torch.randint(124,131,(N,K//32),dtype=torch.uint8,device=dev)
fp4=aiter.dtypes.fp4x2; e8=aiter.dtypes.fp8_e8m0
xq=x.view(fp4); wq=w.view(fp4)
wsh=shuffle_weight(wq,layout=(16,16)); xss=e8m0_shuffle(xs.view(e8)); wss=e8m0_shuffle(ws.view(e8))
out=torch.empty(M,N,dtype=torch.bfloat16,device=dev)
def bench(name,f,warm=1000,it=200):
    for _ in range(warm): f()
    torch.cuda.synchronize(); e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True); r=[]
    for _ in range(3):
        e0.record()
        for _ in range(it): f()
        e1.record(); torch.cuda.synchronize(); r.append(e0.elapsed_time(e1)*1e3/it)
    us=sorted(r)[1]; print(f"{name:28s} {us:7.1f} us  {2*M*N*K/us/1e9/1e3:.2f} PFLOPS  {['%.1f'%v for v in r]}", flush=True)
    return out.clone()
ra=bench("aiter asm gemm_a4w4 bpreshuf", lambda: gemm_a4w4_asm(xq,wsh,xss,wss,out,'',None,1.0,0.0,True))
#rt=bench("triton gemm_afp4wfp4", lambda: gemm_afp4wfp4(x,w,xs,ws,torch.bfloat16,out))
#d=(ra.float()-rt.float()); print("asm vs triton rel diff:", (d.norm()/rt.float().norm()).item())
