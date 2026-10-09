#!/usr/bin/env python3
"""Report measured cache effects without attributing unknown historical differences."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_owner(name):
    events=[json.loads(line) for line in (OUT/name).read_text().splitlines()]
    ends=[x for x in events if x['event']=='end']
    assert len(ends)==8 and all(x['returncode']==0 and not x['contamination'] for x in ends)
    assert not any(x['event']=='external_work_started' for x in events)
    return {'children':8,'all_return0':True,'monitor_samples':sum(x['event']=='monitor' for x in events)}
def main():
    rows=[];raw=[];ring=[];hashes={}
    for path in sorted(OUT.glob('shard[0-7].json')):
        data=json.loads(path.read_text());assert data['status']=='completed';hashes[path.name]=sha(path)
        for entry in data['rows']:
            assert len(entry['protocol_records'])==56 and len(entry['ring_records'])==24
            case=entry['case'];row={**case,'gpu_bdf':data['gpu_bdf'],'input_bytes_per_copy':entry['ring_input_bytes_per_copy'],'tensor_bytes_per_copy':entry['ring_pool_bytes_per_copy']}
            for name,value in entry['protocol_summary'].items():row[name+'_us']=value['median_us']
            for name,value in entry['ring_summary'].items():row['pool'+name+'_us']=value['median_us']
            row['reuse50_vs_upstream_pct']=(row['reuse_50_200_us']/row['us_upstream']-1)*100
            row['reuse5_vs_rotate5_pct']=(row['reuse_5_51_us']/row['rotate_5_51_us']-1)*100
            row['pool64_vs16_pct']=(row['pool64_us']/row['pool16_us']-1)*100
            rows.append(row)
            for point in entry['protocol_records']:assert point['errRatio']<=.05;raw.append({**case,'gpu_bdf':data['gpu_bdf'],**point})
            for point in entry['ring_records']:assert point['errRatio']<=.05;ring.append({**case,'gpu_bdf':data['gpu_bdf'],**point})
    assert len(rows)==16
    frame=pd.DataFrame(rows)
    plateau_raw=[];plateau_rows=[]
    for path in sorted(OUT.glob('plateau[0-7].json')):
        data=json.loads(path.read_text());assert data['status']=='completed';hashes[path.name]=sha(path)
        for entry in data['rows']:
            assert len(entry['ring_records'])==18
            case=entry['case'];row={key:case[key] for key in ['M','N','K','libtype']};row['plateau_gpu_bdf']=data['gpu_bdf']
            for name,value in entry['ring_summary'].items():row['extended_pool'+name+'_us']=value['median_us']
            row['pool256_vs128_pct']=(row['extended_pool256_us']/row['extended_pool128_us']-1)*100
            row['pool256_vs64_pct']=(row['extended_pool256_us']/row['extended_pool64_us']-1)*100
            plateau_rows.append(row)
            for point in entry['ring_records']:assert point['errRatio']<=.05;plateau_raw.append({**case,'gpu_bdf':data['gpu_bdf'],**point})
    assert len(plateau_rows)==16
    frame=frame.merge(pd.DataFrame(plateau_rows),on=['M','N','K','libtype'],validate='one_to_one')
    assert frame.gpu_bdf.eq(frame.plateau_gpu_bdf).all()
    frame.to_csv(OUT/'protocol_comparison.csv',index=False)
    pd.DataFrame(raw).to_csv(OUT/'protocol_raw.csv',index=False)
    pd.DataFrame(ring).to_csv(OUT/'ring_pool_raw.csv',index=False)
    pd.DataFrame(plateau_raw).to_csv(OUT/'extended_pool_raw.csv',index=False)
    selected=pd.read_csv(ROOT/'reports/opus_compiler_ab_baseline_review_20261008/compiler_ab_per_shape.csv');selected=selected[selected.role.eq('selected_unpinned')]
    routed=selected.kernelId.isin([9042,9052])
    compiler_policy={'default':'minimal Clang23 46fcb339','pin24_required':[9000,9001,9010,9011],'pin24_recommended_per_candidate_after_AB':[9042,9052],'recommendation_applied_to_build_routing':False,'evidence_scope':'589 current unpinned OPUS-only winners, same-ID23/24AB; not all legal shapes/candidates or a new compiler-specific full tune','projected_fixed_winner_latency_GM_change_pct_589':float((np.exp(np.where(routed,np.log(selected.clang24_median_us/selected.clang23_median_us),0).mean())-1)*100)}
    summary={'status':'completed','physical_gpus':8,'cases':16,'cases_by_backend':frame.libtype.value_counts().to_dict(),'protocol_measurements':len(raw),'ring_measurements':len(ring),'extended_pool_measurements':len(plateau_raw),'total_measurements':len(raw)+len(ring)+len(plateau_raw),'ownership':{'initial':validate_owner('claim_eight.jsonl'),'extended':validate_owner('claim_plateau.jsonl')},'cache_effect_scope':'Same-binary/input/GPU protocol variation and ring-pool sensitivity are established. Original upstream address policy, clock and build unconfirmed. Do not assign every historical difference to cache.','hardware_cache_kib':{'L2':4096,'L3':262144},'cache_policy_recommendation':'Keep latency definitions explicit. Streaming input test uses shared balanced ring exceeding L3 and measured pool-size plateau; steady-state inference test rotates A/output while keeping real reused B fixed. Use identical protocol per compared backend; direct profiler device time; do not time cache eviction/memcpy as GEMM.','compiler_policy_recommendation':compiler_policy,'input_sha256':hashes,'rows':frame.to_dict('records')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def fmt_table(data):
        return '\n'.join(f"| {r.libtype.upper()} {r.M}×{r.N}×{r.K} | {r.us_upstream:.3f} | {r.us_new:.3f} | {r.reuse_5_51_us:.3f} | {r.reuse_50_200_us:.3f} | {r.reuse50_vs_upstream_pct:+.1f}% |" for r in data.itertuples(index=False))
    slow=frame[frame.new_latency_change_pct.gt(5)]
    plateau_table='\n'.join(f"| {r.libtype.upper()} {r.M}×{r.N}×{r.K} | {r.input_bytes_per_copy/2**20:.2f} | {r.extended_pool64_us:.3f} | {r.extended_pool128_us:.3f} | {r.extended_pool256_us:.3f} | {r.pool256_vs128_pct:+.2f}% |" for r in frame.itertuples(index=False))
    text=f'''# OPUS编译器建议与CK/ASM缓存计时核对

建议OPUS继续默认Clang23，9000/9001/9010/9011使用pin LLVM24；
同卡A/B有稳定收益的9042/9052建议单独使用同一个pin LLVM24编译。
整体全切24并无明显收益（589同候选GM23比24慢0.131%，未改代码参照也波动0.190%）；
23在9024等候选更快，9042/9052有局部显著退化。固定589winner，仅将这两个候选
换成A/B24值的GM延迟预测变化为{compiler_policy['projected_fixed_winner_latency_GM_change_pct_589']:.3f}%。
该预测不是重tune结果，尚未修改编译器路由或745行配置；用户此前要求无pin统一23仍保留。
详见[编译器同卡A/B](../opus_compiler_ab_baseline_review_20261008/README.md)。

## 缓存影响如何控制

地址复用本身是实际运行条件：steady-state权重B可能常驻缓存；流式输入可能需从HBM读。
若要避免benchmark仅反复同地址得到偏乐观值，要轮换足够多的独立存储地址，
明确预热/迭代，固定相同输入池与平衡候选顺序，并扩大地址池直到延迟稳定。
这台MI355X通过AMD-SMI核对L2=4MiB、L3=256MiB（`hardware_cache_info.json`），
只看4MiBL2或固定8地址不能保证消除缓存复用。
可在计时外读写eviction buffer、同步后只记录GEMM device time，但这定义的是人为冷缓存；
`torch.cuda.empty_cache()`仅释放allocator空闲块，不是GPU硬件缓存清除。
实际推理性能评估应额外保留“B固定、A/out轮换”的条件，避免去掉真实权重缓存收益。
所有后端必须按同一声明的协议比较；上游未知协议的历史us不能直接混进新选型。

## CK/ASM是否也受缓存影响

8卡共测16个代表shape（CK8/ASM8）：慢例8、快例4、近基线4。
每个shape固定本轮Clang23二进制、候选身份、FP32 scales/数据和GPU。
5/51与50/200的自动轮换/单地址复用4协议各14次观测取中位数，
另用显式1/4/16/64地址池统一128warmup/256iters、每池6次观测检查容量变化。
追加64/128/256地址池统一256warmup/512iters、每池6次观测。
总共{len(raw)+len(ring)+len(plateau_raw)}条计时与输出复查通过，两轮8子进程均return0，
物理卡锁/KFD监控无外部进程干扰；原完整tune与发布745行未修改。

| 后端/shape | 上游µs | 全量tuneµs | 复用5/51µs | 复用50/200µs | 复用50/200相对上游 |
| --- | ---: | ---: | ---: | ---: | ---: |
{fmt_table(slow)}

缓存复用能解释部分差距：ASM64×6144×7168复用50/200达到14.84µs，
上游14.99；ASM112×768×7168达到9.54，上游9.29；ASM992×768×7168达到19.02，上游18.53。
CK256×7168×384达到5.91，上游5.80。
但CK96×7168×7168仍27.71vs23.24（约+19%），
ASM240×768×7168仍12.87vs10.81（约+19%），不是仅换复用模式就能对齐。
CK608×7168×768这类新测约16µs/上游36µs的快例，四种协议均明显快于历史值；
因此不能给所有CK/ASM或CKTile历史差距统一归因/乘修正系数。
未解释残差包括尚无收据的原工具链/CK版本/编译参数/时钟/计时与运行环境。
ASM设备`.co`保持本地既有二进制，host封装重编，原上游运行时二进制未取得；
“本地未重编”不等于已证明与上游同一哈希。

## 扩大地址池后的稳定性

此表所有输入/输出随池轮换，数值是固定候选256warmup/512iters设备时间中位数；
输入量只计A/B/scale，不计输出。不同shape单份输入量差异很大，避免固定池数假设。
接近平台并非严格证明每级缓存完全冷；输出写入也影响工作集。

| 后端/shape | 单份输入MiB | 64地址µs | 128地址µs | 256地址µs | 256相对128变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
{plateau_table}

完整16项协议对照在`protocol_comparison.csv`；原始值分别在`protocol_raw.csv`、
`ring_pool_raw.csv`和`extended_pool_raw.csv`；构建身份、归属、计时输出在plan/shard/claim文件。
所有结论限制于这些固定代表shape；没有据此宣称全部409CK/133ASM逐项已复现上游。
'''
    (OUT/'README.md').write_text(text)
    print(frame[['libtype','M','N','K','reuse50_vs_upstream_pct','pool256_vs128_pct']].to_string(index=False))
if __name__=='__main__':main()
