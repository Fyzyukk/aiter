#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    baseline=json.loads((OUT/'baseline_review.json').read_text());ab=json.loads((OUT/'compiler_ab_summary.json').read_text())
    protocols=[];raw=[];artifacts={}
    for path in sorted(OUT.glob('cktile_timing[0-7].json')):
        data=json.loads(path.read_text());assert data['status']=='completed';assert len(data['records'])==56;assert all(x['errRatio']<=.05 for x in data['records']);artifacts[path.name]=sha(path)
        case=data['case'];row={**case,'gpu_bdf':data['gpu_bdf']}
        for name,entry in data['summary'].items():row[name+'_us']=entry['median_us']
        row['reuse_5_51_change_vs_rotation_pct']=(row['reuse_5_51_us']/row['rotate_5_51_us']-1)*100
        row['reuse_50_200_change_vs_upstream_pct']=(row['reuse_50_200_us']/row['upstream_us']-1)*100
        protocols.append(row)
        for entry in data['records']:raw.append({**case,'gpu_bdf':data['gpu_bdf'],**entry})
    assert len(protocols)==8
    events=[json.loads(x) for x in (OUT/'claim_cktile.jsonl').read_text().splitlines()];ends=[x for x in events if x['event']=='end'];assert len(ends)==8 and all(x['returncode']==0 and not x['contamination'] for x in ends);assert not any(x['event']=='external_work_started' for x in events)
    pd.DataFrame(protocols).to_csv(OUT/'cktile_timing_protocol_comparison.csv',index=False);pd.DataFrame(raw).to_csv(OUT/'cktile_timing_protocol_raw.csv',index=False)
    (OUT/'cktile_timing_summary.json').write_text(json.dumps({'status':'completed','cases':8,'raw_measurements':448,'method':'fixed current Clang23 binary; 7 forward/reverse blocks per shape, 14 observations/protocol; all4 protocols on same GPU and input data; profiler timing; reuse1 vs automatic rotation (large outputs cap8)','ownership_clean':True,'ownership_monitor_samples':sum(x['event']=='monitor' for x in events),'input_artifact_sha256':artifacts,'protocol_rows':protocols,'interpretation':'Current compiler can achieve upstream or better times on the 5 selected slower cases when reusing addresses. Same-binary protocol variation is proven. Upstream rotation/cache policy remains unknown; cannot assert every historical row used hot cache or all178 differences are explained.'},indent=2)+'\n')
    latest=Path(baseline['latest_745_file']);assert sha(latest)==baseline['latest_file_sha256']
    baseline_table='\n'.join(f"| {name.upper()} | {s['upstream_rows']} | {s['geomean_new_latency_change_pct']:+.3f}% | {s['median_new_latency_change_pct']:+.3f}% | {s['within_5pct']} | {s['slower_over5pct']} | {s['faster_over5pct']} |" for name in ['ck','cktile','asm'] for s in [baseline['same_execution_comparison'][name]])
    timing_table='\n'.join(f"| {r['M']}×{r['N']}×{r['K']} | {r['kid']} | {r['upstream_us']:.3f} | {r['full_tune_us']:.3f} | {r['rotate_5_51_us']:.3f} | {r['reuse_5_51_us']:.3f} | {r['rotate_50_200_us']:.3f} | {r['reuse_50_200_us']:.3f} |" for r in protocols)
    per_shape=pd.read_csv(OUT/'compiler_ab_per_shape.csv');unpin=per_shape[per_shape.role.eq('selected_unpinned')]
    outliers=unpin[unpin.clang23_latency_change_pct.abs().gt(5)].sort_values('clang23_latency_change_pct',ascending=False)
    examples=pd.concat([outliers.head(4),outliers.tail(2)])
    example_table='\n'.join(f"| {r.M}×{r.N}×{r.K} | {r.kernelId} | {r.clang23_median_us:.4f} | {r.clang24_median_us:.4f} | {r.clang23_latency_change_pct:+.2f}% | {r.clang23_slower_rounds}/6 |" for r in examples.itertuples(index=False))
    text=f'''# 745-shape 保存位置、上游基线核对与 OPUS 编译器对照

本轮完整 tune 已结束，745 行保存到仓库配置
`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv`。
它与 `reports/opus_clang23_mixed_retune_20261008/tuned_all_config.csv` 完全一致，
包含 OPUS 693 / ASM 34 / CKTile 10 / CK 8。文件名含 opus，但内容是全后端最优。
OPUS-only 745 行为 `tuned_opus.csv`，外部最优 745 行为 `tuned_external.csv`；
所有 79,504 条候选原始记录为该目录的 `profile.csv`。
本页新增的 A/B 与 CKTile 诊断结果单独留档，没有修改完成的 tune/raw 或配置选择。

## 同执行候选相对上游

对上游 `dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv` 筛选 gfx950/256CU，
得到 CK409 / CKTile178 / ASM133 / Triton25，共745行。
前三种的720个实际执行候选都有本轮有效测量；Triton25不在此次请求的候选范围。
ASM按名字和splitK匹配，42个数字ID只是枚举重排；CKTile必须同时匹配ID，
避免同名ID8/9错误合并。CK4行splitK标签不同，但原封装计算KBatch后不传入
dispatch，实际执行不变。完整身份规则在 `baseline_review.json`。

延迟变化为 `(本轮/上游-1)*100%`，正值表示本轮更慢。

| 后端 | 对应行 | 延迟变化GM | 延迟变化中位数 | ±5%内 | 慢>5% | 快>5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{baseline_table}

CK/ASM整体GM接近，不意味着每行都对齐。CKTile的严重Clang20 spill问题已消除，
但也不能把GM变快当成178行逐项复现。上游原tune工具链和二进制仍无可靠收据。
本轮最小Clang23是按用户授权建立的新测量基线。

## CKTile为什么偏差更明显

已经验证的因素是**计时条件和地址复用**。上游62µs样例由PR#5283写入，
公开描述50warmup/200iters、7组ABBA取中位数；未提供缓存轮换、原二进制或
完整时钟/构建信息。本轮tune用5warmup/51iters、自动地址轮换，超大输出cap8。
历史CSV行来自多次更新，不能假定所有行的原测量条件统一。
参考：[PR#5283](https://github.com/ROCm/aiter/pull/5283)、
[先前计时调查](../cktile_timing_diagnosis_20260925/README.md)。

为避免仅引用旧机器结果，本次固定当前Clang23 CKTile二进制，8卡各测1个shape；
每个shape的4种协议在同卡、同数据上做7个正反序块，每协议14次观测取中位数。
448项输出复查均满足外部原精度门槛，物理卡锁和KFD监控均干净。

| M×N×K | ID | 上游µs | 本轮tuneµs | 轮换5/51 | 复用5/51 | 轮换50/200 | 复用50/200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{timing_table}

前三项和另外两个选取的慢例在复用地址后都接近或快于上游。
ID27由79.23降到60.81，进一步58.71µs，说明该当前二进制有达到62µs的能力；
全量tune的78.29与本次自动轮换79.23也基本对应。该实例新Clang23热循环
VGPR234、scratch0、VGPR/SGPR spills0，旧Clang20热循环曾有604个VGPR spill。
因此这些约25%差距有直接的计时/缓存解释，不能直接称为本次kernel生成退化。
但未公开的上游地址策略仍未知，不能断言上游全用热缓存，也不能把8项诊断外推到178项。
短K的两项本次所有协议均明显快于上游，进一步说明历史值不是统一乘一个系数即可对齐。
其他剩余原因如原LLVM/CK版本、编译参数、时钟和测试方法仍未完全复现。
ASM设备`.co`未重新编译，少一个编译器变量；CK没有此前CKTile同类严重spill。
这可以解释两者相对稳定，但并非证明CK/ASM完全不受计时条件影响。

## OPUS Clang23 vs pin-Clang24

覆盖当前OPUS-only最优中全部589个无pin shape，固定相同kernel ID；另测16个
代码未改变的pin候选参照。每个shape两版在同一进程、同卡、同8地址池交替6轮，
每次5warmup/51iters，并重置轮换起点。固定split-K使用共享预分配workspace，
两版每个shape均通过signed FP8/native E8M0误差区间检查，共1210次输出校验。
605项×12次=7260条计时，8个子进程返回0且无外部进程干扰。

构建保留相同host dispatch和其他181个object；只重编24个无pin BF16 device TU。
pybind封装用相同Clang23源码/参数，仅改模块名宏以避免CPython单相扩展缓存复用；
分别核对两库文件、PyInit地址和进程映射。具体argv和SHA见 `build_manifest.json`。
23使用自身resource23，24因ROCm7.0兼容性使用resource20；对比的是这两个具体
工具链配置，不代表所有Clang23/24版本，也没有为每版独立重tune全部候选。

| 范围 | Shape数 | Clang23相对24延迟GM | 中位数 | ±5%内 | 23慢>5% | 23快>5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无pin最优同候选 | 589 | +0.131% | +0.033% | 565 | 12 | 12 |
| 代码未改pin参照 | 16 | +0.190% | +0.185% | 16 | 0 | 0 |

**整体接近持平，没有证据说23整体明显更慢。** 0.13%与未改代码参照波动相当；
此前跨轮同候选+0.87%不能直接归因于编译器。但9042和9052存在局部稳定退化：
9042的3项慢约38%，9052有9项慢超过5%，这12项全部6轮均为23更慢。
部分其他候选23更快，例如9024六项GM约-10.50%。

| M×N×K | ID | 23µs | 24µs | 23延迟变化 | 23更慢轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
{example_table}

这是固定本次选中候选的对照；仍有156个OPUS-only最优本来就是pin24，没有切到23。
诊断未改当前编译器路由，也未将新测量混入745行tune配置。

## 留档

- `upstream_same_execution_review.csv`、`upstream_outside_5pct.csv`：720个基线候选逐项核对与偏差。
- `upstream_vs_latest_selection.csv`、`backend_selection_transitions.csv`：745行旧新选型。
- `compiler_ab_per_shape.csv`、`compiler_ab_per_kid.csv`、`compiler_ab_raw.csv`、`compiler_ab_summary.json`：589项编译器对照和16项参照。
- `cktile_timing_protocol_comparison.csv`、`cktile_timing_protocol_raw.csv`、`cktile_timing_summary.json`：固定本轮二进制计时诊断。
- `build_manifest.json`、`plan.json`、`shard*.json`、`claim_eight.jsonl`：编译、二进制、计时和GPU归属证据。
- `cktile_plan.json`、`cktile_timing*.json`、`claim_cktile.jsonl`：8项CKTile运行证据。

原始配置bw是TB/s，raw/tuned原始CSV是GB/s，不应直接把单位差当性能差。
配置SHA256：`{sha(latest)}`。
'''
    (OUT/'README.md').write_text(text)
    hashes={str(path.relative_to(ROOT)):sha(path) for path in [latest,ROOT/'reports/opus_clang23_mixed_retune_20261008/profile.csv',OUT/'README.md',OUT/'baseline_review.json',OUT/'compiler_ab_summary.json',OUT/'cktile_timing_summary.json']}
    (OUT/'review_manifest.json').write_text(json.dumps({'status':'completed','file_sha256':hashes},indent=2)+'\n')
    print(pd.DataFrame(protocols).to_string(index=False))
if __name__=='__main__':main()
