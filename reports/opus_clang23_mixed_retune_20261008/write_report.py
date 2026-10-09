from pathlib import Path
import collections,hashlib,json,math,os,shutil,tempfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
summary=json.loads((OUT/'summary.json').read_text());assert summary['status']=='completed'
new=pd.read_csv(OUT/'tuned_all.csv');opus=pd.read_csv(OUT/'tuned_opus.csv');ext=pd.read_csv(OUT/'tuned_external.csv')
cols=['gfx','cu_num','M','N','K','libtype','kernelId','splitK','us','kernelName','tflops','bw','errRatio']
assert list(new.columns)==cols and len(new)==len(opus)==len(ext)==745
# Raw profiler reports GB/s; repository model configs use TB/s.
for name in ['all','opus','external','ck','cktile','asm']:
 data=pd.read_csv(OUT/('tuned_'+name+'.csv'));data['bw']=data['bw']/1000
 data.to_csv(OUT/('tuned_'+name+'_config.csv'),index=False)
assert not new.duplicated(['gfx','cu_num','M','N','K']).any()
assert new.us.gt(0).all() and new.us.map(math.isfinite).all()
old_hashes=json.loads((OUT/'old_artifact_hashes.json').read_text())
for name in ['reports/opus_retune28_20261008/profile.csv','aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv']:
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==old_hashes[name]
# Preserve the previous published all-backend table before replacing it.
target=ROOT/'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv'
backup=OUT/'previous_published_all_backend.csv'
assert hashlib.sha256(target.read_bytes()).hexdigest()==old_hashes[str(target.relative_to(ROOT))]
shutil.copy2(target,backup)
with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.opus-mixed-publish-',suffix='.csv',delete=False) as stream:
 temporary=Path(stream.name)
try:
 temporary.write_bytes((OUT/'tuned_all_config.csv').read_bytes());os.replace(temporary,target)
finally:temporary.unlink(missing_ok=True)
counts=summary['all_backend_counts'];policy=summary['compiler_policy'];same=summary['same_run_opus_vs_external'];oldnew=summary['old_new_best_descriptive_only'];backendold=summary['old_new_each_backend_best_descriptive_only'];upstream=summary['upstream_same_execution_descriptive_only']
def speed(record):return f"{record['geomean_speedup']:.6f}×"
def change(record):return f"{100*(1/record['geomean_speedup']-1):+.3f}%"
backend_table='\n'.join(f"| {name.upper()} | {counts.get(name,0)} |" for name in ['opus','ck','cktile','asm'])
old_table='\n'.join(f"| {name.upper()}最优 | {record['shapes']} | {speed(record)} | {change(record)} |" for name,record in backendold.items())
upstream_table='\n'.join(f"| {name.upper()} | {record['upstream_rows']} | {record['same_execution_measured']} | {record['same_execution_valid']} | {change(record) if record.get('shapes') else '未测'} |" for name,record in upstream.items())
readme=f"""# Clang 23 + pin-AGPR LLVM 全后端重测

完整 **745/745** 个 gfx950/256-CU shape 在 8 张物理 GPU 上测量完成。
全部 28 个 OPUS 候选保留，**13,027/13,027 个合法配置均有效，errRatio=0**。
CK/CKTile/ASM 候选重新完整 tune，总计 {summary['profile_rows']:,} 条原始记录。
同一 shape 的所有后端候选在同一 GPU 上依次比较；8 个进程都核对实际加载库，
KFD/AMD-SMI监控未发现非所属进程干扰。

| 最优后端 | Shape 数 |
| --- | ---: |
{backend_table}

OPUS 相对本轮 CK/CKTile/ASM 最快有效者，几何平均加速 **{speed(same)}**，
{same['faster']} 项更快、{same['slower']} 项更慢。
这是本轮同 shape/同 GPU 数据的比较，公共shuffle和reference不计时；
OPUS使用FP8与native E8M0 scale，外部后端保留原tuner的FP32 scale和输入分布。
小差距受单轮筛选噪声影响；不把历史时间混入新表选型。

## 编译器记录

CK、CKTile、ASM主机封装及24个不使用pin AGPR的OPUS候选使用最小
Clang23 `46fcb339fb61119b337f973c7ca9e710a319fdd0`，
路径 `/opt/rocm-llvm23-46fcb339/bin`。
只有9000/9001/9010/9011使用pin LLVM Clang24
`49c41889681640665400cb01c9fbb4c0a024cde4`，
路径 `/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin`。
ASM设备kernel为既有预编译`.co`。同一`/opt/rocm` SDK；
pin24仅使用ROCm7.0的clang20 resource headers，Clang23使用自己的resource headers。
原上游CSV的tune编译器仍未找到可靠收据，本轮创建新的可复查基线。

28个OPUS对象逐个核对编译器；4个pin候选的设备机器码与上轮完全一致。
88个外部/core对象为Clang23；6个ASM设备二进制身份已保存。
全28候选signed/cancellation校验加5个尾部样例通过，共66次实际输出校验；
60个CPU/JIT检查及10个tuner编译环境检查通过。

## 与上轮重测的描述性比较

此前CK/CKTile/ASM封装用Clang20，OPUS全部用pin24。
下列“上轮时间/新时间”的几何平均同时受GPU分片、地址池、时间漂移及编译器影响，
不能将所有差异只归因于编译器。

| 比较 | 对应Shape | 旧/新GM加速 | 新延迟GM变化 |
| --- | ---: | ---: | ---: |
{old_table}

外部后端最优整体旧/新GM {speed(oldnew['external'])}；
OPUS最优旧/新GM {speed(oldnew['opus'])}；
全后端最优旧/新GM {speed(oldnew['all'])}。
各shape明细、原候选对照及无效记录均保留。
CKTile已摆脱此前Clang20的严重spill退化；样例M4096 N2048 K7168 ID27
初次profiler数值检查80.0586µs，旧Clang20约3.2ms。
该样例不同Event/profiler时间不混用；全量选型以本轮profile为准。

## 与上游保存候选对应

| 上游后端 | 保存Shape | 同执行已测 | 有效 | 新延迟相对上游GM变化 |
| --- | ---: | ---: | ---: | ---: |
{upstream_table}

ASM按kernelName/splitK匹配，数字ID为枚举序号。CKTile同时匹配数字ID，
因为部分候选名字相同。CK B-preshuffle封装接收splitK并计算KBatch，
实际调用未使用它；因此原4行splitK标签不一致不是kernel执行缺失，
全部409个上游CK执行都有对应记录。严格CSV参数匹配仍单独保存。
上游25个Triton shape未在本次候选集合中测量。
这些上游历史时间的对照不证明已复现原工具链或原测试环境。

## 输出与核验

- `tuned_all.csv`：本轮全部后端最优745行；`tuned_all_config.csv`将bw从GB/s转为TB/s，已保存到仓库模型配置CSV。
- `tuned_opus.csv`：OPUS28内部最优745行。
- `tuned_external.csv`：CK/CKTile/ASM最快有效者745行；另有各后端最优CSV。
- `profile.csv`、`batches/*`：完整raw、逐卡进度、合法候选清单。
- `shape_comparison.csv`、`old_new_same_call_comparison.csv`：同轮和历史对照。
- `upstream_same_execution_comparison.csv`、`upstream_same_call_comparison.csv`：上游实际执行/CSV参数对照。
- `compiler_manifest.json`、`build_manifest.json`、`*_commands.txt`、`*_per_object_compiler_audit.json`：构建记录。
- `pin_machine_identity.json`、`asm_device_binary_hashes.json`：设备指令和二进制身份。
- `plan.json`、`queue_eight.json`、`claim_eight.jsonl`、`runtime_mapped_libraries_audit.json`：执行和实际库加载。
- `summary.json`：完整统计；`previous_published_all_backend.csv`保留此前已发布745行。

测量warmup5/iters51、原tuner的torch profiler；大输出地址池上限8并按空闲显存限额，
reference和误差检查按256行分块，表达式不变。OPUS完整FP32/BF16累加范围零outlier；
外部后端保留原误差门槛。无效项按后端统计为 `{summary['invalid_by_backend']}`，全部留在raw中。
本次未枚举Triton。tune时间不能直接作为compute/memory bandwidth/memory latency/dispatch分类证据。
旧 `reports/opus_retune28_20261008/` 原始记录保持不变。
"""
(OUT/'README.md').write_text(readme)
# Rewrite latest-policy status in handoff, leaving historical sections intact.
handoff=ROOT/'HANDOFF_MXFP8.md';text=handoff.read_text();start=text.index('## 2026-10-08：Clang 23 / pin LLVM 分流全量重测');end=text.index('\n## ',start+4)
block=f"""## 2026-10-08：Clang 23 / pin LLVM 分流全量重测已完成

745/745 shape、28个OPUS候选、13,027个合法OPUS配置全部完成且errRatio0；
全后端共{summary['profile_rows']:,}条raw，8张物理卡同时运行，同shape全候选同卡，实际加载库SHA和KFD归属均通过。
本轮最优后端 `{counts}`；OPUS相对同轮外部最优GM {speed(same)}，{same['faster']}项更快、{same['slower']}项更慢。

CK、CKTile、ASM主机封装及24个无pin的OPUS候选用最小Clang23
`/opt/rocm-llvm23-46fcb339/bin`（46fcb339fb61119b337f973c7ca9e710a319fdd0）；
仅9000/9001/9010/9011用pin24
`/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin`
（49c41889681640665400cb01c9fbb4c0a024cde4）。ASM设备保持预编译.co。
原上游tune编译器仍未确认，新Clang23测量建立新基线。
使用混合构建时设置`HIP_CLANG_PATH`和`OPUS_BASELINE_HIP_CLANG_PATH`为Clang23 bin、
`OPUS_HIP_CLANG_PATH`为pin bin，并使用全新`AITER_JIT_DIR`。
4个pin候选机器码与上轮相同；60个CPU/JIT、10个编译环境检查和66次signed/tail输出检查通过。

完整报告：[README](reports/opus_clang23_mixed_retune_20261008/README.md)。
已将`tuned_all_config.csv`写入`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv`；
`tuned_opus.csv`和`tuned_external.csv`另存。旧raw和旧发布表保留。
CKTile全后端选型已恢复有效竞争。CK B-preshuffle的splitK参数实际被忽略：
原4行CSV splitK不同也对应同kernel执行，上游409CK/178CKTile/133ASM共720执行均有对应；
25Triton未测。历史时间变化不是编译器单因素A/B；bound分类需要原ATT/counter证据。
本节取代下文旧策略和旧发布表统计。
"""
handoff.write_text(text[:start]+block+text[end:])
readme_path=ROOT/'csrc/opus_gemm/README.md';text=readme_path.read_text();start=text.index('The [complete tuned CSV]');end=text.index('\n',text.index('The 13-column format is retained;',start)) if 'The 13-column format is retained;' in text[start:] else start
# Preserve historical explanation as a dated section while making current export explicit.
end=text.index('The table can be passed to `-i`',start)
newblock=f"""The [complete tuned CSV](../../aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv)
contains the latest 745-shape compiler-split all-backend results from
[tuned_all_config.csv](../../reports/opus_clang23_mixed_retune_20261008/tuned_all_config.csv):
{counts.get('opus',0)} OPUS, {counts.get('ck',0)} CK, {counts.get('cktile',0)} CKTile, and {counts.get('asm',0)} ASM rows.
OPUS uses native E8M0 scales; CK/CKTile/ASM use FP32 scales; all outputs are BF16.
The production table reports bandwidth in TB/s; raw reports retain GB/s.
The 13-column schema is retained. See the [complete measured report](../../reports/opus_clang23_mixed_retune_20261008/README.md)
for compiler receipts, same-run comparison, historical timings, and raw accuracy results.
The original upstream tune compiler remains unconfirmed; this Clang23 run is a new baseline.

The earlier [Clang20 external-backend run](../../reports/opus_retune28_20261008/README.md)
is retained as historical evidence; its anomalous CKTile timings and mixed
five-round/screening output do not define the current table. The fresh table
uses the complete new screening profile for all 745 selections.

"""
readme_path.write_text(text[:start]+newblock+text[end:])
manifest={'status':'completed','published_path':str(target),'published_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'previous_published_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256((OUT/'profile.csv').read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),'old_raw_unchanged':True,'upstream_csv_unchanged':True,'all_backend_counts':counts,'shapes':745,'opus_valid_tasks':13027}
(OUT/'publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
