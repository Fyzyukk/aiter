# 恢复 745 shape 的 OPUS / FlyDSL BMM 优化

2026-10-12 提交归档说明：本目录保留 10 月 9 日实验的源码、配置、冻结
`plan.json`、原始结果和收据。下文时间、PID、等待状态及阶段进度均为当时记录。
归档时 `smoke_progress.json` 有 9/9 项，`fused_smoke_progress.json` 有 10/10 项，
`baseline_progress.json` 仅有 14/745 项；这些进度文件不能替代完整收据审计，
也不是新增候选完成 745 项调优或回放的证明。统一 tuner 此后已修改，
恢复绑定源码 SHA 的实验前须生成新 plan/目录。编译产物、JIT/FlyDSL 缓存、
日志和 `activity.jsonl` 保留在原工作区，未纳入 Git；构建与资源收据保留在本目录。

2026-10-09 用户恢复 GPU 测试并要求持续优化，目标是全部 745 个 shape 的 OPUS
逐项超过 FlyDSL BMM。恢复起点为 `4aea3e5e`，使用当前 5 pipeline 参数化代码。
此前停止测试的要求已被本次恢复指令撤销；历史源码、CSV、库和报告保持原始证据。

按冻结 BMM 行单独核验，历史最快合法 OPUS 为 **428 快 / 317 慢**。
此前 411/334 是 BMM 与 compact mxpsh 的最小值口径。两组数字都是历史结果；
本目录当前有 6/9 个源码 smoke shape 通过，共 124 个数值 variant、118 个
公开 API variant 和 1,364 次 signed 检查；有效 worker 均无数值或 guard 失败。
最初两项是 1x768x7168、16x2048x7168，后续四项及收据见下一段。
这些为数值结果，尚无当前源码有效性能比较，不能据历史结果宣布达到目标。

18:14–18:15的新空闲窗口又完成48x7168x16384、112x768x7168、256x7168x7168、
672x7168x384四个有效smoke，当前6/9、124数值variant/118公开API/1364signed检查。
`validated_smoke_first6.json`及`evidence_audit_smoke6.json`核对全部当前有效收据，
排除三份外部占用中断记录；性能仍0/745、private数值待验证。
最新动态恢复为`RESUME_GPU_PROGRESS_1816.md`，代码/build恢复仍为v19。
18:41 UTC 再次恢复核验与等待，没有新增 GPU 结果；六份有效收据复核通过，
最新等待记录是 `RESUME_WAIT_1841.md` 和 `verified_wait_1841.json`。
18:43 UTC 连续三轮同一 GPU 占用阻塞经独立 AMD-SMI/KFD 查询确认；目标标记 blocked，
原 watcher PID2059925保持运行，空闲后自动续跑。最新恢复为 `RESUME_BLOCKED_1843.md`。

`plan.json` 绑定完整 745 shape、每项合法最快历史 OPUS parent、冻结 FlyDSL BMM 名称、
全部合法候选和 runtime split、当前源码和输入 SHA。baseline 使用该 parent 的当前源码
重编译，不能自动等同于历史机器码或历史性能。

204 组 compile config 包括默认配置和 133 组新增合法参数，覆盖 347,658 个合法
shape/config/split 组合。第一轮 screening 对 runtime split 先测 1/2/4/8/16；完整合法
split 集合保存在 plan，后续可依实测邻域继续细化。新增参数包含 register queue 深度、
local K wave、短 K tile/panel、A LDS + B register ahead、fine tile/stage、C chunk。
不会为新增 tuple 增加公开数字 ID。

CPU 131 项现有回归通过。133 组 extra tuple 已 fresh HIP 编译成功，解析出的 207 个
kernel 实例均无 scratch、VGPR spill 或 SGPR spill。编译入口和完整 argv/object SHA 在
`build_extra.py`、`extra_device_build/receipt.json`；资源解析在 `audit_extra.py`、
`extra_resources.json`。资源通过不能替代 GPU 数值与性能检查。

一次物理GPU0短暂空闲窗口中，worker已取得lock与明确KFD owner并成功构建/加载11个
当前独立配置模块；外部PID随后占卡，worker被监控停止，receipt无效，没有可用数值或计时。
`build_cache.py` 随后在CPU上预构建完整204组实际runtime模块，调用build而不调用
get_module/ensure_config，不加载配置库或提交GPU工作。`cache_build.json`、`cache_build.log`
保存进展。204/204组实际模块已经完成，编译失败0；`audit_cache.py` 核验全部模块SHA
并独立解析309个设备kernel。没有scratch或VGPR spill；旧9023/9024各一个runtime分支
有22个SGPR spill到VGPR，fixedK分支不受影响。全部实际模块无spill的说法不成立。
完整资源证据为`cache_resources.json`，较早partial报告保持不变。

`build_flydsl_cache.py` 用官方BMM compile入口和`COMPILE_ONLY=1`预编译453组
不同对照组合；隐藏全部GPU，指定`FLYDSL_GPU_ARCH=gfx950`，每项检查torch CUDA
未初始化且没有打开`/dev/kfd`。同一cache build lock串行预编译并让monitor继续采样，
锁持有期间不会启动性能worker。453/453已成功，cache文件SHA保存在
`flydsl_cache_manifest.json`。另一个全新进程以RUN_ONLY+COMPILE_ONLY重载全部453组，
缓存逐字未变、GPU上下文和kernel启动均0，见`flydsl_cache_reload.json`。

等待新同卡结果期间，在`fused_register/`独立准备last-arrival split-K机制，
保持当前register计算体/FP32累加，改为tile-major lane fragment partial；
每producer wave发布agent release，再由workgroup原子计数，最后到达者按split次序
读取partial完成唯一BF16转换并重置counter。一个8-tile分组的1D grid组织各split。
`build_fused_register.py`生成并离线编译10个tile/prefetch配置，不修改生产源码。
10个device kernel无scratch/spill；`isa_gate.json`保存release wait、agent atomic、
sc0/sc1 partial load指令检查。单wave版本的多余barrier由编译器消除，多wave保留3次。
`build_fused_controls.py`另编译10个参数完全相同的公开OPUS配置用于隔离fusion收益。
这些属于待GPU数值和性能验证候选，尚无收益结论。

`fused_lds/`另有6个独立LDS last-arrival候选，复用当前LDS计算体和相同发布/汇总协议，
M48xN64、M64xN64、M64xN128各stages3/4。公开匹配参数已在原204个缓存中。
`build_fused_lds.py`离线编译6/6通过，VGPR46–102，scratch/spill0；
`audit_fused_candidates.py`对两种private库共16个配置检查2496个完整grid枚举、
83936个合法split/resource边界、tile fragment完整不重叠和实际C ABI adapter选库。
静态ISA发布/原子/sc0-sc1 load/reset检查通过；动态同步、数值和性能尚待GPU。
v12在同一smoke和fusion tune中测试register与LDS，CSV区分
`private_register_fused`和`private_lds_fused`，两者都需要生产集成后再验证。

`watch.py` 持续检测 GPU，延续最近的设备约束只使用物理 SMI 0..3。
空进程列表、连续三次 gfx≤2%、使用显存≤8GiB、可用≥16GiB、物理 GPU lock 和
KFD host PID handshake 都通过后才启动一个 worker；SMI/HIP/BDF 映射实际查询。
v17还要求对应物理BDF的KFD owner为空：任何queue或超过8MiB的显存分配都算占用。
观察到的2MiB且无queue的持久注册可以忽略。读取缺失但PID仍存在时视为未知，
不会领取GPU；worker运行中出现外部KFD owner或未知状态则停止本worker并作废收据。
这样可识别AMD-SMI进程列表暂为空但仍有38GiB分配的实际占用。
监控每两秒检查归属；外部任务重新占卡则停止本 worker，保留无效 receipt，重新等待。
不会停止外部进程。

自动流程是代表 shape数值smoke → 745 当前 parent 基线 → 新实测慢项/不稳定项调优 →
仍慢项的完整合法runtime split集合调优 → M≤2048未解决项的private fusion调优 →
745 完整回放。register fusion限制M≤512，LDS fusion允许M≤2048；每项按公开配置合法域筛选。
private fusion另有十shape数值smoke，覆盖745集合全部七种K及M1088，验证signed/reference、输出和
partial/counter前后guard、每次重复后counter归零。private候选胜出会明确标记
`opus_private/private_register_fused`或`private_lds_fused`，不能直接作为公开参数CSV使用；仍须生产集成和
新的745公开API回放才能完成任务。smoke按shape独立保存，
覆盖29种机制组合、145个验证任务，不进行性能测量。
每项先验证实际公开参数 API，再测 PreparedConfig 的同一 producer/reducer
完整调用；kernel 指标不包含 CPU launch 开销。OPUS/BMM 共享原生 E8M0 scale 底层地址，
BMM 连续描述符为零拷贝 view；公共 shuffle、参考计算在计时外，BMM 内部 split/counter
等 GPU 活动完整计入。保持 profiler warmup5/iters51、reuse1/实际pool≤8、六轮正反序。

v10补齐公开入口覆盖：seed29对每个公开候选（含replay选中的参数）验证公开API，
seed101对所有公开finalist再次验证；输出保存`public_api_checks`。
`audit_public_api.py`在CPU上执行实际AST提取的adapter，检查公开/Prepared参数顺序、
workspace精确切片、direct不传workspace，以及数值/输出guard/workspace越界拒绝。
private实验仍不能声明公开API通过。旧基线numerical失败按实际split标签立即终止。
旧v9脚本SHA归档于`harness_revisions/`；本次只有CPU合同检查，GPU结果仍为0。
v11扩展private smoke到9个shape/949项OPUS任务（不含BMM），
其中470项为private，479项为公开配置；清单检查在`fused_smoke_coverage.json`。
v12加入6个LDS候选后为1513项任务，其中752项private、761项公开；
当前覆盖清单为`fused_smoke_coverage_v12.json`，v11清单保留为历史证据。
v13按各候选合法M范围筛选并加入M1088 smoke，目前10shape/1562项任务，
其中776项private、786项公开；当前清单为`fused_smoke_coverage_v13.json`。
v14每轮计时结束后复查本轮实际访问的所有地址输出，按原FP32误差界验证OPUS，
BMM沿用共同误差比例；检查在profiler计时之外。输出写入每轮的
`post_timing_error_ratios`，失败立即保存`post_timing_numerical`记录并停止批次。
这用于检查连续重复调用与workspace/counter复用，GPU结果仍待空闲卡。

v15新增`fused_wavek/`6组register local WaveK2/4 + global last-arrival候选，
保持原workgroup内FP32求和，由K-wave0发布全局partial，所有wave留在后续barrier。
tile16x16、16x32、32x32各WaveK2/4，prefetch3、Bcache2。
`build_fused_wavek.py`6/6离线编译，VGPR72–138，静态LDS1028–12292字节，scratch/spill0。
`build_wavek_controls.py`6/6精确公开对照已预编译，无库加载、GPU上下文或kernel启动。
当前共22组private配置、3个独立库，沿用相同FP32 partial/counter工作区合同。
scalar/ISA adapter检查更新为3432次完整grid、106088项合法split/resource检查，
包含global partition再分配至local wave时的空分区。
smoke当前10shape/2126项OPUS任务，1058private、1068公开，见`fused_smoke_coverage_v15.json`。
所有private候选数值/动态同步/性能仍待GPU，不声明新性能胜项。
v16缩短owner-free采样轮询间隔至2秒，保持三次活动采样、empty processes和显存门。
baseline745与replay745按shape独立worker/receipt，避免短窗口中8shape整批被作废。
benchmark源码SHA未改，已有两项有效smoke自动复用。
v17的性能收据绑定`kfd_owners.py`和watcher SHA；性能结果复用必须匹配KFD helper SHA。
已有两项数值smoke保持有效。`audit_kfd_owners.py`验证无SMI的显存占用、仅queue占用、
自身PID过滤、未知读取拒绝以及物理前四卡限制，不创建GPU上下文或启动kernel。
最新恢复入口为`RESUME_V17.md`，实际运行状态仍以`status.json`及存活PID为准。
v18进一步区分KFD队列已消失与队列仍存在但元数据缺失；后者保持未知，阻止claim并作废运行中批次。
新增回归检查通过。最新恢复更新为`RESUME_V18.md`，PID2052695、exec session39225。
v19修复private host合同：register限制M≤512；register/WaveK counter要求4bytes对齐；
LDS sfa/sfb要求16/4bytes对齐。旧库完整保存于`private_libraries_before_v19/`。
三库离线重编译通过，22个device镜像SHA和资源不变；无新增scratch/spill。
`audit_fused_host.py`提取实际C++ traits/参数头/完整host launcher，仅替换device函数声明和HIP launch，
host-only HIP编译后plain C++链接；readelf确认测试程序不依赖HIP/HSA再在CPU执行。
88合法launch计划及279非法调用拒绝通过，host/device参数块均120bytes；grid/LDS/指针/尺寸匹配。
v19 启动时数值 smoke 为 2/9；后续进度以本文开头及 `RESUME_GPU_PROGRESS_1816.md` 为准。
代码/build 恢复为 `RESUME_V19.md`，PID 2059925、exec session 23410；当前性能 0/745。

数值检查使用独立 FP32 参考与绝对乘积误差界，signed seed29/101，NaN 预填，重复调用，
输出/workspace 前后各 4096 元素 guard；FP32 参考按 M256 chunk 构造和检查，
B的dequant和绝对值跨M chunk复用。三个signed CPU样本的两张参考平面与原函数逐值一致。
候选数值失败不参与选型。先两轮 screening，再对 parent/前三名与 BMM 六轮确认，最后
完整回放避免单次最小值偏差。稳定胜出要求两种协议 median 都快且各至少五轮快。

启动或恢复（成功且 SHA 一致的子任务自动跳过）：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -u \
  reports/opus_optimize745_20261009/watch.py
```

同目录 watcher lock 防止重复启动。`status.json` 是最新状态，`activity.jsonl` 保留全部
采样与启动/结束记录；每个子任务有 cases、raw JSON、log 和 receipt。
源码或输入发生变化时拒绝混用结果，下一轮须使用新 plan/目录。
只有有效 receipt 的完整结果参与聚合和回放；最后输出可由参数API读取的
`selected745.csv`，保存完整参数、split、实测时间、BMM时间与稳定胜出状态。
private fusion行仅为实验选型，正式参数API暂不识别。
正式 tuned CSV 尚未写入。

`audit_results.py` 在CPU上重新核对收据与源码/模块/结果SHA、物理卡映射、signed seeds、
公开API检查、每轮计时后数值检查，并由原始六轮重算两协议median与稳定胜负。
最终还核对745行selected CSV的参数/split/时间与对应replay结果；不完整记录不能通过。
公开选型逐shape调用真实CPU `select_config`，要求source为tuned且参数identity/split精确一致。
该工具只审计候选证据，生产集成及正式tuned选型公开API回放仍需要单独最终验证。
输出使用新文件名保存，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  reports/opus_optimize745_20261009/audit_results.py \
  --output reports/opus_optimize745_20261009/evidence_audit_latest.json
```

exit2表示证据尚未齐，exit1表示审计失败，exit0表示745个公开候选证据齐备。
`target_completed`保持false，避免将候选证据自动当作生产目标验收。

如果第一轮参数搜索后仍有慢项，`summary.json` 会保存完整 unresolved shape 与实测差距，
状态设为 `more_kernel_optimization_required`。这表示继续源码机制优化的交接点，
不表示任务目标已完成，也不自动声称 745 项全部超过 BMM。
