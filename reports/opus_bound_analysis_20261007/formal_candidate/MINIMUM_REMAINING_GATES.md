# 三文件待采用差异的最小充分验证范围

本审查只读取已有源码、计划、二进制及CPU汇总；未运行GPU，未改动态计划。它限定本次采用决策，避免把新的接口或新的机制引入当前验证。

## 1. 本次改动和已经充分的证据

| 改动 | 需要防止的错误 | 已有证据 |
|---|---|---|
| 9021/9022 prologue顺序与partial wait | K0/scale未就绪、K1在途发布、环槽换代错误、loops1错误、M尾块或第五repeat异常 | 32项机制用例全部signed8rep通过；包括K128/256、奇偶、4096/4224、8192/8320、16384及M尾块 |
| 9042/9053/9054三个alias显式B-scale复用 | N方向复用了错误scale、queue覆盖或wave-K短尾错误 | 模板断言`128 % BlockN == 0`；BlockN32/64，tile起点`col=block_x×BlockN`，每个tile完全在一个N128 scale组内，故`floor((col+nr)/128)=floor(col/128)`。已有真实shape Event及数值；还需执行现成5项mechanism与有限支持域 |
| 上述body进入正式TU | 选错alias/variant、ABI/资源变化、编译器生成不同代码 | 隔离正式CPU构建26parent/56entry通过，5改变entry匹配private指令/完整metadata/归一化descriptor，51不变entry匹配baseline；generated TU/host impl均相同 |

B-scale静态等价证明依赖当前三个BlockN32/64 alias及N128外部scale契约。改变的不是K累加顺序或输出归约，现有独立FP32误差区间契约保持适用。三个alias默认值仍为false；K7168 fixed9071/9073/9072不改变。

## 2. Rank3 batch1不属于必要正向GPU验证

当前公开`opus_gemm`在 [__init__.py](../../../aiter/ops/opus/__init__.py) 传入rank2；[dispatch.py](../../../aiter/ops/opus/dispatch.py:134) 要求所有矩阵和scale与该rank一致。`opus_bmm`传rank3，但同一dispatch的 [221行](../../../aiter/ops/opus/dispatch.py:221) 对`a8w8_blockscale_bpreshuffle`所在family明确要求`operation == 'opus_gemm'`，否则报GEMM-only错误。

Generated C++ launcher含rank2/rank3 batch1检查是底层遗留能力，不能据此声称当前9000系列公开支持rank3。此次三文件没有修改Python接口、rank处理、host参数或batch逻辑，因此**不新增rank3正向GPU采用条件**。如果以后要公开支持rank3，这是独立API工作，须另行设计与测试。

## 3. 16B storage offset无需成为本次强制扩大项

当前host要求XQ/WQ/Y的`data_ptr()%16==0`，9021/9022还检查SFA16B对齐。它直接取tensor `data_ptr()`，按m/n/k设置相对stride；见 [codegen](../../../csrc/opus_gemm/codegen/gen_instances_gfx950.py:3199)。此次源码差异没有改地址表达式、buffer extent、指针获取、对齐假设、bounds或向量宽度。16B偏移仍保持所需对齐，改变顺序及B-scale复用不依赖allocation base低位。

现有共享51地址池已经使两侧覆盖不同allocation。显式storage-offset调用可作为轻量API补充，但**不是此次采用必须新增的GPUgate**。只有正式smoke出现指针/布局异常，或后续修改地址/bounds/对齐机制时，才需要围绕该风险追加测试。

## 4. 真正尚未完成的有限条件

### 4.1 机制正确性与实际使用路径，不要求完成全部fallback穷举

Prologue32项机制正确性已经完成；截至本CPU快照，另有528项支持域数值全部clean，覆盖M1..2048、7种K和6种N。这远多于最初机制最小集。两处改动仅重排单WG起步预取及partial wait，不改变mainloop、grid、地址公式或scale布局；大M增加独立WG数量，没有新的跨WG状态。因此**不把剩余1426总组合的全fallback数值扫完作为采用硬条件**。201个实际赢家Event命令本身还含8次数值/guard检查，并覆盖两parent实际使用的K/N/grid。

Register现成5项机制用例仍应执行：它们补K partition/queue full-tail邻界、M33、N384多个N128组，9042 K1664和9053 K2176不在745集合的K列表中。9054的K1024是exact full，实际赢家/support的K384/768/1536已可覆盖空wave、波间不均和full+tail，无需新增K1152。B-scale等价证明、完整数值及实际6赢家覆盖提供核心证据；646项runtime支持计划是额外拓展，**不要求为采用把全部646扫完**。224个不变fixed组合已有机器身份，可保留正式smoke中的有限control。

已完成的支持域数值保留有效，未完成或污染组合明确标partial，不声称全域通过。有限目标充分之后停止可选拓展，不为了“所有合法fallback都零回退”的不可证明表述重复穷举。若特定支持路径出现与当前机制相矛盾的数值失败，应修复对应机制；否则不因shape未逐项枚举就阻止采用。

### 4.2 完成当前赢家的完整Event和screen异常复核

- Prologue：现成201个当前赢家（42+159）共享池51calls×5AB/BA。可按parent+shape及相同device身份计入已经完成的8个独立实际赢家，剩余193个；若主线程选择统一新批次则仍执行原201队列，不能将重复调用算作新增shape覆盖。
- Register：6个实际改变赢家已经完整Event。它们使用原full实验.so，但这三个entry在full、scoped及正式candidate的指令、完整metadata和归一化descriptor均相同；无需仅为.so整体SHA变化重做同一device实验。
- 支持域非赢家路径：对已经观察到的trace候选时间增加≥5%信号做有限Event确认。阈值只决定复核顺序，不能据trace直接拒绝。当前7项fallback计划覆盖最大21.83%信号及M1/16/64/96/144/256的K384/768短任务，只有7个目标，可直接完成它们，不扩成全fallback Event。M480K384真实赢家已有正向Event，继续引用该证据。完成现有信号后没有新证据要求就停止追加筛选；register已观察到的异常也按同样有限原则处理。
- 每个parent根据有限集合的完整Event、逐轮方向与波动决定。明确回退则关闭该parent改动或提出新范围设计；接近零则不能声称该shape有收益。不要持续补轮直到出现正值。需要独立复测时，只针对具体影响决策且现有证据矛盾的shape，固定一次复测后决定保留基线或候选。

Counter/ATT可以说明机制；已经有的有限counter证据可入文档。**绝对利用率时间窗校准、更多counter组合、额外ATT、固定时钟不是功能或整体时间采用的必需新增条件。** 当前GPU空闲监测/物理PCI、同地址配对、数值/guard及实际module身份仍必须可靠。

### 4.3 正式API集成smoke与最终身份

现成15项正式smoke已经CPU审核：覆盖5个改变entry、loops1、short/longK、M-tail、register wave-K/scale组及四个不变分支对照。执行现有check-only8rep即可，确认公开rank2/nativeE8M0数值与private一致，并由原runner核对实际加载module路径和SHA。无需因功能测试重复全部201 Event，也无需修改原runner支持rank3或offset。

正式15项没有完整覆盖所有机制K边界，已经由相同body的32+5 private机制计划覆盖，CPU exact identity将两者连接起来。其职责是证明host/API实际调用正确entry；不再扩成另一轮重复device机制测试。

若最终仅采用部分parent/alias，需要在隔离worktree移除被拒绝部分、CPU重建，并按新最终范围再次核对changed/private及unchanged/baseline身份；正式smoke只复核最终保留entry和必要邻接控制。若三文件全部保留，现成new.so身份及15smoke可直接作为最终候选证据。把差异带入原checkout后做CPU源码一致性检查；若编译生成模块，核对最终actualmodule身份。

## 5. 明确停止点

当机制正确性充分、当前赢家Event与现有筛选异常已有最终结论、正式API身份/数值通过、最终保留差异与审核候选一致时，采用验证结束。完整1426/646支持域及每个fallback Event不是硬条件；报告保留partial范围说明，不声称未测形状都通过或都更快。只给收益与回退结论，不强求每一级硬件根因都已唯一确定。

本轮已经拒绝的narrow/fine/fixed_n32不再追加测量。继续工作集中在上述未完成条件；GPU占用造成的污染只重试原用例，不引入新形状集合或新优化机制。
