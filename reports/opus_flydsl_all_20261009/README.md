# OPUS 全落后系列第一轮优化

原先693个正式OPUS选型中落后FlyDSL的294个shape，现已全部配置独立优化实现。四包共62个新候选，覆盖26个公开parent、1,280个shape/candidate组合。离线编译、CPU布局/契约检查和独立复核通过；GPU测试依用户要求保持停止，尚无新数值或性能结果，不能据此声称已超过FlyDSL。

| 优化包 | 294中shape数 | 新候选数 | shape/candidate组合 | 已实现的变化 |
|---|---:|---:|---:|---|
| [main_variants](main_variants/README.md) | 109 | 24 | 465 | BM64/96/128几何、短K与固定K面板、pin固定K、9024调度顺序 |
| [hybrid_small](hybrid_small/README.md) | 85 | 24 | 255 | 保留A的LDS协作，将B改为寄存器队列，提前1/2/3步 |
| [small_split](small_split/README.md) | 90 | 12 | 540 | register全局splitK4及完整FP32 reducer；fine新tile及splitK1/2 |
| [large9030](large9030/README.md) | 10 | 2 | 20 | 固定K1536/panel16；B直读、A三stage、C按96行分块输出 |
| 合计 | **294** | **62** | **1,280** | 四包精确分割，无漏项或跨包重复 |

main的24个新候选里21个与原294集合相交，其余3个保留作合法支持域对照。hybrid的候选标识是“actual ID + library variant”，并未引入新公开数字ID。额外的9070同源Clang23/24对照不计入62个新算法；此前短K9022两个原型也不计入本轮62。

## 清单与来源

[coverage294.csv](coverage294.csv)保留每项M/N/K、历史OPUS parent/actual路径、FlyDSL最快配置/时间、差距和现有合法profile配置数，并附本轮优化包与全部候选键。[candidate_cases.csv](candidate_cases.csv)列出1,280个具体组合；[candidate_catalog.json](candidate_catalog.json)列出62个新实现。候选键只用于私有实验，正式选型未注册这些实现。

历史基线仍为：745项正式选择中OPUS693/ASM34/CKTile10/CK8；693个OPUS历史399快/294慢；全部745项采用最快合法OPUS时411快/334慢。294项此前已经各自取13,027条合法OPUS测量中的最小值，因此本轮需要新实现，重新选择已有配置不能直接消除差距。

全部745项纯OPUS目标还包括正式选型为ASM32/CK8的额外40个历史OPUS落后shape。[additional40.csv](additional40.csv)保存profile最小值、实际dispatch与候选；[additional40_audit.json](additional40_audit.json)从79,504条profile记录中核验13,027条合法OPUS记录，并以实际四包contract完成1,800项CPU查询。40项均有同系列已编译候选，共172个组合（main4、hybrid66、small102），无需新增kernel。另1,705个满足契约的组合包含跨系列显式重映射，独立列出，不混作历史dispatch。

[coverage334.csv](coverage334.csv)逐shape列出完整334项及首选候选；[candidate_cases334.csv](candidate_cases334.csv)列出原1,280加额外172，共1,452个组合。完整覆盖为main110/hybrid107/small107/large10个shape。该扩展表示已有实现覆盖了全部历史慢例，不是新增GPU测量或胜率改善。

FlyDSL比较取每shape已保存的BMM与可用mxpsh行最小值；原294项中272个FlyDSL赢家为raw-scale BMM，22个为compact mxpsh。mxpsh调用方A-scale预重排成本未包含。这些时间来自历史CSV，不是当前干净同卡实验。新候选无计时，不会把旧的编译器局部收益拼入该比较。

## 可检查的资源变化

| 局部源码重编译对照 → 候选 | VGPR变化 | LDS bytes变化 |
|---|---:|---:|
| 9022 runtime160行 → BM96/fixed384或768/panel8 | 194 → 124 | 81,184 → 59,912 |
| 9049保留A协作、B去除LDS | 依ahead变化，详见coverage | 81,920 → 16,384 |
| 9055保留A协作、B去除LDS | 依ahead变化，详见coverage | 153,912 → 52,536 |
| 9030 runtime → direct-B/A3/chunk96 | 203 → 170 | 143,360 → 79,168 |

全部114个最终编译kernel（包括局部对照、reducer及两种编译器对照）scratch、VGPR spill与SGPR spill均为0。静态资源改善尚不代表实际提速：tile改变会改变workgroup数与重复B读取，splitK增加workspace和reducer调用，direct-B需要确认寄存器压力与等待开销，9024顺序变化需验证缓存效果。

局部baseline是当前冻结源码按对应实验编译器/flags重编译，不宣称与历史正式module机器码一致。正式CSV、正式module和冻结FlyDSL CSV均匹配原SHA。

## 检查结论与限制

最终CPU检查使用实际冻结布局与候选表达式，覆盖A/B地址、scale消费、K循环/队列、尾行、输出、split分区和ABI边界。独立main深查核对1,966,080个matrix地址字节及913,408/228,352个pin SFA/SFB消费字节；hybrid核对2,424,832个布局地址字节；small独立split查核对5,248个分区（83个空分区）与全部540个组合。CPU检查支持地址/契约的一致性，不能替代GPU数值及同步验证。

审查中修复main launcher的batch scale stride未赋值、pin短面板的SFB lane保护；9011初版unroll4与9030整tile B队列出现spill，均已弃用并保留失败记录，最终产物无spill。small的候选traits只启用OUTPUT3；复制源码里未实例化的OUTPUT4分支需要适配FP32 workspace指针语义后才能使用。

所有最终host检查均先编译object，再以普通C++链接；readelf确认无HIP/HSA动态依赖后执行。large9030与small_split早期host-only直接链接曾继承并加载libamdhip64，相关检查已被上述最终检查替代。早期程序没有调用HIP函数，没有加载实验.so、启动GPU kernel或执行计时。该记录保留在包内失败说明中，不能宣称整个过程中从未加载HIP库。

[total_read_only_review.json](total_read_only_review.json)独立验证最新源码、编译器、对象、library、完整ELF function/资源、CPU executable依赖及四包manifest。所有最新结果通过；failed/spilled/superseded产物独立保留，不计入通过项。[summary.json](summary.json)、[verification.json](verification.json)和[artifact_manifest.json](artifact_manifest.json)提供聚合统计与SHA绑定。

[额外40项独立复核](additional40_read_only_review.json)另行重算全部13,027个OPUS profile最小值、334−294集合、40项dispatch、1,800项contract结果以及1,705个可用候选的workspace/调用/library映射，无发现问题。[聚合独立复核](aggregate_read_only_review.json)重建全部CSV与候选键，确认294/1,280以及334/1,452两套清单精确对应源码。

## 后续验收入口

GPU测试保持停止，没有后台等待或自动续跑。各包保留独立ABI、launcher、编译参数与源码，可在用户恢复测试后先做signed/reference数值、边界和重复调用检查，再逐shape同卡比较旧OPUS与新候选，并计入完整split-K/reducer及各FlyDSL scale布局成本。只有实测后才选择赢家、更新快慢数量及考虑正式注册；最终还需要745项回归。

聚合脚本[aggregate.py](aggregate.py)仅使用Python标准库读已有产物、校验哈希并生成清单，不导入aiter/torch，不查询GPU或加载实验库。[manifest.py](manifest.py)绑定本报告全部产物及交接/正式输入；`python reports/opus_flydsl_all_20261009/manifest.py --verify`可只读复核完整SHA。四个包均保留冻结84个依赖文件和可重复离线构建脚本；已有目标目录的覆盖由包内脚本保护。
