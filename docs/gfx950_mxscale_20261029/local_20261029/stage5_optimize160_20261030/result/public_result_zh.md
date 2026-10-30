本轮结果：304项领先，117项落后（六轮耗时中位数对固定FlyDSL CSV）。

原160个落后项全部做了新一组最终六轮；43项越线。原261个领先项保留本轮不重测。
39项越线时原最优仍落后，4项在本轮原最优自身也越线。105项有稳定候选收益（至少4/6轮比同批原最优快），耗时下降中位数3.03%，最大16.38%。

6个新CO已并回公用manifest，覆盖20个最终最优shape；原48234配置身份完整保留，当前48240配置，JIT base仍780。
新CO IDs：[100045, 100047, 100049, 100050, 100051, 100052]。仅并入最终正确且有稳定成对收益的shape，保留实验、正式数据和合并前manifest。

剩余117项落后中位数2.55%，最大11.44%；25项差不到1%，64项差不到3%，93项差不到5%，116项差不到10%。

最终六轮：160shape、571个候选组合、3426条正确测量，无待确认。整个新campaign共42414条实际测量，全部正确。

输出：public_best_opus421.csv、public_final160_comparisons.csv、public_remaining_losses.csv。结果可运行环境：../public_env.sh；单独JIT缓存保留原大规模tune二进制。

新CO初筛明显偏慢项：new_co_screen_outliers20pct.csv。U4中产生大量spill的3项和CO编译拒绝项保存在../co_round3/build_and_selection.json。

FlyDSL没有tune，基线按gfx/cu_num/M/N/K/w_scale_block六字段匹配。A端2GiB寻址范围没有修改。
