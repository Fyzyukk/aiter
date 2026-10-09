small与fine分支本轮按33个有历史winner的实际producer symbol配置收口，映射18个public parent、30个actual ID和278个历史winner。候选性能实测严格为4个配置、21个actual-winner形状：fixed9062 initial matrix-before-scale覆盖M144/M160两个winner且完整调用包含matching split2 reducer；9046与9055 steady ring issue-before-read各测一个自身winner；runtime9051 scale-first覆盖其全部17个winner。四个exact配置的候选均拒绝并保留原selected：fixed9062两项speedup0.996596/0.998854，9046/9055为0.980340/0.991871；9051完整17项为8项median正、8项退、1项平，仅3项全部5轮更快。所有自身signed8repeat、reference、重复及guard和clean sharedpool完整Event审查通过。两个9051早期正代表仅是进入固定覆盖的screen依据，全17 mixed结果作统一拒绝，不按shape新增阈值或重跑寻找正结果。本轮small没有新增source/device entry采用，9042/9053/9054此前已采用runtime B-scale alias保留，9071草稿在build前停止。

证据范围分别保存：7个exact配置各有一个自身有限形状的局部CU ATT，33个配置各自一个actualwinner的signed2repeat/reference/output/workspace guard数值门；它们分别证明局部机制和正确性，均不扩大为全部winner性能覆盖。9055原CU0 code=null/零wave记录排除，CU1补采独立核过身份；9051的32个ATT wave跨多个WG，聚合同PC barrier跨度不用于WG内到达差。该small正式入口集合为35个producer与5个独立reducer，其中33个producer与历史winner相交、两个兼容/runtime producer无历史winner；5个reducer保留机器身份，未声称每个reducer有独立性能实测。各配置的keep依据自身selected source/type、正式FUNC/fullmetadata/descriptor、数值门及存在时的自身历史exact-config Event；共享ring/register/fine机制说明、静态资源或其他配置reject都不算该配置性能。完整配置记录见 [small审查](../opus_remaining_20261008/small_family_review.json)，范围核验见 [small最终审计](../opus_remaining_20261008/small_final_closure_audit.json)。

| public parent | actual配置（同ID按fixedK区分） | 历史winner数 | 自身本轮候选Event形状 | 自身ATT配置 | 自身数值门配置 | 最终状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 9040 | 9040 | 11 | 0 | 1 | 1 | keep自身当前配置 |
| 9041 | 9041, 9050 | 17 | 0 | 0 | 2 | keep自身当前配置 |
| 9042 | 9042, 9071, 9073 | 15 | 0 | 1 | 3 | 保留已采用runtime alias；其余配置keep；9071保留N48 |
| 9043 | 9043 | 2 | 0 | 0 | 1 | keep自身当前配置 |
| 9044 | 9044, 9056 | 26 | 0 | 0 | 2 | keep自身当前配置 |
| 9045 | 9045 | 17 | 0 | 0 | 1 | keep自身当前配置 |
| 9046 | 9046 | 32 | 1 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9047 | 9047 | 27 | 0 | 1 | 1 | keep自身当前配置 |
| 9049 | 9049 | 6 | 0 | 0 | 1 | keep自身当前配置 |
| 9051 | 9051 | 17 | 17 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9052 | 9052, 9070 | 24 | 0 | 0 | 2 | keep自身当前配置 |
| 9053 | 9053, 9072 | 9 | 0 | 0 | 2 | 保留已采用runtime alias；其余配置keep |
| 9054 | 9054 | 3 | 0 | 0 | 1 | 保留已采用runtime alias；其余配置keep |
| 9055 | 9055 | 17 | 1 | 1 | 1 | 保留；拒绝自身有限候选 |
| 9060 | 9060/K0, 9060/K3072, 9060/K7168 | 9 | 0 | 0 | 3 | keep自身当前配置 |
| 9061 | 9061/K0 | 19 | 0 | 0 | 1 | keep自身当前配置 |
| 9062 | 9062/K16384, 9064/K0 | 6 | 2 | 1 | 2 | 保留；拒绝自身有限候选 |
| 9063 | 9063/K0, 9063/K16384, 9065/K0, 9066/K0, 9067/K0, 9068/K0, 9069/K0 | 21 | 0 | 0 | 7 | keep自身当前配置 |

表中0项candidate Event表示该parent本轮保留自身配置，没有冒称该parent全部winner已测或已获得新性能增益；历史exact-config Event另存逐symbol记录。
