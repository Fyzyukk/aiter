runtime9051 的scale-first请求次序候选已准备，尚未GPU验证。仅匹配原 `<16,32,1,1,Q3,waveK4,OUTPUT3,cache3,K0,false,false>` type，保留原symbol和false B-scale reuse。每tile先发原SFA/SFB byte请求，再发原A/B matrix请求；字节/数量/guard/地址/cache/queue生命周期/compute/FP32归约及输出source保留，只用compiler sched_barrier锁定两个issue组次序。

基线ATT的32完整wave跨多个WG，startupmatrix issue和scaleissue/wait占主要局部时段，精确身份与原LFIFO/translation/cache四组counter可复用。候选检验byte请求提前是否改善same-tile请求供给，不改变整体requests或采用已拒全局reuse。长issue decoderstall不等同单次HBM访问延迟；跨WG barrier跨度不用于WG内arrival结论。

18正式TU/40entry基线均精确Oct8 FUNC/fullmetadata/descriptor；候选39个未选entry完全相同。选中FUNC3012→3000B/VGPR104→102/SGPR55→52，零spill。ISA证明每initialtile由matrix2/SFA1/matrix4/SFB2变为scale3/matrix6，三tile27requests覆盖一致；总matrix48/byte24、MFMA16/LDSwrites2/reads6/barrier1及B sc0nt请求数相同。Compiler operandwait随请求次序重新匹配，首vmcnt1→2、staticwait22→21，这是需要数值/完整Event判定的调度变化。

screen_plan覆盖external/longqueue/shortK实际赢家以及Mtail/emptywave/unevenpartition/Q3refill。representative_event_plan仅external[1,65536,1536]和longqueue[16,7168,3072]两个实际赢家，使用同物理卡51共享池5轮AB/BA完整call Event；若代表无正收益就关闭，只有正证据才扩展17个实际winner覆盖并走正式API检查。

<!-- EVENT_DECISION -->

两个代表的完整sharedpoolEvent均5/5轮更快，signed8repeat及51池outputrepeat/reference/guards通过。当前允许继续17个actualwinner完整Event coverage，尚未采用runtime9051。

| shape M,N,K | baseline median us | candidate median us | speedup | 更快轮数 |
| --- | ---: | ---: | ---: | ---: |
| 1,65536,1536 | 16.677040 | 16.537431 | 1.008442 | 5/5 |
| 16,7168,3072 | 6.724000 | 6.643980 | 1.012044 | 5/5 |

独立automaticrotation screen中longqueue计时曾显示退步，sharedpoolEvent两代表则全部正值；保留两种原值，以同池完整Event作为采用性能门。不得把两代表当作17个赢家范围收益；须完整coverage和正式API数值门。

<!-- FULL17_COVERAGE_DECISION -->

17个冻结实际winner的完整sharedpool Event审查已完成：keep_current_selected_reject_this_candidate。8/17 median正值、3/17全部5轮更快、7/17两个order组median正值；所有signed8repeat/reference/outputguard和每shape510池outputrepeat/reference/guard通过。见 [coverage_analysis.json](coverage_analysis.json)。

| shape M,N,K | baseline median us | candidate median us | speedup | 更快轮数 | AB / BA paired medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,7168,768 | 3.120020 | 3.141980 | 0.993011 | 0/5 | 0.991762 / 0.996386 |
| 1,7168,1024 | 3.424333 | 3.410216 | 1.004140 | 3/5 | 0.993602 / 1.054957 |
| 1,16384,1536 | 5.882392 | 5.999255 | 0.980520 | 0/5 | 0.973422 / 0.988443 |
| 1,65536,1536 | 16.647216 | 16.577392 | 1.004212 | 5/5 | 1.001041 / 1.009395 |
| 2,7168,768 | 3.204726 | 3.261980 | 0.982448 | 0/5 | 0.986352 / 0.972625 |
| 2,7168,1024 | 3.527863 | 3.516902 | 1.003117 | 2/5 | 0.943769 / 1.058161 |
| 2,7168,16384 | 23.126490 | 22.995509 | 1.005696 | 5/5 | 1.007879 / 1.008483 |
| 2,16384,1536 | 6.134176 | 6.148294 | 0.997704 | 4/5 | 1.004830 / 1.021850 |
| 2,65536,1536 | 16.688784 | 16.600157 | 1.005339 | 4/5 | 1.007182 / 1.001518 |
| 4,7168,1024 | 3.596882 | 3.696490 | 0.973053 | 3/5 | 0.949460 / 1.038253 |
| 4,16384,1536 | 6.206333 | 6.262020 | 0.991107 | 2/5 | 0.981005 / 1.005781 |
| 8,7168,768 | 3.484725 | 3.425118 | 1.017403 | 3/5 | 0.983055 / 1.037140 |
| 8,7168,1024 | 3.829843 | 3.829843 | 1.000000 | 2/5 | 0.975215 / 1.032124 |
| 16,7168,768 | 3.592196 | 3.603961 | 0.996736 | 2/5 | 0.986991 / 1.005512 |
| 16,7168,1024 | 3.982784 | 3.996118 | 0.996663 | 3/5 | 1.000589 / 1.003031 |
| 16,7168,3072 | 6.786745 | 6.730255 | 1.008393 | 4/5 | 1.008344 / 1.011678 |
| 16,16384,1536 | 6.556138 | 6.466725 | 1.013827 | 5/5 | 1.012139 / 1.017231 |

本次以整个exact runtime9051类型统一判定，不按结果新增shape子集或阈值；不重跑以寻找正结果。当前未修改production。split1完整调用包含producer内部waveK LDS/FP32归约与输出，不含单独reducer/globalworkspace。
