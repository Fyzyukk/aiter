9046/9055 的共享small LDS ring单机制候选已准备，尚未GPU验证。只匹配当前9046 S6/C2和9055 S12/C4原traits，保留原symbol。每个ring step把原未来matrix prefetch从当前operand LDS reads之后放到之前；原cluster wait/barrier仍在两者之前，future slot/index/kt/guard保持相同。初始scale/matrix handoff、scales、短K、drain、输出均保留。

9046 baselineATT的普通tile boundary median316 clocks、cluster boundary648；cluster后首LDS operand read常见84/168clocks，原未来matrix issue在10个matrix LDS reads及scale读后。9055 CU1显示S12/C4 cluster boundary median492、group内216；同样未来matrix issue在current operands之后。候选检验这一可交换请求次序对后续tile供给是否有益，不改已被拒绝的fine_wait阈值或N48/N128几何。

CPU构建18个正式small-family TU，40entry基线逐项精确Oct8 FUNC/fullmetadata/descriptor；候选仅两producer变化，38个其它entry完全不变。9046VGPR80→62/SGPR55保持/FUNC8044→7972；9055VGPR56→48/SGPR73→72/FUNC10356→10204，零spill。两侧VMEM/LDS读写/MFMA opcode计数及vmcnt阈值序列一致，s_barrier各4。ISA证明各cluster后未来matrix issue在operand read前；编译器因此将fragment读交错MFMA，静态waitcnt各多12条。这些resource/调度差异需要完整Event判断，不构成收益证据。

screen_plan包含两个ATT实际赢家、K1024/K3072、Mtail实际赢家、短K无ring控制和首ring partial-cluster尾部。representative_event_plan仅两个有限ATT赢家，以同物理卡共享pool5ABBA完整调用Event分别决策。若代表退步，该机制在对应family关闭；若正收益，才扩大该family实际winner范围并完成正式API检查后采用。

<!-- EVENT_DECISION -->

完整调用Event决定：两type均拒绝此候选，保留原9046/9055 selected。两个有限代表实际赢家的signed8repeat/reference/output guards和51池地址的逐轮repeat/guard全部通过，split1无global workspace。clean owner/物理PCI65/HIP2、同一共享池5轮AB/BA完整kernel+output Event均验证。runner通用includes字段写producer+reducer，但本plan为workspace:false/split1，精确host只有producer调用。

| shape M,N,K | baseline median us | candidate median us | speedup | 更快轮数 |
| --- | ---: | ---: | ---: | ---: |
| 256,7168,16384 | 56.357473 | 57.487684 | 0.980340 | 0/5 |
| 64,7168,7168 | 16.653510 | 16.790000 | 0.991871 | 0/5 |

两项median均无正改善。请求提前导致compiler fragment读/MFMA调度和更多waitcnt改变；没有candidateATT不能归因到某条wait。两type此方向均关闭，逐轮原值见results_analysis.json；后续runtime9051请求顺序为独立register机制。
