9062 的单机制候选已准备，尚未进行 GPU 验证。只匹配 `fine_traits<80,1,4,4,1,2,4,128,0,16384>`，把原来的三个初始矩阵 tile 放到原 scale 生产者路径之前，保留四槽 ring。原 SFA/SFB helper、字节、尾行保护、publish、所有消费者/退休/barrier、split2 partial 与 matching reducer 均保留。

基线 ATT 的四波中，慢波首 barrier 到达比最早波晚2308 shader clocks；它执行第二个 SFA pass，包含约1600 clocks 的尾行/对齐保护和打包路径，再串行等待 SFA 与 SFB，导致矩阵请求开始更晚。这是有限单CU的发布到达差，不构成整体 GPU 百分比结论。

CPU构建使用18个原正式 small-family device TU 和同一编译参数，40个device entry基线均精确匹配 Oct8 FUNC/fullmetadata/normalized descriptor；候选只改 fixed9062 producer，39个未选 entry 和所有reducers保持逐字节身份。选中 VGPR86/SGPR70/AGPR0不变，零spill，FUNC由11052B变为10996B。两侧静态global/LDS/store/MFMA opcode计数和全部vmcnt阈值序列一致，硬件barrier均5个。

ISA证明21个静态初始matrix请求位置移到首scale vmcnt之前；这些位置包含每波A的有条件分支，实际请求仍依原wave guard。编译器额外生成一个 `lgkmcnt(0)`，并且原scale `vmcnt(0)`现在包含此前matrix请求，会提前full-drain。这是需要Event解决的性能风险，不能宣称提前请求已产生有效重叠。

`screen_plan.json`包括两项实际赢家、M1/M79/M80尾行与M81未选分支控制、runtime split2短K/不平衡分区控制。`winner_event_plan.json`只含两项固定9062实际赢家。GPU由root统一运行，调用库均不在timed launch中分配workspace，完整调用包含producer与reducer。采用需signed reference/repeat/guards和同物理卡共享池5ABBA完整Event；若失败或退步，保留原selected。

<!-- EVENT_DECISION -->

完整调用Event决定：拒绝此候选，保留原fixed9062 selected。两个实际赢家的signed8repeat/reference/output+workspace guards和51池地址的逐轮repeat/guard全部通过。clean owner/物理PCI65/HIP2、同一共享池5轮AB/BA完整producer+reducer Event均验证。

| shape M,N,K | baseline median us | candidate median us | speedup | 更快轮数 |
| --- | ---: | ---: | ---: | ---: |
| 144,7168,16384 | 44.552569 | 44.704727 | 0.996596 | 0/5 |
| 160,7168,16384 | 45.145512 | 45.197295 | 0.998854 | 1/5 |

两项median均无正改善。原scale vmcnt0可能full-drain早先matrix和新增generated LGKM wait仍是风险线索；没有candidateATT不能把微小退步归因到某条wait。此方向关闭，逐轮原值见results_analysis.json；后续ring consumer请求顺序为独立机制。
