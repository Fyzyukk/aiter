# 主系列独立优化候选

GPU测试保持停止。24个新候选与8个当前源码对照均完成离线编译；CPU布局、边界和独立审查通过，所有32个kernel无scratch或spill。尚无GPU数值或性能验证，未注册或修改正式选型。

本组覆盖109个历史落后shape：9022共72、9021共13、9020共5、9024共6、9000共5、9001共7、9011共1。覆盖表465个shape/candidate组合；24个新候选中21个与当前落后集合相交，另外3个作为合法支持域对照。

- BM64/96/128、BN128、双matrix stage的独立几何；短K384/768 panel8，K1536 panel16，固定K3072/7168。保持原160行pipeline的同步与scale/output顺序，只泛化各BM相关常量和SFA pass数量。
- 9024固定7168保持64×64与scale面板，只比较groupM8/16与现groupM4的launch顺序。
- 9000/9001保留pin分配和half-tile数学，固定K和panel16/32/64；所有SFB写入新增lane<panel保护。
- 9011固定7168并关闭主循环unroll；初版fixedK/unroll4产生1776B scratch、715 VGPR spill，已弃用并保留记录。最终scratch/spill均0。

| Kernel traits | VGPR | SGPR | LDS B | Scratch B | ISA B |
|---|---:|---:|---:|---:|---:|
| opus_private_geometry_traits<64, 2, 32, 0 | 116 | 62 | 52768 | 0 | 3596 |
| opus_private_geometry_traits<96, 2, 32, 0 | 140 | 58 | 62240 | 0 | 4368 |
| opus_private_geometry_traits<128, 2, 32, 0 | 166 | 58 | 71712 | 0 | 5016 |
| opus_private_geometry_traits<96, 2, 8, 384 | 124 | 46 | 59912 | 0 | 3640 |
| opus_private_geometry_traits<96, 2, 8, 768 | 124 | 46 | 59912 | 0 | 3628 |
| opus_private_geometry_traits<128, 2, 16, 1536 | 150 | 46 | 69648 | 0 | 4200 |
| opus_private_geometry_traits<160, 2, 16, 1536 | 176 | 46 | 78608 | 0 | 4908 |
| opus_private_geometry_traits<96, 2, 32, 3072 | 124 | 46 | 62240 | 0 | 3628 |
| opus_private_geometry_traits<96, 2, 32, 7168 | 140 | 58 | 62240 | 0 | 4088 |
| opus_private_geometry_traits<128, 2, 32, 7168 | 166 | 59 | 71712 | 0 | 4664 |
| opus_private_geometry_traits<128, 2, 32, 3072 | 150 | 46 | 71712 | 0 | 4200 |
| opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64, 7168, 8 | 264 | 62 | 71744 | 0 | 18360 |
| opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64, 7168, 16 | 264 | 62 | 71744 | 0 | 18360 |
| opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) | 194 | 62 | 81184 | 0 | 6696 |
| opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) | 162 | 99 | 105504 | 0 | 7960 |
| opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 128, 0 | 203 | 56 | 143360 | 0 | 8864 |
| opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128, 128, 64, 0 | 96 | 53 | 76032 | 0 | 4620 |
| opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64, 7168, 4 | 264 | 62 | 71744 | 0 | 18360 |
| opus_private_pin_fixed_traits<384, 16, false | 484 | 58 | 139392 | 0 | 10068 |
| opus_private_pin_fixed_traits<1536, 16, false | 464 | 54 | 139392 | 0 | 12808 |
| opus_private_pin_fixed_traits<3072, 32, false | 464 | 52 | 143616 | 0 | 13088 |
| opus_private_pin_fixed_traits<7168, 64, false | 464 | 56 | 152064 | 0 | 13580 |
| opus_private_pin_fixed_traits<16384, 64, false | 468 | 62 | 152064 | 0 | 14984 |
| opus_private_pin_fixed_traits<384, 16, true | 484 | 58 | 139392 | 0 | 10080 |
| opus_private_pin_fixed_traits<1536, 16, true | 464 | 54 | 139392 | 0 | 12820 |
| opus_private_pin_fixed_traits<3072, 32, true | 464 | 52 | 143616 | 0 | 13104 |
| opus_private_pin_fixed_traits<7168, 64, true | 464 | 56 | 152064 | 0 | 13644 |
| opus_private_pin_fixed_traits<16384, 64, true | 468 | 62 | 152064 | 0 | 15120 |
| opus_private_pad_pin_fixed_traits<7168 | 464 | 58 | 152064 | 0 | 13948 |
| opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) | 477 | 70 | 152064 | 0 | 20744 |
| opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) | 469 | 70 | 152064 | 0 | 20792 |
| opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) | 492 | 80 | 152064 | 0 | 26964 |

静态资源减少不是速度或occupancy结论。例BM96/shortK可减少M padding，但workgroup数量和B重复加载也变动。9000/9001这些历史慢例只输compact mxpsh，对raw BMM已快，最终需计入调用方scale布局前提。9024只是调度顺序对照，尚无证据它带来改善。

Clang23编译main几何/9024，pin-Clang24/resource20编译pin族。源码依赖84文件冻结；局部baseline为当前源码重编译，不宣称等同历史正式.so。两种library各自导出launch，ID只是私有ABI开关，没有production注册。

[variants.json](variants.json)列出全部ID/traits；[coverage.csv](coverage.csv)逐shape列候选；[build_receipt.json](build_receipt.json)保存argv/编译器/源码/二进制SHA；[cpu_audit.json](cpu_audit.json)保存ELF/CPU结果。

CPU scale检查使用真实冻结布局覆盖BM64/96/128/160、panel8/16/32、多种K和M尾行；独立深审查另覆盖1,966,080 A/B matrix地址、913,408 pin SFA与228,352 pin SFB消费字节，验证DPP/perm和panel边界。独立审查发现launcher SFA descriptor stride未赋值，已修复全batch strides并重新编译、复核。详见[独立修复审查](../hybrid_small/main_variants_review_fix.json)及[深布局审查](../hybrid_small/main_deep_cpu_v1/receipt.json)。

所有最终CPU检查均HIP host-only编译成普通host object，再普通链接；readelf确认没有HIP/HSA依赖后才执行。原型.so未加载。

早期编译/CPU设置失败及首个pin spill产物在build_attempt*/cpu_audit_attempt*保留；最终结果为build/与cpu_audit/。重现脚本拒绝已有目标目录。

后续GPU验证需要signed/reference、每项输出和guard、循环调用、完整同卡基线/FlyDSL成本以及745回归；目前停止状态不变。

