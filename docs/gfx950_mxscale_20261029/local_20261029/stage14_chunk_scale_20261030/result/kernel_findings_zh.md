第十四轮完整完成：336领先/85落后。66个宽N项新的六轮，355历史行逐字段保持，5446条性能记录全部正确、pending0、controller退出0。

scale refill移到按cache边界分段的外层，inner MFMA loop不含refill；保持tile/stage/future matrix请求连续及原所有硬件wait/barrier。CPU 42组边界与drain模型验证通过，Panel16 VGPR spill从785~795降到0~2，但没有Panel16候选进入最终公开最优。

真实kernel收益6项，比当前control快0.5978%~3.9774%，均通过同配置local-scale原循环matched control的>=4/6与>=0.5%成对收益；这些6项仍落后CSV，新增越线0。整合4个CO100901/100906/100908/100911。旧48255配置完整保持，公共registry48259、JIT base780。公开421项身份/编译及4个新CO独立正确性检查通过。comparison-only的8个对照禁止整合。

用户新要求按剩余pipeline类别优先：最终85项中pin_agpr55、pingpong_lds20、prefetch_b_vgpr_overlap4、wave_interleave3、prefetch_b_vgpr3。先优化pin_agpr55项，包含9个窄N项。
