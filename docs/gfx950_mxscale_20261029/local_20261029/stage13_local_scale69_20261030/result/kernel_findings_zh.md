第十三轮完整完成：336领先/85落后。69个宽N项新的六轮，352历史行逐字段保持；5622条性能记录正确、pending0、controller退出0，公开421配置及新CO正确性校验通过。整合CO100808，旧48254配置保持，当前48255、JIT base780。

局部SFA staging真正收益2项，均6/6快：M32768/N6144/K7168比原392快3.4941%，仍落后CSV；M40960/N7168/K3072比原392快2.4438%，新kernel越线。两项相对同panel32、unroll8、GroupM16及AGPR-form旧body100807也6/6快，paired speedup分别1.3028%和1.4964%，故能确认局部化代码本身有效。其余2项新越线是原control复测，分别M1792/N6144/K7168、M57344/N7168/K4096。

load_sfa_panel每次返回新的局部array，publish接收这次array；prologue和refill不共享跨循环旧值。Panel32的GroupM16版本spill由4降0，但Panel16仍785~795 spill。原硬件wait、线程barrier、地址边界和LDS布局不变。8个comparison-only旧body禁止集成，候选必须同时超过matched baseline和当前control的4/6及0.5%门槛。

下一轮将refill移到按scale cache边界分段的外层，每个MFMA innerloop不包含refill；stage、tile、future matrix请求连续，CPU边界和epilogue交接验证通过。
