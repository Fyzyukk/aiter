第九轮窄N主循环优化已完成，325领先 /96落后。原323领先和76未测试落后共399行逐字段保留，22个窄N落后项新的六轮完成。

真实新kernel收益只有1项：M1536/N768/K7168，kid100413取代41672，waves2、B预取深度7、流式A、静态B环、主循环周期4；原seed的自然B队列深度8、环9，原body周期36。组合修改六轮6/6快，耗时19.384874us→18.981940us（下降2.0786%，成对中位speedup2.0425%）。该shape原control仍落后，新kernel越过固定FlyDSL CSV。不能把全部收益归因于单独缩短PERIOD，因为B队列和A流式也同时改变。

另1项新增领先M10240/N1536/K7168由原kid100072复测越线。其余2项stable improvement来自既有kid41816更快，均尚未越线。其余15个设计未集成，候选诊断、ISA/build资源及初筛明显慢候选保留，不能按设计数宣称收益。

独立新header只服务新CO，旧48250个配置和名称保持，公共registry48251，JIT base780。所有硬件wait、barrier、tile descriptor和地址位宽保留；A输入2GiB guard未修改。741个CPU环复用/producer-consumer/wait计数检查及1670条数值测试正确，public CO重新加载正确性也通过。

下一轮针对B全局预取提前到A LDS读取前，并以同几何/同队列/同A stages的成对对照分离指令顺序的实际收益。
