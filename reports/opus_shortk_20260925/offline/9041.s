	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
	.protected	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950 ; -- Begin function _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.globl	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.p2align	8
	.type	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,@function
_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950: ; @_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_load_dword s22, s[0:1], 0x18
	s_load_dwordx4 s[4:7], s[0:1], 0x28
	s_load_dwordx4 s[12:15], s[0:1], 0x40
	s_load_dwordx2 s[18:19], s[0:1], 0x50
	v_readfirstlane_b32 s30, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s3, 0xc0
	v_lshlrev_b32_e32 v1, 4, v0
	s_movk_i32 s0, 0x48
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.1:
	v_mul_u32_u24_e32 v2, 0xaab, v1
	v_lshrrev_b32_e32 v2, 19, v2
	v_mul_lo_u16_e32 v2, 0xc0, v2
	v_sub_u16_e32 v2, v1, v2
	v_add_u32_e32 v6, s7, v2
	v_cmp_gt_i32_e32 vcc, s22, v6
	v_mov_b32_e32 v2, 0x7f
	v_mov_b32_e32 v9, 0x7f
	v_mov_b32_e32 v8, 0x7f
	v_mov_b32_e32 v7, 0x7f
	v_mov_b32_e32 v3, 0x7f
	v_mov_b32_e32 v12, 0x7f
	v_mov_b32_e32 v11, 0x7f
	v_mov_b32_e32 v10, 0x7f
	v_mov_b32_e32 v4, 0x7f
	v_mov_b32_e32 v15, 0x7f
	v_mov_b32_e32 v14, 0x7f
	v_mov_b32_e32 v13, 0x7f
	v_mov_b32_e32 v5, 0x7f
	v_mov_b32_e32 v18, 0x7f
	v_mov_b32_e32 v17, 0x7f
	v_mov_b32_e32 v16, 0x7f
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	s_and_b32 s25, s13, 0xffff
	s_mov_b32 s27, 0x20000
	s_mov_b32 s26, -1
	s_mov_b32 s24, s12
	v_mul_lo_u16_e32 v2, 0xab, v0
	v_lshrrev_b16_e32 v2, 11, v2
	v_mad_u64_u32 v[2:3], s[12:13], s18, v2, v[6:7]
	buffer_load_dwordx4 v[2:5], v2, s[24:27], 0 offen
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v7, 24, v2
	v_lshrrev_b32_e32 v8, 16, v2
	v_lshrrev_b32_e32 v9, 8, v2
	v_lshrrev_b32_e32 v10, 24, v3
	v_lshrrev_b32_e32 v11, 16, v3
	v_lshrrev_b32_e32 v12, 8, v3
	v_lshrrev_b32_e32 v13, 24, v4
	v_lshrrev_b32_e32 v14, 16, v4
	v_lshrrev_b32_e32 v15, 8, v4
	v_lshrrev_b32_e32 v16, 24, v5
	v_lshrrev_b32_e32 v17, 16, v5
	v_lshrrev_b32_e32 v18, 8, v5
.LBB0_3:
	s_or_b64 exec, exec, s[20:21]
	s_mov_b32 s3, 0xc0c0004
	v_perm_b32 v2, v2, v9, s3
	v_perm_b32 v6, v8, v7, s3
	v_lshl_or_b32 v2, v6, 16, v2
	v_perm_b32 v3, v3, v12, s3
	v_perm_b32 v6, v11, v10, s3
	v_lshl_or_b32 v3, v6, 16, v3
	v_perm_b32 v4, v4, v15, s3
	v_perm_b32 v6, v14, v13, s3
	v_lshl_or_b32 v4, v6, 16, v4
	v_perm_b32 v5, v5, v18, s3
	v_perm_b32 v6, v17, v16, s3
	v_lshl_or_b32 v5, v6, 16, v5
	v_add_u32_e32 v6, 0x1ce00, v1
	ds_write_b128 v6, v[2:5]
.LBB0_4:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_and_b32 s25, s15, 0xffff
	s_mov_b32 s27, 0x20000
	s_mov_b32 s26, -1
	s_mov_b32 s24, s14
	v_cmp_lt_u32_e32 vcc, 5, v0
	s_nop 1
	v_cndmask_b32_e64 v3, 0, 1, vcc
	v_subrev_co_u32_e32 v2, vcc, 6, v0
	s_nop 1
	v_cndmask_b32_e32 v2, v2, v0, vcc
	v_lshl_or_b32 v3, s2, 1, v3
	v_mad_u64_u32 v[2:3], s[12:13], s19, v3, v[2:3]
	buffer_load_ubyte v3, v2, s[24:27], 0 offen
	v_add_u32_e32 v2, 0x1d280, v0
	s_waitcnt vmcnt(0)
	ds_write_b8 v2, v3
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s12, s2, 8
	s_mul_i32 s13, s4, s7
	s_mul_i32 s0, s5, s12
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	s_ashr_i32 s10, s13, 31
	s_add_u32 s8, s8, s13
	s_addc_u32 s9, s9, s10
	s_sub_i32 s14, s22, s7
	s_mul_i32 s10, s14, s4
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, s3
	s_lshr_b32 s13, s30, 8
	s_mul_i32 s31, s13, 0x840
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 4, v2
	s_bfe_u32 s15, s30, 0x20006
	s_mul_i32 s18, s5, s15
	s_lshl_b32 s18, s18, 5
	s_mul_i32 s19, s5, s13
	s_lshl_b32 s19, s19, 7
	s_add_i32 s18, s18, s19
	s_lshl4_add_u32 s5, s5, s18
	s_add_i32 s19, s5, 0x400
	v_add_u32_e32 v101, s19, v2
	v_add_u32_e32 v102, s5, v2
	s_addk_i32 s18, 0x400
	v_add_u32_e32 v103, s18, v2
	v_add_u32_e32 v104, 0xfffffc00, v103
	s_mul_i32 s5, s15, s4
	s_mul_i32 s18, s4, s13
	s_lshl_b32 s18, s18, 5
	s_add_i32 s5, s5, s18
	v_and_b32_e32 v1, 0x70, v1
	v_bfe_u32 v3, v0, 3, 3
	v_mul_lo_u32 v3, s4, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_add3_u32 v106, s5, v1, v3
	v_lshl_add_u32 v105, s4, 7, v106
	v_lshl_add_u32 v107, s4, 6, v106
	s_mul_i32 s4, s13, 0x1080
	s_mul_i32 s29, s15, 0x420
	s_add_i32 s29, s29, s4
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], 0 offen lds
	s_add_i32 s24, s29, 0x2100
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v107, s[8:11], 0 offen lds
	s_add_i32 s23, s29, 0x4200
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], 0 offen lds
	s_mul_i32 s4, s13, 0x4200
	s_mul_i32 s5, s15, 0x1080
	s_add_i32 s25, s5, s4
	s_add_i32 s25, s25, 0xc600
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v104, s[0:3], 0 offen lds
	s_add_i32 s26, s25, 0x420
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v103, s[0:3], 0 offen lds
	s_add_i32 s27, s25, 0x840
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], 0 offen lds
	s_add_i32 s28, s25, 0xc60
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], 0 offen lds
	s_add_i32 s22, s29, 0x6300
	s_movk_i32 s4, 0x80
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s4 offen lds
	s_add_i32 s5, s29, 0x8400
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v107, s[8:11], s4 offen lds
	s_add_i32 s18, s29, 0xa500
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s4 offen lds
	s_add_i32 s4, s25, 0x8400
	s_movk_i32 s33, 0x800
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v104, s[0:3], s33 offen lds
	s_add_i32 s20, s25, 0x8820
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v103, s[0:3], s33 offen lds
	s_add_i32 s21, s25, 0x8c40
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s33 offen lds
	s_add_i32 s19, s25, 0x9060
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s33 offen lds
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v1, 15, v0
	v_lshl_or_b32 v100, s15, 4, v1
	v_or_b32_e32 v3, 0x1ce00, v100
	ds_read_u8 v3, v3
	v_or_b32_e32 v4, 0x1ce40, v100
	ds_read_u8 v4, v4
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v3, v4, 8, v3
	v_or_b32_e32 v4, 0x1ce80, v100
	ds_read_u8 v4, v4
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v4, 16, v3
	v_mov_b32_e32 v3, 0x1d280
	ds_read_u8 v110, v3
	v_mov_b32_e32 v3, 0x1d286
	ds_read_u8 v111, v3
	v_and_b32_e32 v3, 3, v0
	s_lshr_b32 s30, s30, 5
	v_and_or_b32 v3, s30, 4, v3
	v_mul_u32_u24_e32 v3, 0x420, v3
	v_lshlrev_b32_e32 v4, 5, v100
	v_and_b32_e32 v4, 0x380, v4
	v_and_b32_e32 v5, 48, v0
	v_add3_u32 v98, v5, v3, v4
	ds_read_b128 v[30:33], v98
	ds_read_b128 v[34:37], v98 offset:64
	ds_read_b128 v[62:65], v98 offset:8448
	ds_read_b128 v[66:69], v98 offset:8512
	ds_read_b128 v[112:115], v98 offset:16896
	ds_read_b128 v[116:119], v98 offset:16960
	v_add_u32_e32 v108, s31, v2
	v_add_u32_e32 v99, 0xc600, v108
	ds_read_b128 v[82:85], v108 offset:50688
	ds_read_b128 v[86:89], v108 offset:51744
	ds_read_b128 v[90:93], v108 offset:54912
	ds_read_b128 v[94:97], v108 offset:55968
	ds_read_b128 v[120:123], v108 offset:59136
	ds_read_b128 v[124:127], v108 offset:60192
	ds_read_b128 v[128:131], v108 offset:63360
	ds_read_b128 v[132:135], v108 offset:64416
	ds_read_b128 v[136:139], v99 offset:16896
	ds_read_b128 v[140:143], v99 offset:17952
	ds_read_b128 v[166:169], v99 offset:21120
	ds_read_b128 v[170:173], v99 offset:22176
	ds_read_b128 v[174:177], v99 offset:25344
	ds_read_b128 v[178:181], v99 offset:26400
	ds_read_b128 v[190:193], v99 offset:29568
	ds_read_b128 v[194:197], v99 offset:30624
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_movk_i32 s30, 0x100
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s30 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v107, s[8:11], s30 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s30 offen lds
	s_movk_i32 s30, 0x1000
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v104, s[0:3], s30 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v103, s[0:3], s30 offen lds
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s30 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s30 offen lds
	v_or_b32_e32 v2, 0x1cec0, v100
	ds_read_u8 v2, v2
	v_or_b32_e32 v3, 0x1cf00, v100
	ds_read_u8 v3, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v2, v3, 8, v2
	v_or_b32_e32 v3, 0x1cf40, v100
	ds_read_u8 v3, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v188, v3, 16, v2
	v_mov_b32_e32 v2, 0x1d281
	ds_read_u8 v164, v2
	v_mov_b32_e32 v2, 0x1d287
	ds_read_u8 v189, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[82:89], v[30:37], 0, v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[90:97], v[30:37], 0, v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[120:127], v[30:37], 0, v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[128:135], v[30:37], 0, v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[136:143], v[30:37], 0, v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[166:173], v[30:37], 0, v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[174:181], v[30:37], 0, v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[30:37], 0, v111, v109 op_sel_hi:[0,0,0]
	ds_read_b128 v[198:201], v98 offset:25344
	ds_read_b128 v[202:205], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:89], v[62:69], 0, v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[90:97], v[62:69], 0, v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[120:127], v[62:69], 0, v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[128:135], v[62:69], 0, v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[136:143], v[62:69], 0, v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[166:173], v[62:69], 0, v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[174:181], v[62:69], 0, v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[190:197], v[62:69], 0, v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[74:77], v98 offset:33792
	ds_read_b128 v[78:81], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[82:89], v[112:119], 0, v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[82:85], v99 offset:33792
	ds_read_b128 v[86:89], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:97], v[112:119], 0, v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[90:93], v99 offset:38016
	ds_read_b128 v[94:97], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[120:127], v[112:119], 0, v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v99 offset:42240
	ds_read_b128 v[148:151], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[128:135], v[112:119], 0, v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[206:209], v99 offset:46464
	ds_read_b128 v[210:213], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[136:143], v[112:119], 0, v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[214:217], v99 offset:50688
	ds_read_b128 v[218:221], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], v[166:173], v[112:119], 0, v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[222:225], v99 offset:54912
	ds_read_b128 v[226:229], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], v[174:181], v[112:119], 0, v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[230:233], v99 offset:59136
	ds_read_b128 v[234:237], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], v[190:197], v[112:119], 0, v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[190:193], v99 offset:63360
	ds_read_b128 v[194:197], v99 offset:64416
	ds_read_b128 v[238:241], v98 offset:42240
	ds_read_b128 v[242:245], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s30, 0x180
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s30 offen lds
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v107, s[8:11], s30 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s30 offen lds
	s_movk_i32 s30, 0x1800
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v104, s[0:3], s30 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v103, s[0:3], s30 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s30 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s30 offen lds
	v_or_b32_e32 v109, 0x1cf80, v100
	ds_read_u8 v109, v109
	v_or_b32_e32 v110, 0x1cfc0, v100
	ds_read_u8 v110, v110
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v110, 8, v109
	v_or_b32_e32 v110, 0x1d000, v100
	ds_read_u8 v110, v110
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v110, 16, v109
	v_mov_b32_e32 v110, 0x1d282
	ds_read_u8 v111, v110
	v_mov_b32_e32 v110, 0x1d288
	ds_read_u8 v110, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[82:89], v[198:205], v[2:5], v164, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[90:97], v[198:205], v[6:9], v164, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[144:151], v[198:205], v[10:13], v164, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[206:213], v[198:205], v[14:17], v164, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[214:221], v[198:205], v[18:21], v189, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[222:229], v[198:205], v[22:25], v189, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[230:237], v[198:205], v[26:29], v189, v188 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[198:205], v[30:33], v189, v188 op_sel_hi:[0,0,0]
	ds_read_b128 v[112:115], v98
	ds_read_b128 v[116:119], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:89], v[74:81], v[34:37], v164, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[90:97], v[74:81], v[38:41], v164, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[144:151], v[74:81], v[42:45], v164, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[74:81], v[46:49], v164, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[214:221], v[74:81], v[50:53], v189, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[222:229], v[74:81], v[54:57], v189, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[230:237], v[74:81], v[58:61], v189, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[190:197], v[74:81], v[62:65], v189, v188 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[120:123], v98 offset:8448
	ds_read_b128 v[124:127], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[82:89], v[238:245], v[66:69], v164, v188 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v108 offset:50688
	ds_read_b128 v[132:135], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:97], v[238:245], v[70:73], v164, v188 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v108 offset:54912
	ds_read_b128 v[140:143], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[144:151], v[238:245], v[152:155], v164, v188 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v108 offset:59136
	ds_read_b128 v[148:151], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[206:213], v[238:245], v[156:159], v164, v188 op_sel_hi:[0,1,0]
	s_nop 3
	ds_read_b128 v[152:155], v108 offset:63360
	s_nop 1
	ds_read_b128 v[156:159], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[214:221], v[238:245], v[160:163], v189, v188 op_sel_hi:[0,1,0]
	s_nop 6
	ds_read_b128 v[160:163], v99 offset:16896
	ds_read_b128 v[164:167], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[222:229], v[238:245], v[168:171], v189, v188 op_sel_hi:[0,1,0]
	s_nop 6
	ds_read_b128 v[168:171], v99 offset:21120
	ds_read_b128 v[172:175], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[230:237], v[238:245], v[176:179], v189, v188 op_sel_hi:[0,1,0]
	s_nop 6
	ds_read_b128 v[176:179], v99 offset:25344
	ds_read_b128 v[180:183], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[190:197], v[238:245], v[184:187], v189, v188 op_sel_hi:[0,1,0]
	s_nop 6
	ds_read_b128 v[184:187], v99 offset:29568
	ds_read_b128 v[188:191], v99 offset:30624
	ds_read_b128 v[192:195], v98 offset:16896
	ds_read_b128 v[196:199], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s30, 0x200
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s30 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v107, s[8:11], s30 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s30 offen lds
	s_movk_i32 s23, 0x2000
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v104, s[0:3], s23 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v103, s[0:3], s23 offen lds
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s23 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s23 offen lds
	v_or_b32_e32 v200, 0x1d040, v100
	ds_read_u8 v200, v200
	v_or_b32_e32 v201, 0x1d080, v100
	ds_read_u8 v201, v201
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v200, v201, 8, v200
	v_or_b32_e32 v201, 0x1d0c0, v100
	ds_read_u8 v201, v201
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v236, v201, 16, v200
	v_mov_b32_e32 v200, 0x1d283
	ds_read_u8 v237, v200
	v_mov_b32_e32 v200, 0x1d289
	ds_read_u8 v238, v200
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], v[128:135], v[112:119], v[2:5], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], v[136:143], v[112:119], v[6:9], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], v[144:151], v[112:119], v[10:13], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], v[152:159], v[112:119], v[14:17], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], v[160:167], v[112:119], v[18:21], v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], v[168:175], v[112:119], v[22:25], v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], v[176:183], v[112:119], v[26:29], v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[184:191], v[112:119], v[30:33], v110, v109 op_sel_hi:[0,0,0]
	ds_read_b128 v[240:243], v98 offset:25344
	ds_read_b128 v[244:247], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[116:119], v[128:135], v[120:127], v[34:37], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], v[136:143], v[120:127], v[38:41], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[144:151], v[120:127], v[42:45], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[152:159], v[120:127], v[46:49], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[160:167], v[120:127], v[50:53], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[120:127], v[54:57], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[176:183], v[120:127], v[58:61], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[184:191], v[120:127], v[62:65], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[248:251], v98 offset:33792
	ds_read_b128 v[252:255], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[128:135], v[192:199], v[66:69], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[2:5], v99 offset:33792
	ds_read_b128 v[6:9], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[136:143], v[192:199], v[70:73], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[120:123], v99 offset:38016
	ds_read_b128 v[124:127], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[144:151], v[192:199], v[74:77], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[140:143], v99 offset:42240
	ds_read_b128 v[144:147], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[152:159], v[192:199], v[78:81], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:46464
	ds_read_b128 v[132:135], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[160:167], v[192:199], v[82:85], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[156:159], v99 offset:50688
	ds_read_b128 v[160:163], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], v[168:175], v[192:199], v[86:89], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[14:17], v99 offset:54912
	ds_read_b128 v[18:21], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], v[176:183], v[192:199], v[90:93], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[22:25], v99 offset:59136
	ds_read_b128 v[26:29], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[184:191], v[192:199], v[94:97], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[30:33], v99 offset:63360
	ds_read_b128 v[34:37], v99 offset:64416
	ds_read_b128 v[38:41], v98 offset:42240
	ds_read_b128 v[42:45], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s23, 0x280
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s23 offen lds
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v107, s[8:11], s23 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s23 offen lds
	s_movk_i32 s5, 0x2800
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v104, s[0:3], s5 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v103, s[0:3], s5 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s5 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s5 offen lds
	v_or_b32_e32 v86, 0x1d100, v100
	ds_read_u8 v86, v86
	v_or_b32_e32 v87, 0x1d140, v100
	ds_read_u8 v87, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v86, v87, 8, v86
	v_or_b32_e32 v87, 0x1d180, v100
	ds_read_u8 v87, v87
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v101, v87, 16, v86
	v_mov_b32_e32 v86, 0x1d284
	ds_read_u8 v106, v86
	v_mov_b32_e32 v86, 0x1d28a
	ds_read_u8 v107, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[2:9], v[240:247], v[200:203], v237, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[120:127], v[240:247], v[204:207], v237, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[140:147], v[240:247], v[208:211], v237, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[128:135], v[240:247], v[212:215], v237, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[156:163], v[240:247], v[216:219], v238, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[14:21], v[240:247], v[220:223], v238, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[22:29], v[240:247], v[224:227], v238, v236 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[30:37], v[240:247], v[112:115], v238, v236 op_sel_hi:[0,0,0]
	ds_read_b128 v[180:183], v98
	ds_read_b128 v[184:187], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[2:9], v[248:255], v[116:119], v237, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], v[120:127], v[248:255], v[228:231], v237, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[140:147], v[248:255], v[232:235], v237, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[128:135], v[248:255], v[46:49], v237, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[156:163], v[248:255], v[50:53], v238, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[14:21], v[248:255], v[54:57], v238, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[22:29], v[248:255], v[58:61], v238, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[30:37], v[248:255], v[62:65], v238, v236 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[188:191], v98 offset:8448
	ds_read_b128 v[192:195], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[2:9], v[38:45], v[66:69], v237, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[196:199], v108 offset:50688
	ds_read_b128 v[200:203], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[120:127], v[38:45], v[70:73], v237, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[118:121], v108 offset:54912
	ds_read_b128 v[122:125], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[140:147], v[38:45], v[74:77], v237, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[204:207], v108 offset:59136
	ds_read_b128 v[208:211], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[128:135], v[38:45], v[78:81], v237, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[126:129], v108 offset:63360
	ds_read_b128 v[130:133], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[156:163], v[38:45], v[82:85], v238, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[156:159], v99 offset:16896
	ds_read_b128 v[160:163], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[14:21], v[38:45], v[172:175], v238, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[212:215], v99 offset:21120
	ds_read_b128 v[216:219], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[22:29], v[38:45], v[176:179], v238, v236 op_sel_hi:[0,1,0]
	s_nop 3
	ds_read_b128 v[172:175], v99 offset:25344
	s_nop 1
	ds_read_b128 v[176:179], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[30:37], v[38:45], v[10:13], v238, v236 op_sel_hi:[0,1,0]
	ds_read_b128 v[220:223], v99 offset:29568
	ds_read_b128 v[224:227], v99 offset:30624
	ds_read_b128 v[228:231], v98 offset:16896
	ds_read_b128 v[232:235], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_or_b32_e32 v14, 0x1d1c0, v100
	ds_read_u8 v14, v14
	v_or_b32_e32 v15, 0x1d200, v100
	ds_read_u8 v15, v15
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v14, v15, 8, v14
	v_or_b32_e32 v15, 0x1d240, v100
	ds_read_u8 v15, v15
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v140, v15, 16, v14
	v_mov_b32_e32 v14, 0x1d285
	ds_read_u8 v141, v14
	v_mov_b32_e32 v14, 0x1d28b
	ds_read_u8 v142, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[196:203], v[180:187], v[86:89], v106, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:125], v[180:187], v[90:93], v106, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[204:211], v[180:187], v[94:97], v106, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[126:133], v[180:187], v[102:105], v106, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[156:163], v[180:187], v[148:151], v107, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[212:219], v[180:187], v[152:155], v107, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[172:179], v[180:187], v[164:167], v107, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[220:227], v[180:187], v[110:113], v107, v101 op_sel_hi:[0,0,0]
	ds_read_b128 v[144:147], v98 offset:25344
	s_nop 2
	ds_read_b128 v[148:151], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[196:203], v[188:195], v[114:117], v106, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[118:125], v[188:195], v[168:171], v106, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[204:211], v[188:195], v[136:139], v106, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[126:133], v[188:195], v[46:49], v106, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[156:163], v[188:195], v[50:53], v107, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[212:219], v[188:195], v[54:57], v107, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[172:179], v[188:195], v[58:61], v107, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[220:227], v[188:195], v[62:65], v107, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[164:167], v98 offset:33792
	ds_read_b128 v[168:171], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[196:203], v[228:235], v[66:69], v106, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[180:183], v99 offset:33792
	ds_read_b128 v[184:187], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[118:125], v[228:235], v[70:73], v106, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[188:191], v99 offset:38016
	ds_read_b128 v[192:195], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[204:211], v[228:235], v[74:77], v106, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[196:199], v99 offset:42240
	ds_read_b128 v[200:203], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[126:133], v[228:235], v[78:81], v106, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[120:123], v99 offset:46464
	ds_read_b128 v[124:127], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[156:163], v[228:235], v[82:85], v107, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:50688
	ds_read_b128 v[132:135], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[212:219], v[228:235], v[2:5], v107, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v99 offset:54912
	ds_read_b128 v[156:159], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[172:179], v[228:235], v[6:9], v107, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[172:175], v99 offset:59136
	ds_read_b128 v[176:179], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[220:227], v[228:235], v[10:13], v107, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[204:207], v99 offset:63360
	ds_read_b128 v[208:211], v99 offset:64416
	ds_read_b128 v[212:215], v98 offset:42240
	ds_read_b128 v[216:219], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[180:187], v[144:151], v[14:17], v141, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[188:195], v[144:151], v[18:21], v141, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[196:203], v[144:151], v[22:25], v141, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[116:119], v[120:127], v[144:151], v[26:29], v141, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[128:135], v[144:151], v[30:33], v142, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[152:159], v[144:151], v[34:37], v142, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[172:179], v[144:151], v[38:41], v142, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[204:211], v[144:151], v[42:45], v142, v140 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[180:187], v[164:171], v[86:89], v141, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[188:195], v[164:171], v[90:93], v141, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[196:203], v[164:171], v[94:97], v141, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[120:127], v[164:171], v[46:49], v141, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[128:135], v[164:171], v[50:53], v142, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[152:159], v[164:171], v[54:57], v142, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[172:179], v[164:171], v[58:61], v142, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[204:211], v[164:171], v[62:65], v142, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[180:187], v[212:219], v[66:69], v141, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[188:195], v[212:219], v[70:73], v141, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[196:203], v[212:219], v[74:77], v141, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[120:127], v[212:219], v[78:81], v141, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[128:135], v[212:219], v[82:85], v142, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[152:159], v[212:219], v[2:5], v142, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[172:179], v[212:219], v[6:9], v142, v140 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[204:211], v[212:219], v[100:103], v142, v140 op_sel_hi:[0,1,0]
	s_lshl_b32 s4, s6, 6
	s_mul_i32 s0, s6, s15
	v_mul_lo_u32 v1, s6, v1
	s_lshl_b32 s1, s13, 4
	v_lshrrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v0, 12, v0
	s_lshl4_add_u32 s0, s0, s1
	v_add3_u32 v70, v0, v1, s0
	v_add_u32_e32 v71, s4, v70
	s_mul_i32 s0, s6, s7
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s16, s0
	s_addc_u32 s5, s17, s1
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[0:1], s[12:13], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s5, s1
	s_mul_i32 s2, s6, s14
	s_sub_i32 s2, s2, s12
	s_lshl_b32 s2, s2, 1
	s_and_b32 s1, s1, 0xffff
	v_cvt_pk_bf16_f32 v1, v106, v107
	v_cvt_pk_bf16_f32 v0, v104, v105
	v_lshlrev_b32_e32 v70, 1, v70
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v110, v111
	v_cvt_pk_bf16_f32 v0, v108, v109
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v114, v115
	v_cvt_pk_bf16_f32 v0, v112, v113
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v118, v119
	v_cvt_pk_bf16_f32 v0, v116, v117
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v32, v33
	v_cvt_pk_bf16_f32 v0, v30, v31
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v36, v37
	v_cvt_pk_bf16_f32 v0, v34, v35
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v40, v41
	v_cvt_pk_bf16_f32 v0, v38, v39
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v44, v45
	v_cvt_pk_bf16_f32 v0, v42, v43
	buffer_store_dwordx2 v[0:1], v70, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v88, v89
	v_cvt_pk_bf16_f32 v0, v86, v87
	v_lshlrev_b32_e32 v30, 1, v71
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v92, v93
	v_cvt_pk_bf16_f32 v0, v90, v91
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v96, v97
	v_cvt_pk_bf16_f32 v0, v94, v95
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v48, v49
	v_cvt_pk_bf16_f32 v0, v46, v47
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v52, v53
	v_cvt_pk_bf16_f32 v0, v50, v51
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v56, v57
	v_cvt_pk_bf16_f32 v0, v54, v55
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v60, v61
	v_cvt_pk_bf16_f32 v0, v58, v59
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v64, v65
	v_cvt_pk_bf16_f32 v0, v62, v63
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v68, v69
	v_cvt_pk_bf16_f32 v0, v66, v67
	v_add_lshl_u32 v30, v71, s4, 1
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v28, v29
	v_cvt_pk_bf16_f32 v0, v26, v27
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v24, v25
	v_cvt_pk_bf16_f32 v0, v22, v23
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v20, v21
	v_cvt_pk_bf16_f32 v0, v18, v19
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v16, v17
	v_cvt_pk_bf16_f32 v0, v14, v15
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v12, v13
	v_cvt_pk_bf16_f32 v0, v10, v11
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v8, v9
	v_cvt_pk_bf16_f32 v0, v6, v7
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v4, v5
	v_cvt_pk_bf16_f32 v0, v2, v3
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:448
	s_endpgm
.Lfunc_end0:
	.size	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950, .Lfunc_end0-_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
		.amdhsa_group_segment_fixed_size 119436
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text,"axG",@progbits,_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
                                        ; -- End function
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_vgpr, 256
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_agpr, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.numbered_sgpr, 34
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_named_barrier, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.private_seg_size, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_vcc, 1
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_flat_scratch, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_dyn_sized_stack, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_recursion, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6524
; TotalNumSgprs: 40
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 119436 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_830661443091056b,@object ; @__hip_cuid_830661443091056b
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_830661443091056b
__hip_cuid_830661443091056b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_830661443091056b, 1

	.ident	"clang version 24.0.0git (https://github.com/yuyzhang512/llvm-project.git 49c41889681640665400cb01c9fbb4c0a024cde4)"
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_830661443091056b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 119436
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi768EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu9.50-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
