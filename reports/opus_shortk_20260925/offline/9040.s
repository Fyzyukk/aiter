	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
	.protected	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950 ; -- Begin function _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.globl	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.p2align	8
	.type	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,@function
_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950: ; @_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_load_dword s23, s[0:1], 0x18
	s_load_dwordx4 s[4:7], s[0:1], 0x28
	s_load_dwordx4 s[12:15], s[0:1], 0x40
	s_load_dwordx2 s[18:19], s[0:1], 0x50
	v_readfirstlane_b32 s22, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s3, 0xc0
	v_lshlrev_b32_e32 v1, 4, v0
	v_cmp_gt_u32_e32 vcc, 36, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.1:
	v_mul_u32_u24_e32 v2, 0xaab, v1
	v_lshrrev_b32_e32 v2, 19, v2
	v_mul_lo_u16_e32 v2, 0xc0, v2
	v_sub_u16_e32 v2, v1, v2
	v_add_u32_e32 v6, s7, v2
	v_cmp_gt_i32_e32 vcc, s23, v6
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
	v_cmp_gt_u32_e32 vcc, 6, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_and_b32 s25, s15, 0xffff
	s_mov_b32 s27, 0x20000
	s_mov_b32 s26, -1
	s_mov_b32 s24, s14
	v_cmp_lt_u32_e32 vcc, 2, v0
	s_nop 1
	v_cndmask_b32_e64 v3, 0, 1, vcc
	v_subrev_co_u32_e32 v2, vcc, 3, v0
	s_nop 1
	v_cndmask_b32_e32 v2, v2, v0, vcc
	v_lshl_or_b32 v3, s2, 1, v3
	v_mad_u64_u32 v[2:3], s[12:13], s19, v3, v[2:3]
	buffer_load_ubyte v3, v2, s[24:27], 0 offen
	v_add_u32_e32 v2, 0x1d040, v0
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
	s_sub_i32 s14, s23, s7
	s_mul_i32 s10, s14, s4
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, s3
	s_lshr_b32 s13, s22, 8
	s_mul_i32 s18, s13, 0x840
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 4, v2
	s_bfe_u32 s15, s22, 0x20006
	s_mul_i32 s19, s5, s15
	s_lshl_b32 s19, s19, 5
	s_mul_i32 s20, s5, s13
	s_lshl_b32 s20, s20, 7
	s_add_i32 s19, s19, s20
	s_lshl4_add_u32 s5, s5, s19
	s_add_i32 s20, s5, 0x400
	v_add_u32_e32 v3, s20, v2
	v_add_u32_e32 v4, s5, v2
	s_addk_i32 s19, 0x400
	v_add_u32_e32 v5, s19, v2
	v_add_u32_e32 v6, 0xfffffc00, v5
	s_mul_i32 s5, s15, s4
	s_mul_i32 s19, s4, s13
	s_lshl_b32 s19, s19, 5
	s_add_i32 s5, s5, s19
	v_and_b32_e32 v1, 0x70, v1
	v_bfe_u32 v7, v0, 3, 3
	v_mul_lo_u32 v7, s4, v7
	v_lshlrev_b32_e32 v7, 2, v7
	v_add3_u32 v7, s5, v1, v7
	v_lshl_add_u32 v8, s4, 7, v7
	v_lshl_add_u32 v9, s4, 6, v7
	s_mul_i32 s4, s13, 0x1080
	s_mul_i32 s5, s15, 0x420
	s_add_i32 s4, s5, s4
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v7, s[8:11], 0 offen lds
	s_add_i32 s5, s4, 0x2100
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v9, s[8:11], 0 offen lds
	s_add_i32 s19, s4, 0x4200
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v8, s[8:11], 0 offen lds
	s_mul_i32 s20, s13, 0x4200
	s_mul_i32 s21, s15, 0x1080
	s_add_i32 s20, s21, s20
	s_add_i32 s20, s20, 0xc600
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_add_i32 s21, s20, 0x420
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v5, s[0:3], 0 offen lds
	s_add_i32 s23, s20, 0x840
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 s24, s20, 0xc60
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v3, s[0:3], 0 offen lds
	s_add_i32 m0, s4, 0x6300
	s_movk_i32 s25, 0x80
	buffer_load_dwordx4 v7, s[8:11], s25 offen lds
	s_add_i32 m0, s4, 0x8400
	s_nop 0
	buffer_load_dwordx4 v9, s[8:11], s25 offen lds
	s_add_i32 m0, s4, 0xa500
	s_nop 0
	buffer_load_dwordx4 v8, s[8:11], s25 offen lds
	s_add_i32 m0, s20, 0x8400
	s_movk_i32 s25, 0x800
	buffer_load_dwordx4 v6, s[0:3], s25 offen lds
	s_add_i32 m0, s20, 0x8820
	s_nop 0
	buffer_load_dwordx4 v5, s[0:3], s25 offen lds
	s_add_i32 m0, s20, 0x8c40
	s_nop 0
	buffer_load_dwordx4 v4, s[0:3], s25 offen lds
	s_add_i32 m0, s20, 0x9060
	s_nop 0
	buffer_load_dwordx4 v3, s[0:3], s25 offen lds
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v1, 15, v0
	v_lshl_or_b32 v103, s15, 4, v1
	v_or_b32_e32 v10, 0x1ce00, v103
	ds_read_u8 v10, v10
	v_or_b32_e32 v11, 0x1ce40, v103
	ds_read_u8 v11, v11
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v10, v11, 8, v10
	v_or_b32_e32 v11, 0x1ce80, v103
	ds_read_u8 v11, v11
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v30, v11, 16, v10
	v_mov_b32_e32 v10, 0x1d040
	ds_read_u8 v14, v10
	v_mov_b32_e32 v10, 0x1d043
	ds_read_u8 v31, v10
	v_and_b32_e32 v10, 3, v0
	s_lshr_b32 s22, s22, 5
	v_and_or_b32 v10, s22, 4, v10
	v_mul_u32_u24_e32 v10, 0x420, v10
	v_lshlrev_b32_e32 v11, 5, v103
	v_and_b32_e32 v11, 0x380, v11
	v_and_b32_e32 v12, 48, v0
	v_add3_u32 v98, v12, v10, v11
	ds_read_b128 v[16:19], v98
	ds_read_b128 v[20:23], v98 offset:64
	ds_read_b128 v[104:107], v98 offset:8448
	ds_read_b128 v[108:111], v98 offset:8512
	ds_read_b128 v[160:163], v98 offset:16896
	ds_read_b128 v[164:167], v98 offset:16960
	v_add_u32_e32 v100, s18, v2
	v_add_u32_e32 v99, 0xc600, v100
	ds_read_b128 v[112:115], v100 offset:50688
	ds_read_b128 v[116:119], v100 offset:51744
	ds_read_b128 v[120:123], v100 offset:54912
	ds_read_b128 v[124:127], v100 offset:55968
	ds_read_b128 v[128:131], v100 offset:59136
	ds_read_b128 v[132:135], v100 offset:60192
	ds_read_b128 v[136:139], v100 offset:63360
	ds_read_b128 v[140:143], v100 offset:64416
	ds_read_b128 v[144:147], v99 offset:16896
	ds_read_b128 v[148:151], v99 offset:17952
	ds_read_b128 v[152:155], v99 offset:21120
	ds_read_b128 v[156:159], v99 offset:22176
	ds_read_b128 v[168:171], v99 offset:25344
	ds_read_b128 v[172:175], v99 offset:26400
	ds_read_b128 v[176:179], v99 offset:29568
	ds_read_b128 v[180:183], v99 offset:30624
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_movk_i32 s18, 0x100
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v7, s[8:11], s18 offen lds
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v9, s[8:11], s18 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v8, s[8:11], s18 offen lds
	s_movk_i32 s4, 0x1000
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v6, s[0:3], s4 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v5, s[0:3], s4 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v4, s[0:3], s4 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v3, s[0:3], s4 offen lds
	v_or_b32_e32 v2, 0x1cec0, v103
	ds_read_u8 v2, v2
	v_or_b32_e32 v3, 0x1cf00, v103
	ds_read_u8 v3, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v2, v3, 8, v2
	v_or_b32_e32 v3, 0x1cf40, v103
	ds_read_u8 v3, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v101, v3, 16, v2
	v_mov_b32_e32 v2, 0x1d041
	ds_read_u8 v192, v2
	v_mov_b32_e32 v2, 0x1d044
	ds_read_u8 v102, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[112:119], v[16:23], 0, v14, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[120:127], v[16:23], 0, v14, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[128:135], v[16:23], 0, v14, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[136:143], v[16:23], 0, v14, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[144:151], v[16:23], 0, v31, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[152:159], v[16:23], 0, v31, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[168:175], v[16:23], 0, v31, v30 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[176:183], v[16:23], 0, v31, v30 op_sel_hi:[0,0,0]
	ds_read_b128 v[196:199], v98 offset:25344
	ds_read_b128 v[200:203], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[112:119], v[104:111], 0, v14, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[120:127], v[104:111], 0, v14, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[128:135], v[104:111], 0, v14, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[136:143], v[104:111], 0, v14, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[144:151], v[104:111], 0, v31, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[152:159], v[104:111], 0, v31, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[168:175], v[104:111], 0, v31, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[176:183], v[104:111], 0, v31, v30 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[184:187], v98 offset:33792
	ds_read_b128 v[188:191], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[112:119], v[160:167], 0, v14, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[104:107], v99 offset:33792
	ds_read_b128 v[108:111], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[120:127], v[160:167], 0, v14, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[112:115], v99 offset:38016
	ds_read_b128 v[116:119], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[128:135], v[160:167], 0, v14, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[120:123], v99 offset:42240
	ds_read_b128 v[124:127], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[136:143], v[160:167], 0, v14, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:46464
	ds_read_b128 v[132:135], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[144:151], v[160:167], 0, v31, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v99 offset:50688
	ds_read_b128 v[140:143], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[152:159], v[160:167], 0, v31, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v99 offset:54912
	ds_read_b128 v[148:151], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[168:175], v[160:167], 0, v31, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v99 offset:59136
	ds_read_b128 v[156:159], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[176:183], v[160:167], 0, v31, v30 op_sel_hi:[0,1,0]
	ds_read_b128 v[160:163], v99 offset:63360
	ds_read_b128 v[164:167], v99 offset:64416
	ds_read_b128 v[168:171], v98 offset:42240
	ds_read_b128 v[172:175], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_or_b32_e32 v176, 0x1cf80, v103
	ds_read_u8 v176, v176
	v_or_b32_e32 v177, 0x1cfc0, v103
	ds_read_u8 v177, v177
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v176, v177, 8, v176
	v_or_b32_e32 v103, 0x1d000, v103
	ds_read_u8 v103, v103
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v103, v103, 16, v176
	v_mov_b32_e32 v176, 0x1d042
	ds_read_u8 v193, v176
	v_mov_b32_e32 v176, 0x1d045
	ds_read_u8 v194, v176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[104:111], v[196:203], v[34:37], v192, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[112:119], v[196:203], v[38:41], v192, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[120:127], v[196:203], v[42:45], v192, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[128:135], v[196:203], v[46:49], v192, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[136:143], v[196:203], v[50:53], v102, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[144:151], v[196:203], v[54:57], v102, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[152:159], v[196:203], v[58:61], v102, v101 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[160:167], v[196:203], v[62:65], v102, v101 op_sel_hi:[0,0,0]
	ds_read_b128 v[176:179], v98
	ds_read_b128 v[180:183], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[104:111], v[184:191], v[66:69], v192, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[112:119], v[184:191], v[70:73], v192, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[120:127], v[184:191], v[74:77], v192, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[128:135], v[184:191], v[78:81], v192, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[136:143], v[184:191], v[82:85], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[144:151], v[184:191], v[86:89], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[152:159], v[184:191], v[90:93], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[160:167], v[184:191], v[94:97], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[184:187], v98 offset:8448
	ds_read_b128 v[188:191], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[104:111], v[168:175], v[2:5], v192, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[196:199], v100 offset:50688
	ds_read_b128 v[200:203], v100 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[112:119], v[168:175], v[6:9], v192, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[204:207], v100 offset:54912
	ds_read_b128 v[208:211], v100 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[120:127], v[168:175], v[10:13], v192, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[116:119], v100 offset:59136
	ds_read_b128 v[120:123], v100 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[128:135], v[168:175], v[14:17], v192, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[124:127], v100 offset:63360
	ds_read_b128 v[128:131], v100 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[136:143], v[168:175], v[18:21], v102, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[132:135], v99 offset:16896
	ds_read_b128 v[136:139], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[144:151], v[168:175], v[22:25], v102, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[140:143], v99 offset:21120
	ds_read_b128 v[144:147], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[152:159], v[168:175], v[26:29], v102, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[148:151], v99 offset:25344
	ds_read_b128 v[152:155], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[160:167], v[168:175], v[30:33], v102, v101 op_sel_hi:[0,1,0]
	ds_read_b128 v[156:159], v99 offset:29568
	ds_read_b128 v[160:163], v99 offset:30624
	ds_read_b128 v[164:167], v98 offset:16896
	ds_read_b128 v[168:171], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[196:203], v[176:183], v[34:37], v193, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[204:211], v[176:183], v[38:41], v193, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[116:123], v[176:183], v[42:45], v193, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[124:131], v[176:183], v[46:49], v193, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[132:139], v[176:183], v[50:53], v194, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[140:147], v[176:183], v[54:57], v194, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[148:155], v[176:183], v[58:61], v194, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[156:163], v[176:183], v[62:65], v194, v103 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[196:203], v[184:191], v[66:69], v193, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[204:211], v[184:191], v[70:73], v193, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[116:123], v[184:191], v[74:77], v193, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[124:131], v[184:191], v[78:81], v193, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[132:139], v[184:191], v[82:85], v194, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[140:147], v[184:191], v[86:89], v194, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[148:155], v[184:191], v[90:93], v194, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[156:163], v[184:191], v[94:97], v194, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[196:203], v[164:171], v[2:5], v193, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[204:211], v[164:171], v[6:9], v193, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[116:123], v[164:171], v[10:13], v193, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[124:131], v[164:171], v[14:17], v193, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[132:139], v[164:171], v[104:107], v194, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[140:147], v[164:171], v[108:111], v194, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[148:155], v[164:171], v[112:115], v194, v103 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[156:163], v[164:171], v[30:33], v194, v103 op_sel_hi:[0,1,0]
	s_lshl_b32 s4, s6, 6
	s_mul_i32 s0, s6, s15
	v_mul_lo_u32 v1, s6, v1
	s_lshl_b32 s1, s13, 4
	v_lshrrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v0, 12, v0
	s_lshl4_add_u32 s0, s0, s1
	v_add3_u32 v30, v0, v1, s0
	v_add_u32_e32 v31, s4, v30
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
	v_cvt_pk_bf16_f32 v1, v36, v37
	v_cvt_pk_bf16_f32 v0, v34, v35
	v_lshlrev_b32_e32 v30, 1, v30
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v40, v41
	v_cvt_pk_bf16_f32 v0, v38, v39
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v44, v45
	v_cvt_pk_bf16_f32 v0, v42, v43
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
	v_lshlrev_b32_e32 v30, 1, v31
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v72, v73
	v_cvt_pk_bf16_f32 v0, v70, v71
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v76, v77
	v_cvt_pk_bf16_f32 v0, v74, v75
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v80, v81
	v_cvt_pk_bf16_f32 v0, v78, v79
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v84, v85
	v_cvt_pk_bf16_f32 v0, v82, v83
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v88, v89
	v_cvt_pk_bf16_f32 v0, v86, v87
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v92, v93
	v_cvt_pk_bf16_f32 v0, v90, v91
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v96, v97
	v_cvt_pk_bf16_f32 v0, v94, v95
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v100, v101
	v_cvt_pk_bf16_f32 v0, v98, v99
	v_add_lshl_u32 v30, v31, s4, 1
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
	.size	_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950, .Lfunc_end0-_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
		.amdhsa_group_segment_fixed_size 118854
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
		.amdhsa_next_free_vgpr 212
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 212
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
	.section	.text,"axG",@progbits,_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
                                        ; -- End function
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_vgpr, 212
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_agpr, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.numbered_sgpr, 28
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_named_barrier, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.private_seg_size, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_vcc, 1
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_flat_scratch, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_dyn_sized_stack, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_recursion, 0
	.set .L_Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4072
; TotalNumSgprs: 34
; NumVgprs: 212
; NumAgprs: 0
; TotalNumVgprs: 212
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 118854 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 212
; AccumOffset: 212
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 52
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_f98a8ebb49d4b9ff,@object ; @__hip_cuid_f98a8ebb49d4b9ff
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_f98a8ebb49d4b9ff
__hip_cuid_f98a8ebb49d4b9ff:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f98a8ebb49d4b9ff, 1

	.ident	"clang version 24.0.0git (https://github.com/yuyzhang512/llvm-project.git 49c41889681640665400cb01c9fbb4c0a024cde4)"
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f98a8ebb49d4b9ff
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 118854
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _Z41gemm_a8w8_mxfp8_bpreshuffle_shortk_kernelI50opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950ILi384EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.kd
    .uses_dynamic_stack: false
    .vgpr_count:     212
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu9.50-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
