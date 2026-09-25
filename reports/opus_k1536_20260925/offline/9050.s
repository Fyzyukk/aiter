	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
	.protected	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950 ; -- Begin function _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.globl	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.p2align	8
	.type	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,@function
_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950: ; @_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
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
	v_readfirstlane_b32 s31, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s3, 0xc0
	v_lshlrev_b32_e32 v1, 4, v0
	v_mul_u32_u24_e32 v2, 0x1556, v0
	v_lshrrev_b32_e32 v7, 16, v2
	s_movk_i32 s0, 0x90
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
	v_mov_b32_e32 v10, 0x7f
	v_mov_b32_e32 v9, 0x7f
	v_mov_b32_e32 v8, 0x7f
	v_mov_b32_e32 v3, 0x7f
	v_mov_b32_e32 v13, 0x7f
	v_mov_b32_e32 v12, 0x7f
	v_mov_b32_e32 v11, 0x7f
	v_mov_b32_e32 v4, 0x7f
	v_mov_b32_e32 v16, 0x7f
	v_mov_b32_e32 v15, 0x7f
	v_mov_b32_e32 v14, 0x7f
	v_mov_b32_e32 v5, 0x7f
	v_mov_b32_e32 v19, 0x7f
	v_mov_b32_e32 v18, 0x7f
	v_mov_b32_e32 v17, 0x7f
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	s_and_b32 s25, s13, 0xffff
	s_mov_b32 s27, 0x20000
	s_mov_b32 s26, -1
	s_mov_b32 s24, s12
	v_mad_u64_u32 v[2:3], s[12:13], s18, v7, v[6:7]
	buffer_load_dwordx4 v[2:5], v2, s[24:27], 0 offen
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v8, 24, v2
	v_lshrrev_b32_e32 v9, 16, v2
	v_lshrrev_b32_e32 v10, 8, v2
	v_lshrrev_b32_e32 v11, 24, v3
	v_lshrrev_b32_e32 v12, 16, v3
	v_lshrrev_b32_e32 v13, 8, v3
	v_lshrrev_b32_e32 v14, 24, v4
	v_lshrrev_b32_e32 v15, 16, v4
	v_lshrrev_b32_e32 v16, 8, v4
	v_lshrrev_b32_e32 v17, 24, v5
	v_lshrrev_b32_e32 v18, 16, v5
	v_lshrrev_b32_e32 v19, 8, v5
.LBB0_3:
	s_or_b64 exec, exec, s[20:21]
	s_mov_b32 s3, 0xc0c0004
	v_perm_b32 v2, v2, v10, s3
	v_perm_b32 v6, v9, v8, s3
	v_lshl_or_b32 v2, v6, 16, v2
	v_perm_b32 v3, v3, v13, s3
	v_perm_b32 v6, v12, v11, s3
	v_lshl_or_b32 v3, v6, 16, v3
	v_perm_b32 v4, v4, v16, s3
	v_perm_b32 v6, v15, v14, s3
	v_lshl_or_b32 v4, v6, 16, v4
	v_perm_b32 v5, v5, v19, s3
	v_perm_b32 v6, v18, v17, s3
	v_lshl_or_b32 v5, v6, 16, v5
	v_add_u32_e32 v6, 0x1ce00, v1
	ds_write_b128 v6, v[2:5]
.LBB0_4:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 24, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_and_b32 s25, s15, 0xffff
	s_mov_b32 s27, 0x20000
	s_mov_b32 s26, -1
	s_mov_b32 s24, s14
	v_subrev_co_u32_e32 v2, vcc, 12, v0
	s_nop 1
	v_cndmask_b32_e32 v2, v2, v0, vcc
	v_lshl_add_u32 v3, s2, 1, v7
	v_mad_u64_u32 v[2:3], s[12:13], s19, v3, v[2:3]
	buffer_load_ubyte v3, v2, s[24:27], 0 offen
	v_add_u32_e32 v2, 0x1d700, v0
	s_waitcnt vmcnt(0)
	ds_write_b8 v2, v3
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s12, s2, 8
	s_mul_i32 s0, s4, s7
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s8, s0
	s_addc_u32 s1, s9, s1
	s_sub_i32 s14, s22, s7
	s_mul_i32 s2, s14, s4
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mul_i32 s8, s5, s12
	s_ashr_i32 s9, s8, 31
	s_add_u32 s8, s10, s8
	s_addc_u32 s9, s11, s9
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s10, -1
	s_mov_b32 s11, s3
	s_lshr_b32 s13, s31, 8
	s_mul_i32 s33, s13, 0x840
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v2, 4, v2
	s_bfe_u32 s15, s31, 0x20006
	v_and_b32_e32 v1, 0x70, v1
	s_mul_i32 s18, s5, s15
	s_lshl_b32 s18, s18, 5
	s_mul_i32 s19, s15, s4
	s_mul_i32 s20, s5, s13
	s_lshl_b32 s20, s20, 7
	s_mul_i32 s21, s4, s13
	s_lshl_b32 s21, s21, 5
	s_add_i32 s19, s19, s21
	s_add_i32 s18, s18, s20
	s_movk_i32 s30, 0x400
	s_add_i32 s20, s18, 0x400
	s_lshl4_add_u32 s5, s5, s20
	v_add_u32_e32 v106, s5, v2
	v_add_u32_e32 v4, 0xfffffc00, v106
	v_add_u32_e32 v5, s20, v2
	v_add_u32_e32 v251, s18, v2
	v_bfe_u32 v3, v0, 3, 3
	v_mul_lo_u32 v3, s4, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_add3_u32 v107, s19, v1, v3
	v_lshl_add_u32 v101, s4, 7, v107
	s_lshl_b32 s4, s4, 6
	v_subrev_u32_e32 v250, s4, v101
	s_mul_i32 s4, s13, 0x1080
	s_mul_i32 s29, s15, 0x420
	s_add_i32 s29, s29, s4
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], 0 offen lds
	s_add_i32 s23, s29, 0x2100
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], 0 offen lds
	s_add_i32 s24, s29, 0x4200
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], 0 offen lds
	s_mul_i32 s4, s13, 0x4200
	s_mul_i32 s5, s15, 0x1080
	s_add_i32 s26, s5, s4
	s_add_i32 s26, s26, 0xc600
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], 0 offen lds
	s_add_i32 s25, s26, 0x420
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v5, s[8:11], 0 offen lds
	s_add_i32 s27, s26, 0x840
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v4, s[8:11], 0 offen lds
	s_add_i32 s28, s26, 0xc60
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], 0 offen lds
	s_add_i32 s22, s29, 0x6300
	s_movk_i32 s5, 0x80
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s5 offen lds
	s_add_i32 s18, s29, 0x8400
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], s5 offen lds
	s_add_i32 s4, s29, 0xa500
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s5 offen lds
	s_add_i32 s5, s26, 0x8400
	s_movk_i32 s34, 0x800
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s34 offen lds
	s_add_i32 s21, s26, 0x8820
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v5, s[8:11], s34 offen lds
	v_mov_b32_e32 v6, v5
	s_add_i32 s19, s26, 0x8c40
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v4, s[8:11], s34 offen lds
	v_mov_b32_e32 v8, v4
	s_add_i32 s20, s26, 0x9060
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s34 offen lds
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v1, 15, v0
	scratch_store_dword off, v1, off        ; 4-byte Folded Spill
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
	v_lshl_or_b32 v94, v4, 16, v3
	v_mov_b32_e32 v3, 0x1d700
	ds_read_u8 v78, v3
	v_mov_b32_e32 v3, 0x1d70c
	ds_read_u8 v95, v3
	v_and_b32_e32 v3, 3, v0
	s_lshr_b32 s31, s31, 5
	v_and_or_b32 v3, s31, 4, v3
	v_mul_u32_u24_e32 v3, 0x420, v3
	v_lshlrev_b32_e32 v4, 5, v100
	s_movk_i32 s31, 0x380
	v_and_b32_e32 v4, 0x380, v4
	v_and_b32_e32 v5, 48, v0
	v_add3_u32 v98, v5, v3, v4
	ds_read_b128 v[30:33], v98
	ds_read_b128 v[34:37], v98 offset:64
	ds_read_b128 v[62:65], v98 offset:8448
	ds_read_b128 v[66:69], v98 offset:8512
	ds_read_b128 v[110:113], v98 offset:16896
	ds_read_b128 v[114:117], v98 offset:16960
	v_add_u32_e32 v108, s33, v2
	v_add_u32_e32 v99, 0xc600, v108
	ds_read_b128 v[70:73], v108 offset:50688
	ds_read_b128 v[74:77], v108 offset:51744
	ds_read_b128 v[80:83], v108 offset:54912
	ds_read_b128 v[84:87], v108 offset:55968
	ds_read_b128 v[118:121], v108 offset:59136
	ds_read_b128 v[122:125], v108 offset:60192
	ds_read_b128 v[126:129], v108 offset:63360
	ds_read_b128 v[130:133], v108 offset:64416
	ds_read_b128 v[134:137], v99 offset:16896
	ds_read_b128 v[138:141], v99 offset:17952
	ds_read_b128 v[142:145], v99 offset:21120
	ds_read_b128 v[146:149], v99 offset:22176
	ds_read_b128 v[154:157], v99 offset:25344
	ds_read_b128 v[158:161], v99 offset:26400
	ds_read_b128 v[162:165], v99 offset:29568
	ds_read_b128 v[166:169], v99 offset:30624
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_movk_i32 s33, 0x100
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s33 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], s33 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s33 offen lds
	s_movk_i32 s33, 0x1000
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s33 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v6, s[8:11], s33 offen lds
	v_mov_b32_e32 v9, v6
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v8, s[8:11], s33 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s33 offen lds
	v_or_b32_e32 v2, 0x1cec0, v100
	ds_read_u8 v2, v2
	v_or_b32_e32 v3, 0x1cf00, v100
	ds_read_u8 v3, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v2, v3, 8, v2
	v_or_b32_e32 v3, 0x1cf40, v100
	ds_read_u8 v3, v3
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v184, v3, 16, v2
	v_mov_b32_e32 v2, 0x1d701
	ds_read_u8 v152, v2
	v_mov_b32_e32 v2, 0x1d70d
	ds_read_u8 v185, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], v[70:77], v[30:37], 0, v78, v94 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[80:87], v[30:37], 0, v78, v94 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:125], v[30:37], 0, v78, v94 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[126:133], v[30:37], 0, v78, v94 op_sel_hi:[0,0,0]
	scratch_store_dword off, v0, off offset:4 ; 4-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[134:141], v[30:37], 0, v95, v94 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[142:149], v[30:37], 0, v95, v94 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[154:161], v[30:37], 0, v95, v94 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[162:169], v[30:37], 0, v95, v94 op_sel_hi:[0,0,0]
	ds_read_b128 v[170:173], v98 offset:25344
	ds_read_b128 v[174:177], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[70:77], v[62:69], 0, v78, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[80:87], v[62:69], 0, v78, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[118:125], v[62:69], 0, v78, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[126:133], v[62:69], 0, v78, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 11
	scratch_store_dwordx4 off, v[0:3], off offset:56 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[134:141], v[62:69], 0, v95, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[142:149], v[62:69], 0, v95, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[154:161], v[62:69], 0, v95, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[162:169], v[62:69], 0, v95, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[186:189], v98 offset:33792
	ds_read_b128 v[190:193], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[70:77], v[110:117], 0, v78, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[194:197], v99 offset:33792
	ds_read_b128 v[198:201], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[80:87], v[110:117], 0, v78, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[202:205], v99 offset:38016
	ds_read_b128 v[206:209], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[118:125], v[110:117], 0, v78, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[210:213], v99 offset:42240
	ds_read_b128 v[214:217], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[126:133], v[110:117], 0, v78, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[218:221], v99 offset:46464
	ds_read_b128 v[222:225], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[134:141], v[110:117], 0, v95, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[226:229], v99 offset:50688
	ds_read_b128 v[230:233], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[142:149], v[110:117], 0, v95, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[234:237], v99 offset:54912
	ds_read_b128 v[238:241], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[154:161], v[110:117], 0, v95, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[242:245], v99 offset:59136
	ds_read_b128 v[246:249], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[162:169], v[110:117], 0, v95, v94 op_sel_hi:[0,1,0]
	ds_read_b128 v[38:41], v99 offset:63360
	ds_read_b128 v[42:45], v99 offset:64416
	ds_read_b128 v[0:3], v98 offset:42240
	ds_read_b128 v[4:7], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s33, 0x180
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s33 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], s33 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s33 offen lds
	s_movk_i32 s33, 0x1800
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s33 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v9, s[8:11], s33 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v8, s[8:11], s33 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s33 offen lds
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
	v_mov_b32_e32 v110, 0x1d702
	ds_read_u8 v111, v110
	v_mov_b32_e32 v110, 0x1d70e
	ds_read_u8 v110, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], v[194:201], v[170:177], v[252:255], v152, v184 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[202:209], v[170:177], v[46:49], v152, v184 op_sel_hi:[0,0,0]
	s_nop 11
	scratch_store_dwordx4 off, v[46:49], off offset:8 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[210:217], v[170:177], v[10:13], v152, v184 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[218:225], v[170:177], v[14:17], v152, v184 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[226:233], v[170:177], v[18:21], v185, v184 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[234:241], v[170:177], v[22:25], v185, v184 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[242:249], v[170:177], v[26:29], v185, v184 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[38:45], v[170:177], v[30:33], v185, v184 op_sel_hi:[0,0,0]
	ds_read_b128 v[112:115], v98
	ds_read_b128 v[116:119], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[194:201], v[186:193], v[34:37], v152, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[202:209], v[186:193], v[178:181], v152, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 11
	scratch_store_dwordx4 off, v[46:49], off offset:24 ; 16-byte Folded Spill
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[210:217], v[186:193], v[102:105], v152, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 11
	scratch_store_dwordx4 off, v[46:49], off offset:40 ; 16-byte Folded Spill
	scratch_load_dwordx4 v[46:49], off, off offset:56 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[218:225], v[186:193], v[46:49], v152, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[226:233], v[186:193], v[50:53], v185, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[234:241], v[186:193], v[54:57], v185, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[242:249], v[186:193], v[58:61], v185, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[38:45], v[186:193], v[62:65], v185, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[120:123], v98 offset:8448
	ds_read_b128 v[124:127], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[194:201], v[0:7], v[66:69], v152, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v108 offset:50688
	ds_read_b128 v[132:135], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[202:209], v[0:7], v[70:73], v152, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v108 offset:54912
	ds_read_b128 v[140:143], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[210:217], v[0:7], v[74:77], v152, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v108 offset:59136
	ds_read_b128 v[148:151], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[218:225], v[0:7], v[78:81], v152, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v108 offset:63360
	ds_read_b128 v[156:159], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[226:233], v[0:7], v[82:85], v185, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[160:163], v99 offset:16896
	ds_read_b128 v[164:167], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[234:241], v[0:7], v[86:89], v185, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[168:171], v99 offset:21120
	ds_read_b128 v[172:175], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[242:249], v[0:7], v[90:93], v185, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[176:179], v99 offset:25344
	ds_read_b128 v[180:183], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[38:45], v[0:7], v[94:97], v185, v184 op_sel_hi:[0,1,0]
	ds_read_b128 v[184:187], v99 offset:29568
	ds_read_b128 v[188:191], v99 offset:30624
	ds_read_b128 v[192:195], v98 offset:16896
	ds_read_b128 v[196:199], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s33, 0x200
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s33 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], s33 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s33 offen lds
	s_movk_i32 s33, 0x2000
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s33 offen lds
	s_mov_b32 m0, s25
	v_mov_b32_e32 v102, v9
	buffer_load_dwordx4 v102, s[8:11], s33 offen lds
	s_mov_b32 m0, s27
	v_mov_b32_e32 v103, v8
	buffer_load_dwordx4 v103, s[8:11], s33 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s33 offen lds
	v_or_b32_e32 v200, 0x1d040, v100
	ds_read_u8 v200, v200
	v_or_b32_e32 v201, 0x1d080, v100
	ds_read_u8 v201, v201
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v200, v201, 8, v200
	v_or_b32_e32 v201, 0x1d0c0, v100
	ds_read_u8 v201, v201
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v200, v201, 16, v200
	v_mov_b32_e32 v201, 0x1d703
	ds_read_u8 v201, v201
	v_mov_b32_e32 v202, 0x1d70f
	ds_read_u8 v202, v202
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[128:135], v[112:119], v[252:255], v111, v109 op_sel_hi:[0,0,0]
	scratch_load_dwordx4 v[6:9], off, off offset:8 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[136:143], v[112:119], v[6:9], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[144:151], v[112:119], v[10:13], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[112:119], v[14:17], v111, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[160:167], v[112:119], v[18:21], v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[112:119], v[22:25], v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[176:183], v[112:119], v[26:29], v110, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[184:191], v[112:119], v[30:33], v110, v109 op_sel_hi:[0,0,0]
	ds_read_b128 v[112:115], v98 offset:25344
	ds_read_b128 v[116:119], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[128:135], v[120:127], v[34:37], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	scratch_load_dwordx4 v[38:41], off, off offset:24 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[136:143], v[120:127], v[38:41], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	scratch_load_dwordx4 v[42:45], off, off offset:40 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[144:151], v[120:127], v[42:45], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[152:159], v[120:127], v[46:49], v111, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[160:167], v[120:127], v[50:53], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[120:127], v[54:57], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[176:183], v[120:127], v[58:61], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[184:191], v[120:127], v[62:65], v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[120:123], v98 offset:33792
	ds_read_b128 v[124:127], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[128:135], v[192:199], v[66:69], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:33792
	ds_read_b128 v[132:135], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[136:143], v[192:199], v[70:73], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v99 offset:38016
	ds_read_b128 v[140:143], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[144:151], v[192:199], v[74:77], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v99 offset:42240
	ds_read_b128 v[148:151], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[152:159], v[192:199], v[78:81], v111, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v99 offset:46464
	ds_read_b128 v[156:159], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[160:167], v[192:199], v[82:85], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[160:163], v99 offset:50688
	ds_read_b128 v[164:167], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[192:199], v[86:89], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[168:171], v99 offset:54912
	ds_read_b128 v[172:175], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[176:183], v[192:199], v[90:93], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[176:179], v99 offset:59136
	ds_read_b128 v[180:183], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[184:191], v[192:199], v[94:97], v110, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[184:187], v99 offset:63360
	ds_read_b128 v[188:191], v99 offset:64416
	ds_read_b128 v[204:207], v98 offset:42240
	ds_read_b128 v[208:211], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s33, 0x280
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s33 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], s33 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s33 offen lds
	s_movk_i32 s33, 0x2800
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s33 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v102, s[8:11], s33 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v103, s[8:11], s33 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s33 offen lds
	v_or_b32_e32 v109, 0x1d100, v100
	ds_read_u8 v109, v109
	v_or_b32_e32 v110, 0x1d140, v100
	ds_read_u8 v110, v110
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v110, 8, v109
	v_or_b32_e32 v110, 0x1d180, v100
	ds_read_u8 v110, v110
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v110, 16, v109
	v_mov_b32_e32 v110, 0x1d704
	ds_read_u8 v198, v110
	v_mov_b32_e32 v110, 0x1d710
	ds_read_u8 v199, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[128:135], v[112:119], v[2:5], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[136:143], v[112:119], v[6:9], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[144:151], v[112:119], v[10:13], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[152:159], v[112:119], v[14:17], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[160:167], v[112:119], v[18:21], v202, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[112:119], v[22:25], v202, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[176:183], v[112:119], v[26:29], v202, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[184:191], v[112:119], v[30:33], v202, v200 op_sel_hi:[0,0,0]
	ds_read_b128 v[110:113], v98
	ds_read_b128 v[114:117], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[128:135], v[120:127], v[34:37], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[136:143], v[120:127], v[38:41], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[144:151], v[120:127], v[42:45], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[152:159], v[120:127], v[46:49], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[160:167], v[120:127], v[50:53], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[120:127], v[54:57], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[176:183], v[120:127], v[58:61], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[184:191], v[120:127], v[62:65], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[118:121], v98 offset:8448
	ds_read_b128 v[122:125], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[128:135], v[204:211], v[66:69], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[126:129], v108 offset:50688
	ds_read_b128 v[130:133], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[136:143], v[204:211], v[70:73], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[134:137], v108 offset:54912
	ds_read_b128 v[138:141], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[144:151], v[204:211], v[74:77], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[142:145], v108 offset:59136
	ds_read_b128 v[146:149], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[152:159], v[204:211], v[78:81], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[150:153], v108 offset:63360
	ds_read_b128 v[154:157], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[160:167], v[204:211], v[82:85], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[158:161], v99 offset:16896
	ds_read_b128 v[162:165], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[204:211], v[86:89], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[166:169], v99 offset:21120
	ds_read_b128 v[170:173], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[176:183], v[204:211], v[90:93], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[174:177], v99 offset:25344
	ds_read_b128 v[178:181], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[184:191], v[204:211], v[94:97], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[182:185], v99 offset:29568
	ds_read_b128 v[186:189], v99 offset:30624
	ds_read_b128 v[190:193], v98 offset:16896
	ds_read_b128 v[194:197], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s33, 0x300
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s33 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v250, s[0:3], s33 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s33 offen lds
	s_movk_i32 s33, 0x3000
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s33 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v102, s[8:11], s33 offen lds
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v103, s[8:11], s33 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s33 offen lds
	v_or_b32_e32 v200, 0x1d1c0, v100
	ds_read_u8 v200, v200
	v_or_b32_e32 v201, 0x1d200, v100
	ds_read_u8 v201, v201
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v200, v201, 8, v200
	v_or_b32_e32 v201, 0x1d240, v100
	ds_read_u8 v201, v201
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v200, v201, 16, v200
	v_mov_b32_e32 v201, 0x1d705
	ds_read_u8 v201, v201
	v_mov_b32_e32 v202, 0x1d711
	ds_read_u8 v202, v202
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[126:133], v[110:117], v[2:5], v198, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[134:141], v[110:117], v[6:9], v198, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[142:149], v[110:117], v[10:13], v198, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[150:157], v[110:117], v[14:17], v198, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[158:165], v[110:117], v[18:21], v199, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[166:173], v[110:117], v[22:25], v199, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[174:181], v[110:117], v[26:29], v199, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[182:189], v[110:117], v[30:33], v199, v109 op_sel_hi:[0,0,0]
	ds_read_b128 v[204:207], v98 offset:25344
	ds_read_b128 v[208:211], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[126:133], v[118:125], v[34:37], v198, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[134:141], v[118:125], v[38:41], v198, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[142:149], v[118:125], v[42:45], v198, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[150:157], v[118:125], v[46:49], v198, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[158:165], v[118:125], v[50:53], v199, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[166:173], v[118:125], v[54:57], v199, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[174:181], v[118:125], v[58:61], v199, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[182:189], v[118:125], v[62:65], v199, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[112:115], v98 offset:33792
	ds_read_b128 v[116:119], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[126:133], v[190:197], v[66:69], v198, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[120:123], v99 offset:33792
	ds_read_b128 v[124:127], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[134:141], v[190:197], v[70:73], v198, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:38016
	ds_read_b128 v[132:135], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[142:149], v[190:197], v[74:77], v198, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v99 offset:42240
	ds_read_b128 v[140:143], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[150:157], v[190:197], v[78:81], v198, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v99 offset:46464
	ds_read_b128 v[148:151], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[158:165], v[190:197], v[82:85], v199, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v99 offset:50688
	ds_read_b128 v[156:159], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[166:173], v[190:197], v[86:89], v199, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[160:163], v99 offset:54912
	ds_read_b128 v[164:167], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[174:181], v[190:197], v[90:93], v199, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[168:171], v99 offset:59136
	ds_read_b128 v[172:175], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[182:189], v[190:197], v[94:97], v199, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[176:179], v99 offset:63360
	ds_read_b128 v[180:183], v99 offset:64416
	ds_read_b128 v[192:195], v98 offset:42240
	ds_read_b128 v[196:199], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s31 offen lds
	s_mov_b32 m0, s18
	v_mov_b32_e32 v1, v250
	buffer_load_dwordx4 v1, s[0:3], s31 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s31 offen lds
	s_movk_i32 s31, 0x3800
	s_mov_b32 m0, s5
	v_mov_b32_e32 v0, v251
	buffer_load_dwordx4 v0, s[8:11], s31 offen lds
	s_mov_b32 m0, s21
	v_mov_b32_e32 v250, v102
	buffer_load_dwordx4 v250, s[8:11], s31 offen lds
	s_mov_b32 m0, s19
	v_mov_b32_e32 v251, v103
	buffer_load_dwordx4 v251, s[8:11], s31 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s31 offen lds
	v_or_b32_e32 v109, 0x1d280, v100
	ds_read_u8 v109, v109
	v_or_b32_e32 v110, 0x1d2c0, v100
	ds_read_u8 v110, v110
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v110, 8, v109
	v_or_b32_e32 v110, 0x1d300, v100
	ds_read_u8 v110, v110
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v110, 16, v109
	v_mov_b32_e32 v110, 0x1d706
	ds_read_u8 v186, v110
	v_mov_b32_e32 v110, 0x1d712
	ds_read_u8 v190, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[120:127], v[204:211], v[2:5], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[128:135], v[204:211], v[6:9], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[136:143], v[204:211], v[10:13], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[144:151], v[204:211], v[14:17], v201, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], v[152:159], v[204:211], v[18:21], v202, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[160:167], v[204:211], v[22:25], v202, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[168:175], v[204:211], v[26:29], v202, v200 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[176:183], v[204:211], v[30:33], v202, v200 op_sel_hi:[0,0,0]
	ds_read_b128 v[204:207], v98
	ds_read_b128 v[208:211], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[120:127], v[112:119], v[34:37], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[128:135], v[112:119], v[38:41], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[136:143], v[112:119], v[42:45], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[144:151], v[112:119], v[46:49], v201, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[152:159], v[112:119], v[50:53], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[160:167], v[112:119], v[54:57], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[168:175], v[112:119], v[58:61], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[176:183], v[112:119], v[62:65], v202, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[212:215], v98 offset:8448
	ds_read_b128 v[216:219], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[120:127], v[192:199], v[66:69], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[220:223], v108 offset:50688
	ds_read_b128 v[224:227], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[128:135], v[192:199], v[70:73], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[228:231], v108 offset:54912
	ds_read_b128 v[232:235], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[136:143], v[192:199], v[74:77], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[236:239], v108 offset:59136
	ds_read_b128 v[240:243], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[144:151], v[192:199], v[78:81], v201, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[112:115], v108 offset:63360
	ds_read_b128 v[116:119], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[152:159], v[192:199], v[82:85], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[120:123], v99 offset:16896
	ds_read_b128 v[124:127], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[160:167], v[192:199], v[86:89], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[138:141], v99 offset:21120
	ds_read_b128 v[142:145], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[168:175], v[192:199], v[90:93], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[18:21], v99 offset:25344
	ds_read_b128 v[22:25], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[176:183], v[192:199], v[94:97], v202, v200 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:29568
	ds_read_b128 v[132:135], v99 offset:30624
	ds_read_b128 v[174:177], v98 offset:16896
	ds_read_b128 v[178:181], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s30 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v1, s[0:3], s30 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s30 offen lds
	s_movk_i32 s30, 0x4000
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v0, s[8:11], s30 offen lds
	v_mov_b32_e32 v195, v0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v250, s[8:11], s30 offen lds
	v_mov_b32_e32 v0, v250
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v251, s[8:11], s30 offen lds
	v_mov_b32_e32 v136, v251
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s30 offen lds
	v_or_b32_e32 v110, 0x1d340, v100
	ds_read_u8 v110, v110
	v_or_b32_e32 v111, 0x1d380, v100
	ds_read_u8 v111, v111
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v110, v111, 8, v110
	v_or_b32_e32 v111, 0x1d3c0, v100
	ds_read_u8 v111, v111
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v191, v111, 16, v110
	v_mov_b32_e32 v110, 0x1d707
	ds_read_u8 v192, v110
	v_mov_b32_e32 v110, 0x1d713
	ds_read_u8 v193, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], v[220:227], v[204:211], v[2:5], v186, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], v[228:235], v[204:211], v[6:9], v186, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], v[236:243], v[204:211], v[10:13], v186, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], v[112:119], v[204:211], v[14:17], v186, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], v[120:127], v[204:211], v[252:255], v190, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[138:145], v[204:211], v[102:105], v190, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[18:25], v[204:211], v[26:29], v190, v109 op_sel_hi:[0,0,0]
	s_nop 11
	scratch_store_dwordx4 off, v[2:5], off offset:8 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], v[128:135], v[204:211], v[30:33], v190, v109 op_sel_hi:[0,0,0]
	ds_read_b128 v[2:5], v98 offset:25344
	ds_read_b128 v[6:9], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], v[220:227], v[212:219], v[34:37], v186, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[228:235], v[212:219], v[38:41], v186, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[236:243], v[212:219], v[42:45], v186, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[112:119], v[212:219], v[46:49], v186, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[120:127], v[212:219], v[50:53], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[138:145], v[212:219], v[54:57], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[18:25], v[212:219], v[58:61], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[128:135], v[212:219], v[62:65], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[10:13], v98 offset:33792
	ds_read_b128 v[14:17], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], v[220:227], v[174:181], v[66:69], v186, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[212:215], v99 offset:33792
	ds_read_b128 v[216:219], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], v[228:235], v[174:181], v[70:73], v186, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[26:29], v99 offset:38016
	ds_read_b128 v[30:33], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[236:243], v[174:181], v[74:77], v186, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[34:37], v99 offset:42240
	ds_read_b128 v[38:41], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[112:119], v[174:181], v[78:81], v186, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[42:45], v99 offset:46464
	ds_read_b128 v[46:49], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], v[120:127], v[174:181], v[82:85], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[50:53], v99 offset:50688
	ds_read_b128 v[54:57], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[138:145], v[174:181], v[86:89], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[58:61], v99 offset:54912
	ds_read_b128 v[62:65], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[18:25], v[174:181], v[90:93], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[66:69], v99 offset:59136
	ds_read_b128 v[70:73], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[128:135], v[174:181], v[94:97], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[74:77], v99 offset:63360
	ds_read_b128 v[78:81], v99 offset:64416
	ds_read_b128 v[174:177], v98 offset:42240
	ds_read_b128 v[178:181], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s30, 0x480
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s30 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v1, s[0:3], s30 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s30 offen lds
	s_movk_i32 s30, 0x4800
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v195, s[8:11], s30 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v0, s[8:11], s30 offen lds
	s_mov_b32 m0, s19
	v_mov_b32_e32 v82, v136
	buffer_load_dwordx4 v82, s[8:11], s30 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s30 offen lds
	v_or_b32_e32 v109, 0x1d400, v100
	ds_read_u8 v109, v109
	v_or_b32_e32 v190, 0x1d440, v100
	ds_read_u8 v190, v190
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v190, 8, v109
	v_or_b32_e32 v190, 0x1d480, v100
	ds_read_u8 v190, v190
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v190, 16, v109
	v_mov_b32_e32 v190, 0x1d708
	ds_read_u8 v190, v190
	v_mov_b32_e32 v194, 0x1d714
	ds_read_u8 v194, v194
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[212:219], v[2:9], v[248:251], v192, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[26:33], v[2:9], v[196:199], v192, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[34:41], v[2:9], v[200:203], v192, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[42:49], v[2:9], v[244:247], v192, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[50:57], v[2:9], v[252:255], v193, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[58:65], v[2:9], v[102:105], v193, v191 op_sel_hi:[0,0,0]
	scratch_load_dwordx4 v[18:21], off, off offset:8 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[66:73], v[2:9], v[18:21], v193, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[74:81], v[2:9], v[204:207], v193, v191 op_sel_hi:[0,0,0]
	ds_read_b128 v[196:199], v98
	ds_read_b128 v[200:203], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[212:219], v[10:17], v[208:211], v192, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[26:33], v[10:17], v[146:149], v192, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[34:41], v[10:17], v[150:153], v192, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[42:49], v[10:17], v[154:157], v192, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[50:57], v[10:17], v[158:161], v193, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[58:65], v[10:17], v[162:165], v193, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[66:73], v[10:17], v[166:169], v193, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[74:81], v[10:17], v[170:173], v193, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 5
	ds_read_b128 v[166:169], v98 offset:8448
	ds_read_b128 v[170:173], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[212:219], v[174:181], v[220:223], v192, v191 op_sel_hi:[0,1,0]
	ds_read_b128 v[204:207], v108 offset:50688
	ds_read_b128 v[208:211], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[26:33], v[174:181], v[224:227], v192, v191 op_sel_hi:[0,1,0]
	ds_read_b128 v[212:215], v108 offset:54912
	ds_read_b128 v[216:219], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[34:41], v[174:181], v[182:185], v192, v191 op_sel_hi:[0,1,0]
	s_nop 0
	ds_read_b128 v[220:223], v108 offset:59136
	s_nop 1
	ds_read_b128 v[224:227], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[42:49], v[174:181], v[186:189], v192, v191 op_sel_hi:[0,1,0]
	s_nop 0
	ds_read_b128 v[182:185], v108 offset:63360
	s_nop 4
	ds_read_b128 v[186:189], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[50:57], v[174:181], v[228:231], v193, v191 op_sel_hi:[0,1,0]
	s_nop 6
	ds_read_b128 v[228:231], v99 offset:16896
	ds_read_b128 v[232:235], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[58:65], v[174:181], v[86:89], v193, v191 op_sel_hi:[0,1,0]
	ds_read_b128 v[236:239], v99 offset:21120
	ds_read_b128 v[240:243], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[66:73], v[174:181], v[90:93], v193, v191 op_sel_hi:[0,1,0]
	ds_read_b128 v[244:247], v99 offset:25344
	ds_read_b128 v[248:251], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[74:81], v[174:181], v[94:97], v193, v191 op_sel_hi:[0,1,0]
	ds_read_b128 v[174:177], v99 offset:29568
	ds_read_b128 v[178:181], v99 offset:30624
	ds_read_b128 v[48:51], v98 offset:16896
	ds_read_b128 v[52:55], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s30, 0x500
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s30 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v1, s[0:3], s30 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s30 offen lds
	s_movk_i32 s23, 0x5000
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v195, s[8:11], s23 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v0, s[8:11], s23 offen lds
	s_mov_b32 m0, s27
	v_mov_b32_e32 v56, v82
	buffer_load_dwordx4 v56, s[8:11], s23 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s23 offen lds
	v_or_b32_e32 v46, 0x1d4c0, v100
	ds_read_u8 v46, v46
	v_or_b32_e32 v47, 0x1d500, v100
	ds_read_u8 v47, v47
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v46, v47, 8, v46
	v_or_b32_e32 v47, 0x1d540, v100
	ds_read_u8 v47, v47
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v162, v47, 16, v46
	v_mov_b32_e32 v46, 0x1d709
	ds_read_u8 v163, v46
	v_mov_b32_e32 v46, 0x1d715
	ds_read_u8 v164, v46
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[204:211], v[196:203], v[110:113], v190, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], v[212:219], v[196:203], v[114:117], v190, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[116:119], v[220:227], v[196:203], v[118:121], v190, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[182:189], v[196:203], v[122:125], v190, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[228:235], v[196:203], v[126:129], v194, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[236:243], v[196:203], v[130:133], v194, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[244:251], v[196:203], v[134:137], v194, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[174:181], v[196:203], v[2:5], v194, v109 op_sel_hi:[0,0,0]
	s_nop 2
	ds_read_b128 v[120:123], v98 offset:25344
	ds_read_b128 v[124:127], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[204:211], v[166:173], v[6:9], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[212:219], v[166:173], v[138:141], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[220:227], v[166:173], v[142:145], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[182:189], v[166:173], v[146:149], v190, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[228:235], v[166:173], v[150:153], v194, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[236:243], v[166:173], v[154:157], v194, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[244:251], v[166:173], v[158:161], v194, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[174:181], v[166:173], v[10:13], v194, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[128:131], v98 offset:33792
	ds_read_b128 v[132:135], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[204:211], v[48:55], v[14:17], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v99 offset:33792
	ds_read_b128 v[140:143], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[212:219], v[48:55], v[18:21], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v99 offset:38016
	ds_read_b128 v[148:151], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[220:227], v[48:55], v[22:25], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v99 offset:42240
	ds_read_b128 v[156:159], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[182:189], v[48:55], v[26:29], v190, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[166:169], v99 offset:46464
	ds_read_b128 v[170:173], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[228:235], v[48:55], v[30:33], v194, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[182:185], v99 offset:50688
	ds_read_b128 v[186:189], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[236:243], v[48:55], v[34:37], v194, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[196:199], v99 offset:54912
	ds_read_b128 v[200:203], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[244:251], v[48:55], v[38:41], v194, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[204:207], v99 offset:59136
	ds_read_b128 v[208:211], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[174:181], v[48:55], v[42:45], v194, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[174:177], v99 offset:63360
	ds_read_b128 v[178:181], v99 offset:64416
	ds_read_b128 v[212:215], v98 offset:42240
	ds_read_b128 v[216:219], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_movk_i32 s23, 0x580
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v107, s[0:3], s23 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v1, s[0:3], s23 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s23 offen lds
	s_movk_i32 s0, 0x5800
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v195, s[8:11], s0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v0, s[8:11], s0 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v56, s[8:11], s0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s0 offen lds
	v_or_b32_e32 v101, 0x1d580, v100
	ds_read_u8 v101, v101
	v_or_b32_e32 v102, 0x1d5c0, v100
	ds_read_u8 v102, v102
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v101, v102, 8, v101
	v_or_b32_e32 v102, 0x1d600, v100
	ds_read_u8 v102, v102
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v109, v102, 16, v101
	v_mov_b32_e32 v101, 0x1d70a
	ds_read_u8 v114, v101
	v_mov_b32_e32 v101, 0x1d716
	ds_read_u8 v115, v101
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[136:143], v[120:127], v[110:113], v163, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[144:151], v[120:127], v[252:255], v163, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[152:159], v[120:127], v[116:119], v163, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[166:173], v[120:127], v[58:61], v163, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[182:189], v[120:127], v[62:65], v164, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[196:203], v[120:127], v[66:69], v164, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[204:211], v[120:127], v[70:73], v164, v162 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[174:181], v[120:127], v[2:5], v164, v162 op_sel_hi:[0,0,0]
	ds_read_b128 v[120:123], v98
	ds_read_b128 v[124:127], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[136:143], v[128:135], v[6:9], v163, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[144:151], v[128:135], v[74:77], v163, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[152:159], v[128:135], v[78:81], v163, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[166:173], v[128:135], v[82:85], v163, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[182:189], v[128:135], v[86:89], v164, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[196:203], v[128:135], v[90:93], v164, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[204:211], v[128:135], v[94:97], v164, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[174:181], v[128:135], v[10:13], v164, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[2:5], v98 offset:8448
	ds_read_b128 v[6:9], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[136:143], v[212:219], v[14:17], v163, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v108 offset:50688
	ds_read_b128 v[132:135], v108 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[144:151], v[212:219], v[18:21], v163, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v108 offset:54912
	ds_read_b128 v[140:143], v108 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[152:159], v[212:219], v[22:25], v163, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v108 offset:59136
	ds_read_b128 v[148:151], v108 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[166:173], v[212:219], v[26:29], v163, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v108 offset:63360
	ds_read_b128 v[156:159], v108 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[182:189], v[212:219], v[30:33], v164, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[166:169], v99 offset:16896
	ds_read_b128 v[170:173], v99 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[196:203], v[212:219], v[34:37], v164, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[182:185], v99 offset:21120
	ds_read_b128 v[186:189], v99 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[204:211], v[212:219], v[38:41], v164, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[190:193], v99 offset:25344
	ds_read_b128 v[194:197], v99 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[174:181], v[212:219], v[42:45], v164, v162 op_sel_hi:[0,1,0]
	ds_read_b128 v[174:177], v99 offset:29568
	ds_read_b128 v[178:181], v99 offset:30624
	ds_read_b128 v[198:201], v98 offset:16896
	ds_read_b128 v[202:205], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_or_b32_e32 v101, 0x1d640, v100
	ds_read_u8 v101, v101
	v_or_b32_e32 v106, 0x1d680, v100
	ds_read_u8 v106, v106
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v101, v106, 8, v101
	v_or_b32_e32 v100, 0x1d6c0, v100
	ds_read_u8 v100, v100
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v116, v100, 16, v101
	v_mov_b32_e32 v100, 0x1d70b
	ds_read_u8 v117, v100
	v_mov_b32_e32 v100, 0x1d717
	ds_read_u8 v118, v100
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[128:135], v[120:127], v[46:49], v114, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[136:143], v[120:127], v[50:53], v114, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[144:151], v[120:127], v[54:57], v114, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[152:159], v[120:127], v[58:61], v114, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[166:173], v[120:127], v[62:65], v115, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[182:189], v[120:127], v[66:69], v115, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[190:197], v[120:127], v[70:73], v115, v109 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[174:181], v[120:127], v[102:105], v115, v109 op_sel_hi:[0,0,0]
	ds_read_b128 v[120:123], v98 offset:25344
	ds_read_b128 v[124:127], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[128:135], v[2:9], v[110:113], v114, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[136:143], v[2:9], v[74:77], v114, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[144:151], v[2:9], v[78:81], v114, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[152:159], v[2:9], v[82:85], v114, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[166:173], v[2:9], v[86:89], v115, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[182:189], v[2:9], v[90:93], v115, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[190:197], v[2:9], v[94:97], v115, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[174:181], v[2:9], v[10:13], v115, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[206:209], v98 offset:33792
	ds_read_b128 v[210:213], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[128:135], v[198:205], v[14:17], v114, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[128:131], v99 offset:33792
	ds_read_b128 v[132:135], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[136:143], v[198:205], v[18:21], v114, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[136:139], v99 offset:38016
	ds_read_b128 v[140:143], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[144:151], v[198:205], v[22:25], v114, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[144:147], v99 offset:42240
	ds_read_b128 v[148:151], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[198:205], v[26:29], v114, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[152:155], v99 offset:46464
	ds_read_b128 v[156:159], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[166:173], v[198:205], v[30:33], v115, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[160:163], v99 offset:50688
	ds_read_b128 v[164:167], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[182:189], v[198:205], v[34:37], v115, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[182:185], v99 offset:54912
	ds_read_b128 v[186:189], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[190:197], v[198:205], v[38:41], v115, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[190:193], v99 offset:59136
	ds_read_b128 v[194:197], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[174:181], v[198:205], v[42:45], v115, v109 op_sel_hi:[0,1,0]
	ds_read_b128 v[168:171], v99 offset:63360
	ds_read_b128 v[172:175], v99 offset:64416
	ds_read_b128 v[198:201], v98 offset:42240
	ds_read_b128 v[202:205], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[128:135], v[120:127], v[46:49], v117, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[136:143], v[120:127], v[50:53], v117, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[144:151], v[120:127], v[54:57], v117, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[152:159], v[120:127], v[58:61], v117, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[160:167], v[120:127], v[62:65], v118, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[182:189], v[120:127], v[66:69], v118, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[190:197], v[120:127], v[70:73], v118, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[168:175], v[120:127], v[100:103], v118, v116 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[128:135], v[206:213], v[104:107], v117, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[136:143], v[206:213], v[74:77], v117, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[144:151], v[206:213], v[78:81], v117, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[152:159], v[206:213], v[82:85], v117, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[160:167], v[206:213], v[86:89], v118, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[182:189], v[206:213], v[90:93], v118, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[190:197], v[206:213], v[94:97], v118, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[168:175], v[206:213], v[2:5], v118, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[128:135], v[198:205], v[6:9], v117, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[136:143], v[198:205], v[10:13], v117, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[144:151], v[198:205], v[14:17], v117, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[152:159], v[198:205], v[18:21], v117, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[160:167], v[198:205], v[30:33], v118, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[182:189], v[198:205], v[34:37], v118, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[190:197], v[198:205], v[38:41], v118, v116 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[168:175], v[198:205], v[42:45], v118, v116 op_sel_hi:[0,1,0]
	s_lshl_b32 s4, s6, 6
	s_mul_i32 s0, s6, s15
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v1, s6, v0
	s_lshl_b32 s1, s13, 4
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
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
	v_cvt_pk_bf16_f32 v1, v48, v49
	v_cvt_pk_bf16_f32 v0, v46, v47
	v_lshlrev_b32_e32 v30, 1, v30
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v52, v53
	v_cvt_pk_bf16_f32 v0, v50, v51
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v56, v57
	v_cvt_pk_bf16_f32 v0, v54, v55
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v60, v61
	v_cvt_pk_bf16_f32 v0, v58, v59
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v64, v65
	v_cvt_pk_bf16_f32 v0, v62, v63
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v68, v69
	v_cvt_pk_bf16_f32 v0, v66, v67
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v72, v73
	v_cvt_pk_bf16_f32 v0, v70, v71
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v100, v101
	v_cvt_pk_bf16_f32 v0, v98, v99
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v104, v105
	v_cvt_pk_bf16_f32 v0, v102, v103
	v_lshlrev_b32_e32 v30, 1, v31
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v76, v77
	v_cvt_pk_bf16_f32 v0, v74, v75
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v80, v81
	v_cvt_pk_bf16_f32 v0, v78, v79
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v84, v85
	v_cvt_pk_bf16_f32 v0, v82, v83
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v88, v89
	v_cvt_pk_bf16_f32 v0, v86, v87
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v92, v93
	v_cvt_pk_bf16_f32 v0, v90, v91
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v96, v97
	v_cvt_pk_bf16_f32 v0, v94, v95
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v108, v109
	v_cvt_pk_bf16_f32 v0, v106, v107
	buffer_store_dwordx2 v[0:1], v30, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v112, v113
	v_cvt_pk_bf16_f32 v0, v110, v111
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
	.size	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950, .Lfunc_end0-_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
		.amdhsa_group_segment_fixed_size 120600
		.amdhsa_private_segment_fixed_size 76
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
		.amdhsa_enable_private_segment 1
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
	.section	.text,"axG",@progbits,_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
                                        ; -- End function
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_vgpr, 256
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_agpr, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.numbered_sgpr, 35
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_named_barrier, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.private_seg_size, 76
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_vcc, 1
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_flat_scratch, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_dyn_sized_stack, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_recursion, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11436
; TotalNumSgprs: 41
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 76
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 120600 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.type	__hip_cuid_f305ac295a1e186a,@object ; @__hip_cuid_f305ac295a1e186a
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_f305ac295a1e186a
__hip_cuid_f305ac295a1e186a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f305ac295a1e186a, 1

	.ident	"clang version 24.0.0git (https://github.com/yuyzhang512/llvm-project.git 49c41889681640665400cb01c9fbb4c0a024cde4)"
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f305ac295a1e186a
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 120600
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
    .private_segment_fixed_size: 76
    .sgpr_count:     41
    .sgpr_spill_count: 0
    .symbol:         _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi12EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 22
    .wavefront_size: 64
amdhsa.target:   amdgpu9.50-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
