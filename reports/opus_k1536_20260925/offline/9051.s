	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
	.protected	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950 ; -- Begin function _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.globl	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.p2align	8
	.type	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,@function
_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950: ; @_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
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
	v_readfirstlane_b32 s35, v0
	v_lshlrev_b32_e32 v1, 4, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s3, 0xc0
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
	v_and_b32_e32 v2, 63, v0
	s_bfe_u32 s20, s35, 0x20006
	s_lshr_b32 s19, s35, 8
	v_bfe_u32 v3, v0, 3, 3
	v_and_b32_e32 v1, 0x70, v1
	s_mul_i32 s0, s4, s19
	s_lshl_b32 s0, s0, 5
	v_mul_lo_u32 v3, s4, v3
	v_lshlrev_b32_e32 v3, 2, v3
	s_mul_i32 s1, s20, s4
	s_add_i32 s1, s1, s0
	v_add3_u32 v100, s1, v1, v3
	v_lshl_add_u32 v101, s4, 6, v100
	v_lshl_add_u32 v102, s4, 7, v100
	s_mul_i32 s0, s5, s19
	s_lshl_b32 s0, s0, 7
	s_mul_i32 s1, s5, s20
	s_lshl_b32 s1, s1, 5
	s_add_i32 s1, s1, s0
	v_lshlrev_b32_e32 v1, 4, v2
	v_add_u32_e32 v103, s1, v1
	v_add_u32_e32 v104, 0x400, v103
	v_lshl_add_u32 v105, s5, 4, v103
	v_add_u32_e32 v106, 0x400, v105
	s_mul_i32 s0, s19, 0x840
	v_add_u32_e32 v99, s0, v1
	s_lshl_b32 s18, s2, 8
	s_mul_i32 s0, s4, s7
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s8, s0
	s_addc_u32 s1, s9, s1
	s_sub_i32 s21, s22, s7
	s_mul_i32 s2, s21, s4
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mul_i32 s4, s5, s18
	s_ashr_i32 s5, s4, 31
	s_add_u32 s8, s10, s4
	s_addc_u32 s4, s11, s5
	s_and_b32 s9, s4, 0xffff
	s_mov_b32 s14, -1
	s_mov_b32 s12, s8
	s_mov_b32 s13, s9
	s_mov_b32 s15, s3
	s_mul_i32 s4, s19, 0x1080
	s_mul_i32 s5, s20, 0x420
	s_add_i32 s4, s5, s4
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v100, s[0:3], 0 offen lds
	s_add_i32 s5, s4, 0x2100
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], 0 offen lds
	s_add_i32 s22, s4, 0x4200
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], 0 offen lds
	s_mul_i32 s10, s19, 0x4200
	s_mul_i32 s11, s20, 0x1080
	s_add_i32 s23, s11, s10
	s_add_i32 s23, s23, 0xc600
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v103, s[12:15], 0 offen lds
	s_add_i32 s24, s23, 0x420
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v104, s[12:15], 0 offen lds
	s_add_i32 s25, s23, 0x840
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v105, s[12:15], 0 offen lds
	s_add_i32 s26, s23, 0xc60
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v106, s[12:15], 0 offen lds
	s_add_i32 s27, s4, 0x6300
	s_movk_i32 s10, 0x80
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v100, s[0:3], s10 offen lds
	s_add_i32 s28, s4, 0x8400
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s10 offen lds
	s_add_i32 s29, s4, 0xa500
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s10 offen lds
	s_add_i32 s30, s23, 0x8400
	s_movk_i32 s10, 0x800
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v103, s[12:15], s10 offen lds
	s_add_i32 s31, s23, 0x8820
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v104, s[12:15], s10 offen lds
	s_add_i32 s33, s23, 0x8c40
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v105, s[12:15], s10 offen lds
	s_add_i32 s34, s23, 0x9060
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v106, s[12:15], s10 offen lds
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v1, 15, v0
	v_lshl_or_b32 v2, s20, 4, v1
	v_or_b32_e32 v110, 0x1ce00, v2
	v_or_b32_e32 v3, 0x1ce40, v2
	v_or_b32_e32 v4, 0x1ce80, v2
	ds_read_u8 v5, v110
	ds_read_u8 v3, v3
	ds_read_u8 v4, v4
	s_mov_b32 s10, s14
	s_mov_b32 s11, s3
	s_movk_i32 s12, 0x420
	s_waitcnt lgkmcnt(1)
	v_lshl_or_b32 v3, v3, 8, v5
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v107, v4, 16, v3
	v_mov_b32_e32 v3, 0x1d700
	v_mov_b32_e32 v4, 0x1d70c
	ds_read_u8 v109, v3
	ds_read_u8 v108, v4
	v_and_b32_e32 v3, 3, v0
	s_lshr_b32 s13, s35, 5
	v_and_or_b32 v3, s13, 4, v3
	v_lshlrev_b32_e32 v2, 5, v2
	v_and_b32_e32 v2, 0x380, v2
	v_and_b32_e32 v4, 48, v0
	v_mad_u32_u24 v3, v3, s12, v4
	v_add_u32_e32 v98, v3, v2
	ds_read_b128 v[194:197], v98
	ds_read_b128 v[198:201], v98 offset:64
	ds_read_b128 v[178:181], v98 offset:8448
	ds_read_b128 v[182:185], v98 offset:8512
	ds_read_b128 v[114:117], v98 offset:16896
	ds_read_b128 v[118:121], v98 offset:16960
	v_add_u32_e32 v111, 0xc600, v99
	ds_read_b128 v[186:189], v99 offset:50688
	ds_read_b128 v[190:193], v99 offset:51744
	ds_read_b128 v[170:173], v99 offset:54912
	ds_read_b128 v[174:177], v99 offset:55968
	ds_read_b128 v[162:165], v99 offset:59136
	ds_read_b128 v[166:169], v99 offset:60192
	ds_read_b128 v[154:157], v99 offset:63360
	ds_read_b128 v[158:161], v99 offset:64416
	ds_read_b128 v[146:149], v111 offset:16896
	ds_read_b128 v[150:153], v111 offset:17952
	ds_read_b128 v[138:141], v111 offset:21120
	ds_read_b128 v[142:145], v111 offset:22176
	ds_read_b128 v[130:133], v111 offset:25344
	ds_read_b128 v[134:137], v111 offset:26400
	ds_read_b128 v[122:125], v111 offset:29568
	ds_read_b128 v[126:129], v111 offset:30624
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v30, 0
	s_mov_b32 s12, -2
	s_movk_i32 s13, 0x1000
	s_movk_i32 s14, 0x100
	v_mov_b32_e32 v112, v110
	v_mov_b32_e32 v31, v30
	v_mov_b32_e32 v32, v30
	v_mov_b32_e32 v33, v30
	v_mov_b32_e32 v34, v30
	v_mov_b32_e32 v35, v30
	v_mov_b32_e32 v36, v30
	v_mov_b32_e32 v37, v30
	v_mov_b32_e32 v38, v30
	v_mov_b32_e32 v39, v30
	v_mov_b32_e32 v40, v30
	v_mov_b32_e32 v41, v30
	v_mov_b32_e32 v42, v30
	v_mov_b32_e32 v43, v30
	v_mov_b32_e32 v44, v30
	v_mov_b32_e32 v45, v30
	v_mov_b32_e32 v46, v30
	v_mov_b32_e32 v47, v30
	v_mov_b32_e32 v48, v30
	v_mov_b32_e32 v49, v30
	v_mov_b32_e32 v50, v30
	v_mov_b32_e32 v51, v30
	v_mov_b32_e32 v52, v30
	v_mov_b32_e32 v53, v30
	v_mov_b32_e32 v58, v30
	v_mov_b32_e32 v59, v30
	v_mov_b32_e32 v60, v30
	v_mov_b32_e32 v61, v30
	v_mov_b32_e32 v66, v30
	v_mov_b32_e32 v67, v30
	v_mov_b32_e32 v68, v30
	v_mov_b32_e32 v69, v30
	v_mov_b32_e32 v54, v30
	v_mov_b32_e32 v55, v30
	v_mov_b32_e32 v56, v30
	v_mov_b32_e32 v57, v30
	v_mov_b32_e32 v62, v30
	v_mov_b32_e32 v63, v30
	v_mov_b32_e32 v64, v30
	v_mov_b32_e32 v65, v30
	v_mov_b32_e32 v70, v30
	v_mov_b32_e32 v71, v30
	v_mov_b32_e32 v72, v30
	v_mov_b32_e32 v73, v30
	v_mov_b32_e32 v74, v30
	v_mov_b32_e32 v75, v30
	v_mov_b32_e32 v76, v30
	v_mov_b32_e32 v77, v30
	v_mov_b32_e32 v78, v30
	v_mov_b32_e32 v79, v30
	v_mov_b32_e32 v80, v30
	v_mov_b32_e32 v81, v30
	v_mov_b32_e32 v82, v30
	v_mov_b32_e32 v83, v30
	v_mov_b32_e32 v84, v30
	v_mov_b32_e32 v85, v30
	v_mov_b32_e32 v86, v30
	v_mov_b32_e32 v87, v30
	v_mov_b32_e32 v88, v30
	v_mov_b32_e32 v89, v30
	v_mov_b32_e32 v90, v30
	v_mov_b32_e32 v91, v30
	v_mov_b32_e32 v92, v30
	v_mov_b32_e32 v93, v30
	v_mov_b32_e32 v94, v30
	v_mov_b32_e32 v95, v30
	v_mov_b32_e32 v96, v30
	v_mov_b32_e32 v97, v30
	v_mov_b32_e32 v2, v30
	v_mov_b32_e32 v3, v30
	v_mov_b32_e32 v4, v30
	v_mov_b32_e32 v5, v30
	v_mov_b32_e32 v6, v30
	v_mov_b32_e32 v7, v30
	v_mov_b32_e32 v8, v30
	v_mov_b32_e32 v9, v30
	v_mov_b32_e32 v10, v30
	v_mov_b32_e32 v11, v30
	v_mov_b32_e32 v12, v30
	v_mov_b32_e32 v13, v30
	v_mov_b32_e32 v18, v30
	v_mov_b32_e32 v19, v30
	v_mov_b32_e32 v20, v30
	v_mov_b32_e32 v21, v30
	v_mov_b32_e32 v14, v30
	v_mov_b32_e32 v15, v30
	v_mov_b32_e32 v16, v30
	v_mov_b32_e32 v17, v30
	v_mov_b32_e32 v22, v30
	v_mov_b32_e32 v23, v30
	v_mov_b32_e32 v24, v30
	v_mov_b32_e32 v25, v30
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	v_mov_b32_e32 v28, v30
	v_mov_b32_e32 v29, v30
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v100, s[0:3], s14 offen lds
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s14 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s14 offen lds
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v103, s[8:11], s13 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v104, s[8:11], s13 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s13 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s13 offen lds
	ds_read_u8 v113, v112 offset:192
	ds_read_u8 v202, v112 offset:256
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v113, v202, 8, v113
	ds_read_u8 v202, v112 offset:320
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v202, v202, 16, v113
	s_add_i32 s15, s12, 0x1d700
	v_mov_b32_e32 v203, s15
	ds_read_u8 v204, v203 offset:3
	ds_read_u8 v205, v203 offset:15
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[186:193], v[194:201], v[30:33], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[170:177], v[194:201], v[34:37], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[162:169], v[194:201], v[38:41], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[154:161], v[194:201], v[42:45], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[146:153], v[194:201], v[46:49], v108, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[138:145], v[194:201], v[50:53], v108, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[130:137], v[194:201], v[58:61], v108, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[122:129], v[194:201], v[66:69], v108, v107 op_sel_hi:[0,0,0]
	ds_read_b128 v[194:197], v98 offset:25344
	ds_read_b128 v[198:201], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[186:193], v[178:185], v[54:57], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[170:177], v[178:185], v[62:65], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[162:169], v[178:185], v[70:73], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[154:161], v[178:185], v[74:77], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[146:153], v[178:185], v[78:81], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[138:145], v[178:185], v[82:85], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[130:137], v[178:185], v[86:89], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[122:129], v[178:185], v[90:93], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[178:181], v98 offset:33792
	ds_read_b128 v[182:185], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[186:193], v[114:121], v[94:97], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[186:189], v111 offset:33792
	ds_read_b128 v[190:193], v111 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[170:177], v[114:121], v[2:5], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[170:173], v111 offset:38016
	ds_read_b128 v[174:177], v111 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[162:169], v[114:121], v[6:9], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[162:165], v111 offset:42240
	ds_read_b128 v[166:169], v111 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[154:161], v[114:121], v[10:13], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[154:157], v111 offset:46464
	ds_read_b128 v[158:161], v111 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[146:153], v[114:121], v[18:21], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[146:149], v111 offset:50688
	ds_read_b128 v[150:153], v111 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[138:145], v[114:121], v[14:17], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[138:141], v111 offset:54912
	ds_read_b128 v[142:145], v111 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[130:137], v[114:121], v[22:25], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[130:133], v111 offset:59136
	ds_read_b128 v[134:137], v111 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[122:129], v[114:121], v[26:29], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[114:117], v111 offset:63360
	ds_read_b128 v[118:121], v111 offset:64416
	ds_read_b128 v[122:125], v98 offset:42240
	ds_read_b128 v[126:129], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_add_i32 s15, s14, 0x80
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v100, s[0:3], s15 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v101, s[0:3], s15 offen lds
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v102, s[0:3], s15 offen lds
	s_add_i32 s15, s13, 0x800
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v103, s[8:11], s15 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v104, s[8:11], s15 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v105, s[8:11], s15 offen lds
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v106, s[8:11], s15 offen lds
	v_add_u32_e32 v113, 0x180, v112
	ds_read_u8 v107, v112 offset:384
	ds_read_u8 v108, v112 offset:448
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v107, v108, 8, v107
	ds_read_u8 v108, v112 offset:512
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v107, v108, 16, v107
	ds_read_u8 v109, v203 offset:4
	ds_read_u8 v108, v203 offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[186:193], v[194:201], v[30:33], v204, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[170:177], v[194:201], v[34:37], v204, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[162:169], v[194:201], v[38:41], v204, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[154:161], v[194:201], v[42:45], v204, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[146:153], v[194:201], v[46:49], v205, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[138:145], v[194:201], v[50:53], v205, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[130:137], v[194:201], v[58:61], v205, v202 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[114:121], v[194:201], v[66:69], v205, v202 op_sel_hi:[0,0,0]
	ds_read_b128 v[194:197], v98
	ds_read_b128 v[198:201], v98 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[186:193], v[178:185], v[54:57], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[170:177], v[178:185], v[62:65], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[162:169], v[178:185], v[70:73], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[154:161], v[178:185], v[74:77], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[146:153], v[178:185], v[78:81], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[138:145], v[178:185], v[82:85], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[130:137], v[178:185], v[86:89], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[114:121], v[178:185], v[90:93], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[178:181], v98 offset:8448
	ds_read_b128 v[182:185], v98 offset:8512
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[186:193], v[122:129], v[94:97], v204, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[186:189], v99 offset:50688
	ds_read_b128 v[190:193], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[170:177], v[122:129], v[2:5], v204, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[170:173], v99 offset:54912
	ds_read_b128 v[174:177], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[162:169], v[122:129], v[6:9], v204, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[162:165], v99 offset:59136
	ds_read_b128 v[166:169], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[154:161], v[122:129], v[10:13], v204, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[154:157], v99 offset:63360
	ds_read_b128 v[158:161], v99 offset:64416
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[146:153], v[122:129], v[18:21], v205, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[146:149], v111 offset:16896
	ds_read_b128 v[150:153], v111 offset:17952
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[138:145], v[122:129], v[14:17], v205, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[138:141], v111 offset:21120
	ds_read_b128 v[142:145], v111 offset:22176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[130:137], v[122:129], v[22:25], v205, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[130:133], v111 offset:25344
	ds_read_b128 v[134:137], v111 offset:26400
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:121], v[122:129], v[26:29], v205, v202 op_sel_hi:[0,1,0]
	ds_read_b128 v[122:125], v111 offset:29568
	ds_read_b128 v[126:129], v111 offset:30624
	ds_read_b128 v[114:117], v98 offset:16896
	ds_read_b128 v[118:121], v98 offset:16960
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_add_i32 s12, s12, 2
	s_addk_i32 s13, 0x1000
	s_addk_i32 s14, 0x100
	s_cmp_gt_u32 s12, 7
	v_mov_b32_e32 v112, v113
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	ds_read_u8 v100, v110 offset:2112
	ds_read_u8 v101, v110 offset:2176
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v100, v101, 8, v100
	ds_read_u8 v101, v110 offset:2240
	s_waitcnt lgkmcnt(0)
	v_lshl_or_b32 v110, v101, 16, v100
	v_mov_b32_e32 v100, 0x1d70b
	ds_read_u8 v106, v100
	v_mov_b32_e32 v100, 0x1d717
	ds_read_u8 v111, v100
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[186:193], v[194:201], v[30:33], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[170:177], v[194:201], v[34:37], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[162:169], v[194:201], v[38:41], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[154:161], v[194:201], v[42:45], v109, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[146:153], v[194:201], v[46:49], v108, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[138:145], v[194:201], v[50:53], v108, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[130:137], v[194:201], v[58:61], v108, v107 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[122:129], v[194:201], v[66:69], v108, v107 op_sel_hi:[0,0,0]
	ds_read_b128 v[194:197], v98 offset:25344
	ds_read_b128 v[198:201], v98 offset:25408
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[186:193], v[178:185], v[54:57], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[170:177], v[178:185], v[62:65], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[162:169], v[178:185], v[70:73], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[154:161], v[178:185], v[74:77], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[146:153], v[178:185], v[78:81], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[138:145], v[178:185], v[82:85], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[130:137], v[178:185], v[86:89], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[122:129], v[178:185], v[90:93], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	ds_read_b128 v[178:181], v98 offset:33792
	ds_read_b128 v[182:185], v98 offset:33856
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[186:193], v[114:121], v[94:97], v109, v107 op_sel_hi:[0,1,0]
	v_add_u32_e32 v99, 0xc600, v99
	ds_read_b128 v[186:189], v99 offset:33792
	ds_read_b128 v[190:193], v99 offset:34848
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[170:177], v[114:121], v[2:5], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[170:173], v99 offset:38016
	ds_read_b128 v[174:177], v99 offset:39072
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[162:169], v[114:121], v[6:9], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[162:165], v99 offset:42240
	ds_read_b128 v[166:169], v99 offset:43296
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[154:161], v[114:121], v[10:13], v109, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[154:157], v99 offset:46464
	ds_read_b128 v[158:161], v99 offset:47520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[146:153], v[114:121], v[18:21], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[146:149], v99 offset:50688
	ds_read_b128 v[150:153], v99 offset:51744
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[138:145], v[114:121], v[14:17], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[138:141], v99 offset:54912
	ds_read_b128 v[142:145], v99 offset:55968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[130:137], v[114:121], v[22:25], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[130:133], v99 offset:59136
	ds_read_b128 v[134:137], v99 offset:60192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[122:129], v[114:121], v[26:29], v108, v107 op_sel_hi:[0,1,0]
	ds_read_b128 v[112:115], v99 offset:63360
	ds_read_b128 v[116:119], v99 offset:64416
	ds_read_b128 v[120:123], v98 offset:42240
	ds_read_b128 v[124:127], v98 offset:42304
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[186:193], v[194:201], v[30:33], v106, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[170:177], v[194:201], v[34:37], v106, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[162:169], v[194:201], v[38:41], v106, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[154:161], v[194:201], v[42:45], v106, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[146:153], v[194:201], v[46:49], v111, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[138:145], v[194:201], v[50:53], v111, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[130:137], v[194:201], v[58:61], v111, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[112:119], v[194:201], v[66:69], v111, v110 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[186:193], v[178:185], v[54:57], v106, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[170:177], v[178:185], v[62:65], v106, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[162:169], v[178:185], v[70:73], v106, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[154:161], v[178:185], v[74:77], v106, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[146:153], v[178:185], v[78:81], v111, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[138:145], v[178:185], v[82:85], v111, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[130:137], v[178:185], v[86:89], v111, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[112:119], v[178:185], v[90:93], v111, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[186:193], v[120:127], v[94:97], v106, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[170:177], v[120:127], v[2:5], v106, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[162:169], v[120:127], v[6:9], v106, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[154:161], v[120:127], v[10:13], v106, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[146:153], v[120:127], v[18:21], v111, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[138:145], v[120:127], v[14:17], v111, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[130:137], v[120:127], v[22:25], v111, v110 op_sel_hi:[0,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[112:119], v[120:127], v[26:29], v111, v110 op_sel_hi:[0,1,0]
	s_lshl_b32 s4, s6, 6
	s_mul_i32 s0, s6, s20
	v_mul_lo_u32 v1, s6, v1
	s_lshl_b32 s1, s19, 4
	v_lshrrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v0, 12, v0
	s_lshl4_add_u32 s0, s0, s1
	v_add3_u32 v14, v0, v1, s0
	v_add_u32_e32 v15, s4, v14
	s_mul_i32 s0, s6, s7
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s16, s0
	s_addc_u32 s3, s17, s1
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[0:1], s[18:19], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_mul_i32 s2, s6, s21
	s_sub_i32 s2, s2, s18
	s_lshl_b32 s2, s2, 1
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	v_cvt_pk_bf16_f32 v1, v32, v33
	v_cvt_pk_bf16_f32 v0, v30, v31
	v_lshlrev_b32_e32 v14, 1, v14
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v36, v37
	v_cvt_pk_bf16_f32 v0, v34, v35
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v40, v41
	v_cvt_pk_bf16_f32 v0, v38, v39
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v44, v45
	v_cvt_pk_bf16_f32 v0, v42, v43
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v48, v49
	v_cvt_pk_bf16_f32 v0, v46, v47
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v52, v53
	v_cvt_pk_bf16_f32 v0, v50, v51
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v60, v61
	v_cvt_pk_bf16_f32 v0, v58, v59
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v68, v69
	v_cvt_pk_bf16_f32 v0, v66, v67
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v56, v57
	v_cvt_pk_bf16_f32 v0, v54, v55
	v_lshlrev_b32_e32 v14, 1, v15
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v64, v65
	v_cvt_pk_bf16_f32 v0, v62, v63
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v72, v73
	v_cvt_pk_bf16_f32 v0, v70, v71
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v76, v77
	v_cvt_pk_bf16_f32 v0, v74, v75
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v80, v81
	v_cvt_pk_bf16_f32 v0, v78, v79
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v84, v85
	v_cvt_pk_bf16_f32 v0, v82, v83
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v88, v89
	v_cvt_pk_bf16_f32 v0, v86, v87
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v92, v93
	v_cvt_pk_bf16_f32 v0, v90, v91
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:448
	v_cvt_pk_bf16_f32 v1, v96, v97
	v_cvt_pk_bf16_f32 v0, v94, v95
	v_add_lshl_u32 v14, v15, s4, 1
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v1, v100, v101
	v_cvt_pk_bf16_f32 v0, v98, v99
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:64
	v_cvt_pk_bf16_f32 v1, v104, v105
	v_cvt_pk_bf16_f32 v0, v102, v103
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:128
	v_cvt_pk_bf16_f32 v1, v108, v109
	v_cvt_pk_bf16_f32 v0, v106, v107
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:192
	v_cvt_pk_bf16_f32 v1, v20, v21
	v_cvt_pk_bf16_f32 v0, v18, v19
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v12, v13
	v_cvt_pk_bf16_f32 v0, v10, v11
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:320
	v_cvt_pk_bf16_f32 v1, v8, v9
	v_cvt_pk_bf16_f32 v0, v6, v7
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:384
	v_cvt_pk_bf16_f32 v1, v4, v5
	v_cvt_pk_bf16_f32 v0, v2, v3
	buffer_store_dwordx2 v[0:1], v14, s[0:3], 0 offen offset:448
	s_endpgm
.Lfunc_end0:
	.size	_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950, .Lfunc_end0-_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
		.amdhsa_group_segment_fixed_size 120600
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
		.amdhsa_next_free_vgpr 206
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 208
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
	.section	.text,"axG",@progbits,_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950,comdat,unique,1
                                        ; -- End function
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_vgpr, 206
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_agpr, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.numbered_sgpr, 36
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.num_named_barrier, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.private_seg_size, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_vcc, 1
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.uses_flat_scratch, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_dyn_sized_stack, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_recursion, 0
	.set .L_Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5252
; TotalNumSgprs: 42
; NumVgprs: 206
; NumAgprs: 0
; TotalNumVgprs: 206
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 120600 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 206
; AccumOffset: 208
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 51
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_23277911cc3b7204,@object ; @__hip_cuid_23277911cc3b7204
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_23277911cc3b7204
__hip_cuid_23277911cc3b7204:
	.byte	0                               ; 0x0
	.size	__hip_cuid_23277911cc3b7204, 1

	.ident	"clang version 24.0.0git (https://github.com/yuyzhang512/llvm-project.git 49c41889681640665400cb01c9fbb4c0a024cde4)"
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_23277911cc3b7204
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
    .name:           _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         _Z40gemm_a8w8_mxfp8_bpreshuffle_k1536_kernelI49opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950ILi2EEEv42opus_gemm_mxscale_bpreshuffle_kargs_gfx950.kd
    .uses_dynamic_stack: false
    .vgpr_count:     206
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu9.50-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
