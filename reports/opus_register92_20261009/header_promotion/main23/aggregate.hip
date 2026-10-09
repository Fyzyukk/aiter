#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_large_variants_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_main_variants_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_variants_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_2wave_16x64_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_2wave_32x64_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_2wave_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x128_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_fine_lds_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_geometry_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_large_output_direct_b_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_large_output_panel16_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_split_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_direct_b_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh"
using Traits_92000 = opus_gemm_mxscale_bpreshuffle_geometry_traits<64,2,32>;
static_assert(sizeof(Traits_92000) > 0);
using Traits_92001 = opus_gemm_mxscale_bpreshuffle_geometry_traits<96,2,32>;
static_assert(sizeof(Traits_92001) > 0);
using Traits_92002 = opus_gemm_mxscale_bpreshuffle_geometry_traits<128,2,32>;
static_assert(sizeof(Traits_92002) > 0);
using Traits_92003 = opus_gemm_mxscale_bpreshuffle_geometry_traits<96,2,8,384>;
static_assert(sizeof(Traits_92003) > 0);
using Traits_92004 = opus_gemm_mxscale_bpreshuffle_geometry_traits<96,2,8,768>;
static_assert(sizeof(Traits_92004) > 0);
using Traits_92005 = opus_gemm_mxscale_bpreshuffle_geometry_traits<128,2,16,1536>;
static_assert(sizeof(Traits_92005) > 0);
using Traits_92006 = opus_gemm_mxscale_bpreshuffle_geometry_traits<160,2,16,1536>;
static_assert(sizeof(Traits_92006) > 0);
using Traits_92007 = opus_gemm_mxscale_bpreshuffle_geometry_traits<96,2,32,3072>;
static_assert(sizeof(Traits_92007) > 0);
using Traits_92008 = opus_gemm_mxscale_bpreshuffle_geometry_traits<96,2,32,7168>;
static_assert(sizeof(Traits_92008) > 0);
using Traits_92009 = opus_gemm_mxscale_bpreshuffle_geometry_traits<128,2,32,7168>;
static_assert(sizeof(Traits_92009) > 0);
using Traits_92010 = opus_gemm_mxscale_bpreshuffle_geometry_traits<128,2,32,3072>;
static_assert(sizeof(Traits_92010) > 0);
using Traits_92011 = opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,8>;
static_assert(sizeof(Traits_92011) > 0);
using Traits_92012 = opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,16>;
static_assert(sizeof(Traits_92012) > 0);
using Traits_92020 = opus_gemm_mxscale_bpreshuffle_shortk_traits<384>;
static_assert(sizeof(Traits_92020) > 0);
using Traits_92021 = opus_gemm_mxscale_bpreshuffle_shortk_traits<768>;
static_assert(sizeof(Traits_92021) > 0);
using Traits_92100 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<384,16,false>;
static_assert(sizeof(Traits_92100) > 0);
using Traits_92101 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<1536,16,false>;
static_assert(sizeof(Traits_92101) > 0);
using Traits_92102 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<3072,32,false>;
static_assert(sizeof(Traits_92102) > 0);
using Traits_92103 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<7168,64,false>;
static_assert(sizeof(Traits_92103) > 0);
using Traits_92104 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<16384,64,false>;
static_assert(sizeof(Traits_92104) > 0);
using Traits_92110 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<384,16,true>;
static_assert(sizeof(Traits_92110) > 0);
using Traits_92111 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<1536,16,true>;
static_assert(sizeof(Traits_92111) > 0);
using Traits_92112 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<3072,32,true>;
static_assert(sizeof(Traits_92112) > 0);
using Traits_92113 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<7168,64,true>;
static_assert(sizeof(Traits_92113) > 0);
using Traits_92114 = opus_gemm_mxscale_bpreshuffle_pin_fixed_traits<16384,64,true>;
static_assert(sizeof(Traits_92114) > 0);
using Traits_92120 = opus_gemm_mxscale_bpreshuffle_pad_pin_fixed_traits<7168>;
static_assert(sizeof(Traits_92120) > 0);
using Traits_92200 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9043,1>;
static_assert(sizeof(Traits_92200) > 0);
using Traits_92201 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9043,2>;
static_assert(sizeof(Traits_92201) > 0);
using Traits_92202 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9043,3>;
static_assert(sizeof(Traits_92202) > 0);
using Traits_92203 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9044,1>;
static_assert(sizeof(Traits_92203) > 0);
using Traits_92204 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9044,2>;
static_assert(sizeof(Traits_92204) > 0);
using Traits_92205 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9044,3>;
static_assert(sizeof(Traits_92205) > 0);
using Traits_92206 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9045,1>;
static_assert(sizeof(Traits_92206) > 0);
using Traits_92207 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9045,2>;
static_assert(sizeof(Traits_92207) > 0);
using Traits_92208 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9045,3>;
static_assert(sizeof(Traits_92208) > 0);
using Traits_92209 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9046,1>;
static_assert(sizeof(Traits_92209) > 0);
using Traits_92210 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9046,2>;
static_assert(sizeof(Traits_92210) > 0);
using Traits_92211 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9046,3>;
static_assert(sizeof(Traits_92211) > 0);
using Traits_92212 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9047,1>;
static_assert(sizeof(Traits_92212) > 0);
using Traits_92213 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9047,2>;
static_assert(sizeof(Traits_92213) > 0);
using Traits_92214 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9047,3>;
static_assert(sizeof(Traits_92214) > 0);
using Traits_92215 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9049,1>;
static_assert(sizeof(Traits_92215) > 0);
using Traits_92216 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9049,2>;
static_assert(sizeof(Traits_92216) > 0);
using Traits_92217 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9049,3>;
static_assert(sizeof(Traits_92217) > 0);
using Traits_92218 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9055,1>;
static_assert(sizeof(Traits_92218) > 0);
using Traits_92219 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9055,2>;
static_assert(sizeof(Traits_92219) > 0);
using Traits_92220 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9055,3>;
static_assert(sizeof(Traits_92220) > 0);
using Traits_92221 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9056,1>;
static_assert(sizeof(Traits_92221) > 0);
using Traits_92222 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9056,2>;
static_assert(sizeof(Traits_92222) > 0);
using Traits_92223 = opus_gemm_mxscale_bpreshuffle_small_direct_b_traits<9056,3>;
static_assert(sizeof(Traits_92223) > 0);
using Traits_92310 = opus_gemm_mxscale_bpreshuffle_register_split_traits<16,16,1>;
static_assert(sizeof(Traits_92310) > 0);
using Traits_92311 = opus_gemm_mxscale_bpreshuffle_register_split_traits<16,16,2>;
static_assert(sizeof(Traits_92311) > 0);
using Traits_92320 = opus_gemm_mxscale_bpreshuffle_register_split_traits<16,32,1>;
static_assert(sizeof(Traits_92320) > 0);
using Traits_92321 = opus_gemm_mxscale_bpreshuffle_register_split_traits<16,32,2>;
static_assert(sizeof(Traits_92321) > 0);
using Traits_92330 = opus_gemm_mxscale_bpreshuffle_register_split_traits<32,32,1>;
static_assert(sizeof(Traits_92330) > 0);
using Traits_92340 = opus_gemm_mxscale_bpreshuffle_register_split_traits<32,64,1>;
static_assert(sizeof(Traits_92340) > 0);
using Traits_92410 = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<48,64,1,4,1>;
static_assert(sizeof(Traits_92410) > 0);
using Traits_92411 = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<48,64,1,4,2>;
static_assert(sizeof(Traits_92411) > 0);
using Traits_92420 = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<64,128,2,2,1>;
static_assert(sizeof(Traits_92420) > 0);
using Traits_92421 = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<64,128,2,2,2>;
static_assert(sizeof(Traits_92421) > 0);
using Traits_92430 = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<96,128,2,2,1>;
static_assert(sizeof(Traits_92430) > 0);
using Traits_92431 = opus_gemm_mxscale_bpreshuffle_narrow_fine_traits<96,128,2,2,2>;
static_assert(sizeof(Traits_92431) > 0);
using Traits_92500 = opus_gemm_mxscale_bpreshuffle_large_output_panel16_traits;
static_assert(sizeof(Traits_92500) > 0);
using Traits_92501 = opus_gemm_mxscale_bpreshuffle_large_output_direct_b_traits;
static_assert(sizeof(Traits_92501) > 0);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92000>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92001>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92002>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92003>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92004>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92005>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92006>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92007>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92008>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92009>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_geometry_kernel<Traits_92010>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<Traits_92011>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_4wave_64x64_kernel<Traits_92012>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_shortk_kernel<Traits_92020>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_shortk_kernel<Traits_92021>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92200>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92201>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92202>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92203>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92204>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92205>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92206>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92207>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92208>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92209>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92210>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92211>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92212>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92213>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92214>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92215>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92216>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92217>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92218>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92219>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92220>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92221>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92222>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_small_direct_b_kernel<Traits_92223>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_register_split_kernel<Traits_92310>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_register_split_kernel<Traits_92311>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_register_split_kernel<Traits_92320>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_register_split_kernel<Traits_92321>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_register_split_kernel<Traits_92330>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_register_split_kernel<Traits_92340>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<Traits_92410>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<Traits_92411>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<Traits_92420>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<Traits_92421>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<Traits_92430>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel<Traits_92431>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_large_output_panel16_kernel<Traits_92500>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void opus_gemm_mxscale_bpreshuffle_large_output_direct_b_kernel<Traits_92501>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
