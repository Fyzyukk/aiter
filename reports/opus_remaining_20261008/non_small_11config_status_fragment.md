# Non-small eleven configuration status fragment

7×9020 +1×9022 +1×9023runtime +1×9024fixed +1×9030 =11 actualwinner configurations. Legal9023fixed/9024runtime controls are not historicalwinner inventory entries.

| Parent | Existing traits | Historical winners | Decision | Measured scope |
|---|---|---:|---|---|
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128, 128, 64, 0> | 3 | keep_current_reject_candidate_outside_existing_fixed384 | one representative screen |
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 128, 0> | 31 | keep_current_reject_candidate_outside_existing_fixed384 | one representative screen |
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 32, 1536> | 9 | keep_current_reject_candidate_outside_existing_fixed384 | one representative screen |
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 32, 3072> | 5 | keep_current_reject_candidate_outside_existing_fixed384 | one representative screen |
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 64, 7168> | 16 | keep_current_reject_candidate_outside_existing_fixed384 | one representative screen |
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 8, 384> | 7 | adopt_existing_fixed384_pending_root_official_gates | all7 winner coverage |
| 9020 | opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192, 256, 8, 768> | 5 | keep_current_reject_candidate_outside_existing_fixed384 | one representative screen |
| 9022 | opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 | 159 | keep_current_reject_global_rawscale_candidate | all159 winners |
| 9023 | opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<3, 32, 0> | 4 | adopt_existing_runtime_pending_root_official_gates | all4 runtime winners+boundary |
| 9024 | opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64, 7168, 4> | 9 | adopt_existing_fixed7168_pending_root_official_gates | all9 fixedwinners+runtimecontrol |
| 9030 | opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 | 10 | keep_current_reject_global_midpoint_candidate | all10 winners+5guards+2nonwinnercontrols |

Retained entries require root official linked identity/API gates. Other9020 bodies,9022 and9030 remain current. No new dispatch threshold is introduced.

The9020fixed38412288round2negative Event remains part of every aggregate and final risk.9022sixstablewinnerlosses and9030all10negative ratio medians justify their rejections.

Fullper-entry measuredranges, resources, costs and limits: [non_small_11config_status_fragment.json](non_small_11config_status_fragment.json). The rootfamily_progress file is untouched.
