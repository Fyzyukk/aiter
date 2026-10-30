source /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/env.sh
export OPUS_RUN_REPO=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/co_experiment/.cache/experiment_repo
export PYTHONPATH="$OPUS_RUN_REPO"
export AITER_JIT_DIR=/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/co_experiment/.cache/aiter_jit
export OPUS_COMPILE_FAMILY_ONLY=a8w8_blockscale_bpreshuffle
export ROCR_VISIBLE_DEVICES=1,0,6,7 HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REBUILD=0 OPUS_TARGET_NEW_ONLY=0
cd "$OPUS_RUN_REPO"
