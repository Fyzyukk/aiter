#!/usr/bin/env bash
set -euo pipefail
source "/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/co_experiment/experiment_env.sh"
"$OPUS_RUN_PYTHON" "$OPUS_RUN_REPO/docs/gfx950_mxscale_20261029/local_20261029/run_stage6_batch.py" --plan "/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/verification_plan.json" --output "/root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/verification_six_shapes" --logical-gpus 0,1,2
