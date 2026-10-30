#!/usr/bin/env bash
set -euo pipefail
trap 'task_exit_code=$?; printf "%s\n" "$task_exit_code" > /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/controller_exit_code ' EXIT
bash /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030/co_experiment/build_new_co.sh
bash /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/run_kernel_campaign.sh /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage18_remote_final84_20261030 /root/workspace/aiter-fyz-10-29-aiter/docs/gfx950_mxscale_20261029/local_20261029/stage17_pin_invariants54_20261030/result/public_best_opus421.csv 48261
