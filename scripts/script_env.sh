#!/bin/bash

set -euxo pipefail # https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
export HF_HOME="$HOME/nope/hf_home"
export TOKENIZERS_PARALLELISM=true # https://stackoverflow.com/a/72926996/17347885  # false is really slow
export WANDB_DISABLED="true"
export PYTHONWARNINGS="ignore"

# export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
# export CUDA_VISIBLE_DEVICES='0,1,2,3'

port=$(shuf -i25000-30000 -n1) # random port in [25000, 30000]
# sample only once to prevent collision later when run_ds multiple times in one script
run_ds() {
    # env -u: https://stackoverflow.com/questions/18906350/unset-an-environment-variable-for-a-single-command
    env -u CUDA_VISIBLE_DEVICES deepspeed \
        --include "localhost:${CUDA_VISIBLE_DEVICES}" \
        --master_port "$port" \
        "$@"
}
run_ac() {
    accelerate launch --main_process_port "$port" "$@"
}

run_py() {
    python -u "$@"
}

VIZ_ARG="-o logs/result.json --min_duration 0.2ms --max_stack_depth 30"

# shellcheck disable=SC2086
run_py_viz() {
    viztracer $VIZ_ARG -- "$@"
}

# shellcheck disable=SC2086
run_ds_viz() {
    env -u CUDA_VISIBLE_DEVICES viztracer $VIZ_ARG -- "$(which deepspeed)" \
        --include "localhost:${CUDA_VISIBLE_DEVICES}" \
        --master_port "$port" \
        "$@"
}

DEBUG_ARG="--listen localhost:5678 --wait-for-client"

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT="5" # 5s

# shellcheck disable=SC2086
run_py_dbg() {
    python -u -m debugpy $DEBUG_ARG "$@"
}

# shellcheck disable=SC2086
run_ds_dbg() {
    env -u CUDA_VISIBLE_DEVICES python -m debugpy $DEBUG_ARG "$(which deepspeed)" \
        --include "localhost:${CUDA_VISIBLE_DEVICES}" \
        --master_port "$port" \
        "$@"
}

# Note: wait for launch script (which is directly typed and run on terminal, commonly bash), not some python proc
# because launch script may run multiple python procs
wait_for_pid() {
    tail --pid="$1" -f /dev/null && sleep 30
}
