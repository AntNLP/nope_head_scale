#!/bin/bash
# shellcheck disable=SC2086
{
    # cd to script folder: https://stackoverflow.com/questions/3349105#comment69622329_3355423
    cd "$(dirname "${BASH_SOURCE[0]}")"
    source "script_env.sh"
    cd .. # project root
    export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
    # shellcheck disable=SC2086
    run() {
        set_logging_dir # according to other hyper-params
        ulimit -n 4096  # Solution to `OSError: [Errno 24] Too many open files`
        run_py run_pass.py \
            --base_model $model $args \
            --logging_dir "$logging_dir" |&
            tee "$logging_dir/run.log"
        echo "$task_name"
        echo "done"
    }
    set_logging_dir() {
        DATE=$(LC_TIME=en_US.UTF-8 date +"%b%d_%H-%M-%S") # mimic trainer default
        task_name="${name} (test ${l_name})"
        logging_dir="ckpts/pass/${task_name}|${DATE}"
        mkdir -p "$logging_dir" # for tee
    }

    # wait_for_pid 34244

    # args="--max_tokens 512 --interval 512 --num_tests 10"
    # args="--max_tokens 4096" && l_name="4k"
    # args="--max_tokens 8192" && l_name="8k"
    # args="--max_tokens 16384" && l_name="16k"
    args="--max_tokens 32768" && l_name="32k"
    # args="--max_tokens 65536" && l_name="64k"
    # RoPE
    name="RoPE50k"
    model="TinyLlama/tinyLlama-intermediate-checkpoints --revision step-50k-token-105B"
    # name="RoPE_3T"
    # model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    # NoPE
    # name="NoPE50k"
    # model="ckpts/tnl_50k --nope true"
    # const scale
    # name="ConstScale1.6"
    # model="ckpts/tnl --nope true --scale 1.6"
    # name="HS8k"
    # model="ckpts/tnl_hs_50k_8k --nope --scale_type HS"
    run

    exit
}
