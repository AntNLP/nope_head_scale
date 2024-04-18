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
        run_py run_ppl.py \
            --dataset_name "$HOME/nope/data/$data_name" \
            --base_model $model --max_length $length \
            --logging_dir "$logging_dir" |&
            tee "$logging_dir/run.log"
        echo "$task_name"
        echo "done"
    }
    set_logging_dir() {
        DATE=$(LC_TIME=en_US.UTF-8 date +"%b%d_%H-%M-%S") # mimic trainer default
        task_name="${name}_${length}"
        logging_dir="ckpts/ppl/${data_name}/${task_name}|${DATE}"
        mkdir -p "$logging_dir" # for tee
    }

    # wait_for_pid 41467

    # lengths="2048"
    lengths="2048 4096 8192 16384"
    # lengths="2048 4096"
    # lengths="8192 16384"
    data_names="PG19 proof_pile"

    for data_name in $data_names; do
        for length in $lengths; do
            # NoPE
            name="NoPE50k"
            model="ckpts/tnl_50k --nope true"
            run
            # name="NoPE50k_s1.2"
            # model="ckpts/tnl2_50k --nope true --scale 1.2"
            # run
            # name="NoPE50k_s1.5"
            # model="ckpts/tnl2_50k --nope true --scale 1.5"
            # run
            # name="NoPE50k_s1.8"
            # model="ckpts/tnl2_50k --nope true --scale 1.8"
            # run
            # name="HS50k_4k"
            # model="ckpts/tnl_hs_50k_4k --nope true --scale_type HS"
            # run

            # RoPE
            # name="RoPE50k"
            # model="TinyLlama/tinyLlama-intermediate-checkpoints --revision step-50k-token-105B"
            # run
            # name="RoPE1T"
            # model="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"
            # run
            # name="RoPE3T"
            # model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
            # run
            # name="RoPE50k_YaRN2_FT4k"
            # model="ckpts/tl50k_yarn2_FT4k --yarn 2"
            # run
        done
    done

    exit
}
