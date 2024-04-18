#!/bin/bash
{
    # cd to script folder: https://stackoverflow.com/questions/3349105#comment69622329_3355423
    cd "$(dirname "${BASH_SOURCE[0]}")"
    source "script_env.sh"
    cd .. # project root
    export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

    set_logging_dir() {
        DATE=$(LC_TIME=en_US.UTF-8 date +"%b%d_%H-%M-%S") # mimic trainer default
        logging_dir="$output_dir/runs/${lr}_${DATE}"
        mkdir -p "$logging_dir" # for tee
    }

    # shellcheck disable=SC2086
    run() {
        set_logging_dir # according to other hyper-params
        run_ds run_clm.py \
            --config_path "configs/basic.yaml" "configs/tnl.yaml" "configs/$config_name.yaml" \
            --output_dir $output_dir --logging_dir "$logging_dir" \
            --learning_rate $lr $rope $seqlen |&
            tee "$logging_dir/run.log"
    }

    # wait_for_pid 37472

    config_name="tl_pi"
    lr='1e-4'
    rope="--model_name_or_path TinyLlama/tinyLlama-intermediate-checkpoints --model_revision step-50k-token-105B"
    rope+=" --max_steps 200"

    # seqname="16k" && seqlen="--block_size 16384 --scale_factor 8"
    # output_dir="ckpts/tl50k_pi_${seqname}"
    # run
    # seqname="8k" && seqlen="--block_size 8192 --scale_factor 4"
    # output_dir="ckpts/tl50k_pi_${seqname}"
    # run
    seqname="4k" && seqlen="--block_size 4096 --scale_factor 2"
    output_dir="ckpts/tl50k_pi_${seqname}"
    run

    exit # https://stackoverflow.com/a/2358432
}
