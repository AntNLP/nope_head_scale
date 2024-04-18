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
        output_dir="$output_base/${lr}_st${ft_step}/${DATE}"
        mkdir -p "$output_dir" # for tee
    }

    # shellcheck disable=SC2086
    run() {
        set_logging_dir # according to other hyper-params
        # ulimit -n 4096  # Solution to `OSError: [Errno 24] Too many open files`
        run_ds run_clm.py \
            --config_path "configs/basic.yaml" "configs/tnl.yaml" "configs/$config_name.yaml" \
            --output_dir $output_dir --logging_dir "$output_dir" \
            --learning_rate $lr $rope $seqlen --max_steps $ft_step |&
            tee "$output_dir/ft.log"

        echo "$output_dir"
        echo "done"
    }

    # wait_for_pid 37472
    config_name="tl_yarn"
    ft_step="200"
    lr='1e-4'

    # seqname="yarn16_FT32k"
    # seqlen="--block_size 32768 --yarn 16"
    # seqname="yarn8_FT16k"
    # seqlen="--block_size 16384 --yarn 8"
    # seqname="yarn4_FT8k"
    # seqlen="--block_size 8192 --yarn 4"
    seqname="yarn2_FT4k"
    seqlen="--block_size 4096 --yarn 2"

    output_base="ckpts/tl50k_${seqname}"
    rope="--model_name_or_path TinyLlama/tinyLlama-intermediate-checkpoints --model_revision step-50k-token-105B"

    run

    exit # https://stackoverflow.com/a/2358432
}
