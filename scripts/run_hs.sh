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
        output_dir="${output_base}_i${scale}_${lr}/${DATE}"
        mkdir -p "$output_dir" # for tee
    }

    # shellcheck disable=SC2086
    run() {
        set_logging_dir # according to other hyper-params
        ulimit -n 4096  # Solution to `OSError: [Errno 24] Too many open files`

        run_ds run_clm.py \
            --config_path "configs/basic.yaml" "configs/tnl.yaml" "configs/tnl_hs.yaml" \
            --output_dir "$output_dir" --logging_dir "$output_dir" --block_size $length \
            --learning_rate $lr --softmax_scale $scale --max_steps $ft_step \
            $ft_args |&
            tee "$output_dir/ft.log"

        echo "$output_dir"
        echo "done"
    }

    # len_name="4k" && length="4096" && pk_len="4096"
    len_name="8k" && length="8192"
    # len_name="16k" && length="16384"
    # len_name="18k" && length="18432" && pk_len="16384"
    # len_name="32k" && length="32768"

    output_base="ckpts/tnl_hs_50k_${len_name}/relu1"
    ft_args="--model_name_or_path ckpts/tnl_50k --scale_lb 1"

    # wait_for_pid 53040

    scale='1.6'
    ft_step=200
    lr='1e-1'
    run

    lr='5e-2'
    run

    exit # https://stackoverflow.com/a/2358432
}
