import gc
import json
import logging
import math
import os
import re
import warnings
from dataclasses import dataclass
from functools import partial
from queue import Empty
from typing import Optional

import simple_parsing
import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.llama_nope import (
    ModelArguments,
    SoftMaxScaleType,
    monkey_patch_after,
    monkey_patch_before,
)


@dataclass
class Args(simple_parsing.Serializable):
    dataset_name: str
    logging_dir: str
    base_model: str = "/data1/pretrained-models/llama-7b-hf"
    revision: str = "main"  # The specific model version to use (can be a branch name, tag name or commit id).
    overlap: int = 2  # How much parallel runs on one GPU
    max_length: int = 32768  # maximum token length for evaluation
    window_size: int = 256
    add_bos: bool = False  # should be false for llama model
    max_eval_tokens: int = 16384 * 128
    # NoPE
    nope: bool = False
    scale_type: SoftMaxScaleType = SoftMaxScaleType.CONST
    scale: float = 1.0
    window_attn: Optional[int] = None
    # RoPE
    PI: Optional[int] = None
    NTK: Optional[int] = None  # whether to use dynamic NTK
    yarn: Optional[int] = None

    def __post_init__(self):
        if self.nope + (self.PI is not None) + (self.NTK is not None) > 1:
            raise ValueError("Only one of nope, PI, NTK can be set to True")

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PPLResult:
    loss: float
    samples: int


INPUT_IDS = "input_ids"


@torch.no_grad()
def compute_perplexity(
    device: str,
    dataset: Dataset,
    args: Args,
    queue: mp.Queue,
    max_eval_token: int,
    aggressive_memory=True,
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"""
    if "0" in device:
        set_global_logging_level(logging.INFO)
    else:
        set_global_logging_level(logging.ERROR)
    model, tokenizer = load_model(args, device)

    if args.add_bos:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = args.max_length - 1
    else:
        max_tokenized_len = args.max_length
    assert max_eval_token % max_tokenized_len == 0
    eval_token_cnt = 0

    nlls = []
    while True:
        try:
            index: int = queue.get(block=False)
        except Empty:
            break
        encoded_text = dataset[index][INPUT_IDS]
        labels = torch.LongTensor([encoded_text])
        assert labels.size(0) == 1, labels.size()
        seq_len = labels.size(1)

        prev_end_loc = max_tokenized_len - args.window_size
        for end_loc in range(max_tokenized_len, seq_len, args.window_size):
            if eval_token_cnt >= max_eval_token:
                break
            begin_loc = end_loc - max_tokenized_len
            trg_len = end_loc - prev_end_loc
            prev_end_loc = end_loc
            eval_token_cnt += trg_len
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if args.add_bos:
                bos_tokens_tensor = torch.LongTensor([[tokenizer.bos_token_id]]).to(device)
                input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

            if aggressive_memory:
                outputs = None
                input_ids = None
                target_ids = None
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            nlls.append(neg_log_likelihood)

    return PPLResult(torch.stack(nlls).mean().float().item(), len(nlls))


def assign_tasks(queue: mp.Queue, args: Args, dataset: Dataset):
    for i in range(len(dataset)):
        queue.put(i)


# https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def load_model(args: Args, device: str):
    if args.nope:
        monkey_patch_before(
            ModelArguments(
                nope=True,
                use_flash_attention=True,
                softmax_scale_type=args.scale_type,
                softmax_scale=args.scale,
            )
        )
    elif args.yarn is not None:  # scale yarn
        monkey_patch_before(
            ModelArguments(
                use_flash_attention=True,
                softmax_scale_type=args.scale_type,
                softmax_scale=args.scale,
            )
        )

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(
        args.base_model,
        revision=args.revision,
    )

    if args.PI is not None:
        config.rope_scaling = {"type": "linear", "factor": args.PI}
    elif args.NTK is not None:
        config.rope_scaling = {"type": "dynamic", "factor": args.NTK}

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        revision=args.revision,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    if args.yarn is not None:
        monkey_patch_after(model, ModelArguments(yarn=args.yarn))
    return model, tokenizer


def main(args: Args):
    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    dataset = load_from_disk(args.dataset_name)["test"]
    logger.warning("Starting")

    with mp.Manager() as manager, mp.Pool(world_size * args.overlap) as pool:
        queue = manager.Queue()
        assign_tasks(queue, args, dataset)  # type: ignore
        assert args.max_eval_tokens % (world_size * args.overlap) == 0
        max_eval_token_per_gpu = args.max_eval_tokens // (world_size * args.overlap)
        map_func = partial(compute_perplexity, dataset=dataset, args=args, queue=queue, max_eval_token=max_eval_token_per_gpu)  # type: ignore
        procs = pool.map_async(map_func, [f"cuda:{i}" for i in range(world_size)] * args.overlap)
        all_rst = procs.get()
    sum = 0
    cnt = 0
    for item in all_rst:
        sum += item.loss * item.samples
        cnt += item.samples
    avg = sum / cnt
    try:
        ppl = math.exp(avg)
    except OverflowError:
        ppl = float("inf")
    logger.warning(f"Loss: {avg:.4f}, PPL: {ppl:.2f}")
    with open(os.path.join(args.logging_dir, "result.json"), mode="w") as f:
        content = {
            "model_name": args.base_model,
            "logging_dir": args.logging_dir,
            "max_length": args.max_length,
            "loss": avg,
            "ppl": ppl,
        }
        logger.warning(f"Dumps {content}")
        json.dump(content, f)


if __name__ == "__main__":
    args = simple_parsing.parse(
        Args,
        conflict_resolution=simple_parsing.ConflictResolution.NONE,  # do not allow duplicate args
        argument_generation_mode=simple_parsing.ArgumentGenerationMode.FLAT,  # (default)
        add_config_path_arg=True,  # allow `--config_path`
    )
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s|%(name)s:%(lineno)s] >> %(message)s",
        handlers=[logging.StreamHandler()],
        level="INFO",  # INFO by default if not overwritten by huggingface setting
    )
    logger = logging.getLogger(__name__)
    warnings.simplefilter("ignore")

    print(args.to_json_string())
    main(args)
