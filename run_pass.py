# Written by Yukang Chen
# Core code based on https://github.com/CStanKonrad/long_llama
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/dvlab-research/LongLoRA/blob/main/passkey_retrivial.py

import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from functools import partial
from itertools import chain
from queue import Empty
from statistics import fmean
from typing import Any, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s [%(levelname)s|%(name)s:%(lineno)s] >> %(message)s",
    handlers=[logging.StreamHandler()],
    level="INFO",  # INFO by default if not overwritten by huggingface setting
)

import pandas as pd
import simple_parsing
import torch
import torch.multiprocessing as mp
from numpy import random
from objprint import op
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
)

from models.llama_nope import (
    ModelArguments,
    SoftMaxScaleType,
    monkey_patch_after,
    monkey_patch_before,
)
from utils.typing import Tokenizer

Tensor = torch.Tensor


@dataclass
class Args(simple_parsing.Serializable):
    logging_dir: str
    base_model: str = "/data1/pretrained-models/llama-7b-hf"
    revision: str = "main"  # The specific model version to use (can be a branch name, tag name or commit id).
    overlap: int = 2  # How much parallel runs on one GPU
    max_tokens: int = 32768  # maximum token length for evaluation
    num_length: int = 16
    num_depth: int = 10
    num_tests: int = 10  # number of repeat testing for each length
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


text_task = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n"
text_garbage = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
test_question = "\nWhat is the pass key? The pass key is"  # must not ends with a space
template_info = "\nThe pass key is {pass_key}. Remember it. {pass_key} is the pass key.\n"


def generate_prompt_landmark(
    tokenizer: Tokenizer,
    pass_key: str,
    prompt_len: int,
    info_pos: int,
    task_len: int,
    garbage_len: int,
    question_len: int,
):
    """Concat tensors into prompt"""
    text_info = template_info.format(pass_key=pass_key)
    info_len = tokenizer(text_info, return_tensors="pt").input_ids.shape[-1]
    info_pos = max(info_pos, task_len)  # save space for task
    info_pos = min(info_pos, prompt_len - info_len - question_len)  # save space for question
    n_prefix = (info_pos - task_len) // garbage_len
    real_info_pos = task_len + n_prefix * garbage_len
    assert real_info_pos <= info_pos, (real_info_pos, info_pos)
    n_postfix = (prompt_len - real_info_pos - info_len - question_len) // garbage_len
    prompt = text_task + text_garbage * n_prefix + text_info + text_garbage * n_postfix + test_question
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    assert input_ids.shape[-1] <= prompt_len, (input_ids.shape[-1], prompt_len)
    return input_ids, real_info_pos


@torch.no_grad()
def run_inference(
    model: PreTrainedModel, tokenizer: Tokenizer, input_ids: torch.Tensor, answer_ids: torch.Tensor, args: Args
):
    if args.NTK is not None:
        reset_NTK(model)
    len_ans = answer_ids.shape[-1]

    gen_config = model.generation_config
    # gen_config.do_sample = True
    # gen_config.num_beams = 5
    # gen_config.num_beam_groups = 5
    # gen_config.diversity_penalty = 1.0
    gen_config.max_new_tokens = len_ans
    generation_output = model.generate(
        input_ids=input_ids,
        # pad_token_id=tokenizer.eos_token_id,
        generation_config=gen_config,
    )

    model_answer = generation_output[:, -len_ans:]
    # print(answer_ids, model_answer)

    is_correct = int((model_answer == answer_ids).all().item())
    return is_correct, model_answer, answer_ids


@torch.no_grad()
def run_passkey(
    device: str,
    args: Args,
    queue: mp.Queue,
):
    if "0" in device:
        set_global_logging_level(logging.INFO)
    else:
        set_global_logging_level(logging.ERROR)
    model, tokenizer = load_model(args, device)
    # print(type(model.model.layers[0].self_attn))
    task_len = tokenizer(text_task, return_tensors="pt").input_ids.shape[-1]
    garbage_len = tokenizer(text_garbage, return_tensors="pt", add_special_tokens=False).input_ids.shape[-1]
    question_len = tokenizer(test_question, return_tensors="pt", add_special_tokens=False).input_ids.shape[-1]
    rst = []
    while True:
        seq_len: int
        info_positions: list[int]
        pass_keys: list[str]
        try:
            seq_len, depth, info_positions, pass_keys = queue.get(block=False)
        except Empty:
            break

        assert len(info_positions) == len(pass_keys)
        length, position, acc = [], [], []
        for info_pos, pass_key in zip(info_positions, pass_keys):
            answer_ids = tokenizer(pass_key, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            input_ids, real_info_pos = generate_prompt_landmark(
                tokenizer, pass_key, seq_len - answer_ids.shape[-1], info_pos, task_len, garbage_len, question_len
            )
            # print(f"'{tokenizer.decode(answer_ids[0])}'")
            # print(f"'{tokenizer.decode(input_ids[0])}'")
            is_correct, model_answer, ground_truth = run_inference(
                model, tokenizer, input_ids.to(device), answer_ids, args
            )
            length.append(input_ids.shape[-1])
            position.append(real_info_pos)
            acc.append(is_correct)
        assert len(set(length)) == 1, length
        # if 0 in set(acc):
        #     print(f"The correct answer is '{tokenizer.decode(ground_truth[0])}'")
        #     print(f"The model answer is '{tokenizer.decode(model_answer[0])}'")
        # rst.append({"length": input_ids.shape[-1], "position": f"{fmean(position):.0f}", "accuracy": fmean(acc)})
        rst.append(
            {
                "length": seq_len,
                "depth": f"{depth*100:.0f}%",
                "position": f"{fmean(info_positions):.0f}",
                "accuracy": fmean(acc),
            }
        )
        # free cuda memory
        # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898/3
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return rst


def assign_tasks(queue: mp.Queue, args: Args):
    random.seed(42)
    len_interval = args.max_tokens // args.num_length
    for seq_len in reversed(range(len_interval, args.max_tokens + 1, len_interval)):
        depth_interval = seq_len // args.num_depth
        for depth_idx in range(args.num_depth):
            info_pos = depth_idx * depth_interval
            info_positions = [random.randint(info_pos, info_pos + depth_interval) for _ in range(args.num_tests)]
            pass_keys = [str(random.randint(10**4, 10**5)) for _ in range(args.num_tests)]
            queue.put((seq_len, depth_idx / args.num_depth, info_positions, pass_keys))


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
        model_max_length=args.max_tokens,
        # padding_side="right",
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


def reset_NTK(model: PreTrainedModel):
    for each in model.model.layers:
        each.self_attn.rotary_emb.max_seq_len_cached = 0


def main(args: Args):
    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    with mp.Manager() as manager, mp.Pool(world_size * args.overlap) as pool:
        queue = manager.Queue()
        assign_tasks(queue, args)  # type: ignore
        n_tasks = queue.qsize()
        map_func = partial(run_passkey, args=args, queue=queue)  # type: ignore
        rst = pool.map_async(map_func, [f"cuda:{i}" for i in range(world_size)] * args.overlap)
        with tqdm(total=n_tasks) as pbar:
            while not rst.ready():
                pbar.update(n_tasks - queue.qsize() - pbar.n)
                time.sleep(1)
        rst = chain(*rst.get())

    df = pd.DataFrame(rst)
    df = df.groupby(["length", "depth", "position"])["accuracy"].mean().reset_index()
    df.to_csv(os.path.join(args.logging_dir, "result.csv"), index=False)
    print(df)
    acc_mean = df["accuracy"].mean()
    print(f"{acc_mean=}")
    writer = SummaryWriter(log_dir=args.logging_dir)
    writer.add_scalar("eval/passkey", acc_mean, 1)


if __name__ == "__main__":
    # op.config(color=True, line_number=True, arg_name=True)
    # op.install("print")
    args = simple_parsing.parse(
        Args,
        conflict_resolution=simple_parsing.ConflictResolution.NONE,  # do not allow duplicate args
        argument_generation_mode=simple_parsing.ArgumentGenerationMode.FLAT,  # (default)
        add_config_path_arg=True,  # allow `--config_path`
    )

    print(args.to_json_string())
    main(args)
