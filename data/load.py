import glob
import logging
import os
import random
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PretrainedConfig, PreTrainedModel

import models
import utils
from args import args
from utils.typing import Tokenizer

from .dataset import DatasetWithLen, GPTDataset
from .tiny_llama_dataset import CombinedDataset, PackedDataset

logger = logging.getLogger(__name__)


def load_datasets(
    config: PretrainedConfig, tokenizer: Tokenizer, model: PreTrainedModel
) -> Tuple[Optional[DatasetWithLen], Optional[DatasetWithLen], Optional[DatasetWithLen]]:
    if args.data.is_tiny_llama:
        return load_tiny_llama()  # type: ignore
    lm_datasets = load_from_disk(args.data.dataset_path)
    assert isinstance(lm_datasets, DatasetDict)
    block_size = get_block_size(config, tokenizer)
    train_set, dev_set, test_set = get_optional_split(lm_datasets)
    if train_set is not None:
        train_set = GPTDataset(train_set, block_size, args.data.max_train_samples)
    if dev_set is not None:
        dev_set = GPTDataset(dev_set, block_size, args.data.max_dev_samples)
    if test_set is not None:
        test_set = GPTDataset(test_set, block_size, args.data.max_test_samples)
    return train_set, dev_set, test_set


# get split and limit size (full data is preprocessed in prepare, only limit size here according to args)
def get_optional_split(lm_datasets: DatasetDict) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    train_set, dev_set, test_set = None, None, None
    if args.training.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train set")
        train_set = lm_datasets["train"]
    if args.training.do_eval:
        if "dev" not in lm_datasets:
            raise ValueError("--do_eval requires a dev set")
        dev_set = lm_datasets["dev"]
    if args.training.do_predict:
        if "test" not in lm_datasets:
            raise ValueError("--do_predict requires a test set")
        test_set = lm_datasets["test"]
    return train_set, dev_set, test_set


def get_block_size(config: PretrainedConfig, tokenizer: Tokenizer) -> int:
    if args.data.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        block_size = args.data.block_size
        if args.data.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.data.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length})."  # Using block_size={tokenizer.model_max_length}."
            )
        # block_size = min(args.data.block_size, tokenizer.model_max_length)
    return block_size


def load_tiny_llama():
    data_configs = {
        "train": [
            ("train_slim", 0.693584),
            ("train_star", 0.306416),
        ],
        "dev": [
            ("validation", 1.0),
        ],
    }
    train = get_tiny_llama_dataset(data_configs["train"], args.data.max_train_samples, True)
    dev = get_tiny_llama_dataset(data_configs["dev"], args.data.max_dev_samples, False)
    return train, dev, None


def get_tiny_llama_dataset(data_config, max_num_samples: Optional[int], shuffle: bool):
    datasets = []
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(os.path.join(args.data.dataset_path, f"{prefix}*")))
        assert args.training.seed is not None
        random.seed(args.training.seed)
        random.shuffle(filenames)

        assert args.data.block_size is not None
        block_size = args.data.block_size
        # if args.model.yarn is None:
        #     block_size = args.data.block_size
        # else:
        #     block_size = 2048 * args.model.yarn
        #     utils.log_once(logger, f"Overwriting block_size = {block_size}")

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=64 if shuffle else 4,
            block_size=block_size + 1,
            seed=args.training.seed,
            num_processes=args.training.world_size,
            process_rank=args.training.process_index,
            shuffle=shuffle,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {args.data.dataset_path}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(
        datasets=datasets, seed=args.training.seed, weights=weights, max_num_samples=max_num_samples
    )

    return combined_dataset
