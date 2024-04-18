import logging
from itertools import chain
from typing import Dict, List, Optional

import torch.utils.data
from datasets import Dataset

logger = logging.getLogger(__name__)

INPUT_IDS = "input_ids"


class DatasetWithLen(torch.utils.data.Dataset):
    def __len__(self):
        raise NotImplementedError


class GPTDataset(DatasetWithLen):
    def __init__(
        self,
        hf_dataset: Dataset,
        block_size: int,
        max_samples: Optional[int] = None,
    ):
        self.hf_dataset = hf_dataset
        self.block_size = block_size
        self.sample_len = len(hf_dataset[0][INPUT_IDS])
        if (remainder := block_size % self.sample_len) != 0:
            logger.warning(
                f"block_size ({block_size}) is not divisible by sample_len ({self.sample_len}), "
                f"{remainder} tokens will be discarded per sample"
            )
        self.stride = (block_size + self.sample_len - 1) // self.sample_len
        # so `stride` samples will be merged into one
        self.length = len(hf_dataset) // self.stride  # dataset exposed length after merging
        if max_samples is not None:
            self.length = min(self.length, max_samples)
        self.print_once = True

    def __getitem__(self, index):
        if self.print_once:
            self.print_once = False
            logger.info(f"Sanity check: first sample id: {index} (should not be 0 because of the sampler)")
        tokens = self._getitem_with_stride(index)
        sample = self._construct_sample(tokens)
        return sample

    def _getitem_with_stride(self, index):
        index *= self.stride
        samples: dict[str, list[list[int]]] = self.hf_dataset[index : index + self.stride]
        # samples = {'input_ids': shape(self.stride, self.sample_len)}
        tokens = list(chain(*samples[INPUT_IDS]))
        # tokens = {'input_ids': shape(self.stride * self.sample_len)}
        return tokens

    def _construct_sample(self, tokens):
        # save only input_ids on disk, generate other keys on the fly
        if len(tokens) > self.block_size:
            tokens = tokens[: self.block_size]
        sample = {
            INPUT_IDS: tokens,
            "labels": tokens.copy(),
            "attention_mask": [1 for _ in range(self.block_size)],
        }
        return sample

    def __len__(self):
        return self.length
