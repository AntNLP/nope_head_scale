from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Optional, Union

import datasets
import transformers
from datasets import Features, Sequence, Value, load_dataset
from objprint import op
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import CaptureLogger

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@dataclass
class Args:
    """
    Arguments how to load and preprocess the data.
    """

    tokenizer_name: str
    save_path: str
    block_size: int
    """
    Input sequence length after tokenization.
    The training dataset will be truncated in block of this size for training.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    # train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    # dev_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
    # test_file: Optional[str]
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    dev_split_size: Optional[float] = 0.05  # fraction of dev set
    test_split_size: Optional[float] = 0.05  # fraction of test set
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        pass
        # if self.dataset_name is None and self.train_file is None and self.dev_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        #     if self.dev_file is not None:
        #         extension = self.dev_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


args = Args(
    tokenizer_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    save_path="~/nope/data/PG19",
    block_size=256,
    dataset_name="pg19",
    # preprocessing_num_workers=4,
)


def get_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained(args.tokenizer_name)


# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
TOKEN_IDX = "tok_idx"
GROUP_IDX = "grp_idx"
TOK_COLUMN = "input_ids"  # key of the output of tokenizer


def tokenize_function(samples: dict[str, list], indices: list[int], tokenizer: Tokenizer, text_column_name: str):
    assert text_column_name in samples
    assert isinstance(samples[text_column_name], list)  # list[str]
    assert isinstance(samples[text_column_name][0], str)
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(samples[text_column_name])
    # output: {'input_ids': list_of_size(batch_size, different_sample_length), 'attention_mask': list_of_same_size}
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    assert indices[0] + len(indices) == indices[-1] + 1, "indices should be continuous"

    return {TOK_COLUMN: output[TOK_COLUMN], TOKEN_IDX: indices[0]}


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(samples: dict[str, list], block_size: int):
    # Concatenate all texts.
    concatenated_examples = list(chain(*samples[TOK_COLUMN]))
    total_length = len(concatenated_examples)
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        TOK_COLUMN: [concatenated_examples[i : i + block_size] for i in range(0, total_length, block_size)],
        TOKEN_IDX: [samples[TOKEN_IDX] for _ in range(0, total_length, block_size)],  # first idx of each batch
        GROUP_IDX: list(range(total_length // block_size)),
    }
    return result


def map_func(
    samples: dict[str, list], indices: list[int], tokenizer: Tokenizer, text_column_name: str, block_size: int
):
    """for huggingface datasets map()

    Args:
        samples (_type_): input by map()
        tokenizer (PreTrainedTokenizer):
        text_column_name (str): text key in samples
        block_size (int): sequence length to process

    """
    samples = tokenize_function(samples, indices, tokenizer, text_column_name)
    samples = group_texts(samples, block_size)
    return samples


def preprocess(raw_datasets: datasets.DatasetDict):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    tokenizer = get_tokenizer()
    column_names = list(raw_datasets["test"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    features = Features(
        {
            # convert to uint16 to save space, uint16 range: 0~65535
            # ArrowInvalid exception will be raised if data overflow, so it's safe to use without caution.
            # TODO: automate data type
            TOK_COLUMN: Sequence(Value("uint16"), length=args.block_size),
            TOKEN_IDX: Value("int64"),  # for sort
            GROUP_IDX: Value("int64"),
        }
    )

    wrapped_map_func = partial(
        map_func,
        tokenizer=tokenizer,
        text_column_name=text_column_name,
        block_size=args.block_size,
    )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    lm_datasets = raw_datasets.map(
        wrapped_map_func,
        with_indices=True,
        batched=True,
        batch_size=1,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        features=features,
        desc=f"Preprocessing texts in chunks of {args.block_size}",
    )

    lm_datasets = lm_datasets.sort([TOKEN_IDX, GROUP_IDX]).remove_columns([TOKEN_IDX, GROUP_IDX])

    print(f"Dataset after preprocess: {lm_datasets}")
    return lm_datasets


def split_dataset(dataset: datasets.Dataset):
    assert args.dev_split_size is not None and args.test_split_size is not None
    print(f"Splitting dataset in dev {args.dev_split_size} and test {args.test_split_size}")
    dev_split = dataset.train_test_split(args.dev_split_size, seed=42)
    split_datasets = dev_split["train"].train_test_split(args.test_split_size / (1 - args.dev_split_size), seed=137)
    split_datasets["dev"] = dev_split["test"]

    # split_datasets = split_datasets.flatten_indices()
    print(f"Dataset after splits: {split_datasets}")
    return split_datasets


def main():
    assert args.dataset_name is not None  # load from file not implemented
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        num_proc=args.preprocessing_num_workers,
        keep_in_memory=True,  # prevent random access on disk after shuffle (by split)
    )
    print(f"{raw_datasets=}")
    assert isinstance(raw_datasets, datasets.DatasetDict)
    # raw_datasets["train"] = raw_datasets["train"].select(range(10000))
    # if list(raw_datasets.keys()) == ["train"]:
    #     assert False, "PG19 should have val and test"
    #     raw_datasets = split_dataset(raw_datasets["train"])
    raw_datasets.pop("train")
    raw_datasets.pop("validation")

    lm_datasets = preprocess(raw_datasets)

    lm_datasets.save_to_disk(args.save_path, num_proc=args.preprocessing_num_workers)


if __name__ == "__main__":
    datasets.utils.logging.set_verbosity_info()
    op.config(color=True, line_number=True, arg_name=True)
    print(args)
    main()
