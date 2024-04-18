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

    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})

    def __post_init__(self):
        pass


# args = Args(
#     tokenizer_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
#     save_path="~/nope/data/PG19",
#     dataset_name="pg19",
# )
args = Args(
    tokenizer_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    save_path="~/nope/data/proof_pile",
    dataset_name="hoskinson-center/proof-pile",
)


def get_tokenizer() -> Tokenizer:
    return AutoTokenizer.from_pretrained(args.tokenizer_name)


# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
TOK_COLUMN = "input_ids"  # key of the output of tokenizer


def tokenize_function(samples: dict[str, str], tokenizer: Tokenizer, text_column_name: str):
    assert text_column_name in samples
    assert isinstance(samples[text_column_name], str)
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(samples[text_column_name], return_attention_mask=False)
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )

    return {TOK_COLUMN: output[TOK_COLUMN]}


def preprocess(raw_datasets: datasets.DatasetDict):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    tokenizer = get_tokenizer()
    column_names = list(raw_datasets["test"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    features = Features(
        {
            TOK_COLUMN: Sequence(Value("uint16")),
        }
    )

    wrapped_map_func = partial(
        tokenize_function,
        tokenizer=tokenizer,
        text_column_name=text_column_name,
    )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    lm_datasets = raw_datasets.map(
        wrapped_map_func,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        features=features,
        desc=f"Preprocessing texts",
    )

    print(f"Dataset after preprocess: {lm_datasets}")
    return lm_datasets


def main():
    assert args.dataset_name is not None  # load from file not implemented
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        keep_in_memory=True,  # prevent random access on disk after shuffle (by split)
    )
    print(f"{raw_datasets=}")
    assert isinstance(raw_datasets, datasets.DatasetDict)
    # raw_datasets["train"] = raw_datasets["train"].select(range(10000))
    # if list(raw_datasets.keys()) == ["train"]:
    #     assert False, "PG19 should have val and test"
    #     raw_datasets = split_dataset(raw_datasets["train"])
    # raw_datasets.pop("train")
    # raw_datasets.pop("validation")
    assert list(raw_datasets.keys()) == ["test"], raw_datasets.keys()

    lm_datasets = preprocess(raw_datasets)

    lm_datasets.save_to_disk(args.save_path)


if __name__ == "__main__":
    datasets.utils.logging.set_verbosity_info()
    op.config(color=True, line_number=True, arg_name=True)
    print(args)
    main()
