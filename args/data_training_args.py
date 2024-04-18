from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvalMetric(Enum):
    default = "default"
    separate = "separate"
    pos = "pos"
    entropy = "entropy"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: str  # dataset that previously saved using `save_to_disk()`
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_dev_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = None
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = None
    """
    Optional input sequence length that feed into the model.
    Automatically merge dataset samples into one long sequence of length block_size.
    Default to the model max input length for single sentence inputs (take into account special tokens).
    """
    eval_metric: EvalMetric = EvalMetric.default
    is_tiny_llama: bool = False  # whether to use tiny llama dataset

    def __post_init__(self):
        pass
