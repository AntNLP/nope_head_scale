from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class TrainingArgumentsExtend(TrainingArguments):
    save_model: bool = True  # whether to save model at the end of training
