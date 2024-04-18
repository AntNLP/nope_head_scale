import json
from dataclasses import dataclass
from typing import Any, Dict

import simple_parsing

from .data_training_args import DataTrainingArguments, EvalMetric
from .model_args import ModelArguments
from .training_args import TrainingArgumentsExtend


@dataclass
class Args(simple_parsing.Serializable):
    """
    Args combines all other argument dataclasses
    """

    # config: ConfigArguments
    model: ModelArguments
    data: DataTrainingArguments
    training: TrainingArgumentsExtend

    def __post_init__(self):
        # other checks or initializations here
        self.check_eval_pos()
        pass

    def to_dict(self):
        _d = super().to_dict()  # simple_parsing.Serializable.to_dict()

        def mask_token(d: Dict[str, Any]):  # Adapted from transformers/training_args.py (v4.28.0)
            for k, v in d.items():
                if isinstance(v, dict):
                    mask_token(v)
                if k.endswith("_token"):  # Obfuscates the token values by removing their value.
                    d[k] = f"<{k.upper()}>"

        mask_token(_d)
        return _d

    def check_eval_pos(self):
        # if self.data.eval_metric == EvalMetric.pos and self.training.do_train:
        #     raise ValueError("Cannot train with POS evaluation metric")
        if not self.model.use_flash_attention:
            if self.data.eval_metric != EvalMetric.entropy or not self.model.output_attentions:
                raise ValueError("Only use normal attention with entropy metric and output_attentions")

    # adapted from transformers/training_args.py (v4.28.0)
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)


class _Args:
    """Wraps Args to allow lazy initialization"""

    def __init__(self):
        self._args_cache = None

    @property
    def _args(self):
        if self._args_cache is None:
            self._disable_parsing_log()
            self._args_cache = simple_parsing.parse(
                Args,
                conflict_resolution=simple_parsing.ConflictResolution.NONE,  # do not allow duplicate args
                argument_generation_mode=simple_parsing.ArgumentGenerationMode.FLAT,  # (default)
                add_config_path_arg=True,  # allow `--config_path`
            )
        return self._args_cache

    def __getattr__(self, name):
        return getattr(self._args, name)

    @staticmethod
    def _disable_parsing_log():
        simple_parsing.parsing.logger.setLevel("WARNING")
        simple_parsing.wrappers.dataclass_wrapper.logger.setLevel("WARNING")

    def __str__(self):
        return self._args.to_json_string()


# args are not parsed here
args: Args = _Args()  # type: ignore
