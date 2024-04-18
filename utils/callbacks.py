import json
import logging
import os
from collections import OrderedDict

from transformers import (
    PreTrainedModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.integrations import TensorBoardCallback, is_tensorboard_available

from args import Args
from args import args as all_args

from .cb_metrics import (
    MemoryMetricCallback,
    TrainProfilerCallback,
    TrainRunTimeCallback,
    get_flops,
    get_memory,
    get_train_runtime,
)

logger = logging.getLogger(__name__)

_callbacks: OrderedDict[str, TrainerCallback] = OrderedDict()


def get_callbacks(model: PreTrainedModel):
    if len(_callbacks) == 0:
        if not all_args.training.skip_memory_metrics:  # use memory metrics flag as profiler flag
            _callbacks["TrainRunTimeCallback"] = TrainRunTimeCallback()
            _callbacks["TrainProfilerCallback"] = TrainProfilerCallback(all_args)
            _callbacks["MemoryMetricCallback"] = MemoryMetricCallback()
        if is_tensorboard_available():
            _callbacks["ArgsTensorBoardCallBack"] = ArgsTensorBoardCallBack(all_args)
        _callbacks["ParamClampCallback"] = ParamClampCallback(model)
    return list(_callbacks.values())


def collect_callback_metrics(args: Args, metrics: dict):
    if args.training.process_index == 0 and not args.training.skip_memory_metrics:
        metrics["corrected_train_runtime"] = get_train_runtime(
            _callbacks["TrainRunTimeCallback"]
        )  # only world process zero records this data
        metrics["flops_by_profiler"] = get_flops(_callbacks["TrainProfilerCallback"])
        metrics["train_mem_model"], metrics["train_mem_peak"] = get_memory(_callbacks["MemoryMetricCallback"])


def dump_metrics(args: Args, metrics: dict):
    if args.training.process_index == 0:
        print(json.dumps(metrics))


class ArgsTensorBoardCallBack(TensorBoardCallback):
    """
    log all args to TensorBoard TEXT section
    """

    def __init__(self, args: Args):
        super().__init__()
        self.all_args = args

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # pasted from transformers/integrations.py (4.28.0)
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)  # type: ignore

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        # log args
        if self.tb_writer is not None:
            self.tb_writer.add_text("all_args", self.all_args.to_json_string())


class ParamClampCallback(TrainerCallback):
    """
    clamp scale_param
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        from . import log_once

        if all_args.model.scale_lb is None:
            return
        for name, param in self.model.named_parameters():
            if "scale_param" in name:
                clamp_value = all_args.model.scale_lb
                param.data.clamp_min_(clamp_value)  # ReLU 1
                log_once(logger, f"clamped scale_param to {clamp_value}")
