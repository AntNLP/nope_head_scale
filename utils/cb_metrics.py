import time

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from args import Args


class TrainRunTimeCallback(TrainerCallback):
    """
    Records the first and the last step end. Duration accounts for (#steps - 1) steps, skipping the first step.
    CUDA kernel is initialized during the first step, so it's inaccurate to use overall training time as metric.
    """

    def __init__(self):
        self.start_time = -1.0
        self.end_time = 0.0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            now = time.time()
            if self.start_time < 0:
                self.start_time = now  # first step end
            self.end_time = now  # last step end


# get train runtime and update metrics
def get_train_runtime(cb: TrainerCallback):
    assert isinstance(cb, TrainRunTimeCallback)
    runtime = cb.end_time - cb.start_time
    return runtime


class TrainProfilerCallback(TrainerCallback):
    """
    Add pytorch profiler
    """

    def __init__(self, args: Args):
        assert isinstance(args.training.logging_dir, str)  # None is filled by default dir in post_init
        self.profiler = torch.profiler.profile(
            profile_memory=True,
            with_flops=True,
            # record_shapes=True,
            # with_stack=True,  # huge file size
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.training.logging_dir, use_gzip=True),
            schedule=torch.profiler.schedule(
                wait=0, warmup=1, active=999, skip_first=0  # only warmup matters
            ),  # should not use skip_first, do warmup instead and the profiler warmup along with CUDA
        )

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.profiler.start()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.profiler.step()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.profiler.stop()


# This is about 1/4 of the calculated value, code reference: https://github.com/pytorch/pytorch/issues/69782
def get_flops(cb: TrainerCallback):
    from torch.autograd.profiler_util import EventList, FunctionEvent

    assert isinstance(cb, TrainProfilerCallback)
    events = cb.profiler.events()
    assert isinstance(events, list)
    assert all(isinstance(evt, FunctionEvent) for evt in events)
    flops = [evt.flops for evt in events if evt.flops is not None]
    print("#event:", len(events))
    print("#event with flops:", len(flops))
    return sum(flops)


class MemoryMetricCallback(TrainerCallback):
    def __init__(self):
        self.model_mem: int = 0
        self.peak_mem: int = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # self.model_mem = torch.cuda.max_memory_allocated()
        self.model_mem = torch.cuda.max_memory_reserved()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # self.peak_mem = torch.cuda.max_memory_allocated()
        self.peak_mem = torch.cuda.max_memory_reserved()


def get_memory(cb: TrainerCallback):
    assert isinstance(cb, MemoryMetricCallback)
    return cb.model_mem, cb.peak_mem
