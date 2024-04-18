import os

import numpy as np
from torch import Tensor
from transformers import EvalPrediction

from args import args

from .eval_clm import acc_module, compute_metrics_clm


def compute_metrics_pos(eval_preds: EvalPrediction):
    metrics = compute_metrics_clm(eval_preds)
    rst, labels = eval_preds
    assert isinstance(rst, np.ndarray)  # [#samples, seq_len - 1, 2]
    assert isinstance(labels, np.ndarray)  # [#samples, seq_len]
    # preds = rst[:, :, 0]  # [#samples, seq_len - 1]
    loss = rst[:, :, 1]
    # preds are shifted after the argmax(-1) has been calculated by preprocess_logits_for_metrics
    # we need to shift the labels
    labels = labels[:, 1:]
    loss = loss.mean(axis=0)
    # assert acc.shape == loss.shape
    assert args.training.logging_dir is not None
    # np.save(os.path.join(args.training.logging_dir, "acc.npy"), acc)
    np.save(os.path.join(args.training.logging_dir, "loss.npy"), loss)
    return metrics
