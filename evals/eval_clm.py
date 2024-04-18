import os
from typing import Union

import evaluate
import numpy as np
import torch
from evaluate import EvaluationModule
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import EvalPrediction

from args import args

assert args.training.logging_dir is not None
acc_module: EvaluationModule = evaluate.load("accuracy", experiment_id=os.path.basename(args.training.logging_dir))


def preprocess_logits_for_metrics_clm(outputs: Union[Tensor, tuple[Tensor, ...]], labels: Tensor):
    # logits: [bs, seq_len, vocab_size]
    # labels: [bs, seq_len]
    if isinstance(outputs, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = outputs[0]
        if args.model.output_attentions:
            attns = outputs[1]  # [bs, n_heads, seq_len]
            assert isinstance(attns, tuple)
            # attns = [attn.cpu() for attn in attns]  # prevent OOM on GPU
            attn = torch.stack(attns, dim=1)  # [bs, n_layers, n_heads, seq_len]
            return attn
    else:
        logits = outputs

    argmax = logits.argmax(dim=-1)[:, :-1]  # [bs, seq_len - 1]

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")  # get loss for each token, reduce later
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.reshape(argmax.shape)
    return torch.stack([argmax, loss], dim=-1)  # [bs, seq_len - 1, 2]


def compute_metrics_clm(eval_preds: EvalPrediction):
    """
    preds should have the same start position as labels but one less token at the end
    """
    rst, labels = eval_preds
    assert isinstance(rst, np.ndarray)  # [#samples, seq_len - 1, 2]
    assert isinstance(labels, np.ndarray)  # [#samples, seq_len]
    preds = rst[:, :, 0]  # [#samples, seq_len - 1]
    loss = rst[:, :, 1]
    # preds are shifted after the argmax(-1) has been calculated by preprocess_logits_for_metrics
    # we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds.reshape(-1)  # already shifted
    metrics = {}
    acc_metric = acc_module.compute(predictions=preds, references=labels)
    assert acc_metric is not None
    metrics.update(acc_metric)
    metrics["metric_loss"] = loss.mean()
    assert set(metrics.keys()) == {"accuracy", "metric_loss"}
    return metrics


def compute_metrics_clm_no_early(eval_preds: EvalPrediction):
    preds, labels = eval_preds
    assert isinstance(preds, np.ndarray)  # [#samples, seq_len - 1, 2]
    assert isinstance(labels, np.ndarray)  # [#samples, seq_len]
    early_len = 256
    seq_len = labels.shape[1]
    preds = preds[:, early_len:, :]
    labels = labels[:, early_len:]
    metrics = compute_metrics_clm(EvalPrediction(preds, labels))
    rst = {}
    for key in metrics:
        rst[f"{key}_{early_len}-{seq_len}"] = metrics[key]
    return rst


def compute_metrics_clm_separate(eval_preds: EvalPrediction):
    preds, labels = eval_preds
    assert isinstance(preds, np.ndarray)  # [#samples, seq_len - 1, 2]
    assert isinstance(labels, np.ndarray)  # [#samples, seq_len]
    origin_len = 1024
    early_len = 256
    seq_len = labels.shape[1]

    preds0 = preds
    labels0 = labels
    preds1 = preds[:, early_len:origin_len, :]
    labels1 = labels[:, early_len : origin_len + 1]
    preds2 = preds[:, origin_len:, :]
    labels2 = labels[:, origin_len:]

    metric0 = compute_metrics_clm(EvalPrediction(preds0, labels0))
    metric1 = compute_metrics_clm(EvalPrediction(preds1, labels1))
    metric2 = compute_metrics_clm(EvalPrediction(preds2, labels2))
    rst = {}
    for metric, name in zip(
        [metric0, metric1, metric2],
        ["", f"_{early_len}-{origin_len}", f"_{origin_len}-{seq_len}"],
    ):
        for key in metric:
            rst[f"{key}{name}"] = metric[key]
    return rst
