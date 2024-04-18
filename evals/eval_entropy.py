import os

import numpy as np
from torch import Tensor
from transformers import EvalPrediction

from args import args


def compute_metrics_entropy(eval_preds: EvalPrediction):
    entropy, _ = eval_preds
    assert isinstance(entropy, np.ndarray)  # [#samples, n_layers, n_heads, seq_len]
    assert args.training.logging_dir is not None
    entropy = entropy.mean(axis=0)
    # entropy = entropy.std(axis=0)
    np.save(os.path.join(args.training.logging_dir, "entropy.npy"), entropy)
    return {}
