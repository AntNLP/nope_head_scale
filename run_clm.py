#!/usr/bin/env python
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s|%(name)s:%(lineno)s] >> %(message)s",
    handlers=[logging.StreamHandler()],
    level="INFO",  # INFO by default if not overwritten by huggingface setting
)

import math

import torch
from transformers import default_data_collator, set_seed
from transformers.trainer_utils import TrainOutput

import data
import evals
import models
import utils
from args import EvalMetric, args
from trainer import TrainerLRGroup

logger = logging.getLogger(__name__)


def main():
    utils.init_logger(args)
    utils.init_op()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.training.local_rank}, device: {args.training.device}, n_gpu: {args.training.n_gpu}, "
        + f"distributed training: {args.training.parallel_mode.value == 'distributed'}, 16-bits training: {args.training.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Set seed before initializing model.
    set_seed(args.training.seed)

    config, tokenizer, model = models.auto_model()

    train_set, dev_set, test_set = data.load_datasets(config, tokenizer, model)

    if args.data.eval_metric == EvalMetric.default:
        compute_metrics = evals.compute_metrics_clm_no_early
    elif args.data.eval_metric == EvalMetric.separate:
        compute_metrics = evals.compute_metrics_clm_separate
    elif args.data.eval_metric == EvalMetric.pos:
        compute_metrics = evals.compute_metrics_pos
    elif args.data.eval_metric == EvalMetric.entropy:
        compute_metrics = evals.compute_metrics_entropy
    else:
        assert False, f"Unknown eval_metric: {args.data.eval_metric}"

    # Initialize our Trainer
    trainer = TrainerLRGroup(
        model=model,
        args=args.training,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=utils.get_callbacks(model=model),
        preprocess_logits_for_metrics=evals.preprocess_logits_for_metrics_clm,
    )

    # Training
    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint := utils.detect_last_ckpt() is not None:
            checkpoint = last_checkpoint

        train_result: TrainOutput = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        # assert isinstance(train_set, data.DatasetWithLen)
        # metrics["train_samples"] = len(train_set)
        trainer.log_metrics("train", metrics)

        if args.training.save_model:
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    # Evaluation after training
    if args.training.do_eval:  # and args.training.do_train:
        logger.info("*** Evaluate on dev set ***")

        metrics = trainer.evaluate()

        # assert isinstance(dev_set, data.DatasetWithLen)
        # metrics["eval_samples"] = len(dev_set)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("dev", metrics)

        if args.training.save_model:
            trainer.save_metrics("dev", metrics)


if __name__ == "__main__":
    main()
