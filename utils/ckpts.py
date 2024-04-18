import logging
import os
from typing import Optional

from transformers.trainer_utils import get_last_checkpoint

import args

logger = logging.getLogger(__name__)


def detect_last_ckpt(args=args.args) -> Optional[str]:
    last_checkpoint = None
    if os.path.isdir(args.training.output_dir) and args.training.do_train and not args.training.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.training.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.training.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint
