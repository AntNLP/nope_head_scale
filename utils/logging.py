import logging

import datasets
import transformers

import args


# Pass args to init huggingface logging
def init_logger(args=args.args) -> None:
    if args.training.should_log:
        # The default of args.training.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = args.training.get_process_log_level()
    logging.getLogger().setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
