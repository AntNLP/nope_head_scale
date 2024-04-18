import logging

import torch
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

import args
from utils.typing import Tokenizer

from . import llama_nope

logger = logging.getLogger(__name__)


def auto_model(args=args.args):
    llama_nope.monkey_patch_before(args.model)

    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "token": args.model.token,
        "trust_remote_code": args.model.trust_remote_code,
    }
    config_kwargs.update(
        {
            "nope": args.model.nope,
            "output_attentions": args.model.output_attentions,
        }
    )
    if args.model.scale_type is not None:
        config_kwargs["rope_scaling"] = {
            "type": args.model.scale_type,
            "factor": args.model.scale_factor,
        }
    if args.model.config_name:
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif args.model.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model.model_name_or_path, **config_kwargs)
    else:
        assert args.model.model_type is not None
        config = CONFIG_MAPPING[args.model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.model.config_overrides is not None:
            logger.info(f"Overriding config: {args.model.config_overrides}")
            config.update_from_string(args.model.config_overrides)
            logger.info(f"New config: {config}")
    assert isinstance(config, PretrainedConfig)

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "token": args.model.token,
        "trust_remote_code": args.model.trust_remote_code,
    }
    # Useless because pretrained length overwrite it. See transformers/tokenization_utils_base.py:2146 (v4.34.1)
    # if args.model.n_extend is not None:
    #     # used in gpt2pe
    #     tokenizer_kwargs.update({"model_max_length": args.model.n_extend + n_positions})
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_name, **tokenizer_kwargs)
    elif args.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # assert isinstance(tokenizer, Tokenizer)  # python >= 3.10

    if args.model.model_name_or_path:
        torch_dtype = (
            args.model.torch_dtype
            if args.model.torch_dtype == "auto" or args.model.torch_dtype is None
            else getattr(torch, args.model.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model.model_name_or_path,
            from_tf=bool(".ckpt" in args.model.model_name_or_path),
            config=config,
            cache_dir=args.model.cache_dir,
            revision=args.model.model_revision,
            token=args.model.token,
            trust_remote_code=args.model.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.model.low_cpu_mem_usage,
            use_flash_attention_2=args.model.use_flash_attention,
        )
        assert isinstance(model, PreTrainedModel)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.model.trust_remote_code)
        assert isinstance(model, PreTrainedModel)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]  # type: ignore
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    llama_nope.monkey_patch_after(model, args.model)

    if model.config.model_type == "llama":
        llama_nope.prepare_for_training(model, args.model)

    return config, tokenizer, model
