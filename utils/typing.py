from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# There are methods missing in PreTrainedTokenizerBase, hence not suitable for typing.
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
