import functools
import logging

import tiktoken
from transformers import AutoProcessor, AutoTokenizer

log = logging.getLogger(__name__)

TIKTOKEN_PREFIX = "tiktoken/"
# Multimodal model families that require AutoProcessor for correct tokenization.
# Extend this tuple as support for more multimodal models is added.
_PROCESSOR_PATTERNS = ("/gemma-",)


@functools.lru_cache(maxsize=None)
def _get_cached_encoding(encoding_name: str, is_tiktoken: bool, is_processor: bool = False):
    if is_tiktoken:
        return tiktoken.get_encoding(encoding_name[len(TIKTOKEN_PREFIX):])
    if is_processor:
        return AutoProcessor.from_pretrained(encoding_name)
    return AutoTokenizer.from_pretrained(encoding_name)


class Tokenizer:
    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        self._is_tiktoken = encoding_name.startswith(TIKTOKEN_PREFIX)
        self._is_processor = (not self._is_tiktoken
                              and any(p in encoding_name for p in _PROCESSOR_PATTERNS))
        self._load_encoding(use_cache=False)

    def _load_encoding(self, use_cache: bool = False):
        if use_cache:
            self._enc = _get_cached_encoding(self.encoding_name, self._is_tiktoken,
                                             self._is_processor)
        elif self._is_tiktoken:
            self._enc = tiktoken.get_encoding(self.encoding_name[len(TIKTOKEN_PREFIX):])
        elif self._is_processor:
            self._enc = AutoProcessor.from_pretrained(self.encoding_name)
            log.info("Loaded tokenizer via AutoProcessor for %s", self.encoding_name)
        else:
            self._enc = AutoTokenizer.from_pretrained(self.encoding_name)
            log.info("Loaded tokenizer via AutoTokenizer for %s", self.encoding_name)

    def __getstate__(self):
        # The underlying encoder is not guaranteed to be picklable, which breaks
        # multiprocessing (e.g. Pool.imap). Only persist what is needed to rebuild
        # it and reconstruct the encoder lazily in __setstate__.
        return {"encoding_name": self.encoding_name,
                "_is_tiktoken": self._is_tiktoken,
                "_is_processor": self._is_processor}

    def __setstate__(self, state):
        self.encoding_name = state["encoding_name"]
        self._is_tiktoken = state["_is_tiktoken"]
        self._is_processor = state.get("_is_processor", False)
        # Pool.imap unpickles the tokenizer once per chunk in each worker, so cache
        # the loaded encoder at module level to avoid reloading it from disk/network
        # on every chunk.
        self._load_encoding(use_cache=True)

    def _chat_host(self):
        """Return the object carrying the chat template (tokenizer > processor), or None."""
        tokenizer = getattr(self._enc, "tokenizer", self._enc)
        if getattr(tokenizer, "chat_template", None):
            return tokenizer
        if getattr(self._enc, "chat_template", None):
            return self._enc
        return None

    def tokenize(self, text: str, append_eot: bool = True) -> list[int]:
        """Tokenize text into token ids.

        :param append_eot: Append the end-of-text/EOS token. Needed as a
            document separator when preparing training datasets, but must be
            False when encoding a generation prompt: a trailing EOS marks a
            document boundary, making the model start fresh unrelated text
            instead of continuing the prompt.
        """
        if self._is_tiktoken:
            tokens = self._enc.encode_ordinary(text)
            return tokens + [self._enc.eot_token] if append_eot else tokens
        if self._is_processor:
            chat_host = self._chat_host()
            if chat_host is not None:
                result = chat_host.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True,
                )
                # Processors may return a BatchEncoding/dict with input_ids + attention_mask
                if isinstance(result, dict) or hasattr(result, "input_ids"):
                    ids = result["input_ids"]
                    # Strip batch dim if present (tensor of shape [1, seq] or list[list])
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    return ids
                return result
            enc = getattr(self._enc, "tokenizer", self._enc)
        else:
            enc = self._enc
        if not append_eot:
            # Generation prompt: let the tokenizer add its native special
            # tokens (e.g. the leading <bos> Gemma models require) and do
            # not terminate with EOS.
            return enc.encode(text, add_special_tokens=True)
        eos_token_id = enc.eos_token_id
        return enc.encode(text, add_special_tokens=False) + (
            [eos_token_id] if eos_token_id is not None else []
        )

    def decode(self, tokens: list[int]) -> str:
        if self._is_processor:
            if self._chat_host() is not None:
                # Strip chat-format wrapping from the model response
                return self._enc.parse_response(
                    self._enc.decode(tokens, skip_special_tokens=False))
            return getattr(self._enc, "tokenizer", self._enc).decode(tokens)
        return self._enc.decode(tokens)
