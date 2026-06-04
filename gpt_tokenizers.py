import tiktoken
from transformers import AutoTokenizer

TIKTOKEN_PREFIX = "tiktoken/"

class Tokenizer:
    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        self._is_tiktoken = encoding_name.startswith(TIKTOKEN_PREFIX)
        self._load_encoding()

    def _load_encoding(self):
        if self._is_tiktoken:
            self._enc = tiktoken.get_encoding(self.encoding_name[len(TIKTOKEN_PREFIX):])
        else:
            self._enc = AutoTokenizer.from_pretrained(self.encoding_name)

    def __getstate__(self):
        # The underlying encoder is not guaranteed to be picklable, which breaks
        # multiprocessing (e.g. Pool.imap). Only persist what is needed to rebuild
        # it and reconstruct the encoder lazily in __setstate__.
        return {"encoding_name": self.encoding_name, "_is_tiktoken": self._is_tiktoken}

    def __setstate__(self, state):
        self.encoding_name = state["encoding_name"]
        self._is_tiktoken = state["_is_tiktoken"]
        self._load_encoding()

    def tokenize(self, text: str) -> list[int]:
        if self._is_tiktoken:
            return self._enc.encode_ordinary(text) + [self._enc.eot_token]
        eos_token_id = self._enc.eos_token_id
        return self._enc.encode(text, add_special_tokens=False) + (
            [eos_token_id] if eos_token_id is not None else []
        )

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)
