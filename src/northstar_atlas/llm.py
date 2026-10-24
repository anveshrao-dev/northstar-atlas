"""Provider-independent text-generation contracts and verified Qwen byte bound."""
from dataclasses import dataclass, field
import json
from typing import Protocol
import unicodedata

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class GenerationLimits:
    context_tokens: int = 32768
    output_tokens: int = 1024
    margin_tokens: int = 512

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.context_tokens, self.output_tokens, self.margin_tokens)):
            raise ValueError("generation limits must be positive integers")
        if self.context_tokens > 262144 or self.output_tokens + self.margin_tokens >= self.context_tokens:
            raise ValueError("invalid Qwen context/output budget")


class QwenTokenBound:
    """Conservative text bound, not estimated billing usage or a MiniLM tokenizer.

    The selected Qwen2 tokenizer uses NFC + UTF-8 byte BPE. Each token consumes
    at least one byte; merges only decrease token count. Bound both unnormalized
    and normalized text. Include literal ChatML markers (overcounts special tokens).
    Only two text messages are supported, with no tools, images or conversation history.
    Hosted template differences still need confirmation in an approved live call.
    """
    name = "qwen3-instruct-2507-utf8-nfc-byte-bound-v1"

    def __init__(self, model: str = MODEL_ID):
        if model != MODEL_ID:
            raise ValueError("token bound has not been verified for this model")

    def count_tokens(self, text: str) -> int:
        return max(len(text.encode("utf-8")), len(unicodedata.normalize("NFC", text).encode("utf-8")))

    def request_bound(self, system: str, user: str, schema: dict) -> int:
        framed = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        # Schema is also in system instructions. Reserve a second copy in case
        # the hosted structured-output implementation adds it to its prompt.
        return self.count_tokens(framed) + self.count_tokens(canonical_json(schema))


@dataclass(frozen=True)
class GenerationRequest:
    system: str = field(repr=False)
    user: str = field(repr=False)
    schema: dict = field(repr=False)
    model: str
    output_tokens: int
    context_tokens: int
    input_token_bound: int


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class GenerationResponse:
    text: str = field(repr=False)
    model: str
    finish_reason: str
    usage: TokenUsage | None
    latency_seconds: float
    attempts: int
    # Hosted immutable weights/runtime revision is unknown unless documented.
    revision: str | None = None


class LLMProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
