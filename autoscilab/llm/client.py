"""
LLM client with two backends:

- Together AI (default) for Together-hosted models (e.g. Llama)
- OpenAI for models like `gpt-4o-mini` using OPENAI_API_KEY

Uses JSON mode + schema-in-prompt for structured output.
"""
import json
import os
import random
import re
import time
from typing import Type

from pydantic import BaseModel
from together import Together

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def _resolve_openai_compatible_key(base_url: str | None, api_key: str | None = None) -> str:
    """Resolve an API key for OpenAI-compatible endpoints.

    DeepInfra exposes an OpenAI-compatible API but uses a different credential.
    Prefer the explicit key when provided, then provider-specific env vars.
    """
    if api_key:
        return api_key
    url = (base_url or "").lower()
    if "deepinfra.com" in url:
        return (
            os.environ.get("DEEPINFRA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "local"
        )
    return os.environ.get("OPENAI_API_KEY", "") or "local"


def _strip_descriptions(schema: object) -> object:
    """Recursively remove 'description' and 'title' keys from a JSON schema.

    This keeps only the structural info (types, required, properties, $defs),
    making the injected schema much shorter and avoiding Together AI's
    grammar-compilation 422 errors caused by overly long system prompts.
    """
    if isinstance(schema, dict):
        return {
            k: _strip_descriptions(v)
            for k, v in schema.items()
            if k not in ("description", "title", "examples")
        }
    if isinstance(schema, list):
        return [_strip_descriptions(item) for item in schema]
    return schema


def _schema_to_prompt(schema: dict) -> str:
    """Convert a JSON schema to a compact prompt (descriptions stripped)."""
    return json.dumps(_strip_descriptions(schema), indent=2)


def _is_thinking_model(model: str) -> bool:
    """Return True for models that emit <think>...</think> reasoning blocks.

    These models must NOT use response_format=json_object — the thinking tokens
    consume max_tokens before the JSON is generated, yielding empty content.
    """
    name = model.lower()
    return (
        "thinking" in name
        or "qwen3" in name          # all Qwen3 variants use thinking mode
        or "deepseek-r" in name
    )


def _normalize_bounds_payload(obj: object) -> object:
    """Normalize `{min,max}` bound objects into `[min, max]` lists.

    Some models return bounds in object form even when the schema expects a
    2-element list. Normalize only under keys named `bounds`.
    """
    if isinstance(obj, list):
        return [_normalize_bounds_payload(item) for item in obj]
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key == "bounds" and isinstance(value, dict):
                norm_bounds = {}
                for bk, bv in value.items():
                    if isinstance(bv, dict) and "min" in bv and "max" in bv:
                        norm_bounds[bk] = [bv["min"], bv["max"]]
                    else:
                        norm_bounds[bk] = _normalize_bounds_payload(bv)
                out[key] = norm_bounds
            else:
                out[key] = _normalize_bounds_payload(value)
        return out
    return obj


def _api_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_retryable_api_error(exc: Exception) -> bool:
    code = _api_status_code(exc)
    if code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    markers = (
        "rate limit",
        "model busy",
        "retry later",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "server error",
        "overloaded",
    )
    return any(marker in text for marker in markers)


def _retry_sleep_seconds(attempt: int) -> float:
    base = min(20.0, 1.5 * (2 ** attempt))
    return base + random.uniform(0.0, 0.5)


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
        base_url: str | None = None,
        max_completion_tokens: int = 8192,
        temperature: float | None = None,
    ):
        self._model = model
        self._max_retries = max_retries
        self._max_completion_tokens = max_completion_tokens
        self._temperature = temperature
        self._is_thinking = _is_thinking_model(model)

        # If a custom base_url is provided (e.g. local vLLM/transformers server),
        # always use an OpenAI-compatible client pointing there.
        if base_url:
            from openai import OpenAI  # type: ignore
            self._oa_client = OpenAI(
                base_url=base_url,
                api_key=_resolve_openai_compatible_key(base_url, api_key),
            )
            self._tg_client = None
            self._use_openai = True
            return

        # Route to OpenAI when using OpenAI-branded models (e.g. gpt-4o-mini)
        self._use_openai = "gpt-" in model or model.startswith("o1") or model.startswith("o3")

        if self._use_openai:
            # Lazy import so Together-only users don't need openai installed
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:  # pragma: no cover - import-time failure path
                raise ImportError(
                    "openai package is required to use OpenAI models like 'gpt-4o-mini'. "
                    "Install with:\n  pip install openai"
                ) from e

            # For OpenAI, always prefer OPENAI_API_KEY from the environment.
            oa_key = os.environ.get("OPENAI_API_KEY", "")
            self._oa_client = OpenAI(api_key=oa_key)
            self._tg_client = None
        else:
            self._oa_client = None
            # For Together, keep using TOGETHER_API_KEY (or explicit api_key).
            self._tg_client = Together(
                api_key=api_key or os.environ.get("TOGETHER_API_KEY", ""),
                timeout=360,  # Qwen3-235B thinking blocks can take 2-3 min
            )

    def complete(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int | None = None,
    ) -> str:
        """Basic completion — returns raw text."""
        _max = max_tokens if max_tokens is not None else self._max_completion_tokens
        full_messages = [{"role": "system", "content": system}] + messages

        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                if self._use_openai:
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": _max,
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._oa_client.chat.completions.create(**kwargs)
                else:
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": _max,
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._tg_client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                last_err = exc
                if attempt < self._max_retries - 1 and _is_retryable_api_error(exc):
                    time.sleep(_retry_sleep_seconds(attempt))
                    continue
                raise
        else:  # pragma: no cover
            assert last_err is not None
            raise last_err

        raw = response.choices[0].message.content or ""
        # Strip <think>...</think> blocks (Qwen3 reasoning model)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    def complete_json(
        self,
        messages: list[dict],
        system: str,
        schema: Type[BaseModel],
        tool_name: str = "output",
        max_tokens: int | None = None,
    ) -> BaseModel:
        """
        Structured JSON output using JSON mode + schema-in-prompt.
        Injects the JSON schema into the system prompt and forces JSON output.

        For thinking models (Qwen3, DeepSeek-R, etc.) response_format is skipped:
        the <think>...</think> tokens consume max_tokens before the JSON is
        generated, yielding empty content.  Schema-in-prompt is sufficient.
        Thinking models also get a larger token budget (32768) so the reasoning
        chain does not crowd out the actual JSON answer.
        """
        schema_dict = schema.model_json_schema()
        schema_str = _schema_to_prompt(schema_dict)
        fields_desc = ", ".join(
            f'"{k}"' for k in schema_dict.get("properties", {})
        )

        json_system = (
            f"{system}\n\n"
            f"You MUST respond with a single valid JSON object with these fields: {fields_desc}. "
            f"Fill in every field with actual content — do NOT echo or return the schema itself.\n\n"
            f"SCHEMA:\n{schema_str}"
        )

        full_messages = [{"role": "system", "content": json_system}] + messages

        # Thinking models need more tokens (reasoning chain + JSON answer).
        _base_max = max_tokens if max_tokens is not None else self._max_completion_tokens
        effective_max_tokens = 32768 if self._is_thinking else _base_max

        for attempt in range(self._max_retries):
            try:
                if self._use_openai and not self._is_thinking:
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": effective_max_tokens,
                        "response_format": {"type": "json_object"},
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._oa_client.chat.completions.create(**kwargs)
                elif self._use_openai:
                    # Skip JSON mode for thinking models on OpenAI-compatible
                    # endpoints (e.g. DeepInfra-hosted Qwen3). The reasoning
                    # tokens can crowd out the JSON answer.
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": effective_max_tokens,
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._oa_client.chat.completions.create(**kwargs)
                elif self._is_thinking:
                    # Skip response_format for thinking models — JSON mode +
                    # thinking tokens = truncation → empty content.
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": effective_max_tokens,
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._tg_client.chat.completions.create(**kwargs)
                else:
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": effective_max_tokens,
                        "response_format": {"type": "json_object"},
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._tg_client.chat.completions.create(**kwargs)
            except Exception as api_err:
                # Together AI returns 422 "grammar is not valid" when the system
                # prompt is too long and JSON-mode grammar compilation fails.
                # Fall back to no-format mode and rely on schema-in-prompt alone.
                if (not self._use_openai) and ("422" in str(api_err) or "grammar" in str(api_err).lower()):
                    kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": effective_max_tokens,
                    }
                    if self._temperature is not None:
                        kwargs["temperature"] = self._temperature
                    response = self._tg_client.chat.completions.create(**kwargs)
                elif attempt < self._max_retries - 1 and _is_retryable_api_error(api_err):
                    time.sleep(_retry_sleep_seconds(attempt))
                    continue
                else:
                    raise
            raw = (response.choices[0].message.content or "").strip()

            # Strip <think>...</think> blocks (Qwen3 reasoning model emits these)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            try:
                data = _normalize_bounds_payload(json.loads(raw))
                # Detect schema-echoing: model returned schema definition, not values
                if "properties" in data and "title" in data and len(data) <= 5:
                    # Looks like the model returned the schema, not a filled object
                    # Inject a more explicit reminder into the next attempt
                    if attempt < self._max_retries - 1:
                        full_messages = full_messages + [{
                            "role": "assistant", "content": raw
                        }, {
                            "role": "user",
                            "content": (
                                "That is the schema definition, NOT a filled-in object. "
                                f"Return a JSON object with actual values for: {fields_desc}."
                            )
                        }]
                    continue
                return schema.model_validate(data)
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise RuntimeError(
                        f"Failed to parse LLM JSON after {self._max_retries} attempts.\n"
                        f"Raw output: {raw[:500]}\nError: {e}"
                    ) from e

        raise RuntimeError("LLM did not return valid JSON.")
