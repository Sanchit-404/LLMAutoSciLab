"""
EnsembleClient: samples K hypotheses from a local vLLM server in parallel.

Uses the OpenAI-compatible API exposed by vLLM (http://localhost:8001/v1).
Each call sends K requests concurrently with temperature sampling to generate
structurally diverse hypotheses for the MEI pipeline.

Usage
-----
    client = EnsembleClient(base_url="http://localhost:8001/v1", k=5)
    results = client.complete_json_k(
        messages=[{"role": "user", "content": prompt}],
        system=SYSTEM_PROMPT,
        schema=HypothesisBundle,
    )
    # → list of up to K HypothesisBundle objects
"""
import asyncio
import json
import os
import re
import time
from typing import Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from autoscilab.llm.client import _strip_descriptions, _schema_to_prompt

DEFAULT_VLLM_URL = "http://localhost:8001/v1"
DEFAULT_SMALL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _resolve_api_key(base_url: str, api_key: str | None) -> str:
    if api_key:
        return api_key
    url = (base_url or "").lower()
    if "deepinfra.com" in url:
        return (
            os.environ.get("DEEPINFRA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
    return "EMPTY"


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort extraction of a single JSON object from model output."""
    text = raw.strip()
    if not text:
        return None

    candidates: list[str] = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first:last + 1])

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


class EnsembleClient:
    """
    Wraps a local vLLM OpenAI-compatible server and samples K completions
    in parallel via asyncio.gather.

    Parameters
    ----------
    base_url : str
        Base URL of the vLLM server (e.g. "http://localhost:8001/v1").
    model : str
        Model name as registered in vLLM (must match --model flag at launch).
    k : int
        Number of samples per ensemble call.
    temperature : float
        Sampling temperature — higher means more structural diversity.
        0.9 is a good default; go up to 1.2 for more exploration.
    max_tokens : int
        Max tokens per completion.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_VLLM_URL,
        model: str = DEFAULT_SMALL_MODEL,
        k: int = 5,
        temperature: float = 0.9,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        self._model = model
        self._k = k
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=_resolve_api_key(base_url, api_key),
        )

    # ------------------------------------------------------------------ #
    #  Internal async helpers                                              #
    # ------------------------------------------------------------------ #

    async def _complete_one(
        self,
        messages: list[dict],
        temperature: float,
        max_retries: int = 2,
    ) -> str:
        """Single async chat completion with retry — returns raw text or empty string."""
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=temperature,
                )
                raw = response.choices[0].message.content or ""
                return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            except Exception as exc:
                if attempt < max_retries:
                    await asyncio.sleep(1.5 ** attempt)  # 1s, 1.5s backoff
                else:
                    # Final attempt failed — return empty string so gather succeeds
                    return ""

    async def _complete_k_async(
        self,
        messages: list[dict],
        temperature: float,
        k: int | None = None,
    ) -> list[str]:
        """Fire K requests concurrently; individual failures return empty string."""
        n = k if k is not None else self._k
        tasks = [self._complete_one(messages, temperature) for _ in range(n)]
        # return_exceptions=True so one failure doesn't cancel all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Convert exceptions to empty strings
        return [r if isinstance(r, str) else "" for r in results]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def complete_k(
        self,
        messages: list[dict],
        system: str,
        temperature: float | None = None,
    ) -> list[str]:
        """
        Sample K raw text completions from the ensemble in parallel.

        Parameters
        ----------
        messages : list[dict]
            User/assistant turn messages (without system).
        system : str
            System prompt injected as the first message.
        temperature : float | None
            Override instance temperature for this call.

        Returns
        -------
        list[str]
            Up to K raw text completions (may be shorter if some fail).
        """
        t = temperature if temperature is not None else self._temperature
        full_messages = [{"role": "system", "content": system}] + messages
        return asyncio.run(self._complete_k_async(full_messages, t))

    def complete_json_k(
        self,
        messages: list[dict],
        system: str,
        schema: Type[BaseModel],
        temperature: float | None = None,
        min_valid: int = 1,
        k_override: int | None = None,
    ) -> list[BaseModel]:
        """
        Sample K structured JSON completions in parallel.

        Injects the JSON schema into the system prompt (same approach as
        LLMClient.complete_json), parses each response, and returns all
        successfully parsed objects. Failed parses are silently dropped.

        Parameters
        ----------
        messages : list[dict]
            User/assistant turn messages.
        system : str
            Base system prompt (schema will be appended).
        schema : Type[BaseModel]
            Pydantic model to validate against.
        temperature : float | None
            Override instance temperature.
        min_valid : int
            Raise RuntimeError if fewer than min_valid responses parse correctly.

        Returns
        -------
        list[BaseModel]
            List of successfully parsed Pydantic objects (length ≤ K).
        """
        schema_dict = schema.model_json_schema()
        schema_str = _schema_to_prompt(schema_dict)
        fields_desc = ", ".join(f'"{k}"' for k in schema_dict.get("properties", {}))

        json_system = (
            f"{system}\n\n"
            f"You MUST respond with a single valid JSON object with these fields: "
            f"{fields_desc}. Fill every field with actual content — do NOT return "
            f"the schema itself.\n\nSCHEMA:\n{schema_str}"
        )

        t = temperature if temperature is not None else self._temperature
        full_messages = [{"role": "system", "content": json_system}] + messages

        raw_outputs: list[str] = asyncio.run(
            self._complete_k_async(full_messages, t, k=k_override)
        )

        results: list[BaseModel] = []
        for raw in raw_outputs:
            raw = raw.strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                data = _extract_json_object(raw)
                if data is None:
                    raise ValueError("no JSON object found")
                # Reject schema echoes (model returned schema definition, not values)
                if "properties" in data and "title" in data and len(data) <= 5:
                    continue
                results.append(schema.model_validate(data))
            except Exception:
                pass  # silently drop malformed responses

        n_requested = k_override if k_override is not None else self._k
        if len(results) < min_valid:
            raise RuntimeError(
                f"EnsembleClient: only {len(results)}/{n_requested} responses parsed "
                f"successfully (need at least {min_valid})."
            )

        return results
