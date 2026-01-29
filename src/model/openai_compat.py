from __future__ import annotations

from typing import Dict, List, Optional


class OpenAICompatChatWrapper:
    """OpenAI-compatible chat wrapper.

    This is intended for *local* OpenAI-compatible servers such as vLLM's
    OpenAI API server ("vllm serve ..."), so you can run open-source models
    without loading them in this process.

    The rest of the codebase expects an "agent" object.
    For this wrapper we expose:
      - kind = "openai_compat"
      - model_name: str
      - complete(messages, ...) -> str
    """

    kind = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        try:
            # openai>=1.0 provides OpenAI client
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "OpenAICompatChatWrapper requires the official OpenAI Python SDK. "
                "Install with: pip install 'openai>=1.0.0'"
            ) from e

        # vLLM OpenAI server uses the same endpoints as OpenAI.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
