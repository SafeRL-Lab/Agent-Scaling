import os
from typing import Dict, List, Optional


class AzureOpenAIWrapper:
    """Azure OpenAI chat wrapper.

    The rest of this codebase expects an "agent" object. Local agents expose
    `huggingface_model` + `tokenizer`. For API agents we expose:

      - kind = "azure_openai"
      - model_name: str
      - complete(messages: List[Dict[str, str]]) -> str

    This keeps the integration minimal and allows heterogeneous agent lists.
    """

    kind = "azure_openai"

    def __init__(
        self,
        model_name: str,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2025-04-01-preview",
        timeout: Optional[float] = None,
    ):
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.timeout = timeout

        try:
            # openai>=1.0 provides AzureOpenAI
            from openai import AzureOpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "AzureOpenAIWrapper requires the official OpenAI Python SDK. "
                "Install with: pip install 'openai>=1.0.0'"
            ) from e

        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Call Azure Chat Completions.

        `messages` follows the OpenAI chat format:
          [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
        """

        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
