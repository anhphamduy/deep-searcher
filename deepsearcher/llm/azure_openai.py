from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class AzureOpenAI(BaseLLM):
    def __init__(
        self,
        model: str,
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        **kwargs,
    ):
        self.model = model
        import os

        from openai import AsyncAzureOpenAI, AzureOpenAI

        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_KEY")
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs,
        )

        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs,
        )

    def chat(self, messages: List[Dict], json_mode=False) -> ChatResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**kwargs)
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
            usage_metadata=completion.usage,
        )

    async def achat(self, messages: List[Dict], json_mode=False) -> ChatResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = await self.async_client.chat.completions.create(**kwargs)
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
