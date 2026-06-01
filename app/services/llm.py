"""LLM service — Azure OpenAI GPT-4o client wrapper with warmup."""
from __future__ import annotations
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self._client = None

    async def warmup(self):
        try:
            from langchain_openai import AzureChatOpenAI
            self._client = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                temperature=0.1,
                max_tokens=1500,
            )
            logger.info("✅ Azure OpenAI GPT-4o client initialized")
        except Exception as e:
            logger.warning("LLM warmup failed: %s", e)

    @property
    def client(self):
        return self._client

    async def close(self):
        pass
