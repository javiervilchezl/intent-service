from app.core.config import settings
from app.providers.base import LLMProvider
from app.providers.groq_provider import GroqProvider
from app.providers.openai_provider import OpenAIProvider


def get_provider() -> LLMProvider:
    if settings.provider.lower() == "groq":
        return GroqProvider()
    return OpenAIProvider()
