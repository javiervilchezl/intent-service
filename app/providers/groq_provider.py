from importlib import import_module

from app.core.config import settings
from app.providers.base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self) -> None:
        groq_module = import_module("groq")
        self.client = groq_module.AsyncGroq(api_key=settings.groq_api_key)

    async def generate(self, prompt: str, system_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
