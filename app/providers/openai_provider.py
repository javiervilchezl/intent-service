from openai import AsyncOpenAI

from app.core.config import settings
from app.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, prompt: str, system_prompt: str) -> str:
        response = await self.client.responses.create(
            model=settings.openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text
