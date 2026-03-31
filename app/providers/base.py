from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str) -> str:
        raise NotImplementedError
