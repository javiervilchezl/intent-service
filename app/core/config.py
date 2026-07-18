from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "intent-service"
    app_env: str = "development"
    log_level: str = "INFO"
    provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    llm_max_input_tokens: int = 1200
    llm_overlap_tokens: int = 80
    llm_chars_per_token: int = 4
    llm_max_chunks: int = 64
    internal_api_key: str = ""
    internal_api_key_header: str = "X-Service-API-Key"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
