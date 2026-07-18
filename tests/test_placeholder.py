from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app
import app.api
import app.core
import app.providers
import app.schemas
import app.services
from app.api.routes import get_intent_service, verify_internal_api_key
from app.core.config import settings
from app.main import app as fastapi_app
from app.providers.base import LLMProvider
from app.providers.factory import get_provider
from app.providers.groq_provider import GroqProvider
from app.providers.openai_provider import OpenAIProvider
from app.schemas.intent import IntentRequest
from app.services.intent import (
    IntentDetectionService,
    InvalidProviderResponseError,
    ProviderContextLimitError,
    ProviderRateLimitError,
)


class DummyProvider(LLMProvider):
    async def generate(self, prompt: str, system_prompt: str) -> str:
        return await super().generate(prompt, system_prompt)


class StubService:
    async def detect(self, payload: IntentRequest):
        return {
            "intent": "book_flight",
            "entities": {"destination": payload.text},
        }


class InvalidResponseStubService:
    async def detect(self, payload: IntentRequest):
        raise InvalidProviderResponseError(
            "AI provider returned an invalid response for intent detection"
        )


class FailingStubService:
    async def detect(self, payload: IntentRequest):
        raise RuntimeError("provider boom")


class RateLimitStubService:
    async def detect(self, payload: IntentRequest):
        raise ProviderRateLimitError("rate")


class ContextLimitStubService:
    async def detect(self, payload: IntentRequest):
        raise ProviderContextLimitError("context")


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    fastapi_app.dependency_overrides.clear()
    yield
    fastapi_app.dependency_overrides.clear()


def test_package_exports_are_importable() -> None:
    assert app.__all__ == []
    assert app.api.__all__ == []
    assert app.core.__all__ == []
    assert app.providers.__all__ == []
    assert app.schemas.__all__ == []
    assert app.services.__all__ == []


def _internal_headers() -> dict[str, str]:
    if not settings.internal_api_key:
        return {}
    return {settings.internal_api_key_header: settings.internal_api_key}


@pytest.mark.asyncio
async def test_base_provider_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await DummyProvider().generate("prompt", "system")


def test_get_provider_returns_openai(monkeypatch) -> None:
    from app.providers import factory

    monkeypatch.setattr(factory.settings, "provider", "openai")
    monkeypatch.setattr(factory, "OpenAIProvider", lambda: "openai")
    assert get_provider() == "openai"


def test_get_provider_returns_groq(monkeypatch) -> None:
    from app.providers import factory

    monkeypatch.setattr(factory.settings, "provider", "groq")
    monkeypatch.setattr(factory, "GroqProvider", lambda: "groq")
    assert get_provider() == "groq"


def test_get_intent_service(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.get_provider",
        lambda: "provider-instance",
    )
    service = get_intent_service()
    assert service.provider == "provider-instance"


@pytest.mark.asyncio
async def test_openai_provider_generate(monkeypatch) -> None:
    class FakeResponses:
        async def create(self, **kwargs):
            return SimpleNamespace(output_text="intent-openai")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(
        "app.providers.openai_provider.AsyncOpenAI",
        FakeClient,
    )
    provider = OpenAIProvider()
    assert await provider.generate("hello", "system") == "intent-openai"


@pytest.mark.asyncio
async def test_groq_provider_generate(monkeypatch) -> None:
    class FakeCompletions:
        async def create(self, **kwargs):
            message = SimpleNamespace(content="intent-groq")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_module = SimpleNamespace(AsyncGroq=FakeClient)
    monkeypatch.setattr(
        "app.providers.groq_provider.import_module",
        lambda _: fake_module,
    )
    provider = GroqProvider()
    assert await provider.generate("hello", "system") == "intent-groq"


@pytest.mark.asyncio
async def test_detect_uses_provider() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            assert "flight" in prompt
            assert "intent detection engine" in system_prompt
            return (
                '{"intent":"book_flight",'
                '"entities":{"destination":"Madrid"}}'
            )

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="flight to Madrid"))
    assert result.intent == "book_flight"
    assert result.entities == {"destination": "Madrid"}


@pytest.mark.asyncio
async def test_detect_allows_complex_entities() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return (
                '{"intent":"refund_request",'
                '"entities":{"refund_reason":["damaged","late"],'
                '"refund_amount":null,'
                '"order":{"id":"123","priority":true}}}'
            )

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="refund request"))

    assert result.intent == "refund_request"
    assert result.entities["refund_reason"] == ["damaged", "late"]
    assert result.entities["refund_amount"] is None
    assert result.entities["order"] == {"id": "123", "priority": True}


@pytest.mark.asyncio
async def test_detect_fallbacks_to_unknown_when_intent_missing() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return '{"entities":{"destination":"Madrid"}}'

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="flight to Madrid"))

    assert result.intent == "unknown"
    assert result.entities == {"destination": "Madrid"}


@pytest.mark.asyncio
async def test_detect_normalizes_alias_intent() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return '{"intent":"customer support","entities":{}}'

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="necesito ayuda con mi pedido"))

    assert result.intent == "customer_support"


@pytest.mark.asyncio
async def test_detect_infers_maintenance_intent_when_provider_returns_unknown() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return '{"intent":"unknown","entities":{}}'

    service = IntentDetectionService(provider=FakeProvider())
    text = (
        "Se realizaron reparaciones de bombas, sustitucion de componentes, "
        "ajustes de seguridad y limpieza de circuitos neumaticos."
    )
    result = await service.detect(IntentRequest(text=text))

    assert result.intent == "corrective_maintenance"


@pytest.mark.asyncio
async def test_detect_splits_long_text_into_chunks() -> None:
    captured = {}
    calls = {"count": 0}

    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            calls["count"] += 1
            captured.setdefault("prompts", []).append(prompt)
            return '{"intent":"book_flight","entities":{}}'

    service = IntentDetectionService(provider=FakeProvider())
    long_text = ("flight to madrid " * 6000) + "tail"

    result = await service.detect(IntentRequest(text=long_text))

    assert result.intent == "book_flight"
    assert calls["count"] > 1
    assert any("tail" in prompt for prompt in captured["prompts"])


@pytest.mark.asyncio
async def test_detect_accepts_fenced_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return '```json\n{"intent":"book_flight","entities":{}}\n```'

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="flight to Madrid"))

    assert result.intent == "book_flight"
    assert result.entities == {}


@pytest.mark.asyncio
async def test_detect_rejects_invalid_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return "not-json"

    service = IntentDetectionService(provider=FakeProvider())

    with pytest.raises(InvalidProviderResponseError):
        await service.detect(IntentRequest(text="flight to Madrid"))


@pytest.mark.asyncio
async def test_detect_accepts_embedded_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return 'intent:{"intent":"book_flight","entities":{}}'

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="flight to Madrid"))

    assert result.intent == "book_flight"
    assert result.entities == {}


@pytest.mark.asyncio
async def test_detect_rejects_invalid_embedded_json() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return 'intent:{"intent":bad-json}'

    service = IntentDetectionService(provider=FakeProvider())

    with pytest.raises(InvalidProviderResponseError):
        await service.detect(IntentRequest(text="flight to Madrid"))


@pytest.mark.asyncio
async def test_detect_accepts_fenced_json_without_language_tag() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            return '```\n{"intent":"book_flight","entities":{}}\n```'

    service = IntentDetectionService(provider=FakeProvider())
    result = await service.detect(IntentRequest(text="flight to Madrid"))

    assert result.intent == "book_flight"
    assert result.entities == {}


@pytest.mark.asyncio
async def test_detect_maps_provider_rate_limit_error() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            raise RuntimeError("tokens per minute exceeded")

    service = IntentDetectionService(provider=FakeProvider())

    with pytest.raises(ProviderRateLimitError):
        await service.detect(IntentRequest(text="refund"))


@pytest.mark.asyncio
async def test_detect_maps_provider_context_limit_error() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            raise RuntimeError("context_length_exceeded")

    service = IntentDetectionService(provider=FakeProvider())

    with pytest.raises(ProviderContextLimitError):
        await service.detect(IntentRequest(text="refund"))


@pytest.mark.asyncio
async def test_detect_reraises_unknown_provider_error() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            raise RuntimeError("boom")

    service = IntentDetectionService(provider=FakeProvider())

    with pytest.raises(RuntimeError, match="boom"):
        await service.detect(IntentRequest(text="refund"))


@pytest.mark.asyncio
async def test_detect_chunk_reraises_original_when_mapper_does_not_raise(
    monkeypatch,
) -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            raise RuntimeError("boom")

    service = IntentDetectionService(provider=FakeProvider())
    monkeypatch.setattr(service, "_raise_provider_limit_error", lambda exc: None)

    with pytest.raises(RuntimeError, match="boom"):
        await service._detect_chunk("refund")


@pytest.mark.asyncio
async def test_detect_maps_provider_rate_limit_keyword_error() -> None:
    class FakeProvider:
        async def generate(self, prompt: str, system_prompt: str) -> str:
            raise RuntimeError("rate_limit")

    service = IntentDetectionService(provider=FakeProvider())

    with pytest.raises(ProviderRateLimitError):
        await service.detect(IntentRequest(text="refund"))


def test_prepare_text_compacts_whitespace() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    cleaned = service._prepare_text("  flight\n\n to\tMadrid  ")
    assert cleaned == "flight to Madrid"


def test_chunk_text_raises_when_input_is_too_large_for_max_chunks(monkeypatch) -> None:
    monkeypatch.setattr("app.services.intent.settings.llm_max_input_tokens", 1)
    monkeypatch.setattr("app.services.intent.settings.llm_chars_per_token", 1)
    monkeypatch.setattr("app.services.intent.settings.llm_overlap_tokens", 0)
    monkeypatch.setattr("app.services.intent.settings.llm_max_chunks", 2)

    service = IntentDetectionService(provider=SimpleNamespace())
    with pytest.raises(ProviderContextLimitError):
        service._chunk_text("x" * 10)


def test_chunk_text_returns_empty_chunk_for_empty_input() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    assert service._chunk_text("") == [""]


def test_merge_entities_non_dict_right_keeps_left() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    left = {"destination": "Madrid"}
    assert service._merge_entities(left, "not-a-dict") == left


def test_merge_entities_covers_all_merge_paths() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    left = {
        "empty": "",
        "same": "A",
        "list_existing": ["x"],
        "list_incoming": "z",
        "dict": {"a": 1},
        "scalar": "base",
    }
    right = {
        "empty": "filled",
        "same": "A",
        "list_existing": "x",
        "list_incoming": ["z", "y"],
        "dict": {"b": 2},
        "scalar": "new",
    }

    merged = service._merge_entities(left, right)

    assert merged["empty"] == "filled"
    assert merged["same"] == "A"
    assert merged["list_existing"] == ["x"]
    assert merged["list_incoming"] == ["z", "y"]
    assert merged["dict"] == {"a": 1, "b": 2}
    assert merged["scalar"] == ["base", "new"]


def test_raise_provider_limit_error_handles_too_many_requests() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    with pytest.raises(ProviderRateLimitError):
        service._raise_provider_limit_error(RuntimeError("too many requests"))


def test_infer_intent_returns_incident_for_failure_signals() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    text = "Reporte de mantenimiento con falla critica en bomba principal"
    assert service._infer_intent_from_keywords(text) == "technical_incident"


def test_infer_intent_returns_corrective_for_replacement_signals() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    text = "Mantenimiento correctivo por reemplazo de valvula y reparacion"
    assert service._infer_intent_from_keywords(text) == "corrective_maintenance"


def test_infer_intent_returns_preventive_for_inspection_signals() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    text = "Mantenimiento preventivo e inspeccion trimestral del circuito"
    assert service._infer_intent_from_keywords(text) == "preventive_maintenance"


def test_infer_intent_returns_maintenance_report_for_generic_signals() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    text = "Mantenimiento de bombas y circuito neumatico en sala blanca"
    assert service._infer_intent_from_keywords(text) == "maintenance_report"


def test_normalize_intent_falls_back_to_unknown_for_unlisted_intent() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    assert service._normalize_intent("something_else", "texto neutro") == "unknown"


def test_infer_intent_returns_none_without_keywords() -> None:
    service = IntentDetectionService(provider=SimpleNamespace())
    assert service._infer_intent_from_keywords("hello world") is None


def test_health_endpoint() -> None:
    client = TestClient(fastapi_app)
    response = client.get("/health")
    assert response.status_code == 200


def test_detect_intent_endpoint() -> None:
    fastapi_app.dependency_overrides[get_intent_service] = StubService
    client = TestClient(fastapi_app)
    response = client.post(
        "/detect-intent",
        json={"text": "Madrid"},
        headers=_internal_headers(),
    )
    assert response.status_code == 200
    assert response.json() == {
        "intent": "book_flight",
        "entities": {"destination": "Madrid"},
    }


def test_detect_intent_endpoint_with_complex_entities() -> None:
    class ComplexStubService:
        async def detect(self, payload: IntentRequest):
            return {
                "intent": "refund_request",
                "entities": {
                    "reasons": ["damaged", "late"],
                    "order": {"id": "123"},
                    "refund_amount": None,
                },
            }

    fastapi_app.dependency_overrides[get_intent_service] = ComplexStubService
    client = TestClient(fastapi_app)
    response = client.post(
        "/detect-intent",
        json={"text": "refund"},
        headers=_internal_headers(),
    )
    assert response.status_code == 200
    assert response.json()["entities"]["reasons"] == ["damaged", "late"]
    assert response.json()["entities"]["order"] == {"id": "123"}
    assert response.json()["entities"]["refund_amount"] is None


def test_detect_intent_endpoint_handles_invalid_provider_response() -> None:
    fastapi_app.dependency_overrides[get_intent_service] = (
        InvalidResponseStubService
    )
    client = TestClient(fastapi_app)
    response = client.post(
        "/detect-intent",
        json={"text": "refund"},
        headers=_internal_headers(),
    )
    assert response.status_code == 502


def test_detect_intent_endpoint_handles_rate_limit() -> None:
    fastapi_app.dependency_overrides[get_intent_service] = RateLimitStubService
    client = TestClient(fastapi_app)
    response = client.post(
        "/detect-intent",
        json={"text": "refund"},
        headers=_internal_headers(),
    )
    assert response.status_code == 429


def test_detect_intent_endpoint_handles_context_limit() -> None:
    fastapi_app.dependency_overrides[get_intent_service] = ContextLimitStubService
    client = TestClient(fastapi_app)
    response = client.post(
        "/detect-intent",
        json={"text": "refund"},
        headers=_internal_headers(),
    )
    assert response.status_code == 413


def test_detect_intent_endpoint_handles_unexpected_error() -> None:
    fastapi_app.dependency_overrides[get_intent_service] = FailingStubService
    client = TestClient(fastapi_app)
    response = client.post(
        "/detect-intent",
        json={"text": "refund"},
        headers=_internal_headers(),
    )
    assert response.status_code == 502


def test_detect_intent_endpoint_requires_internal_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", "svc-key")
    monkeypatch.setattr(settings, "internal_api_key_header", "X-Service-API-Key")
    fastapi_app.dependency_overrides[get_intent_service] = StubService
    client = TestClient(fastapi_app)

    response = client.post(
        "/detect-intent",
        json={"text": "Madrid"},
    )
    assert response.status_code == 401


def test_detect_intent_endpoint_accepts_internal_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", "svc-key")
    monkeypatch.setattr(settings, "internal_api_key_header", "X-Service-API-Key")
    fastapi_app.dependency_overrides[get_intent_service] = StubService
    client = TestClient(fastapi_app)

    response = client.post(
        "/detect-intent",
        json={"text": "Madrid"},
        headers={"X-Service-API-Key": "svc-key"},
    )
    assert response.status_code == 200


def test_verify_internal_api_key_accepts_matching_value(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", "svc-key")
    verify_internal_api_key(api_key="svc-key")


def test_verify_internal_api_key_returns_when_key_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "internal_api_key", "")
    verify_internal_api_key(api_key=None)
