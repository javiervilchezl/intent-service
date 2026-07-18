import json
import re
from collections import defaultdict
from typing import Any

from app.core.config import settings
from app.providers.base import LLMProvider
from app.schemas.intent import IntentRequest, IntentResponse


class InvalidProviderResponseError(ValueError):
    pass


class ProviderRateLimitError(ValueError):
    pass


class ProviderContextLimitError(ValueError):
    pass


class IntentDetectionService:
    ALLOWED_INTENTS = {
        "book_flight",
        "customer_support",
        "refund_request",
        "request_refund",
        "document_review",
        "maintenance_report",
        "technical_incident",
        "preventive_maintenance",
        "corrective_maintenance",
        "unknown",
    }

    INTENT_ALIASES = {
        "customer support": "customer_support",
        "support_request": "customer_support",
        "atencion al cliente": "customer_support",
        "solicitud_soporte": "customer_support",
        "request refund": "request_refund",
        "solicitud_reembolso": "request_refund",
        "document review": "document_review",
        "revision_documental": "document_review",
        "documento_tecnico": "document_review",
        "maintenance": "maintenance_report",
        "maintenance report": "maintenance_report",
        "reporte_mantenimiento": "maintenance_report",
        "incident": "technical_incident",
        "incidente_tecnico": "technical_incident",
        "preventive maintenance": "preventive_maintenance",
        "mantenimiento_preventivo": "preventive_maintenance",
        "corrective maintenance": "corrective_maintenance",
        "mantenimiento_correctivo": "corrective_maintenance",
    }

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def detect(self, payload: IntentRequest) -> IntentResponse:
        text = self._prepare_text(payload.text)
        chunks = self._chunk_text(text)
        intents = defaultdict(int)
        merged_entities: dict[str, Any] = {}

        for chunk in chunks:
            parsed = await self._detect_chunk(chunk)
            intent = self._normalize_intent(parsed.get("intent"), chunk)
            intents[intent] += 1
            merged_entities = self._merge_entities(
                merged_entities,
                parsed.get("entities", {}),
            )

        detected_intent = "unknown"
        if intents:
            detected_intent = max(intents.items(), key=lambda item: item[1])[0]

        return IntentResponse(intent=detected_intent, entities=merged_entities)

    async def _detect_chunk(self, text: str) -> dict:
        allowed = ", ".join(sorted(self.ALLOWED_INTENTS))
        prompt = (
            "Detect the user intent and entities from operational text. "
            "Return valid JSON with this exact shape: "
            '{"intent":"...","entities":{}}.\n'
            "Use one intent from this list only: "
            f"{allowed}.\n"
            "If the text is unclear, use unknown.\n\n"
            f"Text:\n{text}"
        )
        try:
            content = await self.provider.generate(
                prompt=prompt,
                system_prompt=(
                    "You are an intent detection engine. "
                    "Respond only with valid JSON and never add prose."
                ),
            )
        except Exception as exc:
            self._raise_provider_limit_error(exc)
            raise

        return self._parse_provider_response(content)

    def _chunk_text(self, text: str) -> list[str]:
        max_tokens = max(1, settings.llm_max_input_tokens)
        overlap_tokens = max(0, settings.llm_overlap_tokens)
        chars_per_token = max(1, settings.llm_chars_per_token)
        max_chunks = max(1, settings.llm_max_chunks)

        chunk_chars = max_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token
        overlap_chars = min(overlap_chars, chunk_chars - 1)

        chunks = []
        start = 0
        while start < len(text):
            if len(chunks) >= max_chunks:
                raise ProviderContextLimitError(
                    "Input too large for configured token budget. "
                    "Increase LLM_MAX_INPUT_TOKENS or LLM_MAX_CHUNKS."
                )

            end = min(len(text), start + chunk_chars)
            chunks.append(text[start:end])
            if end >= len(text):
                break

            next_start = max(0, end - overlap_chars)
            start = next_start

        return chunks or [text]

    def _raise_provider_limit_error(self, exc: Exception) -> None:
        message = str(exc).lower()
        if (
            "rate_limit" in message
            or "too many requests" in message
            or "tokens per minute" in message
        ):
            raise ProviderRateLimitError(
                "Rate limit exceeded in LLM provider. "
                "Try again later or reduce LLM_MAX_INPUT_TOKENS."
            ) from exc

        if (
            "context_length_exceeded" in message
            or "request too large" in message
            or "please reduce the length" in message
        ):
            raise ProviderContextLimitError(
                "LLM context limit exceeded. "
                "Reduce LLM_MAX_INPUT_TOKENS or upgrade model limits."
            ) from exc

        raise

    def _merge_entities(self, left: dict[str, Any], right: Any) -> dict[str, Any]:
        if not isinstance(right, dict):
            return left

        merged = dict(left)
        for key, value in right.items():
            if key not in merged or merged[key] in (None, "", []):
                merged[key] = value
                continue

            existing = merged[key]
            if existing == value:
                continue

            if isinstance(existing, list):
                items = existing + (value if isinstance(value, list) else [value])
                merged[key] = self._dedupe_list(items)
                continue

            if isinstance(value, list):
                merged[key] = self._dedupe_list([existing] + value)
                continue

            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = self._merge_entities(existing, value)
                continue

            merged[key] = self._dedupe_list([existing, value])

        return merged

    def _dedupe_list(self, values: list[Any]) -> list[Any]:
        unique = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique

    def _prepare_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_intent(self, raw_intent: Any, text: str) -> str:
        value = str(raw_intent or "unknown").strip().lower().replace("-", "_")
        value = re.sub(r"\s+", " ", value)

        if value in self.INTENT_ALIASES:
            value = self.INTENT_ALIASES[value]

        if value not in self.ALLOWED_INTENTS:
            value = "unknown"

        if value == "unknown":
            inferred = self._infer_intent_from_keywords(text)
            if inferred:
                return inferred

        return value

    def _infer_intent_from_keywords(self, text: str) -> str | None:
        normalized = text.lower()

        maintenance_terms = [
            "mantenimiento",
            "reparacion",
            "sustitucion",
            "bomba",
            "circuito",
            "neumatic",
            "sala blanca",
            "seguridad de puertas",
            "ajuste de seguridad",
        ]
        incident_terms = [
            "falla",
            "averia",
            "incidencia",
            "error",
            "alarma",
            "riesgo",
        ]
        preventive_terms = [
            "preventivo",
            "inspeccion",
            "calibracion",
        ]
        corrective_terms = [
            "correctivo",
            "reemplazo",
            "sustitu",
            "repar",
        ]

        has_maintenance = any(term in normalized for term in maintenance_terms)
        has_incident = any(term in normalized for term in incident_terms)
        has_preventive = any(term in normalized for term in preventive_terms)
        has_corrective = any(term in normalized for term in corrective_terms)

        if has_incident and has_maintenance:
            return "technical_incident"
        if has_preventive and has_maintenance:
            return "preventive_maintenance"
        if has_corrective and has_maintenance:
            return "corrective_maintenance"
        if has_maintenance:
            return "maintenance_report"

        return None

    def _parse_provider_response(self, content: str) -> dict:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        raise InvalidProviderResponseError(
            "AI provider returned an invalid response for intent detection"
        )
