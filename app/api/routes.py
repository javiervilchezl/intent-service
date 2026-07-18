from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.config import settings
from app.providers.factory import get_provider
from app.schemas.intent import IntentRequest, IntentResponse
from app.services.intent import (
    IntentDetectionService,
    InvalidProviderResponseError,
    ProviderContextLimitError,
    ProviderRateLimitError,
)

router = APIRouter()


def get_intent_service() -> IntentDetectionService:
    return IntentDetectionService(provider=get_provider())


def verify_internal_api_key(
    api_key: str | None = Header(
        default=None,
        alias=settings.internal_api_key_header,
    ),
) -> None:
    if not settings.internal_api_key:
        return
    if api_key == settings.internal_api_key:
        return
    raise HTTPException(status_code=401, detail="Missing or invalid internal API key")


@router.post("/detect-intent", response_model=IntentResponse)
async def detect_intent(
    payload: IntentRequest,
    _: None = Depends(verify_internal_api_key),
    service: IntentDetectionService = Depends(get_intent_service),
) -> IntentResponse:
    try:
        return await service.detect(payload)
    except ProviderRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ProviderContextLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidProviderResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
