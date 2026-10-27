"""
LOS Data Integration API (FastAPI)

Scope:
- Fetch LOS Bureau/CIBIL, GST, Platform and BSA data.
- Validate stage inputs and source response shapes.
- Return a normalized, versioned data snapshot with clear errors.
- Accept an underwriting result and optionally POST it to a configured LOS result endpoint.

Important:
- The Underwriting Engine is not implemented here.
- No live result-update URL was provided, so result posting is configurable and disabled until LOS confirms the endpoint.
- loanProductId values are configurable; verify the product-to-ID mapping with LOS Tech Team.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
# ----------------------------- Configuration -----------------------------
# Load .env from the same directory as main.py
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

LOS_BASE_URL = os.getenv("LOS_BASE_URL", "").rstrip("/")
LOS_AUTHORIZATION_VALUE = os.getenv("LOS_AUTHORIZATION_VALUE" , "").strip()
LOS_TIMEOUT_SECONDS = float(os.getenv("LOS_TIMEOUT_SECONDS", ""))
print("Environment file exists:", env_path.exists())
print("LOS base URL:", LOS_BASE_URL)
print("Authorization loaded:", bool(LOS_AUTHORIZATION_VALUE))
# These are examples shared for the current integration; confirm product mapping.
DEFAULT_LOAN_PRODUCT_IDS = {
    "BUREAU": int(os.getenv("LOS_BUREAU_LOAN_PRODUCT_ID", "138")),
    "PLATFORM": int(os.getenv("LOS_PLATFORM_LOAN_PRODUCT_ID", "81")),
    "BSA": int(os.getenv("LOS_BSA_LOAN_PRODUCT_ID", "81")),
}


# Optional endpoint to post a result/status update back to LOS.
# Keep blank until LOS Tech Team confirms the actual URL and request schema.
LOS_RESULT_POST_URL = os.getenv("LOS_RESULT_POST_URL", "").strip()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("los-integration")

app = FastAPI(
    title="LOS Data Integration API",
    version="1.0.0",
    description=(
        "Fetches and validates LOS source data for underwriting stages. "
        "This service does not execute credit policy or make underwriting decisions."
    ),
)

# ----------------------------- Error contract -----------------------------

class APIError(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)
    retryable: bool = False

class ErrorResponse(BaseModel):
    request_id: str
    correlation_id: str
    status: str = "ERROR"
    error: APIError

def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: Optional[list[dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> JSONResponse:
    body = ErrorResponse(
        request_id=request_id or str(uuid.uuid4()),
        correlation_id=correlation_id or str(uuid.uuid4()),
        error=APIError(
            code=code,
            message=message,
            retryable=retryable,
            details=details or [],
        ),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request, exc: RequestValidationError):
    details = [
        {
            "field": ".".join(str(part) for part in item.get("loc", [])),
            "issue": item.get("msg", "Invalid value"),
        }
        for item in exc.errors()
    ]
    return error_response(
        422,
        "DATA_CONTRACT_BREACH",
        "Request data failed validation.",
        details=details,
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    # Preserve FastAPI's status but expose a consistent JSON error structure.
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code", "REQUEST_ERROR"))
        message = str(detail.get("message", "Request could not be completed."))
        retryable = bool(detail.get("retryable", False))
        details = detail.get("details", [])
    else:
        code = "REQUEST_ERROR"
        message = str(detail)
        retryable = exc.status_code in (429, 500, 502, 503, 504)
        details = []
    return error_response(exc.status_code, code, message, retryable=retryable, details=details)

# ----------------------------- Request schemas -----------------------------

class Stage(str, Enum):
    LEAD_ELIGIBILITY = "LEAD_ELIGIBILITY"
    PRODUCT_UNDERWRITING = "PRODUCT_UNDERWRITING"

class FetchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rfid: int = Field(gt=0, description="LOS application reference")
    stage: Stage
    # Optional product-specific overrides. Use only after confirming the mapping.
    loan_product_ids: dict[str, int] = Field(default_factory=dict)
    include_raw: bool = True

    @field_validator("loan_product_ids")
    @classmethod
    def validate_ids(cls, value):
        allowed = {"BUREAU", "PLATFORM", "BSA"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unsupported source key(s): {', '.join(sorted(unknown))}")
        if any(not isinstance(v, int) or v <= 0 for v in value.values()):
            raise ValueError("loan_product_ids values must be positive integers")
        return value

class SourceMetadata(BaseModel):
    source_type: str
    provider: str
    schema_version: str
    content_type: str = "application/json"
    fetched_at: str
    status: str = "FETCHED"

class UnderwritingResult(BaseModel):
    """Contract for receiving a result once the engine/result producer exists."""
    model_config = ConfigDict(extra="forbid")

    request_id: str
    rfid: int = Field(gt=0)
    stage: Stage
    status: str  # e.g. COMPLETED, PENDING, TECHNICAL_ERROR
    decision: Optional[str] = None  # e.g. PASS, FAIL, REVIEW; not computed by this service
    reason_codes: list[str] = Field(default_factory=list)
    reason_messages: list[str] = Field(default_factory=list)
    credit_limit: Optional[dict[str, Any]] = None
    result_schema_version: str = "1.0"
    evaluated_at: Optional[str] = None

    @field_validator("status")
    @classmethod
    def status_uppercase(cls, value):
        allowed = {"COMPLETED", "PENDING", "TECHNICAL_ERROR", "FAILED"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value

    @field_validator("decision")
    @classmethod
    def decision_uppercase(cls, value):
        if value is None:
            return value
        allowed = {"PASS", "FAIL", "REVIEW", "REFER", "REJECT", "PENDING", "CAP1", "CAP2", "CAP3"}
        if value not in allowed:
            raise ValueError(f"decision must be one of {sorted(allowed)}")
        return value

# ----------------------------- LOS API helpers -----------------------------

def los_headers() -> dict[str, str]:
    token = LOS_AUTHORIZATION_VALUE.strip()

    if not token:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "LOS_AUTH_NOT_CONFIGURED",
                "message": "LOS authorization is not configured.",
                "retryable": False,
            },
        )

    # Add Bearer only if it is not already present.
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"

    return {
        "Accept": "application/json",
        "Authorization": token,
    }



def fetch_json(
    url: str,
    *,
    params: Optional[dict[str, str]] = None
) -> dict[str, Any] | list[Any]:

    headers = los_headers()

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=LOS_TIMEOUT_SECONDS
        )

        logger.info(
            "LOS request: method=GET url=%s status=%s",
            response.url,
            response.status_code
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        logger.warning("LOS request timed out")
        raise HTTPException(
            status_code=504,
            detail={
                "code": "LOS_SOURCE_TIMEOUT",
                "message": "LOS did not respond before timeout.",
                "retryable": True
            }
        ) from exc

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502

        # Local debugging only: do not expose raw response data in API errors.
        logger.error(
            "LOS HTTP error: status=%s url=%s response=%s",
            status,
            response.url,
            (response.text or "")[:2000],
        )

        mapped_status = (
            status
            if status in (400, 401, 403, 404, 409, 422, 429)
            else 502
        )

        raise HTTPException(
            status_code=mapped_status,
            detail={
                "code": "LOS_SOURCE_HTTP_ERROR",
                "message": f"LOS source API returned HTTP {status}.",
                "retryable": status == 429 or status >= 500,
            },
        ) from exc

    except requests.RequestException as exc:
        logger.exception("LOS connection error")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LOS_SOURCE_UNAVAILABLE",
                "message": "Could not connect to LOS.",
                "retryable": True
            }
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LOS_INVALID_JSON",
                "message": "LOS returned invalid JSON.",
                "retryable": False
            }
        ) from exc

    if not isinstance(data, (dict, list)):
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LOS_RESPONSE_SCHEMA_INVALID",
                "message": "LOS response must be a JSON object or array.",
                "retryable": False
            }
        )

    return data


def as_object(data: Any, source: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LOS_RESPONSE_SCHEMA_INVALID",
                "message": f"{source} response must be a JSON object.",
                "retryable": False,
                "details": [{"field": source, "issue": "Expected JSON object"}],
            },
        )
    return data

def product_id(req: FetchRequest, source: str) -> int:
    return req.loan_product_ids.get(source, DEFAULT_LOAN_PRODUCT_IDS[source])

def fetch_cam_custom(
    rfid: int,
    loan_product_id: int
) -> dict[str, Any]:
    url = (
        f"https://los.glaas.co/api/application/"
        f"{rfid}/camCustom"
    )

    return as_object(
        fetch_json(
            url,
            params={"loanProductId": str(loan_product_id)},
        ),
        "camCustom",
    )

def get_bureau(rfid: int, loan_product_id: int) -> Any:
    data = fetch_cam_custom(rfid, loan_product_id)
    contacts = data.get("contact")
    if not isinstance(contacts, list) or not contacts:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "REQUIRED_SOURCE_DATA_MISSING",
                "message": "Bureau contact data is missing from the LOS response.",
                "retryable": False,
                "details": [{"field": "contact", "issue": "Expected a non-empty array"}],
            },
        )
    first_contact = contacts[0]
    if not isinstance(first_contact, dict):
        raise HTTPException(
            status_code=422,
            detail={"code": "LOS_RESPONSE_SCHEMA_INVALID",
                    "message": "contact[0] must be an object.", "retryable": False},
        )
    cic_items = first_contact.get("cic")
    if not isinstance(cic_items, list) or not cic_items or not isinstance(cic_items[0], dict):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "REQUIRED_SOURCE_DATA_MISSING",
                "message": "Bureau CIC data is missing from the LOS response.",
                "retryable": False,
                "details": [{"field": "contact[0].cic[0]", "issue": "Expected an object"}],
            },
        )
    analysed_cic = cic_items[0].get("analysedCic")
    if not isinstance(analysed_cic, dict) or not analysed_cic:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "REQUIRED_SOURCE_DATA_MISSING",
                "message": "analysedCic is missing or empty.",
                "retryable": False,
                "details": [{"field": "contact[0].cic[0].analysedCic", "issue": "Expected a non-empty object"}],
            },
        )
    return analysed_cic

def get_platform(rfid: int, loan_product_id: int) -> dict[str, Any]:
    # The provided LOS endpoint returns a camCustom object; validate that it is JSON.
    return fetch_cam_custom(rfid, loan_product_id)

def get_bsa(rfid: int, loan_product_id: int) -> list[Any] | dict[str, Any]:
    data = fetch_cam_custom(rfid, loan_product_id)
    bsa_data = data.get("bsa")
    if bsa_data is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "REQUIRED_SOURCE_DATA_MISSING",
                "message": "BSA key is missing from the LOS response.",
                "retryable": False,
                "details": [{"field": "bsa", "issue": "Key not found"}],
            },
        )
    if not isinstance(bsa_data, (dict, list)):
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LOS_RESPONSE_SCHEMA_INVALID",
                "message": "The bsa field must be a JSON object or array.",
                "retryable": False,
                "details": [{"field": "bsa", "issue": "Expected object or array"}],
            },
        )
    return bsa_data

def get_gst(rfid: int) -> dict[str, Any] | list[Any]:
    url = f"{LOS_BASE_URL}/api/gst/analysis"
    return fetch_json(url, params={"objectId": str(rfid), "objectType": "application"})

def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ----------------------------- API endpoints -----------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "los-data-integration", "version": "1.0.0"}

@app.post("/api/v1/underwriting/data-snapshots")
def fetch_stage_data(
    req: FetchRequest,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
):
    """
    Fetch LOS data only. This does not call an Underwriting Engine or make a credit decision.
    Stage 1: Bureau/CIBIL.
    Stage 2: GST, Platform, BSA.
    """
    request_id = str(uuid.uuid4())
    correlation_id = x_correlation_id or str(uuid.uuid4())
    fetched_at = now_utc()

    try:
        sources: dict[str, Any] = {}
        source_metadata: list[SourceMetadata] = []

        if req.stage == Stage.LEAD_ELIGIBILITY:
            bureau = get_bureau(req.rfid, product_id(req, "BUREAU"))
            sources["bureau"] = bureau
            source_metadata.append(SourceMetadata(
                source_type="BUREAU", provider="CIBIL", schema_version="LOS_CAMCUSTOM_V1",
                fetched_at=fetched_at,
            ))
        elif req.stage == Stage.PRODUCT_UNDERWRITING:
            gst = get_gst(req.rfid)
            platform = get_platform(req.rfid, product_id(req, "PLATFORM"))
            bsa = get_bsa(req.rfid, product_id(req, "BSA"))
            sources.update({"gst": gst, "platform": platform, "bsa": bsa})
            source_metadata.extend([
                SourceMetadata(source_type="GST", provider="LOS_GST_ANALYSIS", schema_version="LOS_GST_V1", fetched_at=fetched_at),
                SourceMetadata(source_type="PLATFORM", provider="LOS_CAMCUSTOM", schema_version="LOS_CAMCUSTOM_V1", fetched_at=fetched_at),
                SourceMetadata(source_type="BANKING", provider="LOS_BSA", schema_version="LOS_BSA_V1", fetched_at=fetched_at),
            ])

        # Build a stable envelope. Raw LOS source payloads are included only when requested.
        response_body = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "rfid": req.rfid,
            "stage": req.stage.value,
            "status": "DATA_READY",
            "schema_version": "1.0",
            "fetched_at": fetched_at,
            "sources": source_metadata,
            "data": sources if req.include_raw else {
                key: {"available": True, "top_level_type": type(value).__name__}
                for key, value in sources.items()
            },
            "engine_status": "NOT_CONNECTED",
            "message": "LOS data fetched and validated. No underwriting decision has been run.",
        }
        return response_body
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return error_response(
            exc.status_code,
            str(detail.get("code", "LOS_FETCH_FAILED")),
            str(detail.get("message", "LOS data fetch failed.")),
            retryable=bool(detail.get("retryable", False)),
            details=detail.get("details", []),
            request_id=request_id,
            correlation_id=correlation_id,
        )

@app.post("/api/v1/underwriting/results")
def receive_and_forward_result(
    result: UnderwritingResult,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-ID"),
):
    """
    Accept a result object from a future Engine/approved producer.
    Forwarding to LOS is disabled until LOS_RESULT_POST_URL is configured and confirmed.
    This endpoint validates schema; it does not calculate a decision.
    """
    correlation_id = x_correlation_id or str(uuid.uuid4())
    result_payload = result.model_dump()
    result_payload["correlation_id"] = correlation_id
    result_payload["received_at"] = now_utc()

    if not LOS_RESULT_POST_URL:
        return {
            "request_id": result.request_id,
            "correlation_id": correlation_id,
            "status": "VALIDATED_NOT_POSTED",
            "message": (
                "Result schema is valid, but it was not posted to LOS because "
                "LOS_RESULT_POST_URL is not configured. Confirm the real LOS result endpoint and payload contract."
            ),
            "result": result_payload,
        }

    try:
        response = requests.post(
            LOS_RESULT_POST_URL,
            json=result_payload,
            headers=los_headers(),
            timeout=LOS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.Timeout:
        return error_response(
            504, "LOS_RESULT_POST_TIMEOUT",
            "Timed out while posting the result to LOS.", retryable=True,
            request_id=result.request_id, correlation_id=correlation_id,
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        return error_response(
            502, "LOS_RESULT_POST_FAILED",
            f"LOS result endpoint returned HTTP {status}.",
            retryable=status == 429 or status >= 500,
            request_id=result.request_id, correlation_id=correlation_id,
        )
    except requests.RequestException:
        return error_response(
            502, "LOS_RESULT_POST_UNAVAILABLE",
            "Could not connect to the LOS result endpoint.", retryable=True,
            request_id=result.request_id, correlation_id=correlation_id,
        )

    return {
        "request_id": result.request_id,
        "correlation_id": correlation_id,
        "status": "POSTED_TO_LOS",
        "los_http_status": response.status_code,
        "message": "Result was sent to the configured LOS endpoint.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )