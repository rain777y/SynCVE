"""
Pydantic request validation models for all SynCVE backend API endpoints.
"""

import json
import uuid as _uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"age", "gender", "emotion", "race"}

class DetectorBackend(str, Enum):
    opencv = "opencv"
    ssd = "ssd"
    dlib = "dlib"
    mtcnn = "mtcnn"
    retinaface = "retinaface"
    mediapipe = "mediapipe"
    yolov8 = "yolov8"
    yunet = "yunet"
    fastmtcnn = "fastmtcnn"
    centerface = "centerface"


class AspectRatio(str, Enum):
    wide = "16:9"
    standard = "4:3"
    square = "1:1"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Validate /analyze POST payload."""

    img: str = Field(..., min_length=1, description="Base64 image, file path, or URL")
    actions: List[str] = Field(
        default=["age", "gender", "emotion", "race"],
        description="Analysis actions to perform",
    )
    detector_backend: Optional[str] = Field(
        default=None,
        description="Detector backend name (e.g. retinaface, mtcnn)",
    )
    anti_spoofing: Optional[bool] = Field(default=None)
    align: Optional[bool] = Field(default=None)
    enforce_detection: Optional[bool] = Field(default=None)
    enable_ensemble: Optional[bool] = Field(default=None)
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_faces: Optional[int] = Field(default=None, ge=1)
    session_id: Optional[str] = Field(default=None)

    @field_validator("actions", mode="before")
    @classmethod
    def parse_and_validate_actions(cls, v: Any) -> List[str]:
        """Accept JSON string, list, or comma-separated string and whitelist."""
        if isinstance(v, str):
            # Try JSON first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    v = parsed
                else:
                    v = [str(parsed)]
            except (json.JSONDecodeError, ValueError):
                # Fallback: strip brackets/quotes and split on comma
                cleaned = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                cleaned = cleaned.replace('"', "").replace("'", "").replace(" ", "")
                v = [item for item in cleaned.split(",") if item]

        if not isinstance(v, list):
            v = [str(v)]

        validated = []
        for action in v:
            action_lower = str(action).strip().lower()
            if action_lower not in VALID_ACTIONS:
                raise ValueError(
                    f"Invalid action '{action_lower}'. Allowed: {sorted(VALID_ACTIONS)}"
                )
            validated.append(action_lower)

        if not validated:
            raise ValueError("At least one action is required")

        return validated

    @field_validator("detector_backend", mode="before")
    @classmethod
    def validate_detector_backend(cls, v: Any) -> Optional[str]:
        if v is None:
            return v
        val = str(v).strip().lower()
        valid_backends = {e.value for e in DetectorBackend}
        if val not in valid_backends:
            raise ValueError(
                f"Invalid detector_backend '{val}'. Allowed: {sorted(valid_backends)}"
            )
        return val


class SessionStartRequest(BaseModel):
    """Validate /session/start POST payload."""

    user_id: Optional[str] = Field(default=None, max_length=255)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("metadata", mode="before")
    @classmethod
    def limit_metadata_size(cls, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        serialized = json.dumps(v)
        max_bytes = 16 * 1024  # 16 KB
        if len(serialized.encode("utf-8")) > max_bytes:
            raise ValueError(f"metadata exceeds maximum size of {max_bytes} bytes")
        return v


class SessionStopRequest(BaseModel):
    """Validate /session/stop POST payload."""

    session_id: str = Field(..., min_length=1)

    @field_validator("session_id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        try:
            _uuid.UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID")
        return v


class ReportRequest(BaseModel):
    """Validate /session/report/emotion POST payload."""

    session_id: str = Field(..., min_length=1)
    raw_vision_data: Optional[List[Dict[str, Any]]] = Field(default=None)
    max_keyframes: int = Field(default=4, ge=1, le=10)

    @field_validator("session_id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        try:
            _uuid.UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID")
        return v


class VisualReportRequest(BaseModel):
    """Validate /session/report/visual POST payload."""

    session_id: str = Field(..., min_length=1)
    raw_vision_data: Optional[List[Dict[str, Any]]] = Field(default=None)
    aspect_ratio: Optional[AspectRatio] = Field(default=None)
    style_preset: Optional[str] = Field(default=None, max_length=100)

    @field_validator("session_id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        try:
            _uuid.UUID(v)
        except ValueError:
            raise ValueError("session_id must be a valid UUID")
        return v
