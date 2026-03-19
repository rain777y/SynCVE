# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np
from pydantic import ValidationError

# project dependencies
from deepface import DeepFace
from src.backend import service, session_manager
from src.backend.config import get_config
from src.backend.validators import (
    AnalyzeRequest,
    SessionStartRequest,
    SessionStopRequest,
    ReportRequest,
    VisualReportRequest,
)
from deepface.commons import image_utils
from deepface.commons.logger import Logger

logger = Logger()

blueprint = Blueprint("routes", __name__)

# pylint: disable=no-else-return, broad-except


def _deepface_cfg():
    """Shorthand accessor for DeepFace config defaults."""
    return get_config().deepface


def _default_model():
    return get_config().deepface.model_name

def _default_detector():
    return get_config().deepface.detector_backend

def _default_distance_metric():
    return get_config().deepface.distance_metric


@blueprint.route("/")
def home():
    return f"<h1>Welcome to SynCVE Backend (DeepFace API v{DeepFace.__version__})!</h1>"


def extract_image_from_request(img_key: str) -> Union[str, np.ndarray]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
    """

    # Check if the request is multipart/form-data (file input)
    if request.files:
        # request.files is instance of werkzeug.datastructures.ImmutableMultiDict
        # file is instance of werkzeug.datastructures.FileStorage
        file = request.files.get(img_key)

        if file is None:
            raise ValueError(f"Request form data doesn't have {img_key}")

        if file.filename == "":
            raise ValueError(f"No file uploaded for '{img_key}'")

        img = image_utils.load_image_from_file_storage(file)

        return img
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = request.get_json() or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        img = input_args.get(img_key)

        if not img:
            raise ValueError(f"'{img_key}' not found in either json or form data request")

        return img

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    ) or {}

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=input_args.get("model_name", _default_model()),
        detector_backend=input_args.get("detector_backend", _default_detector()),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    ) or {}

    try:
        img1 = extract_image_from_request("img1")
    except Exception as err:
        return {"exception": str(err)}, 400

    try:
        img2 = extract_image_from_request("img2")
    except Exception as err:
        return {"exception": str(err)}, 400

    verification = service.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=input_args.get("model_name", _default_model()),
        detector_backend=input_args.get("detector_backend", _default_detector()),
        distance_metric=input_args.get("distance_metric", _default_distance_metric()),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    ) or {}

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    # Validate actions (and other fields) via Pydantic
    try:
        validated = AnalyzeRequest(
            img="placeholder",  # image already extracted above
            actions=input_args.get("actions", ["age", "gender", "emotion", "race"]),
            detector_backend=input_args.get("detector_backend"),
            enforce_detection=input_args.get("enforce_detection"),
            align=input_args.get("align"),
            anti_spoofing=input_args.get("anti_spoofing"),
            enable_ensemble=input_args.get("enable_ensemble"),
            confidence_threshold=input_args.get("confidence_threshold"),
            max_faces=input_args.get("max_faces"),
            session_id=input_args.get("session_id"),
        )
    except ValidationError as ve:
        return {"error": "Invalid request parameters", "details": ve.errors()}, 422

    demographies = service.analyze(
        img_path=img,
        actions=validated.actions,
        detector_backend=validated.detector_backend or _default_detector(),
        enforce_detection=validated.enforce_detection,
        align=validated.align if validated.align is not None else True,
        anti_spoofing=validated.anti_spoofing if validated.anti_spoofing is not None else True,
        enable_ensemble=validated.enable_ensemble if validated.enable_ensemble is not None else True,
        confidence_threshold=validated.confidence_threshold,
    )

    # Session Logging
    session_id = validated.session_id
    if session_id:
        try:
            log_result = session_manager.log_data(session_id, demographies, image_data=img)
            if isinstance(log_result, dict) and log_result.get("error"):
                logger.warn(f"Session log failed for {session_id}: {log_result}")
                if isinstance(demographies, dict):
                    demographies.setdefault("logging_status", log_result)
        except Exception as e:
            logger.error(f"Failed to log session data: {e}")
    else:
        logger.warn("Session ID missing on /analyze; skipping session logging.")

    logger.debug(demographies)

    return demographies


@blueprint.route("/session/start", methods=["POST"])
def start_session():
    """
    Starts a new emotion tracking session.
    Expected JSON: { "user_id": "...", "metadata": {...} }
    """
    input_args = request.get_json() or {}

    try:
        validated = SessionStartRequest(**input_args)
    except ValidationError as ve:
        return {"error": "Invalid request parameters", "details": ve.errors()}, 422

    result = session_manager.start_session(user_id=validated.user_id, metadata=validated.metadata)
    if "error" in result:
        return result, 500
    return result

@blueprint.route("/session/stop", methods=["POST"])
def stop_session():
    """
    Stops a session and generates a report.
    Expected JSON: { "session_id": "..." }
    """
    input_args = request.get_json() or {}

    try:
        validated = SessionStopRequest(**input_args)
    except ValidationError as ve:
        return {"error": "Invalid request parameters", "details": ve.errors()}, 422

    result = session_manager.stop_session(validated.session_id)
    if "error" in result:
        return result, 500
    return result

@blueprint.route("/session/pause", methods=["POST"])
def pause_session():
    """
    Pauses a session and triggers a visual report.
    Expected JSON: { "session_id": "..." }
    """
    input_args = request.get_json() or {}
    session_id = input_args.get("session_id")
    
    if not session_id:
        return {"error": "session_id is required"}, 400
        
    result = session_manager.pause_session(session_id)
    if "error" in result:
        status_code = result.get("status_code", 500)
        result.pop("status_code", None)
        return result, status_code
    return result

@blueprint.route("/session/history", methods=["GET"])
def get_session_history():
    """
    Get recent sessions.
    Query Params: limit (int), user_id (str)
    """
    limit = request.args.get("limit", 10, type=int)
    user_id = request.args.get("user_id")
    
    result = session_manager.get_recent_sessions(user_id, limit)
    if "error" in result:
         return result, 500
    return result

@blueprint.route("/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """
    Get details for a specific session.
    """
    result = session_manager.get_session_details(session_id)
    if "error" in result:
        return result, 404
    return result


@blueprint.route("/session/report/emotion", methods=["POST"])
def generate_emotion_report():
    """
    Generate a two-stage emotion report:
    - Aggregates raw vision scores (payload or fetched from DB)
    - Flash-lite builds a context prompt (text only)
    - Pro Vision consumes the prompt + keyframes from storage to produce Markdown
    Expected JSON: { "session_id": "...", "raw_vision_data": [...], "max_keyframes": 4 }
    """
    input_args = request.get_json() or {}

    try:
        validated = ReportRequest(**input_args)
    except ValidationError as ve:
        return {"error": "Invalid request parameters", "details": ve.errors()}, 422

    try:
        result = session_manager.generate_emotion_report(
            session_id=validated.session_id,
            raw_vision_data=validated.raw_vision_data,
            max_keyframes=validated.max_keyframes,
        )
        return result
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        logger.error(f"Failed to generate emotion report: {e}")
        return {"error": "Failed to generate emotion report"}, 500


@blueprint.route("/session/report/visual", methods=["POST"])
def generate_visual_report():
    """
    Generate a futuristic visual dashboard image (v3.0 data-to-visual pipeline).
    - Aggregates raw scores
    - Flash-Lite (art director) authors an image prompt
    - Pro Image Preview renders the image
    Expected JSON: { "session_id": "...", "raw_vision_data": [...], "aspect_ratio": "16:9", "style_preset": "futuristic" }
    Returns public_url + prompt + stats summary.
    """
    input_args = request.get_json() or {}

    try:
        validated = VisualReportRequest(**input_args)
    except ValidationError as ve:
        return {"error": "Invalid request parameters", "details": ve.errors()}, 422

    try:
        result = session_manager.generate_visual_report_v3(
            session_id=validated.session_id,
            raw_vision_data=validated.raw_vision_data,
            aspect_ratio=validated.aspect_ratio.value if validated.aspect_ratio else session_manager.EMOTION_VISUAL_ASPECT_RATIO,
            style_preset=validated.style_preset or session_manager.EMOTION_VISUAL_STYLE_PRESET,
        )
        return result
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        logger.error(f"Failed to generate visual report: {e}")
        return {"error": "Failed to generate visual report"}, 500
