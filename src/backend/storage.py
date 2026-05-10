"""
Supabase Storage operations for SynCVE backend.
Handles file upload, download, and public URL generation.
"""
import base64
import time
from typing import Optional

from deepface.commons.logger import Logger

from src.backend.config import get_config

logger = Logger()

# ---------------------------------------------------------------------------
# Supabase client (lazy singleton)
# ---------------------------------------------------------------------------

_supabase_client = None
_supabase_initialized = False


def _get_supabase():
    """Return the Supabase client, creating it on first call."""
    global _supabase_client, _supabase_initialized
    if _supabase_initialized:
        return _supabase_client

    _supabase_initialized = True
    cfg = get_config().supabase
    if not cfg.url or not cfg.key:
        logger.warn("Supabase credentials not found in environment.")
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(cfg.url, cfg.key)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
    return _supabase_client


def get_supabase_client():
    """Public accessor for the Supabase client singleton."""
    return _get_supabase()


def upload_to_supabase(
    path: str,
    file_bytes: bytes,
    content_type: str = "image/jpeg",
    bucket: Optional[str] = None,
) -> Optional[str]:
    """
    Upload a file to Supabase Storage.

    Args:
        path: Storage path (e.g. ``{session_id}/frames/{ts}.jpg``).
        file_bytes: Raw file content.
        content_type: MIME type for the upload.
        bucket: Override bucket name (defaults to config value).

    Returns:
        The storage path on success, or ``None`` on failure.
    """
    client = _get_supabase()
    if not client:
        logger.warn("Supabase not configured; skipping upload.")
        return None

    bucket_name = bucket or get_config().supabase.bucket_name
    try:
        client.storage.from_(bucket_name).upload(
            path=path,
            file=file_bytes,
            file_options={"content-type": content_type},
        )
        logger.info(f"Successfully uploaded: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to upload to storage: {e}")
        return None


def download_from_supabase(
    path: str,
    bucket: Optional[str] = None,
) -> Optional[bytes]:
    """
    Download a file from Supabase Storage.

    Returns:
        Raw bytes on success, or ``None`` on failure.
    """
    client = _get_supabase()
    if not client:
        return None

    bucket_name = bucket or get_config().supabase.bucket_name
    try:
        data = client.storage.from_(bucket_name).download(path)
        # Some SDK variants wrap bytes
        if hasattr(data, "data"):
            data = data.data
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, (bytes, bytearray)):
            return None
        return bytes(data)
    except Exception as e:
        logger.warn(f"Failed to download {path}: {e}")
        return None


def get_public_url(
    path: str,
    bucket: Optional[str] = None,
) -> str:
    """Return the public URL for a storage object."""
    client = _get_supabase()
    if not client:
        return ""

    bucket_name = bucket or get_config().supabase.bucket_name
    return client.storage.from_(bucket_name).get_public_url(path)


def list_files(
    folder_path: str,
    limit: int = 10,
    bucket: Optional[str] = None,
):
    """List files in a storage folder, newest first."""
    client = _get_supabase()
    if not client:
        return []

    bucket_name = bucket or get_config().supabase.bucket_name
    try:
        files = client.storage.from_(bucket_name).list(
            path=folder_path,
            limit=limit,
            sortBy={"column": "name", "order": "desc"},
        )
    except TypeError:
        # Older supabase-py signatures
        files = client.storage.from_(bucket_name).list(folder_path)
    return files or []


def upload_frame_to_storage(session_id: str, base64_image: str) -> Optional[str]:
    """
    Decode a base64 image and upload it as a session frame.

    Returns:
        The storage path on success, or ``None`` on failure.
    """
    try:
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]

        image_bytes = base64.b64decode(base64_image)
        timestamp = int(time.time() * 1000)
        file_path = f"{session_id}/frames/{timestamp}.jpg"
        return upload_to_supabase(file_path, image_bytes, content_type="image/jpeg")
    except Exception as e:
        logger.error(f"Failed to upload frame to storage: {e}")
        return None
