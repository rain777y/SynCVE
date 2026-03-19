"""
SynCVE Backend Package
Custom backend for emotion recognition with GPU acceleration and anti-spoofing.

Modules:
    config              - Centralized, typed configuration (AppConfig)
    app                 - Flask application factory
    routes              - HTTP endpoint definitions
    service             - DeepFace analysis (represent / verify / analyze)
    session_manager     - Session CRUD, frame logging, in-memory cache
    emotion_analytics   - Emotion aggregation, statistics, noise filtering
    report_generator    - Text & visual report pipelines (Gemini)
    gemini_client       - Google GenAI SDK wrapper (text, image, retry)
    storage             - Supabase Storage operations (upload, download, URL)
    gpu_utils           - GPU memory management utilities
"""

__version__ = "1.0.0"
