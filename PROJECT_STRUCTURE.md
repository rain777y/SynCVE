# SynCVE Project Structure

## Directory Layout

```
SynCVE/
├── .env.example                    # Environment variable template (secrets, API keys)
├── .gitignore                      # Git ignore rules
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Python pip dependencies
├── settings.yml                    # Application configuration (YAML)
├── PROJECT_STRUCTURE.md            # This file
│
├── docs/                           # Project documentation
│   ├── ethics_analysis.md          # Ethical considerations for emotion recognition
│   ├── experiment_report.md        # Evaluation results and analysis write-up
│   └── model_card.md               # Model card describing capabilities and limitations
│
├── eval/                           # Evaluation and benchmarking framework
│   ├── __init__.py
│   ├── configs/                    # Experiment configuration files (YAML)
│   │   ├── baseline.yml            # Baseline DeepFace evaluation config
│   │   ├── ablation_preprocess.yml # Preprocessing ablation study config
│   │   ├── ablation_detector.yml   # Face detector comparison config
│   │   ├── ablation_postprocess.yml# Temporal post-processing parameter study config
│   │   └── pipeline.yml            # Full pipeline vs baseline comparison config
│   ├── datasets/                   # Dataset storage (gitignored, large files)
│   │   ├── FER2013/                # FER-2013 facial expression dataset
│   │   └── RAF-DB/                 # RAF-DB real-world affective faces dataset
│   ├── results/                    # Evaluation output (gitignored except log)
│   │   ├── .gitkeep
│   │   ├── baseline/              # B0/B1 baseline results (per-detector JSON + plots/)
│   │   ├── ablation/              # Ablation study results (preprocess/detector/postprocess/ensemble)
│   │   └── pipeline/              # Full pipeline vs B0 comparison results
│   ├── cache/                      # Intermediate caching for evaluation runs (gitignored)
│   │   └── .gitkeep
│   ├── experiment_log.py           # Experiment logger (writes to results/experiment_log.jsonl)
│   ├── metrics.py                  # Accuracy, F1, confusion matrix utilities
│   ├── plot_results.py             # Visualization and chart generation
│   ├── benchmark_fer2013.py        # FER-2013 baseline benchmark script
│   ├── benchmark_rafdb.py          # RAF-DB baseline benchmark script
│   ├── ablation_preprocess.py      # Preprocessing ablation experiment runner
│   ├── ablation_detector.py        # Face detector ablation experiment runner
│   ├── ablation_postprocess.py     # Temporal post-processing ablation runner
│   ├── optimize_ensemble_weights.py# Ensemble weight optimization (grid/Bayesian)
│   ├── pipeline_vs_baseline.py     # Full pipeline comparison experiment
│   └── run_all.py                  # Orchestrator to run all evaluations sequentially
│
├── scripts/                        # Developer and CI/CD scripts
│   ├── setup.bat                   # One-click environment setup (Windows)
│   ├── start_backend.bat           # Launch Flask backend server
│   ├── start_frontend.bat          # Launch React dev server
│   ├── stop_service.bat            # Kill running backend/frontend processes
│   ├── run_tests.bat               # Run unit and integration tests
│   ├── run_e2e_tests.bat           # Run end-to-end test suite
│   ├── health_check.py             # System health and dependency checker
│   ├── generate_test_artifacts.bat # Generate test images and videos
│   ├── generate_test_images.py     # Synthetic face image generator for tests
│   └── generate_test_video.py      # Synthetic video generator for tests
│
├── src/                            # Application source code
│   ├── __init__.py
│   ├── backend/                    # Flask REST API + DeepFace emotion service
│   │   ├── __init__.py
│   │   ├── app.py                  # Flask application factory and startup
│   │   ├── config.py               # Configuration loader (settings.yml + .env)
│   │   ├── routes.py               # API route definitions (/detect, /session, /report)
│   │   ├── service.py              # Core emotion detection service (DeepFace wrapper)
│   │   ├── session_manager.py      # Detection session lifecycle management
│   │   ├── temporal_analysis.py    # EMA smoothing, noise floor, temporal post-processing
│   │   ├── validators.py           # Request/response validation schemas
│   │   ├── gpu_utils.py            # GPU detection and CUDA/TensorFlow device config
│   │   ├── gemini_client.py        # Google Gemini API client for session reports
│   │   ├── report_generator.py     # AI-powered session report generation
│   │   ├── emotion_analytics.py    # Emotion statistics and analytics helpers
│   │   └── storage.py              # Supabase storage integration
│   │
│   └── frontend/                   # React single-page application
│       ├── package.json            # Node.js dependencies and scripts
│       ├── public/                 # Static assets served directly
│       └── src/
│           ├── App.js              # Root React component and router
│           ├── index.js            # Application entry point
│           ├── components/         # Reusable UI components
│           │   ├── EmotionDetector.jsx      # Webcam + real-time emotion overlay
│           │   ├── EmotionDetector.css
│           │   ├── EmotionProgressBars.jsx  # Emotion probability bar chart
│           │   ├── EmotionProgressBars.css
│           │   ├── SessionReport.jsx        # AI-generated session report view
│           │   ├── SessionReport.css
│           │   ├── SystemStatus.jsx         # Backend health status indicator
│           │   └── SystemStatus.css
│           ├── pages/              # Route-level page components
│           │   ├── Home.js         # Landing page
│           │   ├── Detection.js    # Live detection page
│           │   └── History.js      # Session history browser
│           ├── contexts/           # React context providers
│           │   ├── AuthContext.js   # Supabase authentication state
│           │   └── ConfigContext.js # Runtime configuration provider
│           ├── hooks/              # Custom React hooks
│           │   ├── useDetectionLoop.js    # Frame capture and API polling loop
│           │   ├── useDetectionSession.js # Session start/stop lifecycle
│           │   └── useWebcam.js           # Webcam access and stream management
│           └── lib/
│               └── supabase.js     # Supabase client initialization
│
└── tests/                          # Test suite
    ├── __init__.py
    ├── pytest.ini                  # Pytest configuration and markers
    ├── artifacts/                  # Generated test fixtures (images, videos)
    │   ├── images/                 # Synthetic face images for testing
    │   ├── videos/                 # Synthetic emotion videos for testing
    │   └── reports/                # Sample report outputs
    ├── unit/                       # Unit tests (mocked dependencies)
    │   ├── __init__.py
    │   ├── conftest.py             # Unit test fixtures and mocks
    │   ├── test_gpu_utils.py
    │   ├── test_routes.py
    │   ├── test_service.py
    │   ├── test_session_manager.py
    │   └── test_temporal_analysis.py
    ├── integration/                # Integration tests (real services, GPU)
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_anti_spoofing_real.py
    │   ├── test_emotion_detection_real.py
    │   ├── test_eval_chain.py
    │   ├── test_gpu_performance_real.py
    │   ├── test_report_generation_real.py
    │   └── test_session_lifecycle_real.py
    ├── e2e/                        # End-to-end tests (full stack)
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_frontend_api_flow.py
    │   ├── test_gemini_direct.py
    │   ├── test_real_environment.py
    │   └── test_video_pipeline.py
    └── regression/                 # Regression tests (accuracy baselines)
        ├── __init__.py
        ├── conftest.py
        └── test_regression.py
```
