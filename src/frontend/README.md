# SynCVE Frontend

React-based UI for the SynCVE emotion recognition system. The app uses Supabase (Google OAuth) for sign-in and sends webcam frames to the backend API for analysis.

## Prerequisites
- Node.js 18+
- Backend API running (default `http://localhost:5005`)
- Supabase project with Google OAuth enabled

## Setup
1) `cd src/frontend`
2) Copy `.env.example` to `.env` and fill:
   - `REACT_APP_SUPABASE_URL`
   - `REACT_APP_SUPABASE_ANON_KEY`
   - `REACT_APP_SERVICE_ENDPOINT` (if not localhost)
   - Optional detection presets (`REACT_APP_DETECTOR_BACKEND`, `REACT_APP_ANTI_SPOOFING`, `REACT_APP_DETECTION_INTERVAL`)
3) Install deps: `npm install`
4) Start dev server: `npm start` (or run `scripts\start_frontend.bat` from repo root)

## Scripts
- `npm start` - dev server at http://localhost:3000
- `npm run build` - production build
- `npm test` - CRA test runner

## Environment reference
- `REACT_APP_SERVICE_ENDPOINT` - backend base URL (default `http://localhost:5005`)
- `REACT_APP_SUPABASE_URL` / `REACT_APP_SUPABASE_ANON_KEY` - Supabase project creds
- `REACT_APP_DETECTOR_BACKEND` - opencv | ssd | mtcnn | dlib | mediapipe | retinaface | yolov8 | yunet | centerface
- `REACT_APP_ANTI_SPOOFING` - `0` (off) or `1` (on)
- `REACT_APP_DETECTION_INTERVAL` - interval in ms between frames (500/1000/1500/2000/3000)
- Optional: `REACT_APP_FACE_RECOGNITION_MODEL`, `REACT_APP_DISTANCE_METRIC`, `REACT_APP_USER_*` base64 entries for face DB experiments

## Notes
- The UI enforces auth; unauthenticated users are redirected to the Home page.
- Backend status, detector choice, and anti-spoofing flags are read from env at startup.
