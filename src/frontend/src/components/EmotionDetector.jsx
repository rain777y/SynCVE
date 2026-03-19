/**
 * EmotionDetector Component
 * 
 * Continuous real-time emotion detection from webcam
 * Implements Phase 1: Continuous Real-time Detection
 * 
 * Features:
 * - Automatic continuous detection at configurable intervals
 * - Start/Stop controls
 * - Real-time emotion updates with progress bars
 * - Memory leak prevention with proper cleanup
 * - Performance optimization
 * 
 * @version 1.1.0
 * @date 2026-03-20
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import EmotionProgressBars from './EmotionProgressBars';
import SessionReport from './SessionReport';
import SystemStatus from './SystemStatus';
import './EmotionDetector.css';

const EmotionDetector = () => {
  const { user } = useAuth();
  // Configuration
  const serviceEndpoint = process.env.REACT_APP_SERVICE_ENDPOINT || 'http://localhost:5005';
  const [faceDetector, setFaceDetector] = useState('retinaface');
  const [antiSpoofing, setAntiSpoofing] = useState(true);
  const [defaultInterval, setDefaultInterval] = useState(2000);

  // Fetch detection settings from backend (single source of truth: settings.yml)
  useEffect(() => {
    fetch(`${serviceEndpoint}/config`)
      .then(res => res.json())
      .then(cfg => {
        setFaceDetector(cfg.detector_backend || 'retinaface');
        setAntiSpoofing(cfg.anti_spoofing !== false);
        setDefaultInterval(cfg.detection_interval || 2000);
        setDetectionInterval(cfg.detection_interval || 2000);
      })
      .catch(() => {}); // fallback to defaults
  }, [serviceEndpoint]);

  // Refs for video and canvas
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const isProcessingRef = useRef(false);
  const loadingTimerRef = useRef(null);
  const sessionIdRef = useRef(null); // Ref to avoid closure issues with setInterval

  // Loading delay threshold (only show spinner if request takes longer than this)
  const LOADING_DELAY = 300; // ms

  // State management
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionInterval, setDetectionInterval] = useState(defaultInterval);
  const [emotionData, setEmotionData] = useState({
    angry: 0,
    neutral: 0,
    happy: 0,
    fear: 0,
    surprise: 0,
    sad: 0,
    disgust: 0
  });
  const [dominantEmotion, setDominantEmotion] = useState('neutral');
  const [confidence, setConfidence] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  const [showSpinner, setShowSpinner] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState(null);

  const [elapsedTime, setElapsedTime] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // Initialize webcam
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const initializeWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }
        });
        video.srcObject = stream;
        await video.play();
        setErrorMessage('');
      } catch (err) {
        console.error('Error accessing webcam:', err);
        setErrorMessage('Failed to access webcam. Please check permissions.');
      }
    };

    initializeWebcam();

    // Cleanup function
    return () => {
      if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  // Session timer
  useEffect(() => {
    if (!isDetecting || !sessionStartTime) return;

    const timerInterval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - sessionStartTime) / 1000));
    }, 1000);

    return () => clearInterval(timerInterval);
  }, [isDetecting, sessionStartTime]);

  // Capture and analyze emotion
  const captureAndAnalyze = useCallback(async () => {
    // Prevent concurrent processing
    if (isProcessingRef.current) {
      console.log('Skipping detection - previous request still processing');
      return;
    }

    if (!sessionIdRef.current) {
      // Avoid hammering backend without a valid session; prompt UI to start a session first.
      console.warn('Session ID not set; skipping analyze call until session is active.');
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState !== video.HAVE_ENOUGH_DATA) {
      console.log('Video not ready');
      return;
    }

    isProcessingRef.current = true;

    // Debounced loading spinner - only show if request takes longer than LOADING_DELAY
    loadingTimerRef.current = setTimeout(() => {
      setShowSpinner(true);
    }, LOADING_DELAY);

    try {
      // Capture frame from video
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // OPTIMIZED: Increased JPEG quality from 0.8 to 0.95 for better facial detail preservation
      // This improves emotion detection accuracy at the cost of slightly larger payload
      const base64Image = canvas.toDataURL('image/jpeg', 0.95);

      // Call backend API
      const requestBody = JSON.stringify({
        detector_backend: faceDetector,
        actions: ['emotion'], // keep payload lean for real-time UX
        align: true,
        img: base64Image,
        enforce_detection: true,
        anti_spoofing: antiSpoofing,
        session_id: sessionIdRef.current, // Use ref to get latest value
      });

      const response = await fetch(`${serviceEndpoint}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: requestBody,
      });

      const data = await response.json();

      if (response.status !== 200) {
        console.error('API error:', data.error);

        // Extract user-friendly error message
        let errorMsg = 'Analysis failed';
        if (data.error) {
          // Check for specific error types
          if (data.error.includes('Spoof detected')) {
            errorMsg = '🚫 Anti-Spoofing Alert: Photo/video detected. Please use live camera.';
          } else if (data.error.includes('No face')) {
            errorMsg = 'No face detected in frame';
          } else {
            // Extract first line of error message (before traceback)
            const firstLine = data.error.split('\n')[0];
            errorMsg = firstLine.replace('Exception while analyzing: ', '');
          }
        }

        setErrorMessage(errorMsg);
        return;
      }

      // Process results
      if (data.results && data.results.length > 0) {
        const result = data.results[0]; // Use first detected face

        // Update emotion data
        setEmotionData(result.emotion);
        setDominantEmotion(result.dominant_emotion);

        // Calculate confidence (max emotion score)
        const maxConfidence = Math.max(...Object.values(result.emotion));
        setConfidence(maxConfidence);

        // Increment detection count
        setDetectionCount(prev => prev + 1);
        setErrorMessage('');
      } else {
        setErrorMessage('No face detected');
      }

    } catch (error) {
      console.error('Exception during emotion detection:', error);
      setErrorMessage('Detection error: ' + error.message);
    } finally {
      // Clear the loading timer and hide spinner
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
        loadingTimerRef.current = null;
      }
      setShowSpinner(false);
      isProcessingRef.current = false;
    }
  }, [serviceEndpoint, faceDetector, antiSpoofing]); // sessionId removed - using sessionIdRef instead

  // Start continuous detection
  const startDetection = useCallback(async () => {
    if (isDetecting) return;

    // Start Session on Backend (must succeed before detections begin)
    let newSessionId = null;

    // Start Session on Backend
    try {
      const response = await fetch(`${serviceEndpoint}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.id,
          metadata: {
            user_email: user?.email,
            start_time: new Date().toISOString()
          }
        })
      });
      const data = await response.json();
      if (data.session_id) {
        newSessionId = data.session_id;
        sessionIdRef.current = data.session_id; // Update ref immediately
        setSessionId(data.session_id);
        console.log("Session started:", data.session_id);
      } else {
        console.warn("Failed to start backend session; detections will not start.");
      }
    } catch (err) {
      console.error("Error starting session:", err);
    }

    if (!newSessionId) {
      setErrorMessage('Unable to start session. Please retry.');
      return;
    }

    setIsDetecting(true);
    setSessionStartTime(Date.now());
    setDetectionCount(0);
    setElapsedTime(0);
    setErrorMessage('');

    // Initial detection (will use new sessionId if set, or null if async delayed - mostly fine for first frame)
    // captureAndAnalyze(); // SKIP first frame immediate call to let state settle or wait for next interval

    // Set up interval for continuous detection
    intervalRef.current = setInterval(() => {
      captureAndAnalyze();
    }, detectionInterval);

    console.log(`Started continuous detection with ${detectionInterval}ms interval`);
  }, [isDetecting, detectionInterval, captureAndAnalyze, serviceEndpoint, user]);

  // Pause Detection & Generate Report
  const pauseDetection = async () => {
    if (!isDetecting || isPaused) return;

    // Stop interval loop temporarily
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    setIsPaused(true);
    setIsDetecting(false); // UI state update (but session active on backend)
    setIsGeneratingReport(true);

    try {
      const response = await fetch(`${serviceEndpoint}/session/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      const data = await response.json();

      if (data.report) {
        setReportData(data.report);
      } else if (data.image_url) {
        // Backward compat: old full-mode response
        setReportData({ visual_report_url: data.image_url });
      } else if (data.error) {
        setErrorMessage(data.error);
      }
    } catch (err) {
      console.error("Error pausing session:", err);
      setErrorMessage("Error generating report.");
    } finally {
      setIsGeneratingReport(false);
    }
  };

  // Resume Detection
  const resumeDetection = () => {
    setReportData(null);
    setIsPaused(false);
    setIsDetecting(true);
    // Manually restart interval without calling /session/start
    captureAndAnalyze();
    intervalRef.current = setInterval(() => {
      captureAndAnalyze();
    }, detectionInterval);
  };



  // Stop continuous detection
  const stopDetection = useCallback(async () => {
    if (!isDetecting && !isPaused) return;

    setIsDetecting(false);
    setIsPaused(false);
    setReportData(null);

    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Stop Session on Backend
    if (sessionId) {
      try {
        console.log("Stopping session:", sessionId);
        await fetch(`${serviceEndpoint}/session/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId })
        });
        setSessionId(null);
        sessionIdRef.current = null; // Clear ref
      } catch (err) {
        console.error("Error stopping session:", err);
      }
    }

    console.log('Stopped continuous detection');
  }, [isDetecting, isPaused, sessionId, serviceEndpoint]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
      }
    };
  }, []);

  // Format elapsed time
  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="emotion-detector-container">
      <header className="emotion-detector-header">
        <h1>Emotion Recognition System</h1>
        <p className="subtitle">Real-time Facial Expression Analysis</p>
      </header>

      <div className="main-content">
        {/* Video Section */}
        <div className="video-section">
          <div className="video-wrapper">
            <video
              ref={videoRef}
              className="webcam-video"
              autoPlay
              playsInline
              muted
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Overlay indicators */}
            {isDetecting && (
              <div className="detection-indicator">
                <span className="pulse-dot"></span>
                <span>Detecting...</span>
              </div>
            )}

            {/* Non-blocking loading indicator - only shows for slow requests */}
            {showSpinner && (
              <div className="loading-indicator">
                <div className="spinner-small"></div>
              </div>
            )}
          </div>

          {/* Dominant Emotion Display */}
          <div className="dominant-emotion-display">
            <h3>Dominant Emotion</h3>
            <div className={`emotion-badge emotion-${dominantEmotion}`}>
              {dominantEmotion.toUpperCase()}
            </div>
            <div className="confidence-display">
              Confidence: {confidence.toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Emotion Progress Bars */}
        <EmotionProgressBars
          emotions={emotionData}
          position="right"
          width={400}
          showPercentage={true}
          animated={isDetecting}
          language="en"
        />
      </div>

      {/* Control Panel */}
      <div className="control-panel">
        <div className="controls-row">
          <button
            onClick={startDetection}
            disabled={isDetecting || isPaused}
            className="btn btn-start"
          >
            ▶ Start Detection
          </button>

          {isDetecting && (
            <button
              onClick={pauseDetection}
              className="btn btn-secondary"
              style={{ marginLeft: '10px', backgroundColor: '#e67e22' }}
            >
              ⏸ Pause & Report
            </button>
          )}

          <button
            onClick={stopDetection}
            disabled={!isDetecting && !isPaused}
            className="btn btn-stop"
          >
            ⏹ Stop Detection
          </button>

          <div className="interval-control">
            <label htmlFor="interval-select">Interval:</label>
            <select
              id="interval-select"
              value={detectionInterval}
              onChange={(e) => setDetectionInterval(parseInt(e.target.value, 10))}
              disabled={isDetecting}
            >
              <option value="500">0.5s</option>
              <option value="1000">1.0s</option>
              <option value="1500">1.5s</option>
              <option value="2000">2.0s</option>
              <option value="3000">3.0s</option>
            </select>
          </div>
        </div>

        {/* Session Info */}
        <div className="session-info">
          <div className="info-item">
            <span className="info-label">Session Time:</span>
            <span className="info-value">{formatTime(elapsedTime)}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Detections:</span>
            <span className="info-value">{detectionCount}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Status:</span>
            <span className={`info-value status-${isDetecting ? 'active' : 'inactive'}`}>
              {isDetecting ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>

        {/* Error Message */}
        {errorMessage && (
          <div className="error-message">
            ⚠ {errorMessage}
          </div>
        )}

        {/* System Status Component */}
        <SystemStatus />

        {/* Report Modal */}
        {(isGeneratingReport || reportData) && (
          <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.85)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
          }}>
            <div className="modal-content" style={{
              background: '#1a1a1a', padding: '24px', borderRadius: '12px', maxWidth: '960px', width: '95%', maxHeight: '90vh', overflow: 'auto',
              border: '1px solid rgba(255,255,255,0.1)'
            }}>
              {isGeneratingReport ? (
                <div style={{ padding: '40px', textAlign: 'center' }}>
                  <div className="spinner-small" style={{ margin: '0 auto 20px' }}></div>
                  <h3 style={{ color: '#fff' }}>Analyzing Session...</h3>
                  <p style={{ color: '#888' }}>Aggregating emotion data</p>
                </div>
              ) : (
                <SessionReport
                  report={reportData}
                  onResume={resumeDetection}
                  onStop={stopDetection}
                />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmotionDetector;
