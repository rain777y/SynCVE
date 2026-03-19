/**
 * EmotionDetector Component
 *
 * Continuous real-time emotion detection from webcam.
 * Composed from custom hooks for webcam, session, and detection loop management.
 *
 * @version 2.0.0
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useConfig } from '../contexts/ConfigContext';
import { useWebcam } from '../hooks/useWebcam';
import { useDetectionSession } from '../hooks/useDetectionSession';
import { useDetectionLoop } from '../hooks/useDetectionLoop';
import EmotionProgressBars from './EmotionProgressBars';
import SessionReport from './SessionReport';
import SystemStatus from './SystemStatus';
import './EmotionDetector.css';

const DEFAULT_EMOTIONS = {
  angry: 0, neutral: 0, happy: 0, fear: 0, surprise: 0, sad: 0, disgust: 0,
};

const EmotionDetector = () => {
  const { user } = useAuth();
  const { config, serviceEndpoint } = useConfig();
  const [consentGiven, setConsentGiven] = useState(false);
  const [detectionInterval, setDetectionInterval] = useState(2000);
  const [sessionStartTime, setSessionStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  const { videoRef, canvasRef, isReady, error: webcamError, start: startWebcam, captureFrame, cleanup: cleanupWebcam } = useWebcam();
  const {
    sessionId, sessionIdRef, startSession, pauseSession, stopSession, resumeSession,
    isPaused, reportData, isGeneratingReport, error: sessionError,
  } = useDetectionSession(serviceEndpoint, user?.id);
  const {
    isDetecting, emotionData, dominantEmotion, confidence, detectionCount, showSpinner,
    startLoop, stopLoop, captureAndAnalyze, error: loopError,
  } = useDetectionLoop(captureFrame, serviceEndpoint, sessionIdRef, detectionInterval, config);

  // Sync config interval
  useEffect(() => {
    if (config?.detection_interval) setDetectionInterval(config.detection_interval);
  }, [config]);

  // Start webcam after consent
  useEffect(() => {
    if (consentGiven) startWebcam();
  }, [consentGiven, startWebcam]);

  // Session timer
  useEffect(() => {
    if (!isDetecting || !sessionStartTime) return;
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - sessionStartTime) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [isDetecting, sessionStartTime]);

  const errorMessage = webcamError || sessionError || loopError || '';

  // Start detection
  const handleStart = useCallback(async () => {
    if (isDetecting) return;
    const sid = await startSession(user?.email);
    if (!sid) return;
    setSessionStartTime(Date.now());
    setElapsedTime(0);
    startLoop();
  }, [isDetecting, startSession, startLoop, user]);

  // Pause detection
  const handlePause = useCallback(async () => {
    if (!isDetecting || isPaused) return;
    stopLoop();
    await pauseSession();
  }, [isDetecting, isPaused, stopLoop, pauseSession]);

  // Resume detection
  const handleResume = useCallback(() => {
    resumeSession();
    captureAndAnalyze();
    startLoop();
  }, [resumeSession, captureAndAnalyze, startLoop]);

  // Stop detection
  const handleStop = useCallback(async () => {
    if (!isDetecting && !isPaused) return;
    stopLoop();
    await stopSession();
    setSessionStartTime(null);
  }, [isDetecting, isPaused, stopLoop, stopSession]);

  // Format elapsed time
  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Consent banner
  if (!consentGiven) {
    return (
      <div className="emotion-detector-container">
        <div className="consent-banner">
          <h3>Emotion Detection Consent</h3>
          <p>This application will:</p>
          <ul>
            <li>Access your webcam to capture video frames</li>
            <li>Analyze facial expressions using machine learning</li>
            <li>Store session data for report generation</li>
          </ul>
          <p><strong>Important:</strong> Emotion predictions are probabilistic estimates based on facial patterns. They do not represent definitive measures of internal emotional states.</p>
          <p>You can stop detection at any time.</p>
          <button onClick={() => setConsentGiven(true)} className="btn btn-start">
            I Understand, Continue
          </button>
        </div>
      </div>
    );
  }

  const currentEmotion = dominantEmotion || 'neutral';
  const currentConfidence = confidence || 0;
  const currentEmotions = emotionData || DEFAULT_EMOTIONS;

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
              aria-label="Webcam video feed for emotion detection"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {isDetecting && (
              <div className="detection-indicator">
                <span className="pulse-dot"></span>
                <span>Detecting...</span>
              </div>
            )}

            {showSpinner && (
              <div className="loading-indicator">
                <div className="spinner-small"></div>
              </div>
            )}
          </div>

          <div className="dominant-emotion-display">
            <h3>Dominant Emotion</h3>
            <div className={`emotion-badge emotion-${currentEmotion}`}>
              {currentEmotion.toUpperCase()}
            </div>
            <div className="confidence-display">
              Confidence: {currentConfidence.toFixed(1)}%
            </div>
          </div>
        </div>

        <EmotionProgressBars
          emotions={currentEmotions}
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
          <button onClick={handleStart} disabled={isDetecting || isPaused} className="btn btn-start">
            ▶ Start Detection
          </button>

          {isDetecting && (
            <button onClick={handlePause} className="btn btn-secondary"
              style={{ marginLeft: '10px', backgroundColor: '#e67e22' }}>
              ⏸ Pause & Report
            </button>
          )}

          <button onClick={handleStop} disabled={!isDetecting && !isPaused} className="btn btn-stop">
            ⏹ Stop Detection
          </button>

          <div className="interval-control">
            <label htmlFor="interval-select">Interval:</label>
            <select id="interval-select" value={detectionInterval}
              onChange={(e) => setDetectionInterval(parseInt(e.target.value, 10))}
              disabled={isDetecting}>
              <option value="500">0.5s</option>
              <option value="1000">1.0s</option>
              <option value="1500">1.5s</option>
              <option value="2000">2.0s</option>
              <option value="3000">3.0s</option>
            </select>
          </div>
        </div>

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

        {errorMessage && (
          <div className="error-message">
            {errorMessage}
          </div>
        )}

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
                  onResume={handleResume}
                  onStop={handleStop}
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
