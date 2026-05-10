/**
 * EmotionDetector Component
 *
 * Continuous real-time emotion detection from webcam.
 * Composed from custom hooks for webcam, session, and detection loop management.
 *
 * @version 2.1.0  (editorial instrument refresh)
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useConfig } from '../contexts/ConfigContext';
import { useWebcam } from '../hooks/useWebcam';
import { useDetectionSession } from '../hooks/useDetectionSession';
import { useDetectionLoop } from '../hooks/useDetectionLoop';
import EmotionProgressBars from './EmotionProgressBars';
import SessionReport from './SessionReport';
import './EmotionDetector.css';

const DEFAULT_EMOTIONS = {
  angry: 0, neutral: 0, happy: 0, fear: 0, surprise: 0, sad: 0, disgust: 0,
};

const EmotionDetector = () => {
  const { user } = useAuth();
  const { config, serviceEndpoint } = useConfig();
  const [consentGiven, setConsentGiven] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const detectionInterval = config?.detection_interval || 2000;
  // Tracks elapsed seconds while a stop/pause job is in flight, so the
  // "Compiling clinical summary…" modal shows live progress and the user
  // doesn't think the UI has frozen during the ~16s Gemini call.
  const [reportElapsed, setReportElapsed] = useState(0);

  const { videoRef, canvasRef, isReady, error: webcamError, start: startWebcam, captureFrame } = useWebcam();
  const {
    sessionId, sessionIdRef, startSession, pauseSession, stopSession, resumeSession,
    isPaused, reportData, isGeneratingReport, error: sessionError,
    startInFlight, pauseInFlight, stopInFlight,
  } = useDetectionSession(serviceEndpoint, user?.id);
  const {
    isDetecting, emotionData, dominantEmotion, confidence, detectionCount, showSpinner,
    samplingStats, startLoop, stopLoop, resetLoopState, captureAndAnalyze, error: loopError,
  } = useDetectionLoop(captureFrame, serviceEndpoint, sessionIdRef, detectionInterval, config);

  useEffect(() => {
    if (consentGiven) startWebcam();
  }, [consentGiven, startWebcam]);

  useEffect(() => {
    if (!isDetecting || !sessionStartTime) return;
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - sessionStartTime) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [isDetecting, sessionStartTime]);

  const errorMessage = webcamError || sessionError || loopError || '';

  const handleStart = useCallback(async () => {
    if (isDetecting || startInFlight) return;
    const webcamReady = isReady || await startWebcam();
    if (!webcamReady) return;
    const sid = await startSession(user?.email);
    if (!sid) return;
    setSessionStartTime(Date.now());
    setElapsedTime(0);
    startLoop();
  }, [isDetecting, startInFlight, isReady, startWebcam, startSession, startLoop, user]);

  const handlePause = useCallback(async () => {
    if (!isDetecting || isPaused) return;
    stopLoop();
    await pauseSession();
  }, [isDetecting, isPaused, stopLoop, pauseSession]);

  const handleResume = useCallback(() => {
    resumeSession();
    captureAndAnalyze();
    startLoop();
  }, [resumeSession, captureAndAnalyze, startLoop]);

  const handleStop = useCallback(async () => {
    if (!isDetecting && !isPaused) return;
    stopLoop();
    await stopSession();
    setSessionStartTime(null);
    setElapsedTime(0);
    resetLoopState();
  }, [isDetecting, isPaused, stopLoop, stopSession, resetLoopState]);

  // While a stop/pause request is generating the report (or the stop is
  // running async on the backend), tick a 1Hz counter so the modal's hint
  // text can show "(Ns elapsed)" — a small reassurance that we haven't hung.
  useEffect(() => {
    const active = isGeneratingReport || stopInFlight;
    if (!active) {
      setReportElapsed(0);
      return undefined;
    }
    setReportElapsed(0);
    const startedAt = Date.now();
    const id = setInterval(() => {
      setReportElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [isGeneratingReport, stopInFlight]);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // ─────────── Consent ───────────
  if (!consentGiven) {
    return (
      <div className="ed-shell">
        <div className="ed-consent">
          <span className="eyebrow">Informed Consent — Required</span>
          <h2 className="ed-consent__title">
            Before we begin, a note <em>on what this measures.</em>
          </h2>
          <hr className="rule" style={{ margin: '24px 0' }} />
          <p>This application will:</p>
          <ul>
            <li>Access your webcam to capture video frames in-browser.</li>
            <li>Analyse facial expressions using machine-learning ensembles.</li>
            <li>Persist session data so reports can be reviewed later.</li>
          </ul>
          <p className="ed-consent__warn">
            <strong>Important.</strong> Emotion predictions are probabilistic estimates
            derived from facial patterns. They are <em>not</em> definitive measures of
            internal emotional states, nor are they a clinical diagnosis. You may stop
            acquisition at any time.
          </p>
          <button onClick={() => setConsentGiven(true)} className="btn-ink btn-ink--lg">
            I Understand — Continue
            <span className="arrow">→</span>
          </button>
        </div>
      </div>
    );
  }

  const currentEmotion = dominantEmotion || 'neutral';
  const currentConfidence = confidence || 0;
  const currentEmotions = emotionData || DEFAULT_EMOTIONS;

  const status = isDetecting ? 'recording' : (isPaused ? 'paused' : 'idle');
  const statusLabel = { recording: 'Recording', paused: 'Paused', idle: 'Standby' }[status];

  return (
    <div className="ed-shell">
      {/* ─────────── Console header ─────────── */}
      <header className="ed-head">
        <div className="ed-head__left">
          <span className="eyebrow">Acquisition · Channel 01</span>
          <h1 className="ed-head__title">
            Continuous Affect <em>Inference</em>
          </h1>
        </div>
        <div className="ed-head__right">
          {status === 'recording' && (
            <span className="signal-pill">REC · LIVE</span>
          )}
          {status === 'paused' && (
            <span className="ed-status-pill ed-status-pill--paused">PAUSED</span>
          )}
          {status === 'idle' && (
            <span className="ed-status-pill">STANDBY</span>
          )}
        </div>
      </header>

      {/* ─────────── Main grid ─────────── */}
      <div className="ed-grid">
        {/* SPECIMEN PANEL ── webcam framed as a "specimen" */}
        <section className="ed-specimen">
          <div className="ed-specimen__cap">
            <span className="mono ed-specimen__cap-l">SPECIMEN · F-CAM 01</span>
            <span className="mono ed-specimen__cap-r">{isDetecting ? '● LIVE' : '○ IDLE'}</span>
          </div>

          <div className="ed-specimen__frame">
            <video
              ref={videoRef}
              className="ed-specimen__video"
              autoPlay
              playsInline
              muted
              aria-label="Webcam video feed for emotion detection"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Crosshair overlay */}
            <div className="ed-specimen__hud" aria-hidden>
              <span className="hud-mark hud-mark--tl" />
              <span className="hud-mark hud-mark--tr" />
              <span className="hud-mark hud-mark--bl" />
              <span className="hud-mark hud-mark--br" />
            </div>

            {showSpinner && (
              <div className="ed-specimen__loading">
                <div className="spinner-small"></div>
              </div>
            )}
          </div>

          <div className="ed-specimen__foot">
            <span className="mono">7-axis affect ensemble · DeepFace ⨯ Gemini</span>
            <span className="mono">
              {samplingStats?.lastLatencyMs != null
                ? `last ${(samplingStats.lastLatencyMs / 1000).toFixed(1)}s`
                : 'adaptive sampling'}
            </span>
          </div>
        </section>

        {/* DOMINANT READOUT ── editorial big-number */}
        <aside className="ed-readout">
          <span className="eyebrow">Dominant Affect</span>
          <div className="ed-readout__name">
            <span
              className={`emotion-${currentEmotion}`}
              style={{ '--dot-color': `var(--e-${currentEmotion})` }}
            >
              {currentEmotion}
            </span>
          </div>

          <hr className="rule" style={{ margin: '16px 0' }} />

          <div className="ed-readout__row">
            <span className="ed-readout__lbl">Confidence</span>
            <span className="ed-readout__val mono">{currentConfidence.toFixed(1)}%</span>
          </div>
          <div className="ed-readout__row">
            <span className="ed-readout__lbl">Status</span>
            <span className={`ed-readout__val ed-readout__val--${status}`}>{statusLabel}</span>
          </div>
          <div className="ed-readout__row">
            <span className="ed-readout__lbl">Elapsed</span>
            <span className="ed-readout__val mono ed-readout__val--lg">{formatTime(elapsedTime)}</span>
          </div>
          <div className="ed-readout__row">
            <span className="ed-readout__lbl">N samples</span>
            <span className="ed-readout__val mono">{detectionCount}</span>
          </div>
        </aside>

        {/* PROGRESS BARS — full-width band beneath */}
        <section className="ed-bars">
          <span className="eyebrow">Distribution · Per-axis Confidence</span>
          <hr className="rule" style={{ margin: '12px 0 18px' }} />
          <EmotionProgressBars
            emotions={currentEmotions}
            position="static"
            width={'100%'}
            showPercentage={true}
            animated={isDetecting}
            language="en"
          />
        </section>
      </div>

      {/* ─────────── Control bench ─────────── */}
      <section className="ed-bench">
        <div className="ed-bench__head">
          <span className="eyebrow">Control Bench</span>
          <span className="mono ed-bench__hint">Operator-side commands</span>
        </div>

        <div className="ed-bench__row">
          <div className="ed-bench__btn-wrap">
            <button
              onClick={handleStart}
              disabled={isDetecting || isPaused || startInFlight}
              className={`btn-ink ed-bench__primary ${(isDetecting || startInFlight) ? 'is-disabled' : ''}`}
            >
              <span className="cmd-glyph">▸</span>
              Start session
            </button>
            {startInFlight && (
              <span className="mono ed-bench__hint ed-bench__hint--inflight">starting…</span>
            )}
          </div>

          {isDetecting && (
            <div className="ed-bench__btn-wrap">
              <button
                onClick={handlePause}
                disabled={pauseInFlight || isGeneratingReport}
                className="btn-ghost"
              >
                <span className="cmd-glyph">‖</span>
                Pause + report
              </button>
              {(pauseInFlight || isGeneratingReport) && (
                <span className="mono ed-bench__hint ed-bench__hint--inflight">pausing…</span>
              )}
            </div>
          )}

          <div className="ed-bench__btn-wrap">
            <button
              onClick={handleStop}
              disabled={(!isDetecting && !isPaused) || stopInFlight}
              className="btn-signal"
            >
              <span className="cmd-glyph">■</span>
              End session
            </button>
            {stopInFlight && (
              <span className="mono ed-bench__hint ed-bench__hint--inflight">stopping…</span>
            )}
          </div>

        </div>

        {errorMessage && (
          <div className="ed-error">
            <span className="mono ed-error__tag">ERR</span>
            <span>{errorMessage}</span>
          </div>
        )}

      </section>

      {/* ─────────── Report modal ─────────── */}
      {(isGeneratingReport || reportData) && (
        <div className="ed-modal-overlay">
          <div className="ed-modal">
            {isGeneratingReport ? (
              <div className="ed-modal__loading">
                <div className="spinner"></div>
                <span className="eyebrow">Compiling clinical summary…</span>
                <p>Aggregating temporal data and ensemble agreement…</p>
                <p className="ed-modal__hint">
                  (typically 5–15s
                  {reportElapsed > 0 ? ` · ${reportElapsed}s elapsed` : ''})
                </p>
              </div>
            ) : (
              <SessionReport
                report={reportData}
                onResume={handleResume}
                onStop={handleStop}
                sessionId={sessionId}
                serviceEndpoint={serviceEndpoint}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionDetector;
