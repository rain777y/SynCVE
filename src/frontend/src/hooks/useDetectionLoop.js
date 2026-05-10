import { useState, useCallback, useRef, useEffect } from 'react';
import { useApiClient } from './useApiClient';

/**
 * Interval-based emotion detection loop.
 *
 * Uses useApiClient so any in-flight /analyze request is aborted on unmount.
 * Per-frame skipping (when an analyze is still processing) is still handled
 * by isProcessingRef, so we deliberately do NOT pass `key` to the call —
 * the existing skip semantic is preferred over replace-latest here.
 */
export function useDetectionLoop(captureFrame, serviceEndpoint, sessionIdRef, interval, config) {
  const [isDetecting, setIsDetecting] = useState(false);
  const [emotionData, setEmotionData] = useState(null);
  const [dominantEmotion, setDominantEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [detectionCount, setDetectionCount] = useState(0);
  const [samplingStats, setSamplingStats] = useState({
    lastLatencyMs: null,
    effectiveIntervalMs: null,
  });
  const [error, setError] = useState(null);

  const intervalRef = useRef(null);
  const timeoutRef = useRef(null);
  const loopActiveRef = useRef(false);
  const isProcessingRef = useRef(false);
  const frameSeqRef = useRef(0);
  const lastCompletedAtRef = useRef(null);
  const loadingTimerRef = useRef(null);
  const [showSpinner, setShowSpinner] = useState(false);

  const LOADING_DELAY = 300;

  const { call, abortAll } = useApiClient(serviceEndpoint);

  const captureAndAnalyze = useCallback(async () => {
    if (isProcessingRef.current) return;
    if (!sessionIdRef.current) return;

    isProcessingRef.current = true;
    loadingTimerRef.current = setTimeout(() => setShowSpinner(true), LOADING_DELAY);

    try {
      const base64Image = captureFrame();
      if (!base64Image) return;

      const clientCaptureTs = new Date().toISOString();
      const clientFrameId = frameSeqRef.current + 1;
      frameSeqRef.current = clientFrameId;
      const requestStartedAt = performance.now();

      const result = await call('/analyze', {
        method: 'POST',
        body: {
          detector_backend: config?.detector_backend || 'retinaface',
          actions: ['emotion'],
          align: true,
          img: base64Image,
          enforce_detection: true,
          anti_spoofing: config?.anti_spoofing !== false,
          session_id: sessionIdRef.current,
          client_capture_ts: clientCaptureTs,
          client_frame_id: clientFrameId,
        },
      });

      if (result.aborted) return;

      const data = result.data || {};

      if (!result.ok) {
        let errorMsg = 'Analysis failed';
        if (data.error) {
          if (data.error.includes('Spoof detected')) {
            errorMsg = 'Anti-Spoofing Alert: Photo/video detected. Please use live camera.';
          } else if (data.error.includes('No face')) {
            errorMsg = 'No face detected in frame';
          } else {
            errorMsg = data.error.split('\n')[0];
          }
        } else if (result.error === 'Network error') {
          errorMsg = 'Detection error: Network error';
        } else if (result.error) {
          errorMsg = result.error;
        }
        setError(errorMsg);
        return;
      }

      if (data.results && data.results.length > 0) {
        const result0 = data.results[0];
        // Prefer EMA-smoothed scores for smoother progress bars
        const displayEmotions = data.smoothed_emotions
          ? Object.fromEntries(
              Object.entries(data.smoothed_emotions).map(([k, v]) => [k, v * 100])
            )
          : result0.emotion;
        const displayEntries = Object.entries(displayEmotions || {})
          .filter(([, value]) => Number.isFinite(Number(value)));
        const [displayDominant, displayConfidence] = displayEntries.length
          ? displayEntries.reduce((best, current) => (
              Number(current[1]) > Number(best[1]) ? current : best
            ))
          : [result0.dominant_emotion, Math.max(...Object.values(result0.emotion || { neutral: 0 }))];
        const completedAt = Date.now();
        setEmotionData(displayEmotions);
        setDominantEmotion(displayDominant || result0.dominant_emotion);
        setConfidence(Number(displayConfidence) || 0);
        setSamplingStats({
          lastLatencyMs: data.frame_ledger?.timing?.inference_latency_ms
            ?? Math.round(performance.now() - requestStartedAt),
          effectiveIntervalMs: lastCompletedAtRef.current
            ? completedAt - lastCompletedAtRef.current
            : null,
        });
        lastCompletedAtRef.current = completedAt;
        setDetectionCount(prev => prev + 1);
        setError(null);
      } else {
        setError('No face detected');
      }
    } finally {
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
        loadingTimerRef.current = null;
      }
      setShowSpinner(false);
      isProcessingRef.current = false;
    }
  }, [captureFrame, sessionIdRef, config, call]);

  // Self-rescheduling chain: schedule the next analysis only after the previous one
  // finishes. Aligns cadence to actual response time so slow networks don't queue
  // overlapping calls (the previous setInterval-based loop silently dropped frames
  // via the isProcessingRef guard when analysis exceeded the interval).
  const scheduleNext = useCallback(async () => {
    if (!loopActiveRef.current) return;
    const startedAt = Date.now();
    try {
      await captureAndAnalyze();
    } finally {
      if (!loopActiveRef.current) return;
      const elapsed = Date.now() - startedAt;
      const delay = Math.max(0, interval - elapsed);
      timeoutRef.current = setTimeout(scheduleNext, delay);
    }
  }, [captureAndAnalyze, interval]);

  const startLoop = useCallback(() => {
    setIsDetecting(true);
    setDetectionCount(0);
    setSamplingStats({ lastLatencyMs: null, effectiveIntervalMs: null });
    setError(null);
    frameSeqRef.current = 0;
    lastCompletedAtRef.current = null;
    loopActiveRef.current = true;
    // Kick off the chain immediately; subsequent calls reschedule themselves.
    timeoutRef.current = setTimeout(scheduleNext, 0);
  }, [scheduleNext]);

  const stopLoop = useCallback(() => {
    setIsDetecting(false);
    setError(null);
    loopActiveRef.current = false;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const resetLoopState = useCallback(() => {
    setEmotionData(null);
    setDominantEmotion(null);
    setConfidence(null);
    setDetectionCount(0);
    setSamplingStats({ lastLatencyMs: null, effectiveIntervalMs: null });
    setError(null);
    setShowSpinner(false);
    frameSeqRef.current = 0;
    lastCompletedAtRef.current = null;
  }, []);

  // Cleanup on unmount: clear timers and abort any in-flight analyze.
  useEffect(() => {
    return () => {
      loopActiveRef.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (loadingTimerRef.current) clearTimeout(loadingTimerRef.current);
      abortAll();
    };
  }, [abortAll]);

  return {
    isDetecting,
    emotionData,
    dominantEmotion,
    confidence,
    detectionCount,
    samplingStats,
    showSpinner,
    startLoop,
    stopLoop,
    resetLoopState,
    captureAndAnalyze,
    error,
  };
}
