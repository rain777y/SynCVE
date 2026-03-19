import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Interval-based emotion detection loop.
 */
export function useDetectionLoop(captureFrame, serviceEndpoint, sessionIdRef, interval, config) {
  const [isDetecting, setIsDetecting] = useState(false);
  const [emotionData, setEmotionData] = useState(null);
  const [dominantEmotion, setDominantEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [detectionCount, setDetectionCount] = useState(0);
  const [error, setError] = useState(null);

  const intervalRef = useRef(null);
  const isProcessingRef = useRef(false);
  const loadingTimerRef = useRef(null);
  const [showSpinner, setShowSpinner] = useState(false);

  const LOADING_DELAY = 300;

  const captureAndAnalyze = useCallback(async () => {
    if (isProcessingRef.current) return;
    if (!sessionIdRef.current) return;

    isProcessingRef.current = true;
    loadingTimerRef.current = setTimeout(() => setShowSpinner(true), LOADING_DELAY);

    try {
      const base64Image = captureFrame();
      if (!base64Image) return;

      const controller = new AbortController();
      const response = await fetch(`${serviceEndpoint}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          detector_backend: config?.detector_backend || 'retinaface',
          actions: ['emotion'],
          align: true,
          img: base64Image,
          enforce_detection: true,
          anti_spoofing: config?.anti_spoofing !== false,
          session_id: sessionIdRef.current,
        }),
      });

      const data = await response.json();

      if (response.status !== 200) {
        let errorMsg = 'Analysis failed';
        if (data.error) {
          if (data.error.includes('Spoof detected')) {
            errorMsg = 'Anti-Spoofing Alert: Photo/video detected. Please use live camera.';
          } else if (data.error.includes('No face')) {
            errorMsg = 'No face detected in frame';
          } else {
            errorMsg = data.error.split('\n')[0];
          }
        }
        setError(errorMsg);
        return;
      }

      if (data.results && data.results.length > 0) {
        const result = data.results[0];
        setEmotionData(result.emotion);
        setDominantEmotion(result.dominant_emotion);
        setConfidence(Math.max(...Object.values(result.emotion)));
        setDetectionCount(prev => prev + 1);
        setError(null);
      } else {
        setError('No face detected');
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Detection error:', err);
        setError('Detection error: ' + err.message);
      }
    } finally {
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
        loadingTimerRef.current = null;
      }
      setShowSpinner(false);
      isProcessingRef.current = false;
    }
  }, [captureFrame, serviceEndpoint, sessionIdRef, config]);

  const startLoop = useCallback(() => {
    setIsDetecting(true);
    setDetectionCount(0);
    setError(null);
    intervalRef.current = setInterval(captureAndAnalyze, interval);
  }, [captureAndAnalyze, interval]);

  const stopLoop = useCallback(() => {
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (loadingTimerRef.current) clearTimeout(loadingTimerRef.current);
    };
  }, []);

  return {
    isDetecting,
    emotionData,
    dominantEmotion,
    confidence,
    detectionCount,
    showSpinner,
    startLoop,
    stopLoop,
    captureAndAnalyze,
    error,
  };
}
