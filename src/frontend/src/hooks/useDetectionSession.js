import { useState, useCallback, useRef } from 'react';

/**
 * Manages session lifecycle with backend API.
 */
export function useDetectionSession(serviceEndpoint, userId) {
  const [sessionId, setSessionId] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [error, setError] = useState(null);
  const sessionIdRef = useRef(null);

  const startSession = useCallback(async (userEmail) => {
    setError(null);
    try {
      const controller = new AbortController();
      const response = await fetch(`${serviceEndpoint}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          user_id: userId,
          metadata: { user_email: userEmail, start_time: new Date().toISOString() },
        }),
      });
      const data = await response.json();
      if (data.session_id) {
        sessionIdRef.current = data.session_id;
        setSessionId(data.session_id);
        setIsPaused(false);
        setReportData(null);
        return data.session_id;
      }
      setError('Failed to start session');
      return null;
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Error starting session:', err);
        setError('Unable to start session. Please retry.');
      }
      return null;
    }
  }, [serviceEndpoint, userId]);

  const pauseSession = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return null;
    setIsGeneratingReport(true);
    setError(null);
    try {
      const controller = new AbortController();
      const response = await fetch(`${serviceEndpoint}/session/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ session_id: sid }),
      });
      const data = await response.json();
      if (data.report) {
        setReportData(data.report);
        setIsPaused(true);
      } else if (data.image_url) {
        setReportData({ visual_report_url: data.image_url });
        setIsPaused(true);
      } else if (data.error) {
        setError(data.error);
      }
      return data;
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Error pausing session:', err);
        setError('Error generating report.');
      }
      return null;
    } finally {
      setIsGeneratingReport(false);
    }
  }, [serviceEndpoint]);

  const stopSession = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      const controller = new AbortController();
      await fetch(`${serviceEndpoint}/session/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ session_id: sid }),
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Error stopping session:', err);
      }
    } finally {
      sessionIdRef.current = null;
      setSessionId(null);
      setIsPaused(false);
      setReportData(null);
    }
  }, [serviceEndpoint]);

  const resumeSession = useCallback(() => {
    setReportData(null);
    setIsPaused(false);
  }, []);

  return {
    sessionId,
    sessionIdRef,
    startSession,
    pauseSession,
    stopSession,
    resumeSession,
    isPaused,
    reportData,
    isGeneratingReport,
    error,
  };
}
