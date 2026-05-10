import { useState, useCallback, useRef } from 'react';
import { useApiClient } from './useApiClient';

/**
 * Manages session lifecycle with backend API.
 *
 * Now uses useApiClient internally so all in-flight calls are aborted on
 * unmount. Tracks per-action `inFlight` booleans so the UI can disable
 * buttons and surface "starting…" / "stopping…" affordances.
 */
export function useDetectionSession(serviceEndpoint, userId) {
  const [sessionId, setSessionId] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [error, setError] = useState(null);
  const sessionIdRef = useRef(null);

  // Per-action in-flight booleans for double-submit prevention + UI feedback.
  const [startInFlight, setStartInFlight] = useState(false);
  const [pauseInFlight, setPauseInFlight] = useState(false);
  const [stopInFlight, setStopInFlight] = useState(false);

  // Refs mirror the booleans so we can guard inside callbacks without
  // re-creating them on every state change.
  const startInFlightRef = useRef(false);
  const pauseInFlightRef = useRef(false);
  const stopInFlightRef = useRef(false);

  const { call, pollJob } = useApiClient(serviceEndpoint);

  // Helper: extract the "old sync /session/stop or /session/pause response"
  // shape from either a sync 200 result or the unwrapped result of an async
  // job. Returns the data dict to inspect for { report, image_url, error }.
  const unwrapJobResult = (jobBody) => {
    // jobBody is the GET /jobs/<id> response body when terminal.
    if (!jobBody) return {};
    if (jobBody.status === 'done') return jobBody.result || {};
    if (jobBody.status === 'error') {
      return { error: jobBody.error || 'Job failed' };
    }
    return {};
  };

  const startSession = useCallback(async (userEmail) => {
    // Reject double-submits.
    if (startInFlightRef.current) return null;
    startInFlightRef.current = true;
    setStartInFlight(true);
    setError(null);
    try {
      const result = await call('/session/start', {
        method: 'POST',
        body: {
          user_id: userId,
          metadata: { user_email: userEmail, start_time: new Date().toISOString() },
        },
      });
      if (result.aborted) return null;
      const data = result.data || {};
      if (result.ok && data.session_id) {
        sessionIdRef.current = data.session_id;
        setSessionId(data.session_id);
        setIsPaused(false);
        setReportData(null);
        return data.session_id;
      }
      console.error('Error starting session:', result.error);
      setError('Unable to start session. Please retry.');
      return null;
    } finally {
      startInFlightRef.current = false;
      setStartInFlight(false);
    }
  }, [call, userId]);

  // Apply the unwrapped data dict (same shape as old sync /session/pause
  // response) to component state.
  const applyPauseResult = useCallback((data, requestOk) => {
    if (data && data.report) {
      setReportData(data.report);
      setIsPaused(true);
    } else if (data && data.image_url) {
      setReportData({ visual_report_url: data.image_url });
      setIsPaused(true);
    } else if (data && data.error) {
      setError(data.error);
    } else if (!requestOk) {
      setError('Error generating report.');
    }
  }, []);

  const pauseSession = useCallback(async (jobIdSetter) => {
    const sid = sessionIdRef.current;
    if (!sid) return null;
    if (pauseInFlightRef.current) return null;
    pauseInFlightRef.current = true;
    setPauseInFlight(true);
    setIsGeneratingReport(true);
    setError(null);
    try {
      // 1. Try async submit.
      const submit = await call('/session/pause_async', {
        method: 'POST',
        body: { session_id: sid },
      });
      if (submit.aborted) return null;

      // Fallback: endpoint missing → fall back to sync /session/pause.
      if (!submit.ok && (submit.status === 404 || submit.status === 405)) {
        const sync = await call('/session/pause', {
          method: 'POST',
          body: { session_id: sid },
        });
        if (sync.aborted) return null;
        const data = sync.data || {};
        applyPauseResult(data, sync.ok);
        return data;
      }

      if (!submit.ok) {
        setError('Error generating report.');
        return null;
      }

      const jobId = submit.data && submit.data.job_id;
      if (!jobId) {
        setError('Error generating report.');
        return null;
      }
      if (typeof jobIdSetter === 'function') jobIdSetter(jobId);

      // 2. Poll until terminal.
      const poll = await pollJob(jobId, { intervalMs: 1500, timeoutMs: 60000 });
      if (poll.aborted) return null;
      if (!poll.ok) {
        setError(poll.error || 'Report generation timed out.');
        return null;
      }
      const data = unwrapJobResult(poll.data);
      applyPauseResult(data, true);
      return data;
    } finally {
      pauseInFlightRef.current = false;
      setPauseInFlight(false);
      setIsGeneratingReport(false);
    }
  }, [call, pollJob, applyPauseResult]);

  const stopSession = useCallback(async (jobIdSetter) => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    if (stopInFlightRef.current) return;
    stopInFlightRef.current = true;
    setStopInFlight(true);
    setError(null);

    try {
      // 1. Submit async stop request.
      const submit = await call('/session/stop_async', {
        method: 'POST',
        body: { session_id: sid },
      });
      if (submit.aborted) return;

      // Fallback: endpoint not deployed → use old sync /session/stop.
      if (!submit.ok && (submit.status === 404 || submit.status === 405)) {
        const sync = await call('/session/stop', {
          method: 'POST',
          body: { session_id: sid },
        });
        if (sync.aborted) return;
        return;
      }

      if (!submit.ok) {
        setError('Error stopping session.');
        return;
      }

      const jobId = submit.data && submit.data.job_id;
      if (!jobId) {
        // No job id → backend probably returned synchronously; treat as done.
        return;
      }
      if (typeof jobIdSetter === 'function') jobIdSetter(jobId);

      // 2. Poll for completion. We keep stopInFlight=true for the full
      //    submit + poll duration so the UI button stays disabled.
      const poll = await pollJob(jobId, { intervalMs: 1500, timeoutMs: 60000 });
      if (poll.aborted) return;
      if (!poll.ok) {
        setError(poll.error || 'Stop request timed out.');
        return;
      }
      const data = unwrapJobResult(poll.data);
      if (data && data.error) {
        setError(data.error);
      }
    } finally {
      // Clear local session state regardless of outcome — the user has
      // signalled "stop", so we should not leave the UI in a half-running
      // state. Errors (if any) are surfaced via setError above.
      sessionIdRef.current = null;
      setSessionId(null);
      setIsPaused(false);
      setReportData(null);
      stopInFlightRef.current = false;
      setStopInFlight(false);
    }
  }, [call, pollJob]);

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
    // In-flight flags
    startInFlight,
    pauseInFlight,
    stopInFlight,
  };
}
