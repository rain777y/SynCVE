/**
 * EventSensitivityPanel — UI for live tuning of the Axis 1A event detector.
 *
 * Drives /session/<id>/events?method=...&z_threshold=...&min_magnitude=...
 * and reports back the count + list of events. Designed for clinicians to
 * explore detection sensitivity post-session without restarting capture.
 */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import './EventSensitivityPanel.css';

const METHODS = [
  {
    value: 'ensemble',
    label: 'Ensemble (recommended)',
    description: 'Clusters votes from sliding, CUSUM, and PELT. Best default when enough frames exist.',
  },
  {
    value: 'sliding',
    label: 'Sliding-window z-score',
    description: 'Finds abrupt local jumps by comparing before/after windows on the smoothed emotion vector.',
  },
  {
    value: 'cusum',
    label: 'CUSUM (drift)',
    description: 'Finds persistent positive or negative valence drift, even when the dominant label stays stable.',
  },
  {
    value: 'pelt',
    label: 'PELT (offline, requires ruptures)',
    description: 'Runs offline multivariate change-point detection on the full session, if the optional dependency is installed.',
  },
];

const EventSensitivityPanel = ({
  sessionId,
  serviceEndpoint,
  onEventsChange,
  initialEvents = null,
  fpsEstimate: fpsEstimateProp = 0.5,
}) => {
  const [method, setMethod] = useState('ensemble');
  const [zThreshold, setZThreshold] = useState(2.5);
  const [minMagnitude, setMinMagnitude] = useState(0.10);
  const [consensus, setConsensus] = useState(2);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState(initialEvents || []);
  const [error, setError] = useState(null);
  const [fpsEstimate, setFpsEstimate] = useState(fpsEstimateProp || 0.5);
  const [diagnostics, setDiagnostics] = useState(null);

  // Sync prop -> state when parent provides a new estimate (e.g. after temporal
  // payload arrives). Backend response below remains the ultimate source of truth.
  useEffect(() => {
    if (fpsEstimateProp && fpsEstimateProp > 0) {
      setFpsEstimate(fpsEstimateProp);
    }
  }, [fpsEstimateProp]);

  // Keep `onEventsChange` in a ref so a fresh callback identity from the
  // parent (e.g. an inline arrow `(evs)=>setX(evs)`) does NOT invalidate
  // the refetch callback and re-fire the effect on every render.
  const onEventsChangeRef = useRef(onEventsChange);
  useEffect(() => {
    onEventsChangeRef.current = onEventsChange;
  }, [onEventsChange]);

  // Buffer the latest desired params; the debounced effect reads from this ref
  // so we always fetch with the *most recent* slider position rather than a
  // stale snapshot taken when the timer was scheduled.
  const pendingParamsRef = useRef({
    method: 'ensemble',
    zThreshold: 2.5,
    minMagnitude: 0.10,
    consensus: 2,
  });
  // Latest in-flight AbortController so a new request cancels the previous.
  const abortRef = useRef(null);
  // Safety fallback: re-enable inputs after 1500ms even if the response is lost.
  const loadingFallbackRef = useRef(null);

  const refetch = useCallback(async () => {
    if (!sessionId) return;
    // Cancel any in-flight request — keep latest only.
    if (abortRef.current) {
      try { abortRef.current.abort(); } catch (_) { /* noop */ }
    }
    const controller = new AbortController();
    abortRef.current = controller;

    const params = pendingParamsRef.current;
    setLoading(true);
    setError(null);
    if (loadingFallbackRef.current) clearTimeout(loadingFallbackRef.current);
    loadingFallbackRef.current = setTimeout(() => {
      setLoading(false);
    }, 1500);

    try {
      const url = new URL(`${serviceEndpoint}/session/${sessionId}/events`);
      url.searchParams.set('method', params.method);
      url.searchParams.set('z_threshold', params.zThreshold);
      url.searchParams.set('min_magnitude', params.minMagnitude);
      url.searchParams.set('consensus_min_methods', params.consensus);
      const r = await fetch(url, { signal: controller.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const evs = data.events || [];
      setEvents(evs);
      setDiagnostics(data.diagnostics || null);
      if (data.fps_estimate && Number(data.fps_estimate) > 0) {
        setFpsEstimate(Number(data.fps_estimate));
      }
      const cb = onEventsChangeRef.current;
      if (cb) cb(evs, data);
    } catch (e) {
      if (e.name === 'AbortError') return; // superseded by newer request
      setError(e.message || 'Failed to fetch events');
    } finally {
      // Only the latest controller flips loading off.
      if (abortRef.current === controller) {
        setLoading(false);
        if (loadingFallbackRef.current) {
          clearTimeout(loadingFallbackRef.current);
          loadingFallbackRef.current = null;
        }
      }
    }
  }, [sessionId, serviceEndpoint]);

  // Mirror state -> ref every render so the debounced fetch reads the latest.
  useEffect(() => {
    pendingParamsRef.current = { method, zThreshold, minMagnitude, consensus };
  }, [method, zThreshold, minMagnitude, consensus]);

  // Re-run on parameter change with a 200ms idle window. Reads from the ref so
  // the very last slider value wins regardless of intermediate renders.
  useEffect(() => {
    if (!sessionId) return undefined;
    const t = setTimeout(() => { refetch(); }, 200);
    return () => clearTimeout(t);
  }, [refetch, sessionId, method, zThreshold, minMagnitude, consensus]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      if (abortRef.current) {
        try { abortRef.current.abort(); } catch (_) { /* noop */ }
      }
      if (loadingFallbackRef.current) clearTimeout(loadingFallbackRef.current);
    };
  }, []);

  const inputsDisabled = loading;
  const methodMeta = METHODS.find((m) => m.value === method) || METHODS[0];
  const selectedMinFrames = diagnostics?.selected_min_required_frames;
  const sampleCount = diagnostics?.sample_count;
  const sampleLimited = (
    Number.isFinite(Number(sampleCount))
    && Number.isFinite(Number(selectedMinFrames))
    && Number(sampleCount) < Number(selectedMinFrames)
  );
  const emptyHint = sampleLimited
    ? `${methodMeta.label} needs ${selectedMinFrames} frames at current settings; this session has ${sampleCount}.`
    : diagnostics?.no_event_reason || 'Try lowering the z-threshold or min Δp slider.';

  return (
    <div className="event-sensitivity-panel">
      <div className="panel-header">
        <h3>Event Detection Sensitivity</h3>
        <span className="panel-status">
          {loading ? 'updating…' : `${events.length} event${events.length === 1 ? '' : 's'}`}
        </span>
      </div>
      <div className="panel-grid">
        <label className="panel-row">
          <span className="panel-label">Method</span>
          <select
            value={method}
            disabled={inputsDisabled}
            onChange={(e) => setMethod(e.target.value)}
          >
            {METHODS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </label>

        <label className="panel-row">
          <span className="panel-label">
            z-threshold <span className="panel-value">{zThreshold.toFixed(2)}</span>
          </span>
          <input
            type="range" min="0.5" max="5.0" step="0.1"
            value={zThreshold}
            disabled={inputsDisabled}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setZThreshold(v);
              pendingParamsRef.current = { ...pendingParamsRef.current, zThreshold: v };
            }}
          />
        </label>

        <label className="panel-row">
          <span className="panel-label">
            min Δp <span className="panel-value">{minMagnitude.toFixed(2)}</span>
          </span>
          <input
            type="range" min="0.02" max="0.50" step="0.01"
            value={minMagnitude}
            disabled={inputsDisabled}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setMinMagnitude(v);
              pendingParamsRef.current = { ...pendingParamsRef.current, minMagnitude: v };
            }}
          />
        </label>

        <label className="panel-row">
          <span className="panel-label">
            consensus methods <span className="panel-value">{consensus}</span>
          </span>
          <input
            type="range" min="1" max="3" step="1"
            value={consensus}
            disabled={inputsDisabled || method !== 'ensemble'}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              setConsensus(v);
              pendingParamsRef.current = { ...pendingParamsRef.current, consensus: v };
            }}
          />
        </label>
      </div>

      <div className="panel-method-note">
        <span>{methodMeta.description}</span>
        {diagnostics && (
          <span>
            Samples {sampleCount ?? 0}
            {selectedMinFrames ? ` · minimum ${selectedMinFrames}` : ''}
            {method === 'pelt' && diagnostics.pelt_available === false ? ' · PELT unavailable' : ''}
          </span>
        )}
      </div>

      {error && <div className="panel-error">{error}</div>}

      <div className="panel-event-list">
        {events.length === 0 && !loading ? (
          <div className="panel-event-empty">
            <span className="empty-mark">∅</span>
            <span>No events at the current sensitivity.</span>
            <span className="empty-hint">
              {emptyHint}
            </span>
          </div>
        ) : (
          <>
            <div className="panel-event-header" role="row">
              <span>#</span>
              <span>Time</span>
              <span>Transition</span>
              <span>Δp</span>
              <span>Conf</span>
              <span className="col-methods">Methods</span>
            </div>
            {events.slice(0, 25).map((ev, i) => (
              <div className="panel-event-row" key={`${ev.frame_idx}-${i}`} role="row">
                <span>{i + 1}</span>
                <span>{(ev.frame_idx / Math.max(0.1, fpsEstimate)).toFixed(1)}s</span>
                <span className="col-transition">
                  <span className="emo">{ev.from_emotion}</span>
                  <span className="arrow">→</span>
                  <span className="emo">{ev.to_emotion}</span>
                </span>
                <span>{Number(ev.magnitude).toFixed(2)}</span>
                <span>{Number(ev.confidence).toFixed(2)}</span>
                <span className="col-methods">{(ev.methods || []).join(', ')}</span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
};

export default EventSensitivityPanel;
