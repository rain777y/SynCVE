/**
 * SessionReport — Data-driven emotion analytics dashboard.
 * Renders structured report data using Recharts, including temporal analysis.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  AreaChart, Area, CartesianGrid, PieChart, Pie, Legend,
} from 'recharts';
import TimelineView from './TimelineView';
import EventSensitivityPanel from './EventSensitivityPanel';
import { resolveServiceEndpoint } from '../lib/serviceEndpoint';
import './SessionReport.css';

const SERVICE_ENDPOINT = resolveServiceEndpoint();

const EMOTION_COLORS = {
  happy: '#FFB020',
  sad: '#4A90D9',
  angry: '#E74C3C',
  fear: '#9B59B6',
  surprise: '#F39C12',
  disgust: '#27AE60',
  neutral: '#95A5A6',
};

const EMOTION_ICONS = {
  happy: '😊', sad: '😢', angry: '😠', fear: '😨',
  surprise: '😲', disgust: '🤢', neutral: '😐',
};

const MeasuredChart = ({ height, children }) => {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const measure = () => {
      const nextWidth = Math.floor(ref.current?.clientWidth || 0);
      setWidth(Math.max(1, nextWidth));
    };

    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(measure);
      if (ref.current) observer.observe(ref.current);
      return () => observer.disconnect();
    }

    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  return (
    <div ref={ref} className="measured-chart" style={{ height }}>
      {width > 0 ? children(width, height) : null}
    </div>
  );
};

const SessionReport = ({ report, onResume, onStop, sessionId = null }) => {
  const [clinicalMetrics, setClinicalMetrics] = useState(null);
  const [clinicalLoading, setClinicalLoading] = useState(false);
  const [clinicalError, setClinicalError] = useState(null);
  const [overrideEvents, setOverrideEvents] = useState(null);

  // Stable callback for the sensitivity panel — passing an inline arrow
  // here causes EventSensitivityPanel.refetch to re-create on every render,
  // which fires its useEffect, which calls setEvents, which re-renders the
  // parent — i.e. an infinite "flashing" loop.
  const handleEventsChange = useCallback((evs) => setOverrideEvents(evs), []);

  // Lazy-load clinical metrics ONCE per sessionId. Don't put loading/data state
  // in the deps — that re-fires the effect on every state transition, and a 4xx
  // response (e.g. empty_session after /session/stop) leaves clinicalMetrics
  // null, which would refetch forever.
  const fetchedFor = useRef(null);
  useEffect(() => {
    if (!sessionId) return;
    if (fetchedFor.current === sessionId) return;
    fetchedFor.current = sessionId;

    let cancelled = false;
    setClinicalLoading(true);
    setClinicalError(null);
    fetch(`${SERVICE_ENDPOINT}/session/${sessionId}/clinical_metrics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then(async (r) => {
        const body = await r.json().catch(() => ({}));
        if (!r.ok) {
          throw new Error(body?.error || `HTTP ${r.status}`);
        }
        return body;
      })
      .then((data) => { if (!cancelled) setClinicalMetrics(data); })
      .catch((e) => { if (!cancelled) setClinicalError(String(e?.message || e)); })
      .finally(() => { if (!cancelled) setClinicalLoading(false); });

    return () => { cancelled = true; };
  }, [sessionId]);

  if (!report) return null;

  const {
    text_summary,
    emotion_ranking = [],
    emotion_timeline = [],
    metrics = {},
    stats_summary = {},
    temporal,
  } = report;

  const reportDownloadUrl = (fmt) => (
    `${SERVICE_ENDPOINT}/session/${sessionId}/clinical_report?format=${fmt}&download=1`
  );

  const dominant = stats_summary.dominant || metrics.dominant || 'neutral';
  const dominantScore = (stats_summary.score || metrics.dominant_score || 0);
  const samples = metrics.samples || 0;

  // Bar chart data (sorted)
  const barData = emotion_ranking.map(({ emotion, score }) => ({
    name: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    score: Math.round(score * 100),
    color: EMOTION_COLORS[emotion] || '#888',
  }));

  // Timeline data
  const timelineData = emotion_timeline.map((entry, i) => {
    const scores = {};
    Object.keys(EMOTION_COLORS).forEach(emo => { scores[emo] = 0; });
    if (entry.emotion) {
      scores[entry.emotion] = 100;
    }
    return { index: i, ...scores, label: entry.emotion || '' };
  });

  // Smoothed timeline from temporal data
  const smoothedTimelineData = temporal?.smoothed_timeline?.map((entry) => {
    const emos = {};
    if (entry.emotions) {
      Object.entries(entry.emotions).forEach(([emo, val]) => {
        emos[emo] = Math.round(val * 100);
      });
    }
    return { frame: entry.frame, ...emos, dominant: entry.dominant };
  }) || [];

  // Duration pie data
  const durationData = temporal?.durations?.map(d => ({
    name: d.emotion.charAt(0).toUpperCase() + d.emotion.slice(1),
    value: d.duration_sec,
    emotion: d.emotion,
  })) || [];

  const dominantColor = EMOTION_COLORS[dominant] || '#888';

  return (
    <div className="session-report">
      {/* Header */}
      <div className="report-header">
        <h2>Session Report</h2>
        <div className="report-badge" style={{ borderColor: dominantColor }}>
          <span className="badge-icon">{EMOTION_ICONS[dominant] || '🔍'}</span>
          <span className="badge-label">{dominant.toUpperCase()}</span>
          <span className="badge-score">{(dominantScore * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Summary */}
      <div className="report-summary">
        <p>{text_summary}</p>
        <span className="sample-count">{samples} frames analyzed</span>
      </div>

      {/* Stability Score */}
      {temporal?.stability_score != null && (
        <div className="stability-gauge" aria-label={`Emotional stability: ${(temporal.stability_score * 100).toFixed(0)}%`}>
          <h4>Emotional Stability</h4>
          <div className="gauge-bar">
            <div className="gauge-fill" style={{ width: `${temporal.stability_score * 100}%` }} />
          </div>
          <span className="gauge-label">{(temporal.stability_score * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* Emotion distribution */}
      <div className="report-charts">
        <div className="chart-card chart-wide">
          <h3>Emotion Distribution</h3>
          <MeasuredChart height={280}>
            {(width, height) => (
              <BarChart width={width} height={height} data={barData} layout="vertical" margin={{ left: 10 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#999' }} />
                <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#ccc', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#222', border: '1px solid #444', borderRadius: 8 }}
                  labelStyle={{ color: '#fff' }} formatter={(value) => [`${value}%`, 'Score']} />
                <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                  {barData.map((entry, i) => (<Cell key={i} fill={entry.color} />))}
                </Bar>
              </BarChart>
            )}
          </MeasuredChart>
        </div>
      </div>

      {/* Fallback: raw timeline if no temporal data */}
      {!smoothedTimelineData.length && timelineData.length > 2 && (
        <div className="chart-card chart-wide">
          <h3>Emotion Timeline</h3>
          <MeasuredChart height={200}>
            {(width, height) => (
              <AreaChart width={width} height={height} data={timelineData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="index" tick={{ fill: '#666', fontSize: 10 }} />
                <YAxis domain={[0, 100]} hide />
                <Tooltip contentStyle={{ background: '#222', border: '1px solid #444', borderRadius: 8 }} />
                {Object.entries(EMOTION_COLORS).map(([emo, color]) => (
                  <Area key={emo} type="monotone" dataKey={emo} stackId="1"
                  stroke={color} fill={color} fillOpacity={0.6} />
                ))}
              </AreaChart>
            )}
          </MeasuredChart>
        </div>
      )}

      {/* Temporal: Transitions + Duration + Trends */}
      {temporal && (
        <div className="report-charts">
          {/* Transitions */}
          {temporal.transitions?.length > 0 && (
            <div className="chart-card">
              <h3>Emotion Transitions ({temporal.transition_count})</h3>
              <div className="transition-flow">
                {temporal.transitions.map((t, i) => (
                  <div key={i} className="transition-item">
                    <span className="from" style={{ color: EMOTION_COLORS[t.from_emotion] || '#888' }}>
                      {t.from_emotion}
                    </span>
                    <span className="arrow" aria-label="transitions to">&rarr;</span>
                    <span className="to" style={{ color: EMOTION_COLORS[t.to_emotion] || '#888' }}>
                      {t.to_emotion}
                    </span>
                    <small className="frame-label">frame {t.frame_idx}</small>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Duration Pie */}
          {durationData.length > 0 && (
            <div className="chart-card">
              <h3>Time per Emotion</h3>
              <MeasuredChart height={220}>
                {(width, height) => (
                  <PieChart width={width} height={height}>
                    <Pie data={durationData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                      outerRadius={70} label={({ name, value }) => `${name} ${value.toFixed(0)}s`}>
                      {durationData.map((d, i) => (
                        <Cell key={i} fill={EMOTION_COLORS[d.emotion] || '#888'} />
                      ))}
                    </Pie>
                    <Tooltip formatter={v => `${v.toFixed(1)}s`} />
                    <Legend />
                  </PieChart>
                )}
              </MeasuredChart>
            </div>
          )}
        </div>
      )}

      {/* Trend Indicators */}
      {temporal?.trends?.filter(t => t.direction !== 'stable').length > 0 && (
        <div className="chart-card chart-wide">
          <h3>Emotion Trends</h3>
          <div className="trends-grid">
            {temporal.trends.filter(t => t.direction !== 'stable').map(t => (
              <div key={t.emotion} className={`trend-item trend-${t.direction}`}>
                <span className="trend-arrow" aria-label={t.direction}>
                  {t.direction === 'increasing' ? '\u2191' : '\u2193'}
                </span>
                <span className="trend-emotion">{t.emotion}</span>
                <small>R\u00B2={t.r_squared.toFixed(2)}</small>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Image (if mode=full) */}
      {report.visual_report_url && (
        <div className="chart-card chart-wide">
          <h3>AI Dashboard</h3>
          <img src={report.visual_report_url} alt="AI Visual Report" style={{ width: '100%', borderRadius: 8 }} />
        </div>
      )}

      {/* --- Axis 1A — Event-level Timeline (clickable markers) --- */}
      {temporal && (temporal.smoothed_timeline?.length || 0) > 2 && (
        <div className="chart-card chart-wide">
          <h3>Event Timeline (Axis 1A)</h3>
          <TimelineView
            temporal={temporal}
            events={overrideEvents ?? temporal.events ?? []}
          />
        </div>
      )}

      {/* --- Axis 4 — Sensitivity panel (post-session exploration) --- */}
      {sessionId && temporal && (
        <div className="chart-card chart-wide">
          <EventSensitivityPanel
            sessionId={sessionId}
            serviceEndpoint={SERVICE_ENDPOINT}
            initialEvents={temporal.events || []}
            onEventsChange={handleEventsChange}
            fpsEstimate={temporal?.fps_estimate || 0.5}
          />
        </div>
      )}

      {/* Report exports */}
      {sessionId && (
        <div className="report-export-actions">
          <a
            className="btn btn-secondary"
            href={reportDownloadUrl('md')}
          >
            Export Markdown
          </a>
          <a
            className="btn btn-secondary"
            href={reportDownloadUrl('pdf')}
          >
            Export PDF
          </a>
        </div>
      )}

      {/* --- Axis 1A — Clinical metrics block --- */}
      {(clinicalMetrics || clinicalLoading || clinicalError) && (
        <div className="chart-card chart-wide">
          <h3>Clinical Metrics</h3>
          {clinicalLoading && <p>Computing clinical metrics…</p>}
          {clinicalError && <p style={{ color: '#ff7575' }}>{clinicalError}</p>}
          {clinicalMetrics && (
            <div className="clinical-metrics-grid">
              <Metric label="Valence (mean)" value={clinicalMetrics.valence_mean} />
              <Metric label="Valence (std)" value={clinicalMetrics.valence_std} />
              <Metric
                label="Drift / min"
                value={clinicalMetrics.valence_drift_per_min}
                ci={clinicalMetrics.valence_drift_ci95}
              />
              <Metric
                label="Affect blunting"
                value={clinicalMetrics.affect_blunting_score}
                hint="0 = full range, 1 = flat"
              />
              <Metric
                label="Reactivity (events/min)"
                value={clinicalMetrics.reactivity_events_per_min}
              />
              <Metric
                label="Suppression index"
                value={clinicalMetrics.suppression_index}
              />
              <Metric
                label="Incongruence"
                value={clinicalMetrics.incongruence_index}
              />
              <Metric
                label="High-conf events"
                value={`${clinicalMetrics.high_confidence_event_count || 0} / ${clinicalMetrics.event_count || 0}`}
                isString
              />
            </div>
          )}
        </div>
      )}

      {/* Disclaimer */}
      {report.disclaimer && (
        <p className="disclaimer-text">{report.disclaimer}</p>
      )}

      {/* Actions (hidden in History view) */}
      {(onResume || onStop) && (
        <div className="report-actions">
          {onResume && <button onClick={onResume} className="btn btn-start">▶ Resume session</button>}
          {onStop && <button onClick={onStop} className="btn btn-stop">⏹ End session</button>}
        </div>
      )}
    </div>
  );
};

/** Tiny helper to render a clinical-metric tile. */
const Metric = ({ label, value, ci, hint, isString }) => {
  let display;
  if (value == null || value === undefined) {
    display = '—';
  } else if (isString) {
    display = value;
  } else {
    display = Number(value).toFixed(3);
  }
  return (
    <div className="clinical-metric-tile">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{display}</div>
      {ci && Array.isArray(ci) && (
        <div className="metric-ci">
          95% CI {Number(ci[0]).toFixed(3)} … {Number(ci[1]).toFixed(3)}
        </div>
      )}
      {hint && <div className="metric-hint">{hint}</div>}
    </div>
  );
};

export default SessionReport;
