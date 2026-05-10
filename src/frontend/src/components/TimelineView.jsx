/**
 * TimelineView — Three-track time-aligned visualisation for clinical review.
 *
 *   Track 1 — Smoothed emotion-probability stack (Recharts area chart).
 *   Track 2 — Event-marker lane (clickable; calls onSeek with the t_sec).
 *   Track 3 — ASR transcript lane (renders as time-anchored chips when
 *             transcript segments are supplied; otherwise hidden).
 *
 * The component is intentionally framework-light: it consumes a single
 * `temporal` shape (the same JSON returned by /session/<id>) plus an
 * optional `events` and `transcript` prop so it can be rendered both
 * standalone and inside SessionReport.
 *
 * Time alignment is derived from `frame_idx / fps_estimate` and rendered
 * along a shared X axis.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  CartesianGrid, ReferenceLine,
} from 'recharts';
import './TimelineView.css';

const EMOTION_COLORS = {
  happy: '#FFB020',
  sad: '#4A90D9',
  angry: '#E74C3C',
  fear: '#9B59B6',
  surprise: '#F39C12',
  disgust: '#27AE60',
  neutral: '#95A5A6',
};

const EMOTIONS = ['angry', 'disgust', 'fear', 'sad', 'surprise', 'happy', 'neutral'];

const formatSec = (s) => {
  if (s == null || Number.isNaN(s)) return '—';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
};

/**
 * Convert temporal summary -> recharts row format.
 */
function buildAreaData(temporal) {
  if (!temporal || !Array.isArray(temporal.smoothed_timeline)) return [];
  const fps = Number(temporal.fps_estimate) || 0.5;
  return temporal.smoothed_timeline.map((row, i) => {
    const out = { t_sec: row.frame / fps, frame: row.frame ?? i };
    EMOTIONS.forEach((e) => {
      out[e] = Number(row.emotions?.[e] || 0);
    });
    return out;
  });
}

const TimelineView = ({
  temporal,
  events: eventsProp,
  transcript = [],
  onSeek,
  selectedTime = null,
  height = 220,
}) => {
  const [hoveredEvent, setHoveredEvent] = useState(null);
  const chartRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(0);
  const fps = Number(temporal?.fps_estimate) || 0.5;
  const events = useMemo(
    () => (eventsProp || temporal?.events || []).map((e) => ({
      ...e,
      t_sec: (e.frame_idx ?? 0) / fps,
    })),
    [eventsProp, temporal, fps],
  );

  const areaData = useMemo(() => buildAreaData(temporal), [temporal]);
  const totalSec = areaData.length ? areaData[areaData.length - 1].t_sec : 0;

  useEffect(() => {
    const measure = () => {
      const nextWidth = Math.floor(chartRef.current?.clientWidth || 0);
      setChartWidth(Math.max(1, nextWidth));
    };

    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(measure);
      if (chartRef.current) observer.observe(chartRef.current);
      return () => observer.disconnect();
    }

    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  // Memoise the events-track JSX so frequent parent re-renders (e.g. timer
  // ticks) don't rebuild N markers when nothing about events/duration changed.
  const eventsTrack = useMemo(() => (
    <div className="timeline-track timeline-track-events">
      <div className="timeline-track-label">Events</div>
      <div className="timeline-track-canvas">
        {events.map((ev, i) => {
          const left = totalSec > 0 ? (ev.t_sec / totalSec) * 100 : 0;
          const conf = Number(ev.confidence || 0);
          return (
            <button
              key={`ev-marker-${i}`}
              className={`timeline-event-marker ${conf >= 0.7 ? 'high' : conf >= 0.5 ? 'mid' : 'low'}`}
              style={{ left: `${left}%` }}
              title={`${formatSec(ev.t_sec)} · ${ev.from_emotion} → ${ev.to_emotion}\nΔp ${ev.magnitude} · conf ${ev.confidence}\nmethods: ${(ev.methods || []).join(', ')}`}
              onMouseEnter={() => setHoveredEvent(ev)}
              onMouseLeave={() => setHoveredEvent(null)}
              onClick={() => onSeek && onSeek(ev.t_sec, ev)}
            >
              <span className="marker-dot" />
            </button>
          );
        })}
        {events.length === 0 && (
          <div className="timeline-event-empty">No consensus events detected.</div>
        )}
      </div>
    </div>
  ), [events, totalSec, onSeek]);

  // Memoise the transcript-track JSX (only re-renders when transcript or
  // duration changes — clicks on events shouldn't rebuild this lane).
  const transcriptTrack = useMemo(() => {
    if (!transcript || transcript.length === 0) return null;
    return (
      <div className="timeline-track timeline-track-transcript">
        <div className="timeline-track-label">Transcript</div>
        <div className="timeline-track-canvas">
          {transcript.map((seg, i) => {
            const start = (seg.t_start_sec ?? seg.t_sec) || 0;
            const end = seg.t_end_sec ?? start + 1;
            const width = totalSec > 0 ? ((end - start) / totalSec) * 100 : 0;
            const left = totalSec > 0 ? (start / totalSec) * 100 : 0;
            return (
              <div
                key={`tr-${i}`}
                className="transcript-chip"
                style={{ left: `${left}%`, width: `${Math.max(width, 1)}%` }}
                title={`${formatSec(start)}–${formatSec(end)}: ${seg.text || ''}`}
                onClick={() => onSeek && onSeek(start, null)}
              >
                {seg.label || seg.text || ''}
              </div>
            );
          })}
        </div>
      </div>
    );
  }, [transcript, totalSec, onSeek]);

  if (!areaData.length) {
    return (
      <div className="timeline-empty">
        <p>No temporal data available for this session yet.</p>
      </div>
    );
  }

  return (
    <div className="timeline-view">
      <div ref={chartRef} className="timeline-track timeline-track-emotions" style={{ height }}>
        {chartWidth > 0 ? (
            <AreaChart width={chartWidth} height={height} data={areaData} margin={{ top: 10, right: 12, bottom: 18, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" />
              <XAxis
                dataKey="t_sec"
                tickFormatter={formatSec}
                type="number"
                domain={[0, Math.max(totalSec, 1)]}
                stroke="#888"
                fontSize={11}
              />
              <YAxis stroke="#888" fontSize={11} domain={[0, 1]} />
              <Tooltip
                labelFormatter={(t) => `t = ${formatSec(t)}`}
                formatter={(v, name) => [`${(v * 100).toFixed(0)}%`, name]}
                contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #444' }}
              />
              {EMOTIONS.map((emo) => (
                <Area
                  key={emo}
                  type="monotone"
                  dataKey={emo}
                  stackId="1"
                  stroke={EMOTION_COLORS[emo]}
                  fill={EMOTION_COLORS[emo]}
                  fillOpacity={0.6}
                />
              ))}
              {events.map((ev, i) => (
                <ReferenceLine
                  key={`ev-${i}`}
                  x={ev.t_sec}
                  stroke="#ffffff"
                  strokeWidth={1.4}
                  strokeDasharray="3 2"
                  ifOverflow="extendDomain"
                />
              ))}
              {selectedTime !== null && (
                <ReferenceLine
                  x={selectedTime}
                  stroke="#ff4081"
                  strokeWidth={2}
                  ifOverflow="extendDomain"
                />
              )}
            </AreaChart>
        ) : <div className="timeline-chart-placeholder" />}
      </div>

      {/* Track 2 — clickable event markers */}
      {eventsTrack}

      {/* Track 3 — ASR / transcript lane (only when supplied) */}
      {transcriptTrack}

      {/* Hover detail popup */}
      {hoveredEvent && (
        <div className="timeline-event-detail">
          <div className="detail-time">{formatSec(hoveredEvent.t_sec)}</div>
          <div className="detail-transition">
            {hoveredEvent.from_emotion} → {hoveredEvent.to_emotion}
          </div>
          <div className="detail-meta">
            Δp {hoveredEvent.magnitude} · conf {hoveredEvent.confidence}
            <span className="detail-methods">
              {' '}
              [{(hoveredEvent.methods || []).join(', ')}]
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// Shallow-compare props so parent re-renders that don't change inputs are
// skipped. (React.memo's default is already shallow; an explicit comparator
// keeps the contract obvious and lets us extend it without surprise.)
function arePropsEqual(prev, next) {
  return (
    prev.temporal === next.temporal
    && prev.events === next.events
    && prev.transcript === next.transcript
    && prev.onSeek === next.onSeek
    && prev.selectedTime === next.selectedTime
    && prev.height === next.height
  );
}

export default React.memo(TimelineView, arePropsEqual);
