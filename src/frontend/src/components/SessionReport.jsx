/**
 * SessionReport — Data-driven emotion analytics dashboard.
 * Renders structured report data using Recharts.
 * Replaces the old AI-generated image approach with instant, interactive charts.
 */
import React from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  AreaChart, Area, CartesianGrid,
} from 'recharts';
import './SessionReport.css';

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

const SessionReport = ({ report, onResume, onStop }) => {
  if (!report) return null;

  const {
    text_summary,
    emotion_ranking = [],
    emotion_timeline = [],
    metrics = {},
    stats_summary = {},
  } = report;

  const dominant = stats_summary.dominant || metrics.dominant || 'neutral';
  const dominantScore = (stats_summary.score || metrics.dominant_score || 0);
  const samples = metrics.samples || 0;
  const averages = metrics.averages || {};

  // Radar chart data
  const radarData = Object.entries(averages).map(([emotion, score]) => ({
    emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: Math.round(score * 100),
    fullMark: 100,
  }));

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

      {/* Charts Grid */}
      <div className="report-charts">
        {/* Radar Chart */}
        <div className="chart-card">
          <h3>Emotion Profile</h3>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="rgba(255,255,255,0.1)" />
              <PolarAngleAxis
                dataKey="emotion"
                tick={{ fill: '#ccc', fontSize: 12 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 100]}
                tick={{ fill: '#666', fontSize: 10 }}
              />
              <Radar
                dataKey="value"
                stroke={dominantColor}
                fill={dominantColor}
                fillOpacity={0.25}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart */}
        <div className="chart-card">
          <h3>Emotion Ranking</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} layout="vertical" margin={{ left: 10 }}>
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#999' }} />
              <YAxis
                type="category"
                dataKey="name"
                width={80}
                tick={{ fill: '#ccc', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{ background: '#222', border: '1px solid #444', borderRadius: 8 }}
                labelStyle={{ color: '#fff' }}
                formatter={(value) => [`${value}%`, 'Score']}
              />
              <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                {barData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Timeline */}
      {timelineData.length > 2 && (
        <div className="chart-card chart-wide">
          <h3>Emotion Timeline</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={timelineData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="index" tick={{ fill: '#666', fontSize: 10 }} />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ background: '#222', border: '1px solid #444', borderRadius: 8 }}
              />
              {Object.entries(EMOTION_COLORS).map(([emo, color]) => (
                <Area
                  key={emo}
                  type="monotone"
                  dataKey={emo}
                  stackId="1"
                  stroke={color}
                  fill={color}
                  fillOpacity={0.6}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* AI Image (if mode=full) */}
      {report.visual_report_url && (
        <div className="chart-card chart-wide">
          <h3>AI Dashboard</h3>
          <img
            src={report.visual_report_url}
            alt="AI Visual Report"
            style={{ width: '100%', borderRadius: 8 }}
          />
        </div>
      )}

      {/* Actions */}
      <div className="report-actions">
        <button onClick={onResume} className="btn btn-start">▶ Resume</button>
        <button onClick={onStop} className="btn btn-stop">⏹ End Session</button>
      </div>
    </div>
  );
};

export default SessionReport;
