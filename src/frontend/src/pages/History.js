
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import SessionReport from '../components/SessionReport';
import { resolveServiceEndpoint } from '../lib/serviceEndpoint';
import './History.css';

/**
 * Build a report prop object from a completed/paused session's persisted data.
 * Returns null if no temporal or report data is available (text-only fallback).
 */
const buildReportFromSession = (session) => {
    const temporal = session.temporal_summary;
    const pauseReport = session.metadata?.pause_report;

    // Prefer pause_report (has full metrics + rankings)
    if (pauseReport) {
        return {
            ...pauseReport,
            temporal: temporal || pauseReport.temporal,
        };
    }

    // Build minimal report from temporal_summary alone
    if (temporal) {
        return {
            text_summary: session.summary || 'Session completed.',
            temporal,
            metrics: { samples: temporal.frame_count },
            stats_summary: {},
            emotion_ranking: [],
            emotion_timeline: [],
        };
    }

    return null;
};

const History = () => {
    const { user, isAuthenticated, signInWithGoogle, signOut, loading: authLoading } = useAuth();
    const navigate = useNavigate();
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedSession, setSelectedSession] = useState(null);
    const [error, setError] = useState(null);

    const serviceEndpoint = resolveServiceEndpoint();

    const handleAuthenticate = async () => {
        try {
            await signInWithGoogle();
        } catch (e) {
            console.error('Sign in error:', e);
            alert('Failed to sign in with Google. Please try again.');
        }
    };

    useEffect(() => {
        if (authLoading) return;
        if (!isAuthenticated) {
            // No redirect — show empty state instead.
            setLoading(false);
            return;
        }

        const fetchHistory = async () => {
            try {
                const url = new URL(`${serviceEndpoint}/session/history`);
                if (user?.id) {
                    url.searchParams.append('user_id', user.id);
                }
                url.searchParams.append('limit', 20);

                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error('Failed to fetch history');
                }

                const data = await response.json();
                if (data.sessions) {
                    setSessions(data.sessions);
                }
            } catch (err) {
                console.error('Error loading history:', err);
                setError('Failed to load emotion history.');
            } finally {
                setLoading(false);
            }
        };

        if (isAuthenticated) {
            fetchHistory();
        }
    }, [isAuthenticated, authLoading, navigate, user, serviceEndpoint]);

    const handleSignOut = async () => {
        try {
            await signOut();
            navigate('/');
        } catch (error) {
            console.error('Sign out error:', error);
        }
    };

    const handleSessionClick = (session) => {
        setSelectedSession(session);
    };

    const closeModal = () => {
        setSelectedSession(null);
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleString(undefined, {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    if (authLoading) return <div className="loading-container"><div className="spinner"></div><p>Authenticating…</p></div>;

    if (!isAuthenticated || !user) {
        return (
            <div className="history-page">
                <nav className="instrument-bar history-bar">
                    <div
                        className="instrument-bar__brand"
                        onClick={() => navigate('/')}
                        role="button"
                        tabIndex={0}
                    >
                        <span className="instrument-bar__mark">SynCVE</span>
                        <span className="instrument-bar__sub">Session Archive</span>
                    </div>
                    <div className="instrument-bar__nav">
                        <button onClick={() => navigate('/')} className="ink-link">Home</button>
                    </div>
                </nav>
                <div className="history-content">
                    <div className="ed-empty-card">
                        <span className="eyebrow">Authentication required</span>
                        <h2 className="ed-empty-card__title">
                            Sign in to see <em>your sessions.</em>
                        </h2>
                        <hr className="rule" style={{ margin: '20px 0' }} />
                        <p className="ed-empty-card__lede">
                            The archive is keyed to your account. Authenticate to load
                            your previous acquisitions.
                        </p>
                        <button onClick={handleAuthenticate} className="btn-ink btn-ink--lg">
                            Authenticate
                            <span className="arrow">→</span>
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    const selectedReport = selectedSession ? buildReportFromSession(selectedSession) : null;

    return (
        <div className="history-page">
            <nav className="instrument-bar history-bar">
                <div
                    className="instrument-bar__brand"
                    onClick={() => navigate('/')}
                    role="button"
                    tabIndex={0}
                >
                    <span className="instrument-bar__mark">SynCVE</span>
                    <span className="instrument-bar__sub">Session Archive</span>
                </div>

                <div className="instrument-bar__meta">
                    <span>RECORDS <b>{sessions.length}</b></span>
                    <span>SCOPE <b>USER</b></span>
                    <span>SORT <b>RECENT FIRST</b></span>
                </div>

                <div className="instrument-bar__nav">
                    <button onClick={() => navigate('/detection')} className="btn-ink">
                        New Session
                        <span className="arrow">→</span>
                    </button>
                    <span className="ink-link ink-link--muted">
                        {user?.user_metadata?.full_name || user?.email}
                    </span>
                    <button onClick={handleSignOut} className="btn-ghost">
                        Sign out
                    </button>
                </div>
            </nav>

            <div className="history-content">
                <header className="history-header">
                    <span className="eyebrow">Volume 04 — Session Archive</span>
                    <h2 className="history-header__title">
                        Past <em>Acquisitions.</em>
                    </h2>
                    <p className="history-header__lede">
                        A reverse-chronological index of recorded sessions, their stability
                        readings, and Gemini-authored summaries. Select a row to expand the
                        full clinical report.
                    </p>
                </header>

                {error && <div className="history-error">{error}</div>}

                {loading ? (
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <p>Loading sessions…</p>
                    </div>
                ) : sessions.length === 0 ? (
                    <div className="empty-state">
                        <span className="eyebrow">No records yet</span>
                        <p className="empty-state__line">
                            No sessions yet — head to{' '}
                            <button
                                type="button"
                                className="ink-link"
                                onClick={() => navigate('/detection')}
                                style={{ display: 'inline', padding: 0 }}
                            >
                                Detection
                            </button>{' '}
                            to start one.
                        </p>
                        <button onClick={() => navigate('/detection')} className="btn-ink btn-ink--lg">
                            Begin First Acquisition
                            <span className="arrow">→</span>
                        </button>
                    </div>
                ) : (
                    <ol className="sessions-table">
                        <li className="sessions-table__head">
                            <span className="col-num">№</span>
                            <span className="col-date">Acquisition Date</span>
                            <span className="col-summary">Synopsis</span>
                            <span className="col-stability">Stability</span>
                            <span className="col-status">Status</span>
                        </li>
                        {sessions.map((session, idx) => {
                            const stability = session.temporal_summary?.stability_score;
                            const stabilityPct = stability != null ? Math.round(stability * 100) : null;
                            const num = String(sessions.length - idx).padStart(3, '0');
                            return (
                                <li
                                    key={session.id}
                                    className="sessions-table__row"
                                    onClick={() => handleSessionClick(session)}
                                    role="button"
                                    tabIndex={0}
                                >
                                    <span className="col-num mono">{num}</span>
                                    <span className="col-date mono">{formatDate(session.created_at)}</span>
                                    <span className="col-summary">
                                        {session.summary || <em className="muted">— no synopsis —</em>}
                                    </span>
                                    <span className="col-stability">
                                        {stabilityPct != null ? (
                                            <span className="stability-readout mono">
                                                <span
                                                    className="stability-readout__bar"
                                                    style={{ '--w': `${stabilityPct}%` }}
                                                />
                                                <span>{stabilityPct}%</span>
                                            </span>
                                        ) : (
                                            <span className="muted">—</span>
                                        )}
                                    </span>
                                    <span className={`col-status status-${session.status || 'completed'}`}>
                                        {session.status || 'completed'}
                                    </span>
                                </li>
                            );
                        })}
                    </ol>
                )}
            </div>

            {selectedSession && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content modal-content-wide" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3 className="modal-title">Session Report</h3>
                            <span className="modal-date">{formatDate(selectedSession.created_at)}</span>
                            <button className="close-btn" onClick={closeModal}>&times;</button>
                        </div>
                        <div className="modal-body">
                            {selectedReport ? (
                                <SessionReport
                                    report={selectedReport}
                                    onResume={null}
                                    onStop={null}
                                    sessionId={selectedSession?.id}
                                    serviceEndpoint={serviceEndpoint}
                                />
                            ) : (
                                /* Text-only fallback for older sessions without temporal data */
                                <>
                                    <div className="detail-section">
                                        <h3>Summary</h3>
                                        <p style={{ lineHeight: '1.6', color: 'var(--text-secondary)' }}>
                                            {selectedSession.summary || "No summary generated for this session."}
                                        </p>
                                    </div>

                                    <div className="detail-section">
                                        <h3>Recommendations</h3>
                                        <div className="recommendation-list">
                                            {selectedSession.recommendations ? (
                                                <div>
                                                    {selectedSession.recommendations.split('\n').map((line, i) => {
                                                        const trimmed = line.trim();
                                                        if (!trimmed) return null;
                                                        return <p key={i} style={{ marginBottom: '0.5rem' }}>{trimmed}</p>;
                                                    })}
                                                </div>
                                            ) : (
                                                <p>No recommendations available.</p>
                                            )}
                                        </div>
                                    </div>

                                    <div className="detail-section" style={{ marginBottom: 0 }}>
                                        <h3>Details</h3>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '0.9rem' }}>
                                            <div><strong>Started:</strong> {formatDate(selectedSession.created_at)}</div>
                                            <div><strong>Ended:</strong> {selectedSession.ended_at ? formatDate(selectedSession.ended_at) : 'Ongoing'}</div>
                                            <div><strong>Status:</strong> <span style={{ textTransform: 'capitalize' }}>{selectedSession.status}</span></div>
                                            <div><strong>ID:</strong> <span style={{ fontFamily: 'monospace' }}>{selectedSession.id.slice(0, 8)}...</span></div>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default History;
