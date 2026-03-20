
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import SessionReport from '../components/SessionReport';
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
    const { user, isAuthenticated, signOut, loading: authLoading } = useAuth();
    const navigate = useNavigate();
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedSession, setSelectedSession] = useState(null);
    const [error, setError] = useState(null);

    const serviceEndpoint = process.env.REACT_APP_SERVICE_ENDPOINT || 'http://localhost:5005';

    useEffect(() => {
        if (!authLoading && !isAuthenticated) {
            navigate('/');
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

    if (authLoading) return <div className="loading-container"><div className="spinner"></div></div>;

    const selectedReport = selectedSession ? buildReportFromSession(selectedSession) : null;

    return (
        <div className="history-page">
            {/* Navigation */}
            <nav className="history-nav">
                <div className="nav-content">
                    <div className="nav-brand" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
                        <h1 className="brand-title">SynCVE</h1>
                        <p className="brand-subtitle">Emotion Recognition System</p>
                    </div>
                    <div className="nav-actions">
                        <button onClick={() => navigate('/detection')} className="btn btn-primary">
                            New Session
                        </button>
                        <div className="user-info">
                            <span className="user-name">{user?.user_metadata?.full_name || user?.email}</span>
                            <button onClick={handleSignOut} className="btn btn-secondary">
                                Sign Out
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            <div className="history-content">
                <header className="page-header">
                    <h2 className="page-title">Emotion History</h2>
                    <p className="page-subtitle">Review your past emotional analysis sessions and insights.</p>
                </header>

                {error && <div className="error-message">{error}</div>}

                {loading ? (
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <p>Loading sessions...</p>
                    </div>
                ) : sessions.length === 0 ? (
                    <div className="empty-state">
                        <p>No sessions recorded yet.</p>
                        <button onClick={() => navigate('/detection')} className="btn btn-primary btn-large" style={{ marginTop: '1rem' }}>
                            Start Your First Session
                        </button>
                    </div>
                ) : (
                    <div className="sessions-grid">
                        {sessions.map(session => (
                            <div
                                key={session.id}
                                className="session-card"
                                onClick={() => handleSessionClick(session)}
                            >
                                <div className="session-header">
                                    <span className="session-date">{formatDate(session.created_at)}</span>
                                    <span className={`session-status status-${session.status || 'completed'}`}>
                                        {session.status || 'completed'}
                                    </span>
                                </div>
                                <div className="session-summary">
                                    {session.summary || "No summary available."}
                                </div>
                                <div className="session-footer">
                                    {session.temporal_summary?.stability_score != null && (
                                        <span className="stability-badge">
                                            Stability: {(session.temporal_summary.stability_score * 100).toFixed(0)}%
                                        </span>
                                    )}
                                    <span>View report &rarr;</span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Session Detail Modal */}
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
                                <SessionReport report={selectedReport} onResume={null} onStop={null} />
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
