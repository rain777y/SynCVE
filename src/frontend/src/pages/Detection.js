import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import EmotionDetector from '../components/EmotionDetector';
import './Detection.css';

const Detection = () => {
  const { user, isAuthenticated, signInWithGoogle, signOut, loading } = useAuth();
  const navigate = useNavigate();

  const handleAuthenticate = async () => {
    try {
      await signInWithGoogle();
    } catch (error) {
      console.error('Sign in error:', error);
      alert('Failed to sign in with Google. Please try again.');
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (error) {
      console.error('Sign out error:', error);
      alert('Failed to sign out. Please try again.');
    }
  };

  const handleGoHome = () => {
    navigate('/');
  };

  if (loading) {
    return (
      <div className="detection-loading">
        <div className="spinner"></div>
        <p>Initialising acquisition station…</p>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <div className="detection-page">
        <nav className="instrument-bar detection-bar">
          <div
            className="instrument-bar__brand"
            onClick={handleGoHome}
            role="button"
            tabIndex={0}
          >
            <span className="instrument-bar__mark">SynCVE</span>
            <span className="instrument-bar__sub">Acquisition Console</span>
          </div>
          <div className="instrument-bar__nav">
            <button onClick={handleGoHome} className="ink-link">Home</button>
          </div>
        </nav>
        <div className="detection-content">
          <div className="ed-shell">
            <div className="ed-empty-card">
              <span className="eyebrow">Authentication required</span>
              <h2 className="ed-empty-card__title">
                Sign in to <em>begin a session.</em>
              </h2>
              <hr className="rule" style={{ margin: '20px 0' }} />
              <p className="ed-empty-card__lede">
                Acquisition sessions are bound to your account so reports can be
                persisted and reviewed in the archive.
              </p>
              <button onClick={handleAuthenticate} className="btn-ink btn-ink--lg">
                Authenticate
                <span className="arrow">→</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const subjectId = (user?.id || user?.email || 'anon').slice(0, 6).toUpperCase();

  return (
    <div className="detection-page">
      <nav className="instrument-bar detection-bar">
        <div
          className="instrument-bar__brand"
          onClick={handleGoHome}
          role="button"
          tabIndex={0}
        >
          <span className="instrument-bar__mark">SynCVE</span>
          <span className="instrument-bar__sub">Acquisition Console</span>
        </div>

        <div className="instrument-bar__meta">
          <span>SUBJECT <b>#{subjectId}</b></span>
          <span>STATION <b>USM-LAB-A</b></span>
          <span>MODE <b>LIVE</b></span>
        </div>

        <div className="instrument-bar__nav">
          <button onClick={() => navigate('/history')} className="ink-link">
            Archive
          </button>
          <span className="ink-link ink-link--muted">
            {user?.user_metadata?.full_name || user?.email}
          </span>
          <button onClick={handleSignOut} className="btn-ghost">
            Sign out
          </button>
        </div>
      </nav>

      <div className="detection-content">
        <EmotionDetector />
      </div>
    </div>
  );
};

export default Detection;
