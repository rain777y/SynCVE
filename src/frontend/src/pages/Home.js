import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import './Home.css';

const Home = () => {
  const { user, isAuthenticated, signInWithGoogle, signOut, loading } = useAuth();
  const navigate = useNavigate();

  const handleGoogleSignIn = async () => {
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
    } catch (error) {
      console.error('Sign out error:', error);
      alert('Failed to sign out. Please try again.');
    }
  };

  const handleStartDetection = () => {
    navigate('/detection');
  };

  if (loading) {
    return (
      <div className="home-page">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Initialising instrument…</p>
        </div>
      </div>
    );
  }

  const todayCode = new Date()
    .toISOString()
    .slice(0, 10)
    .replace(/-/g, '.');

  return (
    <div className="home-page">
      {/* ──────────────── instrument bar ──────────────── */}
      <nav className="instrument-bar home-bar">
        <div
          className="instrument-bar__brand"
          onClick={() => navigate('/')}
          role="button"
          tabIndex={0}
        >
          <span className="instrument-bar__mark">SynCVE</span>
          <span className="instrument-bar__sub">Synthetic Continuous Visual Emotion</span>
        </div>

        <div className="instrument-bar__meta">
          <span>BUILD <b>04 · 2026</b></span>
          <span>STATION <b>USM-LAB-A</b></span>
          <span>UTC <b>{todayCode}</b></span>
        </div>

        <div className="instrument-bar__nav">
          {isAuthenticated ? (
            <>
              <button className="ink-link" onClick={handleStartDetection}>Detection</button>
              <button className="ink-link" onClick={() => navigate('/history')}>Archive</button>
              <span className="ink-link ink-link--muted">
                {user?.user_metadata?.full_name || user?.email}
              </span>
              <button className="btn-ghost" onClick={handleSignOut}>Sign out</button>
            </>
          ) : (
            <button className="btn-ink" onClick={handleGoogleSignIn}>
              <svg className="google-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" opacity="0.55" />
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" opacity="0.7" />
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" opacity="0.85" />
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              Authenticate
            </button>
          )}
        </div>
      </nav>

      {/* ──────────────── hero ──────────────── */}
      <section className="home-hero">
        <header className="home-hero__head">
          <span className="eyebrow">Volume 04 / Issue 09 — Real-time Affective Inference</span>
        </header>

        <div className="home-hero__grid">
          <div className="home-hero__copy">
            <h1 className="home-hero__title">
              <span className="home-hero__line">A research</span>
              <span className="home-hero__line home-hero__line--em">instrument</span>
              <span className="home-hero__line">for the continuous study</span>
              <span className="home-hero__line">of <em>facial affect.</em></span>
            </h1>

            <hr className="rule" style={{ margin: '36px 0 28px' }} />

            <p className="home-hero__lede">
              SynCVE captures, fuses, and reports affective signals on the millisecond.
              Built around a temporally-aware ensemble — DeepFace ⨯ Gemini ⨯ uncertainty
              fusion — for clinical-adjacent pilots that demand more than a single softmax
              vote.
            </p>

            <div className="home-hero__cta">
              {isAuthenticated ? (
                <button className="btn-ink btn-ink--lg" onClick={handleStartDetection}>
                  Begin Acquisition
                  <span className="arrow">→</span>
                </button>
              ) : (
                <button className="btn-ink btn-ink--lg" onClick={handleGoogleSignIn}>
                  Authenticate to Begin
                  <span className="arrow">→</span>
                </button>
              )}
              <a
                className="ink-link"
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
              >
                Read the protocol ↗
              </a>
            </div>
          </div>

          <aside className="home-hero__sidecar">
            <div className="sidecar-block">
              <span className="sidecar-block__num">01</span>
              <h4>Continuous Inference</h4>
              <p>
                Frame-rate-aware sampling at 0.5–3.0 s. EMA smoothing and event detectors
                reduce label thrash without hiding micro-expressions.
              </p>
            </div>
            <div className="sidecar-block">
              <span className="sidecar-block__num">02</span>
              <h4>Uncertainty-Aware</h4>
              <p>
                Logits fused across DeepFace and Gemini, weighted by a calibrated
                confidence prior. Disagreement is reported, not hidden.
              </p>
            </div>
            <div className="sidecar-block">
              <span className="sidecar-block__num">03</span>
              <h4>Clinical-Grade Reporting</h4>
              <p>
                Per-session stability, axis-1 metrics, and a printable transcript that
                respects the limits of behavioural inference.
              </p>
            </div>
          </aside>
        </div>
      </section>

      {/* ──────────────── specimen panel ──────────────── */}
      <section className="home-specimen">
        <div className="home-specimen__head">
          <span className="eyebrow">Specimen Index — 7 Discrete Affects</span>
          <span className="home-specimen__counter mono">N = 7 · Ekman + Neutral</span>
        </div>

        <ol className="home-specimen__list">
          {[
            { code: 'A1', name: 'Anger',    note: 'High arousal · negative valence' },
            { code: 'A2', name: 'Happiness', note: 'High arousal · positive valence' },
            { code: 'A3', name: 'Sadness',   note: 'Low arousal · negative valence' },
            { code: 'A4', name: 'Fear',      note: 'High arousal · negative valence' },
            { code: 'A5', name: 'Surprise',  note: 'Variable valence · transient' },
            { code: 'A6', name: 'Disgust',   note: 'Low arousal · negative valence' },
            { code: 'A7', name: 'Neutral',   note: 'Baseline reference state' },
          ].map((row) => (
            <li key={row.code} className="specimen-row">
              <span className="specimen-row__code mono">{row.code}</span>
              <span className="specimen-row__name">{row.name}</span>
              <span className="specimen-row__note">{row.note}</span>
              <span
                className={`specimen-row__swatch swatch-${row.name.toLowerCase()}`}
                aria-hidden
              />
            </li>
          ))}
        </ol>
      </section>

      <footer className="home-footer">
        <div className="home-footer__inner">
          <span className="mono">© 2026 — SynCVE / FYP, Universiti Sains Malaysia</span>
          <span className="mono">v0.4 · Opus pipeline · Gemini 1.5 · DeepFace 0.0.93</span>
        </div>
      </footer>
    </div>
  );
};

export default Home;
