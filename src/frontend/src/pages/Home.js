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
      <div className="home-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="home-container">
      {/* Navigation */}
      <nav className="home-nav">
        <div className="nav-content">
          <div className="nav-brand">
            <h1 className="brand-title">SynCVE</h1>
            <p className="brand-subtitle">Emotion Recognition System</p>
          </div>
          <div className="nav-actions">
            {isAuthenticated ? (
              <>
                <button onClick={handleStartDetection} className="btn btn-primary">
                  Detection
                </button>
                <button onClick={() => navigate('/history')} className="btn btn-secondary">
                  History
                </button>
                <div className="user-info">
                  <span className="user-name">{user?.user_metadata?.full_name || user?.email}</span>
                  <button onClick={handleSignOut} className="btn btn-secondary">
                    Sign Out
                  </button>
                </div>
              </>
            ) : (
              <button onClick={handleGoogleSignIn} className="btn btn-google">
                <svg className="google-icon" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                </svg>
                Sign in with Google
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-text">
            <h2 className="hero-title">
              Real-time Facial Emotion Recognition
            </h2>
            <p className="hero-description">
              Detect and analyze emotions from facial expressions in real-time using advanced AI technology.
              Our system provides accurate emotion detection with detailed analysis.
            </p>

            <div className="hero-features">
              <div className="feature-item">
                <div className="feature-icon">🎯</div>
                <div className="feature-text">
                  <h3>High Accuracy</h3>
                  <p>Advanced deep learning models for precise emotion detection</p>
                </div>
              </div>
              <div className="feature-item">
                <div className="feature-icon">⚡</div>
                <div className="feature-text">
                  <h3>Real-time Analysis</h3>
                  <p>Continuous emotion detection with configurable intervals</p>
                </div>
              </div>
              <div className="feature-item">
                <div className="feature-icon">🔒</div>
                <div className="feature-text">
                  <h3>Secure & Private</h3>
                  <p>Your data is processed securely with Google authentication</p>
                </div>
              </div>
            </div>

            <div className="hero-cta">
              {isAuthenticated ? (
                <button onClick={handleStartDetection} className="btn btn-large btn-primary">
                  <span className="btn-icon">📹</span>
                  Start Detection
                </button>
              ) : (
                <button onClick={handleGoogleSignIn} className="btn btn-large btn-google">
                  <svg className="google-icon" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                  </svg>
                  Get Started
                </button>
              )}
            </div>
          </div>

          <div className="hero-visual">
            <div className="emotion-preview">
              <div className="emotion-grid">
                <div className="emotion-card">😊 Happy</div>
                <div className="emotion-card">😢 Sad</div>
                <div className="emotion-card">😠 Angry</div>
                <div className="emotion-card">😨 Fear</div>
                <div className="emotion-card">😲 Surprise</div>
                <div className="emotion-card">😐 Neutral</div>
                <div className="emotion-card">🤢 Disgust</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="home-footer">
        <p>© 2025 SynCVE - Emotion Recognition System</p>
      </footer>
    </div>
  );
};

export default Home;

