import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import EmotionDetector from '../components/EmotionDetector';
import './Detection.css';

const Detection = () => {
  const { user, isAuthenticated, signOut, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, loading, navigate]);

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
        <p>Loading...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="detection-page">
      {/* Navigation Bar */}
      <nav className="detection-nav">
        <div className="nav-content">
          <div className="nav-brand" onClick={handleGoHome} style={{ cursor: 'pointer' }}>
            <h1 className="brand-title">SynCVE</h1>
            <p className="brand-subtitle">Emotion Recognition System</p>
          </div>
          <div className="nav-actions">
            <button onClick={() => navigate('/history')} className="btn btn-secondary" style={{ marginRight: '1rem' }}>
              History
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

      {/* Main Content */}
      <div className="detection-content">
        <EmotionDetector />
      </div>
    </div>
  );
};

export default Detection;

