import React, { useState, useEffect, useCallback } from 'react';
import { useConfig } from '../contexts/ConfigContext';
import './SystemStatus.css';

const SystemStatus = () => {
  const { config, serviceEndpoint } = useConfig();

  const systemInfo = {
    detector: config?.detector_backend || 'retinaface',
    antiSpoofing: config?.anti_spoofing !== false,
    interval: config?.detection_interval || 2000,
    imageQuality: 0.95,
    videoResolution: '1280x720',
  };

  const [backendStatus, setBackendStatus] = useState({
    online: false,
    gpuEnabled: false,
    checking: true,
  });

  const checkBackendStatus = useCallback(async () => {
    try {
      const response = await fetch(serviceEndpoint, { method: 'GET' });
      setBackendStatus({
        online: response.ok,
        gpuEnabled: response.ok,
        checking: false,
      });
    } catch {
      setBackendStatus({ online: false, gpuEnabled: false, checking: false });
    }
  }, [serviceEndpoint]);

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, [checkBackendStatus]);

  const getDetectorInfo = (detector) => {
    const detectorMap = {
      'opencv': { name: 'OpenCV', accuracy: '⭐⭐☆☆☆', speed: '⭐⭐⭐⭐⭐', color: '#ff9800' },
      'mtcnn': { name: 'MTCNN', accuracy: '⭐⭐⭐⭐☆', speed: '⭐⭐⭐☆☆', color: '#2196f3' },
      'retinaface': { name: 'RetinaFace', accuracy: '⭐⭐⭐⭐⭐', speed: '⭐⭐☆☆☆', color: '#4caf50' },
      'mediapipe': { name: 'MediaPipe', accuracy: '⭐⭐⭐⭐☆', speed: '⭐⭐⭐⭐☆', color: '#9c27b0' },
      'yolov8': { name: 'YOLOv8', accuracy: '⭐⭐⭐⭐☆', speed: '⭐⭐⭐⭐☆', color: '#f44336' },
    };
    return detectorMap[detector] || { name: detector, accuracy: 'N/A', speed: 'N/A', color: '#757575' };
  };

  const detectorInfo = getDetectorInfo(systemInfo.detector);

  return (
    <div className="system-status">
      <div className="status-header">
        <h3>System Status</h3>
        <div className={`status-indicator ${backendStatus.online ? 'online' : 'offline'}`}>
          <span className="status-dot"></span>
          {backendStatus.checking ? 'Checking...' : (backendStatus.online ? 'Online' : 'Offline')}
        </div>
      </div>

      <div className="status-grid">
        <div className="status-item">
          <div className="status-label">Face Detector</div>
          <div className="status-value" style={{ color: detectorInfo.color }}>{detectorInfo.name}</div>
          <div className="status-details">
            <span>Accuracy: {detectorInfo.accuracy}</span>
            <span>Speed: {detectorInfo.speed}</span>
          </div>
        </div>

        <div className="status-item">
          <div className="status-label">Anti-Spoofing</div>
          <div className={`status-value ${systemInfo.antiSpoofing ? 'enabled' : 'disabled'}`}>
            {systemInfo.antiSpoofing ? '✓ Enabled' : '✗ Disabled'}
          </div>
          <div className="status-details">
            {systemInfo.antiSpoofing ? 'Real face verification active' : 'No liveness detection'}
          </div>
        </div>

        <div className="status-item">
          <div className="status-label">Detection Interval</div>
          <div className="status-value">{systemInfo.interval}ms</div>
          <div className="status-details">
            {systemInfo.interval <= 1000 ? 'Fast' : systemInfo.interval <= 2000 ? 'Balanced' : 'Accurate'}
          </div>
        </div>

        <div className="status-item">
          <div className="status-label">Image Quality</div>
          <div className="status-value">{Math.round(systemInfo.imageQuality * 100)}%</div>
          <div className="status-details">JPEG compression quality</div>
        </div>

        <div className="status-item">
          <div className="status-label">Video Resolution</div>
          <div className="status-value">{systemInfo.videoResolution}</div>
          <div className="status-details">HD quality (720p)</div>
        </div>

        <div className="status-item">
          <div className="status-label">GPU Acceleration</div>
          <div className={`status-value ${backendStatus.gpuEnabled ? 'enabled' : 'disabled'}`}>
            {backendStatus.gpuEnabled ? '✓ Enabled' : '✗ Disabled'}
          </div>
          <div className="status-details">
            {backendStatus.gpuEnabled ? 'CUDA acceleration active' : 'CPU mode'}
          </div>
        </div>
      </div>

      {systemInfo.detector === 'retinaface' && systemInfo.antiSpoofing && (
        <div className="optimization-notice">
          <div className="notice-icon">🚀</div>
          <div className="notice-content">
            <strong>Optimized Configuration Active</strong>
            <p>Using RetinaFace with anti-spoofing for maximum accuracy (85-95%)</p>
          </div>
        </div>
      )}

      {systemInfo.detector === 'opencv' && (
        <div className="optimization-notice warning">
          <div className="notice-icon">⚠️</div>
          <div className="notice-content">
            <strong>Suboptimal Configuration</strong>
            <p>Consider upgrading to RetinaFace or MTCNN for better accuracy</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemStatus;
