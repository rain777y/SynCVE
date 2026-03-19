/**
 * EmotionProgressBars Component
 *
 * Display real-time confidence progress bars for 7 emotions
 * Extracted from PemoFacial library and integrated
 *
 * @version 1.0.0
 * @date 2025-10-25
 */

import React from 'react';
import './EmotionProgressBars.css';

const EmotionProgressBars = ({
  emotions = {
    angry: 0,
    neutral: 0,
    happy: 0,
    fear: 0,
    surprise: 0,
    sad: 0,
    disgust: 0
  },
  position = 'right',
  width = 500,
  showPercentage = true,
  animated = true,
  language = 'en'
}) => {

  // Emotion configuration
  const emotionConfig = [
    {
      key: 'angry',
      label: { en: 'Angry', zh: 'Angry' },
      color: '#FF0000'
    },
    {
      key: 'neutral',
      label: { en: 'Neutral', zh: 'Neutral' },
      color: '#90EE90'
    },
    {
      key: 'happy',
      label: { en: 'Happy', zh: 'Happy' },
      color: '#FFA500'
    },
    {
      key: 'fear',
      label: { en: 'Fear', zh: 'Fear' },
      color: '#ADD8E6'
    },
    {
      key: 'surprise',
      label: { en: 'Surprised', zh: 'Surprised' },
      color: '#FFFF00'
    },
    {
      key: 'sad',
      label: { en: 'Sad', zh: 'Sad' },
      color: '#808080'
    },
    {
      key: 'disgust',
      label: { en: 'Disgusted', zh: 'Disgusted' },
      color: '#FFC0CB'
    }
  ];

  // Position styles
  const positionStyles = {
    right: { position: 'absolute', right: '100px', top: '60px' },
    left: { position: 'absolute', left: '100px', top: '60px' },
    top: { position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)' },
    bottom: { position: 'absolute', bottom: '20px', left: '50%', transform: 'translateX(-50%)' }
  };

  return (
    <div 
      className={`emotion-progress-container ${animated ? 'animated' : ''}`}
      style={{
        ...positionStyles[position],
        width: `${width}px`
      }}
    >
      <h3 className="emotion-progress-title">Emotion Analysis</h3>
      
      {emotionConfig.map(({ key, label, color }) => {
        const value = Math.round(emotions[key] || 0);
        
        return (
          <div key={key} className="emotion-row">
            <label 
              htmlFor={`emotion-${key}`} 
              className="emotion-label"
              style={{ color }}
            >
              {label[language] || label.en}
            </label>
            
            <progress 
              id={`emotion-${key}`}
              className="emotion-progress"
              value={value} 
              max="100"
              style={{
                '--progress-color': color
              }}
            />
            
            {showPercentage && (
              <span className="emotion-value">
                {value}%
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default EmotionProgressBars;

