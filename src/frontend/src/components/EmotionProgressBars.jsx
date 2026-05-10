/**
 * EmotionProgressBars Component
 *
 * Display real-time confidence progress bars for 7 emotions.
 *
 * @version 2.0.0  (editorial palette)
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
  position = 'static',
  width = '100%',
  showPercentage = true,
  animated = true,
  language = 'en'
}) => {
  const emotionConfig = [
    { key: 'angry',    code: 'A1', label: { en: 'Anger',     zh: '愤怒' } },
    { key: 'happy',    code: 'A2', label: { en: 'Happiness', zh: '愉悦' } },
    { key: 'sad',      code: 'A3', label: { en: 'Sadness',   zh: '悲伤' } },
    { key: 'fear',     code: 'A4', label: { en: 'Fear',      zh: '恐惧' } },
    { key: 'surprise', code: 'A5', label: { en: 'Surprise',  zh: '惊讶' } },
    { key: 'disgust',  code: 'A6', label: { en: 'Disgust',   zh: '厌恶' } },
    { key: 'neutral',  code: 'A7', label: { en: 'Neutral',   zh: '平静' } },
  ];

  const positionStyles = {
    right:  { position: 'absolute', right: '100px', top: '60px' },
    left:   { position: 'absolute', left: '100px',  top: '60px' },
    top:    { position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)' },
    bottom: { position: 'absolute', bottom: '20px', left: '50%', transform: 'translateX(-50%)' },
    static: {},
  };

  const widthValue = typeof width === 'number' ? `${width}px` : width;

  return (
    <div
      className={`epb ${animated ? 'epb--animated' : ''}`}
      style={{
        ...(positionStyles[position] || {}),
        width: widthValue,
      }}
    >
      {emotionConfig.map(({ key, code, label }) => {
        const value = Math.round(emotions[key] || 0);
        return (
          <div key={key} className="epb__row" style={{ '--ax-color': `var(--e-${key})` }}>
            <span className="epb__code mono">{code}</span>
            <label htmlFor={`emotion-${key}`} className="epb__label">
              {label[language] || label.en}
            </label>

            <div className="epb__track" aria-hidden="true">
              <span className="epb__fill" style={{ width: `${value}%` }} />
              <span className="epb__tick epb__tick--25" />
              <span className="epb__tick epb__tick--50" />
              <span className="epb__tick epb__tick--75" />
            </div>

            {showPercentage && (
              <span className="epb__value mono">
                {value.toString().padStart(2, '0')}
                <em>%</em>
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default EmotionProgressBars;
