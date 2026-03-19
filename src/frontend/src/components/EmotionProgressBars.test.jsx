/**
 * Tests for the EmotionProgressBars component.
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import EmotionProgressBars from './EmotionProgressBars';

describe('EmotionProgressBars', () => {
  const ALL_EMOTION_LABELS = ['Angry', 'Neutral', 'Happy', 'Fear', 'Surprised', 'Sad', 'Disgusted'];

  test('renders all 7 emotion bars', () => {
    render(<EmotionProgressBars />);
    ALL_EMOTION_LABELS.forEach((label) => {
      expect(screen.getByText(label)).toBeInTheDocument();
    });
  });

  test('renders title', () => {
    render(<EmotionProgressBars />);
    expect(screen.getByText('Emotion Analysis')).toBeInTheDocument();
  });

  test('displays correct percentage for each emotion', () => {
    const emotions = {
      angry: 10,
      neutral: 20,
      happy: 50,
      fear: 5,
      surprise: 8,
      sad: 4,
      disgust: 3,
    };
    render(<EmotionProgressBars emotions={emotions} showPercentage={true} />);

    expect(screen.getByText('10%')).toBeInTheDocument();
    expect(screen.getByText('20%')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('5%')).toBeInTheDocument();
    expect(screen.getByText('8%')).toBeInTheDocument();
    expect(screen.getByText('4%')).toBeInTheDocument();
    expect(screen.getByText('3%')).toBeInTheDocument();
  });

  test('handles zero values without error', () => {
    const emotions = {
      angry: 0,
      neutral: 0,
      happy: 0,
      fear: 0,
      surprise: 0,
      sad: 0,
      disgust: 0,
    };
    render(<EmotionProgressBars emotions={emotions} showPercentage={true} />);
    // All should show 0%
    const zeros = screen.getAllByText('0%');
    expect(zeros.length).toBe(7);
  });

  test('handles missing emotions gracefully (defaults to 0)', () => {
    render(<EmotionProgressBars emotions={{}} showPercentage={true} />);
    const zeros = screen.getAllByText('0%');
    expect(zeros.length).toBe(7);
  });

  test('renders progress elements for each emotion', () => {
    render(<EmotionProgressBars />);
    const progressBars = document.querySelectorAll('progress.emotion-progress');
    expect(progressBars.length).toBe(7);
  });

  test('progress bar max is 100', () => {
    render(<EmotionProgressBars />);
    const progressBars = document.querySelectorAll('progress.emotion-progress');
    progressBars.forEach((bar) => {
      expect(bar.getAttribute('max')).toBe('100');
    });
  });

  test('hides percentages when showPercentage is false', () => {
    const emotions = { angry: 50, neutral: 0, happy: 0, fear: 0, surprise: 0, sad: 0, disgust: 0 };
    render(<EmotionProgressBars emotions={emotions} showPercentage={false} />);
    expect(screen.queryByText('50%')).not.toBeInTheDocument();
  });

  test('applies animated CSS class when animated is true', () => {
    const { container } = render(<EmotionProgressBars animated={true} />);
    expect(container.querySelector('.emotion-progress-container.animated')).toBeTruthy();
  });

  test('does not apply animated CSS class when animated is false', () => {
    const { container } = render(<EmotionProgressBars animated={false} />);
    expect(container.querySelector('.emotion-progress-container.animated')).toBeFalsy();
    expect(container.querySelector('.emotion-progress-container')).toBeTruthy();
  });

  test('applies custom width', () => {
    const { container } = render(<EmotionProgressBars width={600} />);
    const wrapper = container.querySelector('.emotion-progress-container');
    expect(wrapper.style.width).toBe('600px');
  });

  test('dominant emotion has highest value in progress bar', () => {
    const emotions = {
      angry: 5,
      neutral: 10,
      happy: 80,
      fear: 1,
      surprise: 2,
      sad: 1,
      disgust: 1,
    };
    render(<EmotionProgressBars emotions={emotions} showPercentage={true} />);
    expect(screen.getByText('80%')).toBeInTheDocument();
  });
});
