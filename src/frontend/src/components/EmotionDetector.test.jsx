/**
 * Tests for the EmotionDetector component.
 *
 * Mocks: fetch, navigator.mediaDevices, AuthContext, supabase.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';

// ---------- Mocks must be set up BEFORE importing the component ----------

// Mock supabase lib (must come before any component import that touches AuthContext)
jest.mock('../lib/supabase', () => ({
  supabase: {
    auth: {
      getSession: jest.fn().mockResolvedValue({ data: { session: null } }),
      onAuthStateChange: jest.fn().mockReturnValue({
        data: { subscription: { unsubscribe: jest.fn() } },
      }),
      signInWithOAuth: jest.fn().mockResolvedValue({ data: {}, error: null }),
      signOut: jest.fn().mockResolvedValue({ error: null }),
    },
  },
}));

// Mock AuthContext
jest.mock('../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'test-user', email: 'test@test.com' },
    session: {},
    loading: false,
    isAuthenticated: true,
    signInWithGoogle: jest.fn(),
    signOut: jest.fn(),
  }),
}));

import EmotionDetector from './EmotionDetector';

// Stub navigator.mediaDevices
const mockGetUserMedia = jest.fn().mockResolvedValue({
  getTracks: () => [{ stop: jest.fn() }],
});

Object.defineProperty(global.navigator, 'mediaDevices', {
  value: { getUserMedia: mockGetUserMedia },
  writable: true,
});

// Stub HTMLVideoElement.play
HTMLVideoElement.prototype.play = jest.fn().mockResolvedValue(undefined);

// Stub HTMLCanvasElement.getContext
HTMLCanvasElement.prototype.getContext = jest.fn().mockReturnValue({
  drawImage: jest.fn(),
});
HTMLCanvasElement.prototype.toDataURL = jest.fn().mockReturnValue('data:image/jpeg;base64,AAAA');

// ---------- Tests ----------

describe('EmotionDetector', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('renders without crashing', () => {
    render(<EmotionDetector />);
    expect(screen.getByText('Emotion Recognition System')).toBeInTheDocument();
  });

  test('renders start detection button', () => {
    render(<EmotionDetector />);
    const startBtn = screen.getByText(/Start Detection/i);
    expect(startBtn).toBeInTheDocument();
    expect(startBtn).not.toBeDisabled();
  });

  test('renders stop detection button (disabled initially)', () => {
    render(<EmotionDetector />);
    const stopBtn = screen.getByText(/Stop Detection/i);
    expect(stopBtn).toBeInTheDocument();
    expect(stopBtn).toBeDisabled();
  });

  test('displays dominant emotion label', () => {
    render(<EmotionDetector />);
    expect(screen.getByText('Dominant Emotion')).toBeInTheDocument();
    // Default dominant emotion is neutral (uppercased)
    expect(screen.getByText('NEUTRAL')).toBeInTheDocument();
  });

  test('displays confidence value', () => {
    render(<EmotionDetector />);
    expect(screen.getByText(/Confidence:/i)).toBeInTheDocument();
  });

  test('displays session info section', () => {
    render(<EmotionDetector />);
    expect(screen.getByText('Session Time:')).toBeInTheDocument();
    expect(screen.getByText('Detections:')).toBeInTheDocument();
    expect(screen.getByText('Status:')).toBeInTheDocument();
  });

  test('start detection calls session/start endpoint', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      json: jest.fn().mockResolvedValue({ session_id: 'new-session-123' }),
      ok: true,
      status: 200,
    });

    render(<EmotionDetector />);

    const startBtn = screen.getByText(/Start Detection/i);
    await act(async () => {
      fireEvent.click(startBtn);
    });

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/session/start'),
      expect.objectContaining({ method: 'POST' })
    );
  });

  test('shows error when session start fails', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      json: jest.fn().mockResolvedValue({ error: 'DB down' }),
      ok: false,
      status: 500,
    });

    render(<EmotionDetector />);

    const startBtn = screen.getByText(/Start Detection/i);
    await act(async () => {
      fireEvent.click(startBtn);
    });

    await waitFor(() => {
      expect(screen.getByText(/Unable to start session/i)).toBeInTheDocument();
    });
  });

  test('renders interval control select', () => {
    render(<EmotionDetector />);
    const select = screen.getByLabelText(/Interval/i);
    expect(select).toBeInTheDocument();
  });

  test('displays EmotionProgressBars component', () => {
    render(<EmotionDetector />);
    expect(screen.getByText('Emotion Analysis')).toBeInTheDocument();
  });

  test('displays SystemStatus component', () => {
    render(<EmotionDetector />);
    expect(screen.getByText('System Status')).toBeInTheDocument();
  });
});
