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

const mockJsonResponse = (body, ok = true, status = ok ? 200 : 500) => ({
  ok,
  status,
  text: jest.fn().mockResolvedValue(JSON.stringify(body)),
  json: jest.fn().mockResolvedValue(body),
});

const continueConsent = async () => {
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: /i understand/i }));
  });
  return screen.findByRole('button', { name: /start session/i });
};

// ---------- Tests ----------

describe('EmotionDetector', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetUserMedia.mockResolvedValue({
      getTracks: () => [{ stop: jest.fn() }],
    });
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('renders consent gate first', () => {
    render(<EmotionDetector />);
    expect(screen.getByText(/Informed Consent/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /i understand/i })).toBeInTheDocument();
  });

  test('renders detection controls after consent', async () => {
    render(<EmotionDetector />);
    const startBtn = await continueConsent();
    expect(startBtn).not.toBeDisabled();
    expect(screen.getByRole('button', { name: /end session/i })).toBeDisabled();
    expect(screen.getByText('Dominant Affect')).toBeInTheDocument();
    expect(screen.getByText('Confidence')).toBeInTheDocument();
    expect(screen.getByText('Elapsed')).toBeInTheDocument();
    expect(screen.getByText('N samples')).toBeInTheDocument();
    expect(screen.getByText(/Distribution/i)).toBeInTheDocument();
  });

  test('does not render the old system status or manual interval control', async () => {
    render(<EmotionDetector />);
    await continueConsent();
    expect(screen.queryByText('System Status')).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/interval/i)).not.toBeInTheDocument();
  });

  test('start session calls session/start endpoint', async () => {
    global.fetch.mockImplementation((url) => {
      if (String(url).includes('/session/start')) {
        return Promise.resolve(mockJsonResponse({ session_id: 'new-session-123' }));
      }
      if (String(url).includes('/analyze')) {
        return Promise.resolve(mockJsonResponse({ results: [] }));
      }
      return Promise.resolve(mockJsonResponse({}));
    });

    const { unmount } = render(<EmotionDetector />);
    const startBtn = await continueConsent();

    await act(async () => {
      fireEvent.click(startBtn);
    });

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/session/start'),
        expect.objectContaining({ method: 'POST' })
      );
    });
    unmount();
  });

  test('shows error when session start fails', async () => {
    global.fetch.mockResolvedValue(mockJsonResponse({ error: 'DB down' }, false, 500));

    render(<EmotionDetector />);
    const startBtn = await continueConsent();

    await act(async () => {
      fireEvent.click(startBtn);
    });

    await waitFor(() => {
      expect(screen.getByText(/Unable to start session/i)).toBeInTheDocument();
    });
  });

  test('camera denial does not create a backend session', async () => {
    mockGetUserMedia.mockRejectedValue(new DOMException('Permission denied', 'NotAllowedError'));

    render(<EmotionDetector />);
    const startBtn = await continueConsent();

    await act(async () => {
      fireEvent.click(startBtn);
    });

    expect(global.fetch).not.toHaveBeenCalledWith(
      expect.stringContaining('/session/start'),
      expect.anything()
    );
    await waitFor(() => {
      expect(screen.getByText(/Failed to access webcam/i)).toBeInTheDocument();
    });
  });
});
