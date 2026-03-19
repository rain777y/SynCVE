/**
 * Tests for the History page component.
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Mock react-router-dom
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
  BrowserRouter: ({ children }) => <div>{children}</div>,
}));

// Mock supabase
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

// Control auth context per test
let mockAuthValues;

jest.mock('../contexts/AuthContext', () => ({
  useAuth: () => mockAuthValues,
}));

import History from './History';

describe('History', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch = jest.fn();

    mockAuthValues = {
      user: { id: 'u1', email: 'test@test.com' },
      isAuthenticated: true,
      signOut: jest.fn().mockResolvedValue({}),
      loading: false,
    };
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('renders session list when sessions exist', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        sessions: [
          {
            id: 's-1',
            created_at: '2025-01-15T10:00:00Z',
            status: 'completed',
            summary: 'A happy session',
          },
          {
            id: 's-2',
            created_at: '2025-01-14T10:00:00Z',
            status: 'active',
            summary: 'An ongoing session',
          },
        ],
      }),
    });

    await act(async () => {
      render(<History />);
    });

    await waitFor(() => {
      expect(screen.getByText('A happy session')).toBeInTheDocument();
      expect(screen.getByText('An ongoing session')).toBeInTheDocument();
    });
  });

  test('renders empty state when no sessions', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ sessions: [] }),
    });

    await act(async () => {
      render(<History />);
    });

    await waitFor(() => {
      expect(screen.getByText('No sessions recorded yet.')).toBeInTheDocument();
    });
  });

  test('renders empty state with call-to-action button', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ sessions: [] }),
    });

    await act(async () => {
      render(<History />);
    });

    await waitFor(() => {
      expect(screen.getByText('Start Your First Session')).toBeInTheDocument();
    });
  });

  test('shows error when fetch fails', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: false,
      json: jest.fn().mockResolvedValue({}),
    });

    await act(async () => {
      render(<History />);
    });

    await waitFor(() => {
      expect(screen.getByText('Failed to load emotion history.')).toBeInTheDocument();
    });
  });

  test('opens session detail modal when session is clicked', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        sessions: [
          {
            id: 'abcdefgh-1234-5678-9012-abcdefghijkl',
            created_at: '2025-01-15T10:00:00Z',
            status: 'completed',
            summary: 'Detail test session',
            recommendations: 'Take breaks\nPractice mindfulness',
          },
        ],
      }),
    });

    await act(async () => {
      render(<History />);
    });

    await waitFor(() => {
      expect(screen.getByText('Detail test session')).toBeInTheDocument();
    });

    // Click the session card
    await act(async () => {
      const card = screen.getByText('Detail test session').closest('.session-card');
      await userEvent.click(card);
    });

    // Modal should appear
    await waitFor(() => {
      expect(screen.getByText('Session Report')).toBeInTheDocument();
      expect(screen.getByText('Take breaks')).toBeInTheDocument();
      expect(screen.getByText('Practice mindfulness')).toBeInTheDocument();
    });
  });

  test('redirects to home when not authenticated', async () => {
    mockAuthValues = {
      user: null,
      isAuthenticated: false,
      signOut: jest.fn(),
      loading: false,
    };

    await act(async () => {
      render(<History />);
    });

    expect(mockNavigate).toHaveBeenCalledWith('/');
  });

  test('shows loading spinner while auth is loading', () => {
    mockAuthValues = {
      user: null,
      isAuthenticated: false,
      signOut: jest.fn(),
      loading: true,
    };

    render(<History />);
    expect(document.querySelector('.spinner')).toBeTruthy();
  });

  test('renders page header', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ sessions: [] }),
    });

    await act(async () => {
      render(<History />);
    });

    expect(screen.getByText('Emotion History')).toBeInTheDocument();
  });

  test('fetches sessions with user_id param', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ sessions: [] }),
    });

    await act(async () => {
      render(<History />);
    });

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
      const fetchUrl = global.fetch.mock.calls[0][0].toString();
      expect(fetchUrl).toContain('user_id=u1');
    });
  });
});
