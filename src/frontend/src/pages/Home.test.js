/**
 * Tests for the Home page component.
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

// We need to control the auth context per test
let mockAuthValues;

jest.mock('../contexts/AuthContext', () => ({
  useAuth: () => mockAuthValues,
}));

import Home from './Home';

describe('Home', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockAuthValues = {
      user: null,
      isAuthenticated: false,
      signInWithGoogle: jest.fn().mockResolvedValue({}),
      signOut: jest.fn().mockResolvedValue({}),
      loading: false,
    };
  });

  test('renders home page with brand name', () => {
    render(<Home />);
    expect(screen.getByText('SynCVE')).toBeInTheDocument();
  });

  test('renders hero section', () => {
    render(<Home />);
    expect(screen.getByText('Real-time Facial Emotion Recognition')).toBeInTheDocument();
  });

  test('renders feature items', () => {
    render(<Home />);
    expect(screen.getByText('High Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Real-time Analysis')).toBeInTheDocument();
    expect(screen.getByText('Secure & Private')).toBeInTheDocument();
  });

  test('shows loading spinner when loading', () => {
    mockAuthValues.loading = true;
    render(<Home />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  test('shows sign-in button when not authenticated', () => {
    render(<Home />);
    // The nav should have "Get Started" or "Sign in with Google"
    const signInButtons = screen.getAllByText(/Sign in with Google|Get Started/i);
    expect(signInButtons.length).toBeGreaterThan(0);
  });

  test('shows detection button when authenticated', () => {
    mockAuthValues = {
      user: { id: 'u1', email: 'test@test.com', user_metadata: { full_name: 'Test User' } },
      isAuthenticated: true,
      signInWithGoogle: jest.fn(),
      signOut: jest.fn(),
      loading: false,
    };

    render(<Home />);
    expect(screen.getByText('Detection')).toBeInTheDocument();
    expect(screen.getByText('History')).toBeInTheDocument();
    expect(screen.getByText('Start Detection')).toBeInTheDocument();
  });

  test('navigates to /detection when detection button clicked', async () => {
    mockAuthValues = {
      user: { id: 'u1', email: 'test@test.com' },
      isAuthenticated: true,
      signInWithGoogle: jest.fn(),
      signOut: jest.fn(),
      loading: false,
    };

    render(<Home />);
    const startBtn = screen.getByText('Start Detection');
    await userEvent.click(startBtn);
    expect(mockNavigate).toHaveBeenCalledWith('/detection');
  });

  test('navigates to /history when history button clicked', async () => {
    mockAuthValues = {
      user: { id: 'u1', email: 'test@test.com' },
      isAuthenticated: true,
      signInWithGoogle: jest.fn(),
      signOut: jest.fn(),
      loading: false,
    };

    render(<Home />);
    const historyBtn = screen.getByText('History');
    await userEvent.click(historyBtn);
    expect(mockNavigate).toHaveBeenCalledWith('/history');
  });

  test('calls signInWithGoogle when sign-in button clicked', async () => {
    const mockSignIn = jest.fn().mockResolvedValue({});
    mockAuthValues.signInWithGoogle = mockSignIn;

    render(<Home />);
    const signInBtn = screen.getAllByText(/Sign in with Google|Get Started/i)[0];
    await userEvent.click(signInBtn);
    expect(mockSignIn).toHaveBeenCalled();
  });

  test('displays user name when authenticated', () => {
    mockAuthValues = {
      user: { id: 'u1', email: 'test@test.com', user_metadata: { full_name: 'Jane Doe' } },
      isAuthenticated: true,
      signInWithGoogle: jest.fn(),
      signOut: jest.fn(),
      loading: false,
    };

    render(<Home />);
    expect(screen.getByText('Jane Doe')).toBeInTheDocument();
  });

  test('sign out button calls signOut', async () => {
    const mockSignOutFn = jest.fn().mockResolvedValue({});
    mockAuthValues = {
      user: { id: 'u1', email: 'test@test.com' },
      isAuthenticated: true,
      signInWithGoogle: jest.fn(),
      signOut: mockSignOutFn,
      loading: false,
    };

    render(<Home />);
    const signOutBtn = screen.getByText('Sign Out');
    await userEvent.click(signOutBtn);
    expect(mockSignOutFn).toHaveBeenCalled();
  });

  test('renders footer', () => {
    render(<Home />);
    expect(screen.getByText(/SynCVE - Emotion Recognition System/)).toBeInTheDocument();
  });

  test('renders emotion grid cards', () => {
    render(<Home />);
    const emotionCards = document.querySelectorAll('.emotion-card');
    expect(emotionCards.length).toBe(7);
  });
});
