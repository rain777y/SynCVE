/**
 * Tests for the AuthContext provider and useAuth hook.
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Mock the supabase module
const mockGetSession = jest.fn();
const mockOnAuthStateChange = jest.fn();
const mockSignInWithOAuth = jest.fn();
const mockSignOut = jest.fn();

jest.mock('../lib/supabase', () => ({
  supabase: {
    auth: {
      getSession: (...args) => mockGetSession(...args),
      onAuthStateChange: (...args) => mockOnAuthStateChange(...args),
      signInWithOAuth: (...args) => mockSignInWithOAuth(...args),
      signOut: (...args) => mockSignOut(...args),
    },
  },
}));

import { AuthProvider, useAuth } from './AuthContext';

// Helper component that surfaces auth state for assertions
const AuthConsumer = () => {
  const { user, loading, isAuthenticated, signInWithGoogle, signOut } = useAuth();
  return (
    <div>
      <div data-testid="loading">{loading.toString()}</div>
      <div data-testid="authenticated">{isAuthenticated.toString()}</div>
      <div data-testid="user-email">{user?.email || 'none'}</div>
      <button data-testid="login-btn" onClick={signInWithGoogle}>Login</button>
      <button data-testid="logout-btn" onClick={signOut}>Logout</button>
    </div>
  );
};

describe('AuthContext', () => {
  let authChangeCallback;

  beforeEach(() => {
    jest.clearAllMocks();

    // Default: no session
    mockGetSession.mockResolvedValue({
      data: { session: null },
    });

    mockOnAuthStateChange.mockImplementation((callback) => {
      authChangeCallback = callback;
      return {
        data: { subscription: { unsubscribe: jest.fn() } },
      };
    });

    mockSignInWithOAuth.mockResolvedValue({ data: {}, error: null });
    mockSignOut.mockResolvedValue({ error: null });
  });

  test('provides loading=true initially then loading=false after session check', async () => {
    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('false');
    });
  });

  test('provides unauthenticated state when no session', async () => {
    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    await waitFor(() => {
      expect(screen.getByTestId('authenticated').textContent).toBe('false');
      expect(screen.getByTestId('user-email').textContent).toBe('none');
    });
  });

  test('provides authenticated state when session exists', async () => {
    const mockUser = { id: 'u1', email: 'test@example.com' };
    mockGetSession.mockResolvedValue({
      data: { session: { user: mockUser } },
    });

    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    await waitFor(() => {
      expect(screen.getByTestId('authenticated').textContent).toBe('true');
      expect(screen.getByTestId('user-email').textContent).toBe('test@example.com');
    });
  });

  test('login calls supabase signInWithOAuth with google provider', async () => {
    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    await act(async () => {
      await userEvent.click(screen.getByTestId('login-btn'));
    });

    expect(mockSignInWithOAuth).toHaveBeenCalledWith(
      expect.objectContaining({ provider: 'google' })
    );
  });

  test('login throws when supabase returns error', async () => {
    mockSignInWithOAuth.mockResolvedValue({
      data: null,
      error: new Error('OAuth error'),
    });

    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    // Click login - it should throw
    await expect(async () => {
      await act(async () => {
        await userEvent.click(screen.getByTestId('login-btn'));
      });
    }).rejects.toThrow('OAuth error');
  });

  test('logout calls supabase signOut', async () => {
    const mockUser = { id: 'u1', email: 'test@example.com' };
    mockGetSession.mockResolvedValue({
      data: { session: { user: mockUser } },
    });

    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    await act(async () => {
      await userEvent.click(screen.getByTestId('logout-btn'));
    });

    expect(mockSignOut).toHaveBeenCalled();
  });

  test('auth state updates when onAuthStateChange fires', async () => {
    await act(async () => {
      render(
        <AuthProvider>
          <AuthConsumer />
        </AuthProvider>
      );
    });

    // Initially unauthenticated
    await waitFor(() => {
      expect(screen.getByTestId('authenticated').textContent).toBe('false');
    });

    // Simulate auth state change (user logs in)
    await act(async () => {
      authChangeCallback('SIGNED_IN', {
        user: { id: 'u2', email: 'new@example.com' },
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('authenticated').textContent).toBe('true');
      expect(screen.getByTestId('user-email').textContent).toBe('new@example.com');
    });
  });
});
